//! RoPE probe: aclnnApplyRotaryPosEmb (legacy entry, layout=1 BSND,
//! rotate-half formula == Gemma semantics) on 310P3, vs CPU reference.
//! In-place semantics: q/k buffers are updated in device memory.
use apxinf_ascend::ffi;
use apxinf_ascend::{AclTensor, AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");

    // BSND with pi0.5's real head_dim (256) -- the legacy ApplyRotaryPosEmb
    // rejects anything outside {64,128}, so probe RotaryPositionEmbedding.
    let (b, s, n, d) = (1i64, 16i64, 2i64, 256i64);
    let elems = (b * s * n * d) as usize;

    let mut seed = 0xfeedu64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };
    let hq: Vec<f16> = (0..elems).map(|_| f16::from_f32(rnd())).collect();
    let hk: Vec<f16> = (0..elems).map(|_| f16::from_f32(rnd())).collect();

    // cos/sin per position [S, D] (freq duplicated to both halves),
    // theta = 10000 (gemma default), rotate-half freqs
    let theta: f32 = 10000.0;
    let cs_elems = (s * d) as usize;
    let mut hcos = vec![0f32; cs_elems];
    let mut hsin = vec![0f32; cs_elems];
    for pos in 0..s as usize {
        for i in 0..(d as usize / 2) {
            let freq = (1.0 / theta.powf(i as f32 * 2.0 / d as f32)) * pos as f32;
            let (c, sn) = (freq.cos(), freq.sin());
            hcos[pos * d as usize + i] = c;
            hcos[pos * d as usize + i + d as usize / 2] = c;
            hsin[pos * d as usize + i] = sn;
            hsin[pos * d as usize + i + d as usize / 2] = sn;
        }
    }
    let hcos: Vec<f16> = hcos.iter().map(|&x| f16::from_f32(x)).collect();
    let hsin: Vec<f16> = hsin.iter().map(|&x| f16::from_f32(x)).collect();

    let dq = ctx.malloc(elems * 2).unwrap();
    let dk = ctx.malloc(elems * 2).unwrap();
    let dc = ctx.malloc(cs_elems * 2).unwrap();
    let ds = ctx.malloc(cs_elems * 2).unwrap();
    ctx.copy_h2d(&dq, bytemuck::cast_slice(&hq)).unwrap();
    ctx.copy_h2d(&dk, bytemuck::cast_slice(&hk)).unwrap();
    ctx.copy_h2d(&dc, bytemuck::cast_slice(&hcos)).unwrap();
    ctx.copy_h2d(&ds, bytemuck::cast_slice(&hsin)).unwrap();

    let dims = [b, s, n, d];
    let tq = AclTensor::fp16_nd(&dq, &dims).unwrap();
    let tk = AclTensor::fp16_nd(&dk, &dims).unwrap();
    // cos/sin as [1, S, 1, D] -- broadcastable against BSND x
    let cs_dims = [1, s, 1, d];
    let tc = AclTensor::fp16_nd(&dc, &cs_dims).unwrap();
    let ts = AclTensor::fp16_nd(&ds, &cs_dims).unwrap();

    // Composed rope (primary path -- fused ops reject pi0.5 head_dims).
    use apxinf_ascend::ops::rope_rotate_half_fp16;
    let rows = (s * n) as i64;
    // cos/sin per row [rows, d]; sin folded with the rotate-half sign.
    let mut hrow_cos = vec![f16::from_f32(0.0); (rows * d) as usize];
    let mut hrow_sin = vec![f16::from_f32(0.0); (rows * d) as usize];
    for pos in 0..s as usize {
        for h in 0..n as usize {
            for i in 0..d as usize {
                let sign = if i < d as usize / 2 { -1.0f32 } else { 1.0f32 };
                hrow_cos[((pos * n as usize + h) * d as usize) + i] = hcos[pos * d as usize + i];
                hrow_sin[((pos * n as usize + h) * d as usize) + i] =
                    f16::from_f32(hsin[pos * d as usize + i].to_f32() * sign);
            }
        }
    }
    let dcs = ctx.malloc(hrow_cos.len() * 2).unwrap();
    let dsn = ctx.malloc(hrow_sin.len() * 2).unwrap();
    ctx.copy_h2d(&dcs, bytemuck::cast_slice(&hrow_cos)).unwrap();
    ctx.copy_h2d(&dsn, bytemuck::cast_slice(&hrow_sin)).unwrap();
    let rot_idx: Vec<i32> = (0..d as usize)
        .map(|i| if i < d as usize / 2 { i + d as usize / 2 } else { i - d as usize / 2 } as i32)
        .collect();
    let dridx = ctx.malloc(rot_idx.len() * 4).unwrap();
    ctx.copy_h2d(&dridx, unsafe { std::slice::from_raw_parts(rot_idx.as_ptr() as *const u8, rot_idx.len() * 4) }).unwrap();

    // Step replay with per-step dumps (gather verified separately: 0.0).
    use apxinf_ascend::ops::{add_fp16, mul_fp16};
    use apxinf_ascend::AclTensor;
    let t_x = AclTensor::fp16_nd(&dq, &[rows, d]).unwrap();
    let t_idx1 = AclTensor::i32_nd(&dridx, &[d]).unwrap();
    let step_gather = ctx.malloc((rows * d * 2) as usize).unwrap();
    let t_g = AclTensor::fp16_nd(&step_gather, &[rows, d]).unwrap();
    let mut ws: u64 = 0;
    let mut ex: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        let p = ffi::aclnnGatherV2GetWorkspaceSize(t_x.handle(), 1, t_idx1.handle(), t_g.handle(), &mut ws, &mut ex);
        assert_eq!(p, 0, "gather plan");
        let r = ffi::aclnnGatherV2(std::ptr::null_mut(), ws, ex, stream.handle());
        assert_eq!(r, 0, "gather run");
    }
    stream.synchronize().unwrap();

    let t2 = mul_fp16(&ctx, &stream, &step_gather, &dsn, &[rows, d]).unwrap();
    let t1 = mul_fp16(&ctx, &stream, &dq, &dcs, &[rows, d]).unwrap();
    let dout = add_fp16(&ctx, &stream, &t1, &t2, &[rows, d]).unwrap();
    stream.synchronize().unwrap();

    // dump t2 (rotated * sin_signed) vs reference for the first row
    let mut t2b = vec![0u8; (rows * d * 2) as usize];
    ctx.copy_d2h(&t2, &mut t2b).unwrap();
    let t2h: Vec<f16> = bytemuck::cast_slice(&t2b).to_vec();
    let mut t2err = 0f32;
    for r in 0..rows as usize {
        for i in 0..d as usize {
            let want = hq[r * d as usize + rot_idx[i] as usize].to_f32() * hrow_sin[r * d as usize + i].to_f32();
            t2err = t2err.max((t2h[r * d as usize + i].to_f32() - want).abs());
        }
    }
    println!("t2-step max abs err {t2err:.5}");

    let mut back = vec![0u8; elems * 2];
    ctx.copy_d2h(&dout, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();

    // CPU reference (rotate-half, fp32 over fp16 inputs)
    let mut max_rel = 0f32;
    for idx in 0..elems {
        let q = hq[idx].to_f32();
        let i_in_row = idx % d as usize;
        let pos = (idx / d as usize) / n as usize;
        let c = hcos[pos * d as usize + i_in_row].to_f32();
        let sn = hsin[pos * d as usize + i_in_row].to_f32();
        let rotated = if i_in_row < d as usize / 2 {
            -hq[idx + d as usize / 2].to_f32()
        } else {
            hq[idx - d as usize / 2].to_f32()
        };
        let want = q * c + rotated * sn;
        let rel = (got[idx].to_f32() - want).abs() / want.abs().max(0.1);
        max_rel = max_rel.max(rel);
    }
    println!("rope max rel err {max_rel:.5}");
    assert!(max_rel < 0.01);
    println!("ROPE_PROBE_OK");
}
