//! Large-matmul layout probe: the failing production shape
//! [8,2048] x [2048,2560], plain b vs transposed-b descriptor.
//! Isolates the MTE-OOB trigger to the mat2 descriptor's stride.
use apxinf_ascend::ops;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn half_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if ((bits >> 23) & 0xff) == 0 || exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mm = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let mut out = ((exp as u16) << 10) | mm;
    if rem > 0x1000 || (rem == 0x1000 && (mm & 1) == 1) {
        out += 1;
    }
    sign | out
}

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");
    let (m, k, n) = (8i64, 2048i64, 2560i64);
    let elems = (m * k) as usize;
    let belems = (k * n) as usize;

    let mut seed = 0xabcdu64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };
    let ha: Vec<f16> = (0..elems).map(|_| f16::from_f32(rnd())).collect();
    let hb: Vec<f16> = (0..belems).map(|_| f16::from_f32(rnd())).collect(); // [k, n] row-major

    let da = ctx.malloc(elems * 2).unwrap();
    let db = ctx.malloc(belems * 2).unwrap();
    ctx.copy_h2d(&da, bytemuck::cast_slice(&ha)).unwrap();
    ctx.copy_h2d(&db, bytemuck::cast_slice(&hb)).unwrap();

    // CPU reference
    let mut ref_out = vec![0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut acc = 0f32;
            for p in 0..k as usize {
                acc += ha[i * k as usize + p].to_f32() * hb[p * n as usize + j].to_f32();
            }
            ref_out[i * n as usize + j] = acc;
        }
    }

    // Variant 0: AddRmsNorm BEFORE the plain matmul -- does the norm
    // corrupt the process state for the following big matmul?
    println!("-- add_rms_norm then plain b --");
    let zeros = ctx.malloc((m * k * 2) as usize).unwrap();
    ctx.copy_h2d(&zeros, &vec![0u8; (m * k * 2) as usize]).unwrap();
    let gamma = ctx.malloc((k * 2) as usize).unwrap();
    ctx.copy_h2d(&gamma, bytemuck::cast_slice(&vec![f16::from_f32(1.0); k as usize])).unwrap();
    let (normed, _rstd) = ops::add_rms_norm_fp16(&ctx, &stream, &da, &zeros, &gamma, &[m, k], 1e-6).unwrap();
    stream.synchronize().unwrap();
    println!("rms ok");

    // Variant 1: plain [k, n] row-major b (known to MTE-crash at this size)
    println!("-- plain b [k,n] row-major --");
    match ops::matmul_fp16(&ctx, &stream, &normed, [m, k], &db, [k, n]) {
        Ok(out) => {
            stream.synchronize().unwrap();
            let mut back = vec![0u8; (m * n * 2) as usize];
            ctx.copy_d2h(&out, &mut back).unwrap();
            let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
            let mut mr = 0f32;
            for idx in 0..got.len() {
                mr = mr.max((got[idx].to_f32() - ref_out[idx]).abs() / ref_out[idx].abs().max(1.0));
            }
            println!("plain OK, max rel {mr:.5}");
        }
        Err(e) => println!("plain ERR {e:?}"),
    }

    // Variant 2: host-transposed b + [k,n] view with transpose strides
    println!("-- transposed b (torch w.t() layout) --");
    let hb_t = ops::host_transpose(bytemuck::cast_slice(&hb), k, n); // [n, k]
    let dbt = ctx.malloc(hb_t.len()).unwrap();
    ctx.copy_h2d(&dbt, &hb_t).unwrap();
    match ops::matmul_b_t_fp16(&ctx, &stream, &da, [m, k], &dbt, k, n) {
        Ok(out) => {
            stream.synchronize().unwrap();
            let mut back = vec![0u8; (m * n * 2) as usize];
            ctx.copy_d2h(&out, &mut back).unwrap();
            let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
            let mut mr = 0f32;
            let mut bad = 0usize;
            for idx in 0..got.len() {
                let rel = (got[idx].to_f32() - ref_out[idx]).abs() / ref_out[idx].abs().max(1.0);
                if rel > 0.05 {
                    bad += 1;
                }
                mr = mr.max(rel);
            }
            println!("t-b OK, max rel {mr:.5} bad>{bad}");
        }
        Err(e) => println!("t-b ERR {e:?}"),
    }
    // Variant 3: K ladder on the transposed path -- the layer's down proj
    // (K=inter=8192) crashed while K=2048 matmuls passed. Find the cliff.
    println!("-- K ladder (transposed b) --");
    for kk in [4096i64, 8192i64, 16384i64] {
        let n2 = 2048i64;
        let hb2: Vec<f16> = (0..(kk * n2) as usize).map(|_| f16::from_f32(rnd())).collect();
        let db2 = ctx.malloc(hb2.len() * 2).unwrap();
        ctx.copy_h2d(&db2, bytemuck::cast_slice(&hb2)).unwrap();
        let ht = ops::host_transpose(bytemuck::cast_slice(&hb2), kk, n2);
        let dbt2 = ctx.malloc(ht.len()).unwrap();
        ctx.copy_h2d(&dbt2, &ht).unwrap();
        // a: [m, kk]
        let ha2: Vec<f16> = (0..(m * kk) as usize).map(|_| f16::from_f32(rnd())).collect();
        let da2 = ctx.malloc(ha2.len() * 2).unwrap();
        ctx.copy_h2d(&da2, bytemuck::cast_slice(&ha2)).unwrap();
        match ops::matmul_b_t_fp16(&ctx, &stream, &da2, [m, kk], &dbt2, kk, n2) {
            Ok(_) => {
                let _ = stream.synchronize();
                println!("K={kk} OK");
            }
            Err(e) => println!("K={kk} ERR {e:?}"),
        }
    }
    // Variant 4: replicate the smoke sequence's prime suspect --
    // rms -> PFA (BSH, smoke's exact head geometry, tokens=8) ->
    // down-matmul (K=8192). PFA "succeeds" in smoke but may poison
    // later matmuls.
    println!("-- rms -> PFA(BSH) -> down matmul --");
    {
        let heads = 8i64;
        let kv_heads = 1i64;
        let hd = 256i64;
        let qd = heads * hd;
        let kvd = kv_heads * hd;
        let hq2: Vec<f16> = (0..(m * qd) as usize).map(|_| f16::from_f32(rnd())).collect();
        let hk2: Vec<f16> = (0..(m * kvd) as usize).map(|_| f16::from_f32(rnd())).collect();
        let hv2: Vec<f16> = (0..(m * kvd) as usize).map(|_| f16::from_f32(rnd())).collect();
        let dq2 = ctx.malloc(hq2.len() * 2).unwrap();
        let dk2 = ctx.malloc(hk2.len() * 2).unwrap();
        let dv2 = ctx.malloc(hv2.len() * 2).unwrap();
        ctx.copy_h2d(&dq2, bytemuck::cast_slice(&hq2)).unwrap();
        ctx.copy_h2d(&dk2, bytemuck::cast_slice(&hk2)).unwrap();
        ctx.copy_h2d(&dv2, bytemuck::cast_slice(&hv2)).unwrap();
        let attn = ops::prompt_flash_attention_bsh_fp16(&ctx, &stream, &dq2, &dk2, &dv2, m, heads, kv_heads, hd, None);
        match attn {
            Ok(a) => {
                let _ = stream.synchronize();
                println!("PFA ok (len {})", a.len());
                // now the down matmul on its output
                let hb3: Vec<f16> = (0..(8192 * 2048) as usize).map(|_| f16::from_f32(rnd())).collect();
                let db3 = ctx.malloc(hb3.len() * 2).unwrap();
                ctx.copy_h2d(&db3, bytemuck::cast_slice(&hb3)).unwrap();
                let ht3 = ops::host_transpose(bytemuck::cast_slice(&hb3), 8192, 2048);
                let dbt3 = ctx.malloc(ht3.len()).unwrap();
                ctx.copy_h2d(&dbt3, &ht3).unwrap();
                let ha3: Vec<f16> = (0..(m * 8192) as usize).map(|_| f16::from_f32(rnd())).collect();
                let da3 = ctx.malloc(ha3.len() * 2).unwrap();
                ctx.copy_h2d(&da3, bytemuck::cast_slice(&ha3)).unwrap();
                match ops::matmul_b_t_fp16(&ctx, &stream, &da3, [m, 8192], &dbt3, 8192, 2048) {
                    Ok(_) => {
                        let _ = stream.synchronize();
                        println!("post-PFA down matmul OK");
                    }
                    Err(e) => println!("post-PFA down matmul ERR {e:?}"),
                }
            }
            Err(e) => println!("PFA ERR {e:?}"),
        }
    }
    // Variant 5: full layer-sequence replication, segment by segment,
    // in the smoke's exact op order. The last println printed before
    // the fault names the poisoning segment.
    println!("-- full layer sequence replication --");
    {
        use apxinf_ascend::ops as o;
        let (heads, kv_heads, hd) = (8i64, 1i64, 256i64);
        let qd = heads * hd;
        let kvd = kv_heads * hd;
        let inter = 8192i64;
        let mkbuf = |len: usize| ctx.malloc(len).unwrap();
        let rndbuf = |len: usize| -> apxinf_ascend::DeviceBuffer {
            let h: Vec<f16> = (0..len / 2).map(|_| f16::from_f32(rnd())).collect();
            let d = ctx.malloc(len).unwrap();
            ctx.copy_h2d(&d, bytemuck::cast_slice(&h)).unwrap();
            d
        };
        // seg 0: zeros cache
        let zeros = mkbuf((m * k * 2) as usize);
        ctx.copy_h2d(&zeros, &vec![0u8; (m * k * 2) as usize]).unwrap();
        println!("seg0 zeros");
        // seg 1: rms
        let gamma = rndbuf((k * 2) as usize);
        let (normed, _) = o::add_rms_norm_fp16(&ctx, &stream, &da, &zeros, &gamma, &[m, k], 1e-6).unwrap();
        stream.synchronize().unwrap();
        println!("seg1 rms");
        // seg 2: transposed weight cache for qkv (d2h -> transpose -> h2d)
        let qkv_w = rndbuf((k * (qd + 2 * kvd)) as usize * 2);
        let mut host = vec![0u8; qkv_w.len()];
        ctx.copy_d2h(&qkv_w, &mut host).unwrap();
        let t = o::host_transpose(&host, k, qd + 2 * kvd);
        let qkv_wt = mkbuf(t.len());
        ctx.copy_h2d(&qkv_wt, &t).unwrap();
        stream.synchronize().unwrap();
        println!("seg2 nzcache-style transpose");
        // seg 3: qkv matmul (transposed)
        let qkv = o::matmul_b_t_fp16(&ctx, &stream, &normed, [m, k], &qkv_wt, k, qd + 2 * kvd).unwrap();
        stream.synchronize().unwrap();
        println!("seg3 qkv matmul");
        // seg 4: take_rows split
        let q = o::take_rows_fp16(&ctx, &stream, &qkv, 0, m, qd).unwrap();
        let kk = o::take_rows_fp16(&ctx, &stream, &qkv, m, m, kvd).unwrap();
        let vv = o::take_rows_fp16(&ctx, &stream, &qkv, m * 2, m, kvd).unwrap();
        stream.synchronize().unwrap();
        println!("seg4 take_rows");
        // seg 5: kv_bias round trips
        let bias_full = rndbuf(((qd + 2 * kvd) * 2) as usize);
        let mut bh = vec![0u8; bias_full.len()];
        ctx.copy_d2h(&bias_full, &mut bh).unwrap();
        let kb = mkbuf((kvd * 2) as usize);
        ctx.copy_h2d(&kb, &bh[(qd * 2) as usize..((qd + kvd) * 2) as usize]).unwrap();
        let vb = mkbuf((kvd * 2) as usize);
        ctx.copy_h2d(&vb, &bh[((qd + kvd) * 2) as usize..]).unwrap();
        stream.synchronize().unwrap();
        println!("seg5 kv_bias roundtrip");
        // seg 6: rope for q (pos-gather + channel gather + mul + mul + add)
        let rot_idx: Vec<i32> = (0..hd as usize).map(|i| if i < 128 { i + 128 } else { i - 128 } as i32).collect();
        let dridx = mkbuf(hd as usize * 4);
        ctx.copy_h2d(&dridx, unsafe { std::slice::from_raw_parts(rot_idx.as_ptr() as *const u8, hd as usize * 4) }).unwrap();
        let mut cos_t = vec![0u16; (m * qd) as usize];
        let mut sin_t = vec![0u16; (m * qd) as usize];
        for pos in 0..m as usize {
            for h in 0..heads as usize {
                for i in 0..hd as usize / 2 {
                    let f = pos as f32 / 10000f32.powf(i as f32 * 2.0 / hd as f32);
                    let (c, s) = (f.cos(), f.sin());
                    let base = (pos * heads as usize + h) * hd as usize;
                    cos_t[base + i] = half_f16(c);
                    cos_t[base + i + 128] = half_f16(c);
                    sin_t[base + i] = half_f16(-s);
                    sin_t[base + i + 128] = half_f16(s);
                }
            }
        }
        let dc = mkbuf(cos_t.len() * 2);
        let ds = mkbuf(sin_t.len() * 2);
        ctx.copy_h2d(&dc, unsafe { std::slice::from_raw_parts(cos_t.as_ptr() as *const u8, cos_t.len() * 2) }).unwrap();
        ctx.copy_h2d(&ds, unsafe { std::slice::from_raw_parts(sin_t.as_ptr() as *const u8, sin_t.len() * 2) }).unwrap();
        let qb = o::bias_add_fp16(&ctx, &stream, &q, &bias_full, m, qd).unwrap(); // (full-width bias would be wrong numerically; irrelevant for the fault)
        let _ = qb;
        let qr = o::rope_rotate_half_fp16(&ctx, &stream, &q, m * heads, hd, &dc, &ds, &dridx).unwrap();
        let kr = o::rope_rotate_half_fp16(&ctx, &stream, &kk, m * kv_heads, hd, &dc, &ds, &dridx).unwrap();
        stream.synchronize().unwrap();
        println!("seg6 rope");
        // seg 7: PFA
        let attn = o::prompt_flash_attention_bsh_fp16(&ctx, &stream, &qr, &kr, &vv, m, heads, kv_heads, hd, None).unwrap();
        stream.synchronize().unwrap();
        println!("seg7 pfa (len {})", attn.len());
        // seg 8: output proj + residual + rms
        let ow = rndbuf((qd * k) as usize * 2);
        let mut oh = vec![0u8; ow.len()];
        ctx.copy_d2h(&ow, &mut oh).unwrap();
        let ot = o::host_transpose(&oh, qd, k);
        let owt = mkbuf(ot.len());
        ctx.copy_h2d(&owt, &ot).unwrap();
        let proj = o::matmul_b_t_fp16(&ctx, &stream, &attn, [m, qd], &owt, qd, k).unwrap();
        let res = o::add_fp16(&ctx, &stream, &proj, &normed, &[m, k]).unwrap();
        let (norm2, _) = o::add_rms_norm_fp16(&ctx, &stream, &res, &zeros, &gamma, &[m, k], 1e-6).unwrap();
        stream.synchronize().unwrap();
        println!("seg8 output proj + rms");
        // seg 9: gate_up matmul + split + gelu + mul
        let gw = rndbuf((k * (inter * 2)) as usize * 2);
        let mut gh = vec![0u8; gw.len()];
        ctx.copy_d2h(&gw, &mut gh).unwrap();
        let gt = o::host_transpose(&gh, k, inter * 2);
        let gwt = mkbuf(gt.len());
        ctx.copy_h2d(&gwt, &gt).unwrap();
        let gate_up = o::matmul_b_t_fp16(&ctx, &stream, &norm2, [m, k], &gwt, k, inter * 2).unwrap();
        let up = o::take_rows_fp16(&ctx, &stream, &gate_up, 0, m, inter).unwrap();
        let gate = o::take_rows_fp16(&ctx, &stream, &gate_up, m, m, inter).unwrap();
        let gg = o::gelu_fp16(&ctx, &stream, &gate, &[m, inter], true).unwrap();
        let act = o::mul_fp16(&ctx, &stream, &up, &gg, &[m, inter]).unwrap();
        stream.synchronize().unwrap();
        println!("seg9 gate_up + geglu");
        // seg 10: DOWN MATMUL (the smoke's op #25)
        let dw = rndbuf((inter * k) as usize * 2);
        let mut dh2 = vec![0u8; dw.len()];
        ctx.copy_d2h(&dw, &mut dh2).unwrap();
        let dtt = o::host_transpose(&dh2, inter, k);
        let dwt = mkbuf(dtt.len());
        ctx.copy_h2d(&dwt, &dtt).unwrap();
        match o::matmul_b_t_fp16(&ctx, &stream, &act, [m, inter], &dwt, inter, k) {
            Ok(out) => {
                let _ = out;
                let _ = stream.synchronize();
                println!("seg10 DOWN MATMUL OK -- sequence replication passes fully");
            }
            Err(e) => println!("seg10 DOWN MATMUL ERR {e:?}"),
        }
    }
    println!("PROBE_DONE");
}
