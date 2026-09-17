//! Large-matmul layout probe: the failing production shape
//! [8,2048] x [2048,2560], plain b vs transposed-b descriptor.
//! Isolates the MTE-OOB trigger to the mat2 descriptor's stride.
use apxinf_ascend::ops;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

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
    println!("PROBE_DONE");
}
