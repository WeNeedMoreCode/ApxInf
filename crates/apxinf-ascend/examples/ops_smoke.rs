//! Elementwise + fused-norm op smoke: add / silu / add_rms_norm vs CPU
//! references under fp16 tolerance.
//!
//!   source /data/apxinf/rust_env.sh
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ops_smoke --release

use apxinf_ascend::ops::{add_fp16, add_rms_norm_fp16, bias_add_fp16, cat_fp16, euler_update_fp16, gather_rows_fp16, gelu_fp16, layer_norm_fp16, muls_fp16, row_replicate_fp16, silu_fp16};
use apxinf_ascend::AclTensor;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("context");
    let stream = AscendStream::new().expect("stream");

    let rows = 32usize;
    let cols = 256usize;
    let n = rows * cols;

    let mut seed = 0xbeefu64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 2000 - 1000) as f32 / 1000.0
    };
    let ha: Vec<f16> = (0..n).map(|_| f16::from_f32(rnd())).collect();
    let hb: Vec<f16> = (0..n).map(|_| f16::from_f32(rnd())).collect();

    let da = ctx.malloc(n * 2).unwrap();
    let db = ctx.malloc(n * 2).unwrap();
    ctx.copy_h2d(&da, bytemuck::cast_slice(&ha)).unwrap();
    ctx.copy_h2d(&db, bytemuck::cast_slice(&hb)).unwrap();

    // ---- add ----
    let dout = add_fp16(&ctx, &stream, &da, &db, &[rows as i64, cols as i64]).expect("add");
    stream.synchronize().unwrap();
    let mut back = vec![0u8; n * 2];
    ctx.copy_d2h(&dout, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
    let mut max_err = 0f32;
    for i in 0..n {
        let want = ha[i].to_f32() + hb[i].to_f32();
        max_err = max_err.max((got[i].to_f32() - want).abs());
    }
    println!("add   max abs err {max_err:.5}");
    assert!(max_err < 0.01, "add out of tolerance");

    // ---- silu ----
    let ds = silu_fp16(&ctx, &stream, &da, &[rows as i64, cols as i64]).expect("silu");
    stream.synchronize().unwrap();
    ctx.copy_d2h(&ds, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
    let mut max_rel = 0f32;
    for i in 0..n {
        let x = ha[i].to_f32();
        let want = x * (1.0 / (1.0 + (-x).exp()));
        let rel = (got[i].to_f32() - want).abs() / want.abs().max(1.0);
        max_rel = max_rel.max(rel);
    }
    println!("silu  max rel err {max_rel:.5}");
    assert!(max_rel < 0.01, "silu out of tolerance");

    // ---- add_rms_norm (fused) ----
    let hg: Vec<f16> = (0..cols).map(|_| f16::from_f32(rnd())).collect();
    let dg = ctx.malloc(cols * 2).unwrap();
    ctx.copy_h2d(&dg, bytemuck::cast_slice(&hg)).unwrap();
    let eps = 1e-6f64;
    let (dy, _rstd) =
        add_rms_norm_fp16(&ctx, &stream, &da, &db, &dg, &[rows as i64, cols as i64], eps).expect("add_rms_norm");
    stream.synchronize().unwrap();
    ctx.copy_d2h(&dy, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
    let mut max_rel = 0f32;
    for r in 0..rows {
        let mut sum = vec![0f32; cols];
        for c in 0..cols {
            sum[c] = ha[r * cols + c].to_f32() + hb[r * cols + c].to_f32();
        }
        let ms = sum.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let rms = (ms + eps as f32).sqrt();
        for c in 0..cols {
            let want = sum[c] / rms * hg[c].to_f32();
            let rel = (got[r * cols + c].to_f32() - want).abs() / want.abs().max(1.0);
            max_rel = max_rel.max(rel);
        }
    }
    println!("add_rms_norm max rel err {max_rel:.5}");
    assert!(max_rel < 0.02, "add_rms_norm out of tolerance");

    // ---- bias_add (row broadcast) ----
    let hbias: Vec<f16> = (0..cols).map(|i| f16::from_f32(0.25 * (i as f32 % 4.0))).collect();
    let dbias = ctx.malloc(cols * 2).unwrap();
    ctx.copy_h2d(&dbias, bytemuck::cast_slice(&hbias)).unwrap();
    let dbiased = bias_add_fp16(&ctx, &stream, &da, &dbias, rows as i64, cols as i64).expect("bias");
    stream.synchronize().unwrap();
    ctx.copy_d2h(&dbiased, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
    let mut max_err = 0f32;
    for r in 0..rows {
        for c in 0..cols {
            let want = ha[r * cols + c].to_f32() + hbias[c].to_f32();
            max_err = max_err.max((got[r * cols + c].to_f32() - want).abs());
        }
    }
    println!("bias  max abs err {max_err:.5}");
    assert!(max_err < 0.01, "bias out of tolerance");

    // ---- muls + euler_update: x0 + 0.5*(x1-x0) = midpoint ----
    let deuler = euler_update_fp16(&ctx, &stream, &da, &db, 0.5, &[rows as i64, cols as i64]).expect("euler");
    stream.synchronize().unwrap();
    ctx.copy_d2h(&deuler, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
    let mut max_err = 0f32;
    for i in 0..n {
        let a = ha[i].to_f32();
        let b = hb[i].to_f32();
        max_err = max_err.max((got[i].to_f32() - (a + b) / 2.0).abs());
    }
    println!("euler(midpoint) max abs err {max_err:.5}");
    assert!(max_err < 0.01, "euler out of tolerance");

    // ---- cat along dim 0 ----
    let dcat = cat_fp16(
        &ctx,
        &stream,
        &[&da, &db],
        &[vec![rows as i64, cols as i64], vec![rows as i64, cols as i64]],
        0,
        &[(2 * rows) as i64, cols as i64],
    )
    .expect("cat");
    stream.synchronize().unwrap();
    let mut back2 = vec![0u8; 2 * n * 2];
    ctx.copy_d2h(&dcat, &mut back2).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back2).to_vec();
    let mut ok = true;
    for i in 0..n {
        ok = ok && got[i].to_f32() == ha[i].to_f32() && got[n + i].to_f32() == hb[i].to_f32();
    }
    println!("cat exact: {ok}");
    assert!(ok, "cat out of order");

    // ---- gelu (tanh approximation, full cubic form) ----
    let dg = gelu_fp16(&ctx, &stream, &da, &[rows as i64, cols as i64], true).expect("gelu");
    stream.synchronize().unwrap();
    ctx.copy_d2h(&dg, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
    let mut max_rel = 0f32;
    for i in 0..n {
        let x = ha[i].to_f32();
        let u = 0.7978845608 * (x + 0.044715 * x * x * x);
        let want = 0.5 * x * (1.0 + u.tanh());
        let rel = (got[i].to_f32() - want).abs() / want.abs().max(0.1);
        max_rel = max_rel.max(rel);
    }
    println!("gelu  max rel err {max_rel:.5}");
    assert!(max_rel < 0.01, "gelu out of tolerance");

    // ---- gather (embedding lookup) ----
    let vocab = 8i64;
    let dim = 4i64;
    let table: Vec<f16> = (0..(vocab * dim) as usize).map(|i| f16::from_f32(i as f32 * 0.5)).collect();
    let dtable = ctx.malloc((vocab * dim * 2) as usize).unwrap();
    ctx.copy_h2d(&dtable, bytemuck::cast_slice(&table)).unwrap();
    let idx: Vec<i32> = vec![3, 0, 7];
    let didx = ctx.malloc(12).unwrap();
    ctx.copy_h2d(&didx, unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u8, 12) }).unwrap();
    let dgot = gather_rows_fp16(&ctx, &stream, &dtable, vocab, dim, &didx, 3).expect("gather");
    stream.synchronize().unwrap();
    let mut gback = vec![0u8; 24];
    ctx.copy_d2h(&dgot, &mut gback).unwrap();
    let rows_out: Vec<f16> = bytemuck::cast_slice(&gback).to_vec();
    let mut ok = true;
    for (r, &ix) in idx.iter().enumerate() {
        for c in 0..dim as usize {
            let want = table[ix as usize * dim as usize + c].to_f32();
            ok = ok && rows_out[r * dim as usize + c].to_f32() == want;
        }
    }
    println!("gather(embedding) exact: {ok}");
    assert!(ok, "gather wrong rows");

    // ---- layer_norm (fused AddLayerNorm + zeros) ----
    let zeros: Vec<u8> = vec![0; n * 2];
    let dz = ctx.malloc(n * 2).unwrap();
    ctx.copy_h2d(&dz, &zeros).unwrap();
    let hgam: Vec<f16> = (0..cols).map(|i| f16::from_f32(1.0 + 0.01 * i as f32)).collect();
    let hbet: Vec<f16> = (0..cols).map(|i| f16::from_f32(0.001 * i as f32)).collect();
    let dgam = ctx.malloc(cols * 2).unwrap();
    let dbet = ctx.malloc(cols * 2).unwrap();
    ctx.copy_h2d(&dgam, bytemuck::cast_slice(&hgam)).unwrap();
    ctx.copy_h2d(&dbet, bytemuck::cast_slice(&hbet)).unwrap();
    let dln = layer_norm_fp16(&ctx, &stream, &da, &dz, &dgam, &dbet, rows as i64, cols as i64, 1e-6).expect("layer_norm");
    stream.synchronize().unwrap();
    ctx.copy_d2h(&dln, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
    let mut max_err = 0f32;
    for r in 0..rows {
        let row: Vec<f32> = (0..cols).map(|c| ha[r * cols + c].to_f32()).collect();
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        for c in 0..cols {
            let want = (row[c] - mean) / (var + 1e-6).sqrt() * hgam[c].to_f32() + hbet[c].to_f32();
            max_err = max_err.max((got[r * cols + c].to_f32() - want).abs());
        }
    }
    println!("layer_norm max abs err {max_err:.5}");
    assert!(max_err < 0.02, "layer_norm out of tolerance");

    // ---- row_replicate (gather broadcast substitute) ----
    let drep = row_replicate_fp16(&ctx, &stream, &dbias, rows as i64, cols as i64).expect("replicate");
    stream.synchronize().unwrap();
    ctx.copy_d2h(&drep, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
    let ok = (0..rows).all(|r| (0..cols).all(|c| got[r * cols + c].to_f32() == hbias[c].to_f32()));
    println!("row_replicate exact: {ok}");
    assert!(ok);

    // ---- take_rows (D2D split, QKV 用) ----
    let dsplit = apxinf_ascend::ops::take_rows_fp16(&ctx, &stream, &da, (rows / 2) as i64, (rows / 2) as i64, cols as i64).expect("take_rows");
    stream.synchronize().unwrap();
    let mut sback2 = vec![0u8; (rows / 2) * cols * 2];
    ctx.copy_d2h(&dsplit, &mut sback2).unwrap();
    let srow: Vec<f16> = bytemuck::cast_slice(&sback2).to_vec();
    let sok = (0..(rows / 2) * cols).all(|i| srow[i].to_f32() == ha[(rows / 2) * cols + i].to_f32());
    println!("take_rows exact: {sok}");
    assert!(sok);

    println!("OPS_SMOKE_OK");
}
