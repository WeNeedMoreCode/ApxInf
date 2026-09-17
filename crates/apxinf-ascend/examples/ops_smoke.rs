//! Elementwise + fused-norm op smoke: add / silu / add_rms_norm vs CPU
//! references under fp16 tolerance.
//!
//!   source /data/apxinf/rust_env.sh
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ops_smoke --release

use apxinf_ascend::ops::{add_fp16, add_rms_norm_fp16, silu_fp16};
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

    println!("OPS_SMOKE_OK");
}
