//! PFA V3 fused attention smoke: BNSD fp16 vs CPU softmax-attention
//! reference under fp16 tolerance.
//!
//!   source /data/apxinf/rust_env.sh
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example attention_smoke --release

use apxinf_ascend::ops::prompt_flash_attention_fp16;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("context");
    let stream = AscendStream::new().expect("stream");

    // [b, n, s, d] -- S multiple of 16 (310P PFA constraint), D <= 512.
    let (b, n, s, d) = (1usize, 2usize, 64usize, 32usize);

    let mut seed = 0x5eedu64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };
    let total = b * n * s * d;
    let hq: Vec<f16> = (0..total).map(|_| f16::from_f32(rnd())).collect();
    let hk: Vec<f16> = (0..total).map(|_| f16::from_f32(rnd())).collect();
    let hv: Vec<f16> = (0..total).map(|_| f16::from_f32(rnd())).collect();

    let dq = ctx.malloc(total * 2).unwrap();
    let dk = ctx.malloc(total * 2).unwrap();
    let dv = ctx.malloc(total * 2).unwrap();
    ctx.copy_h2d(&dq, bytemuck::cast_slice(&hq)).unwrap();
    ctx.copy_h2d(&dk, bytemuck::cast_slice(&hk)).unwrap();
    ctx.copy_h2d(&dv, bytemuck::cast_slice(&hv)).unwrap();

    let dout = prompt_flash_attention_fp16(
        &ctx,
        &stream,
        &dq,
        &dk,
        &dv,
        [b as i64, n as i64, s as i64, d as i64],
        None,
    )
    .expect("pfa");
    stream.synchronize().expect("sync");

    let mut back = vec![0u8; total * 2];
    ctx.copy_d2h(&dout, &mut back).unwrap();
    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();

    // CPU reference: fp32 softmax(QK^T * scale) V per (b, head).
    let scale = 1.0 / (d as f32).sqrt();
    let at = |v: &Vec<f16>, bi: usize, ni: usize, si: usize, di: usize| -> f32 {
        v[((bi * n + ni) * s + si) * d + di].to_f32()
    };
    let mut max_rel = 0f32;
    for bi in 0..b {
        for ni in 0..n {
            // scores + softmax (fp32)
            let mut probs = vec![0f32; s * s];
            for i in 0..s {
                let mut row = vec![0f32; s];
                let mut mx = f32::MIN;
                for j in 0..s {
                    let mut acc = 0f32;
                    for t in 0..d {
                        acc += at(&hq, bi, ni, i, t) * at(&hk, bi, ni, j, t);
                    }
                    row[j] = acc * scale;
                    mx = mx.max(row[j]);
                }
                let mut sum = 0f32;
                for j in 0..s {
                    row[j] = (row[j] - mx).exp();
                    sum += row[j];
                }
                for j in 0..s {
                    probs[i * s + j] = row[j] / sum;
                }
            }
            for i in 0..s {
                for t in 0..d {
                    let mut acc = 0f32;
                    for j in 0..s {
                        acc += probs[i * s + j] * at(&hv, bi, ni, j, t);
                    }
                    let g = got[((bi * n + ni) * s + i) * d + t].to_f32();
                    let rel = (g - acc).abs() / acc.abs().max(0.05);
                    max_rel = max_rel.max(rel);
                }
            }
        }
    }
    println!("pfa max rel err vs CPU fp32 softmax-attn: {max_rel:.5}");
    assert!(max_rel < 0.02, "pfa out of tolerance: {max_rel}");
    println!("ATTENTION_SMOKE_OK");
}
