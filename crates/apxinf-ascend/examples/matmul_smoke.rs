//! First aclnn compute op: fp16 matmul on 310P3, verified against a CPU
//! reference under fp16 tolerance.
//!
//!   source /data/apxinf/rust_env.sh
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example matmul_smoke --release

use apxinf_ascend::ops::matmul_fp16;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("context");
    let stream = AscendStream::new().expect("stream");

    // Deterministic matrices; multiples of 16 exercise tiling.
    let (m, k, n) = (64usize, 128usize, 96usize);

    let mut seed = 0x1234_5678u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 2000 - 1000) as f32 / 1000.0
    };
    let h_a: Vec<f16> = (0..m * k).map(|_| f16::from_f32(rnd())).collect();
    let h_b: Vec<f16> = (0..k * n).map(|_| f16::from_f32(rnd())).collect();

    let dev_a = ctx.malloc(h_a.len() * 2).expect("malloc a");
    let dev_b = ctx.malloc(h_b.len() * 2).expect("malloc b");
    let a_bytes = bytemuck::cast_slice(&h_a);
    let b_bytes = bytemuck::cast_slice(&h_b);
    ctx.copy_h2d(&dev_a, a_bytes).expect("h2d a");
    ctx.copy_h2d(&dev_b, b_bytes).expect("h2d b");

    let t0 = std::time::Instant::now();
    let dev_out = matmul_fp16(&ctx, &stream, &dev_a, [m as i64, k as i64], &dev_b, [k as i64, n as i64])
        .expect("matmul");
    stream.synchronize().expect("sync");
    println!("matmul {}x{}x{} done in {:?}", m, k, n, t0.elapsed());

    let mut out_bytes = vec![0u8; m * n * 2];
    ctx.copy_d2h(&dev_out, &mut out_bytes).expect("d2h");
    let got: Vec<f16> = bytemuck::cast_slice(&out_bytes).to_vec();

    // CPU reference in fp32 over the fp16-rounded inputs.
    let mut max_rel: f32 = 0.0;
    let mut worst = (0usize, 0f32, 0f32);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += h_a[i * k + p].to_f32() * h_b[p * n + j].to_f32();
            }
            let g = got[i * n + j].to_f32();
            let scale = acc.abs().max(1.0);
            let rel = (g - acc).abs() / scale;
            if rel > max_rel {
                max_rel = rel;
                worst = (i * n + j, acc, g);
            }
        }
    }
    println!(
        "max relative error vs fp32 CPU ref: {max_rel:.5} (worst idx {} ref {:.4} got {:.4})",
        worst.0, worst.1, worst.2
    );
    assert!(max_rel < 0.02, "fp16 matmul out of tolerance: {max_rel}");
    println!("MATMUL_SMOKE_OK");
}
