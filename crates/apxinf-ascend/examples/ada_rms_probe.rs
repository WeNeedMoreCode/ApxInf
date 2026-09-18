//! Fused ada-rms-norm kernel probe: parity vs a host f32 reference
//! (cuda semantics y = rms(x)*(1+s0)+s1, scale pre-folded) and against
//! the production aclnn composition on the same inputs.
//! ADA_RMS_DIAG=1 runs the kernel bring-up passthrough probes instead
//! (diag 1 = shift lane + write-back, diag 2 = x lane).
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ada_rms_probe --release -p apxinf-ascend
use apxinf_ascend::ada_rms;
use apxinf_ascend::context::AscendContext;
use apxinf_ascend::ops;
use apxinf_ascend::stream::AscendStream;
use half::f16;

fn rand_f16(n: usize, seed: &mut u32, scale: f32) -> Vec<f16> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((*seed >> 16) as i32 % 200 - 100) as f32 / 100.0 * scale;
            f16::from_f32(v)
        })
        .collect()
}

fn upload(ctx: &AscendContext, vals: &[f16]) -> apxinf_ascend::DeviceBuffer {
    let bytes: &[u8] = bytemuck::cast_slice(vals);
    let buf = ctx.malloc(bytes.len()).expect("malloc");
    ctx.copy_h2d(&buf, bytes).expect("h2d");
    buf
}

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");
    assert!(ada_rms::available(), "kernel .so not loadable (APXINF_ADA_RMS_LIB)");

    let diag_mode = std::env::var("ADA_RMS_DIAG").is_ok();
    let shapes: Vec<(usize, usize)> = if diag_mode {
        vec![(828, 2048)]
    } else {
        vec![(828, 2048), (50, 1024), (7, 2048), (1, 1024), (64, 2048)]
    };

    let mut seed = 7u32;
    for (rows, cols) in shapes {
        for diag in [
            if diag_mode { 1 } else { 0 },
            if diag_mode { 2 } else { 0 },
            if diag_mode { 3 } else { 0 },
            if diag_mode { 6 } else { 0 },
        ] {
            let n = rows * cols;
            let x_h = rand_f16(n, &mut seed, 3.0);
            let style_h = rand_f16(3 * cols, &mut seed, 1.0);
            let scale_h: Vec<f16> = (0..cols)
                .map(|i| f16::from_f32(style_h[i].to_f32() + 1.0))
                .collect();
            let shift_h: Vec<f16> = (0..cols).map(|i| style_h[cols + i]).collect();

            let x = upload(&ctx, &x_h);
            let scale = upload(&ctx, &scale_h);
            let shift = upload(&ctx, &shift_h);
            let y = upload(&ctx, &vec![f16::from_f32(1.0); n]); // sentinel prefill

            ada_rms::run(&stream, &x, &scale, &shift, &y, rows as i32, cols as i32, 1e-6, 8, diag)
                .expect("kernel run");
            stream.synchronize().expect("sync");
            let mut back = vec![0u8; n * 2];
            ctx.copy_d2h(&y, &mut back).expect("d2h");
            let y_h: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
            let untouched = y_h.iter().filter(|&&v| v.to_f32() == 1.0).count();

            let eps = 1e-6f32;
            let mut max_diff = 0f32;
            for r in 0..rows {
                let sq: f32 = (0..cols).map(|c| x_h[r * cols + c].to_f32().powi(2)).sum();
                let rstd = 1.0 / (sq / cols as f32 + eps).sqrt();
                for c in 0..cols {
                    let want = match diag {
                        1 => 2.0 * shift_h[c].to_f32(),
                        2 => 2.0 * x_h[r * cols + c].to_f32(),
                        3 => f16::from_f32(rstd).to_f32(), // kernel casts rstd to half
                        6 => f16::from_f32(sq).to_f32(),   // kernel casts sum to half
                        _ => x_h[r * cols + c].to_f32() * rstd * scale_h[c].to_f32()
                            + shift_h[c].to_f32(),
                    };
                    max_diff = max_diff.max((want - y_h[r * cols + c].to_f32()).abs());
                }
            }
            println!(
                "[{rows}x{cols} diag={diag}] max_diff={max_diff:.5} untouched_sentinel={untouched}/{n}"
            );
            if diag == 6 {
                let row_sq: f32 = (0..cols).map(|c| x_h[c].to_f32().powi(2)).sum();
                println!(
                    "  y[0..2]={:?} want_sum={row_sq:.2}",
                    &y_h[..2].iter().map(|v| v.to_f32()).collect::<Vec<_>>()
                );
            }
            if diag == 0 {
                assert!(max_diff < 0.02, "fused kernel diverged from f32 reference: {max_diff}");
            } else {
                assert!(untouched < n / 100, "diag {diag} wrote almost nothing");
            }

            if diag == 0 {
                // production composition cross-check on the same inputs
                let zeros = upload(&ctx, &vec![f16::from_f32(0.0); n]);
                let ones = upload(&ctx, &vec![f16::from_f32(1.0); cols]);
                let (normed, _) = ops::add_rms_norm_fp16(
                    &ctx,
                    &stream,
                    &x,
                    &zeros,
                    &ones,
                    &[rows as i64, cols as i64],
                    1e-6,
                )
                .expect("add_rms_norm");
                let scale_mat = upload(&ctx, &scale_h.repeat(rows));
                let shift_mat = upload(&ctx, &shift_h.repeat(rows));
                let scaled = ops::mul_fp16(
                    &ctx,
                    &stream,
                    &normed,
                    &scale_mat,
                    &[rows as i64, cols as i64],
                )
                .expect("mul");
                let comp = ops::add_fp16(
                    &ctx,
                    &stream,
                    &scaled,
                    &shift_mat,
                    &[rows as i64, cols as i64],
                )
                .expect("add");
                stream.synchronize().expect("sync2");
                let mut back2 = vec![0u8; n * 2];
                ctx.copy_d2h(&comp, &mut back2).expect("d2h2");
                let comp_h: Vec<f16> = bytemuck::cast_slice(&back2).to_vec();
                let max_vs_comp = y_h
                    .iter()
                    .zip(&comp_h)
                    .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                    .fold(0f32, f32::max);
                println!(
                    "[{rows}x{cols} diag=0] max_diff_vs_aclnn_comp={max_vs_comp:.5}"
                );
                assert!(
                    max_vs_comp < 0.03,
                    "fused kernel diverged from aclnn composition: {max_vs_comp}"
                );
            }
        }
    }
    println!("ADA_RMS_PROBE_OK");
}
