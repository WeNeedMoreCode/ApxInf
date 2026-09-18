//! Matmul per-task fixed-cost isolation: capture N back-to-back
//! matmul_b_t reps into one graph, replay, time per matmul at several m.
// time = a*m + b -> b is the per-task "start tax" with NO profiler
// attached (the msprof-measured 334us gap needs an independent check).
//!   ASCEND_RT_VISIBLE_DEVICES=5 PROBE_SWEEP=1 cargo run --example matmul_entry_probe --release -p apxinf-ascend
use apxinf_ascend::ops;
use apxinf_core::Backend as _;
use half::f16;

fn main() {
    let be = apxinf_ascend::AscendBackend::new(0).expect("be");
    let ctx = be.ctx();
    let stream = be.stream();
    let (k, n) = (2048i64, 32768i64);

    let mut seed = 7u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        f16::from_f32(((*&seed >> 16) as i32 % 200 - 100) as f32 / 400.0)
    };
    let b_h: Vec<f16> = (0..n * k).map(|_| next()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&b_h);
    let b_t = ctx.malloc(bytes.len()).expect("b malloc");
    ctx.copy_h2d(&b_t, bytes).expect("b h2d");
    drop(stream.synchronize());

    let ms: Vec<i64> = if std::env::var("PROBE_SWEEP").is_ok() {
        vec![64, 128, 256, 512, 832]
    } else {
        vec![64]
    };

    let mut samples: Vec<(i64, f64)> = Vec::new();
    for m in ms {
        let a_h: Vec<f16> = (0..m * k).map(|_| next()).collect();
        let abytes: &[u8] = bytemuck::cast_slice(&a_h);
        let a = ctx.malloc(abytes.len()).expect("a malloc");
        ctx.copy_h2d(&a, abytes).expect("a h2d");
        drop(stream.synchronize());
        apxinf_ascend::flush_pending_frees();

        // warm once, then capture reps into one graph
        drop(ops::matmul_b_t_fp16(ctx, stream, &a, [m, k], &b_t, k, n).expect("warm"));
        drop(stream.synchronize());
        apxinf_ascend::flush_pending_frees();

        let reps: usize = if m >= 512 { 24 } else { 100 };
        let arena = be.ctx().enter_arena(10 << 30).expect("arena");
        be.begin_capture().expect("begin");
        for _ in 0..reps {
            drop(ops::matmul_b_t_fp16(ctx, stream, &a, [m, k], &b_t, k, n).expect("mm"));
        }
        let graph = be.end_capture().expect("end");
        drop(be.ctx().exit_arena());
        be.ctx().clear_arena();
        let mut ts = Vec::new();
        for i in 0..10 {
            let t0 = std::time::Instant::now();
            graph.replay().expect("replay");
            be.synchronize().expect("sync");
            if i >= 2 {
                ts.push(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64);
            }
        }
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = ts[ts.len() / 2];
        println!("m={m:4}: {med:.4} ms/matmul (median of {} reps x{} replays)", reps, ts.len());
        samples.push((m, med));
        drop(graph);
        drop(arena);
        drop(stream.synchronize());
        apxinf_ascend::flush_pending_frees();
    }

    // least-squares fit time = a*m + b
    if samples.len() >= 3 {
        let np = samples.len() as f64;
        let sx: f64 = samples.iter().map(|(m, _)| *m as f64).sum();
        let sy: f64 = samples.iter().map(|(_, t)| *t).sum();
        let sxx: f64 = samples.iter().map(|(m, _)| (*m as f64).powi(2)).sum();
        let sxy: f64 = samples.iter().map(|(m, t)| *m as f64 * t).sum();
        let a = (np * sxy - sx * sy) / (np * sxx - sx * sx);
        let b = (sy - a * sx) / np;
        println!("FIT: {a:.6} ms/row + {b:.4} ms/task  (b*1000 = {:.0}us per-task fixed cost)", b * 1000.0);
    }
    println!("MATMUL_ENTRY_PROBE_DONE");
}
