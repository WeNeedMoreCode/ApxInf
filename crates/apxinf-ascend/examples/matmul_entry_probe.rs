//! Matmul entry-point bake-off under graph replay: msprof shows a 334us
//! average gap BEFORE each MatMulCommon task (324ms of the 680ms chain) --
//! if aclnnMm/aclnnGemm tasks start faster, swapping the entry point is
//! the biggest remaining lever. 100 repeats captured into one graph so
//! replay timing is pure device-side.
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example matmul_entry_probe --release -p apxinf-ascend
use apxinf_ascend::ops;
use apxinf_core::Backend as _;
use half::f16;

fn main() {
    let be = apxinf_ascend::AscendBackend::new(0).expect("be");
    let ctx = be.ctx();
    let stream = be.stream();

    // prefix gate_up shape family; m=64 first to smoke the entries
    let (m, k, n) = match std::env::var("PROBE_BIG").is_ok() {
        true => (832i64, 2048i64, 32768i64),
        false => (64i64, 2048i64, 32768i64),
    };
    let mut seed = 7u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        f16::from_f32(((*seedref(&mut seed) >> 16) as i32 % 200 - 100) as f32 / 400.0)
    };
    let a_h: Vec<f16> = (0..m * k).map(|_| rnd()).collect();
    let b_h: Vec<f16> = (0..n * k).map(|_| rnd()).collect();
    let upload = |vals: &[f16]| -> apxinf_ascend::DeviceBuffer {
        let bytes: &[u8] = bytemuck::cast_slice(vals);
        let buf = ctx.malloc(bytes.len()).expect("malloc");
        ctx.copy_h2d(&buf, bytes).expect("h2d");
        buf
    };
    let a = upload(&a_h);
    let b_t = upload(&b_h);
    let _ = stream.synchronize();

    let reps: usize = if std::env::var("PROBE_BIG").is_ok() { 16 } else { 100 };

    let entries: [(&str, Box<dyn Fn() -> apxinf_ascend::DeviceBuffer>); 3] = [
        (
            "matmul_b_t (MatMulCommon, production)",
            Box::new({
                let ctx = &ctx;
                let stream = &stream;
                let a = &a;
                let b_t = &b_t;
                move || ops::matmul_b_t_fp16(ctx, stream, a, [m, k], b_t, k, n).expect("matmul_b_t")
            }),
        ),
        (
            "mm (aclnnMm)",
            Box::new({
                let ctx = &ctx;
                let stream = &stream;
                let a = &a;
                let b_t = &b_t;
                // perf probe only: [k,n] descriptor over the [n,k] buffer --
                // same shapes/strides cost, data correctness irrelevant here
                move || ops::mm_fp16(ctx, stream, a, [m, k], b_t, [k, n]).expect("mm")
            }),
        ),
        (
            "gemm_b_t (aclnnGemm)",
            Box::new({
                let ctx = &ctx;
                let stream = &stream;
                let a = &a;
                let b_t = &b_t;
                move || ops::gemm_b_t_fp16(ctx, stream, a, [m, k], b_t, k, n).expect("gemm_b_t")
            }),
        ),
    ];

    for (name, run_once) in entries {
        // warm the entry once (plan cache) then capture 100 reps
        drop(run_once());
        let _ = stream.synchronize();
        apxinf_ascend::flush_pending_frees();
        let arena = be.ctx().enter_arena(10 << 30).expect("arena");
        be.begin_capture().expect("begin");
        for _ in 0..reps {
            drop(run_once());
        }
        let graph = match be.end_capture() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[{name}] capture failed: {e:?}");
                be.ctx().clear_arena();
                continue;
            }
        };
        let used = be.ctx().exit_arena();
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
        println!("[{name}] {ts:?} ms/matmul (arena {used}B)");
        drop(graph);
        drop(arena);
        let _ = stream.synchronize();
        apxinf_ascend::flush_pending_frees();
    }
    println!("MATMUL_ENTRY_PROBE_DONE");
}

fn seedref(seed: &mut u32) -> &u32 {
    seed
}
