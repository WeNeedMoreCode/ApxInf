//! GE static-OM POC probe: build a chain of MatMulV2 pairs as a GE graph,
//! compile in-memory, execute on our stream, then
//!  (1) parity vs the aclnn composition on one chain step,
//!  (2) timing per matmul under back-to-back execute -- the number that
//!      decides the C route (aclnn+ACLGraph pays ~475us/task fixed cost).
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ge_poc_probe --release -p apxinf-ascend
use apxinf_core::Backend as _;
use half::f16;
use libloading::{Library, Symbol};

type InitFn = unsafe extern "C" fn(*const std::os::raw::c_char) -> i32;
type BuildFn = unsafe extern "C" fn(i32, i32, i32, i32) -> i32;
type RunFn = unsafe extern "C" fn(*mut (), *mut (), *mut (), *mut (), *mut ()) -> i32;
type FiniFn = unsafe extern "C" fn() -> i32;

fn main() {
    // GE builder init FIRST: the official example runs build in a
    // process without an initialized ACL runtime -- initializing GE after
    // aclInit/SetDevice fails with a bare GRAPH_FAILED
    let lib_path = std::env::var("GE_POC_LIB").unwrap_or_else(|_| {
        "/data/apxinf/ascendc/ge_poc/build/libge_matmul_poc.so".into()
    });
    let lib = unsafe { Library::new(&lib_path) }.expect("dlopen ge_poc lib");
    let ge_init: Symbol<InitFn> = unsafe { lib.get(b"ge_poc_init") }.expect("init sym");
    let ge_build: Symbol<BuildFn> = unsafe { lib.get(b"ge_poc_build") }.expect("build sym");
    let ge_run: Symbol<RunFn> = unsafe { lib.get(b"ge_poc_run") }.expect("run sym");
    let ge_fini: Symbol<FiniFn> = unsafe { lib.get(b"ge_poc_fini") }.expect("fini sym");
    unsafe {
        let soc = b"Ascend310P3\0";
        let rc = ge_init(soc.as_ptr() as *const _);
        assert_eq!(rc, 0, "ge_poc_init rc={rc}");
    }

    let be = apxinf_ascend::AscendBackend::new(0).expect("be");
    let ctx = be.ctx();
    let stream = be.stream();

    // prefix gate_up family: x [832,2048], w1 [2048,32768], w2 [32768,2048]
    let (m, k, n) = (832i32, 2048i32, 32768i32);
    let pairs = 8; // 16 matmuls per execute
    let mut seed = 7u32;
    let mut next = |div: f32| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        f16::from_f32(((*&seed >> 16) as i32 % 200 - 100) as f32 / div)
    };
    let upload = |vals: &[f16]| -> apxinf_ascend::DeviceBuffer {
        let bytes: &[u8] = bytemuck::cast_slice(vals);
        let buf = ctx.malloc(bytes.len()).expect("malloc");
        ctx.copy_h2d(&buf, bytes).expect("h2d");
        buf
    };
    // 权重幅度缩到 ~±0.05：每对 matmul 增益 ≈1，16 连发后仍在 fp16 normal
    // 域（±0.25 权重会让幅值 24×/对爆炸，双路径在 ±65504 饱和异号）
    let x_h: Vec<f16> = (0..m as usize * k as usize).map(|_| next(400.0)).collect();
    let w1_h: Vec<f16> = (0..k as usize * n as usize).map(|_| next(2000.0)).collect();
    let w2_h: Vec<f16> = (0..n as usize * k as usize).map(|_| next(2000.0)).collect();
    let x = upload(&x_h);
    let w1 = upload(&w1_h);
    let w2 = upload(&w2_h);
    let y = ctx.malloc((m * k * 2) as usize).expect("y");
    drop(stream.synchronize());

    unsafe {
        let rc = ge_build(pairs, m, k, n);
        assert_eq!(rc, 0, "ge_poc_build rc={rc}");

        // parity: GE chain vs aclnn composition (same math, 16 matmuls)
        let rc = ge_run(x.as_ptr() as *mut (), w1.as_ptr() as *mut (), w2.as_ptr() as *mut (),
                        y.as_ptr() as *mut (), stream.handle() as *mut ());
        assert_eq!(rc, 0, "ge_poc_run rc={rc}");
        drop(stream.synchronize());
        let mut back = vec![0u8; (m * k * 2) as usize];
        ctx.copy_d2h(&y, &mut back).expect("d2h");
        let y_h: Vec<f16> = bytemuck::cast_slice(&back).to_vec();

        // aclnn reference chain (same data, same order)
        let mut cur = upload(&x_h);
        for _ in 0..pairs {
            let a = apxinf_ascend::ops::matmul_fp16(ctx, stream, &cur, [m as i64, k as i64],
                &w1, [k as i64, n as i64]).expect("mm1");
            cur = apxinf_ascend::ops::matmul_fp16(ctx, stream, &a, [m as i64, n as i64],
                &w2, [n as i64, k as i64]).expect("mm2");
        }
        drop(stream.synchronize());
        let mut back2 = vec![0u8; (m * k * 2) as usize];
        ctx.copy_d2h(&cur, &mut back2).expect("d2h2");
        let ref_h: Vec<f16> = bytemuck::cast_slice(&back2).to_vec();
        let max_diff = y_h.iter().zip(&ref_h)
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0f32, f32::max);
        println!("parity GE-vs-aclnn over {pairs} matmul pairs: max_diff={max_diff:.5}");
        assert!(max_diff < 0.05, "GE chain diverged: {max_diff}");

        // timing: back-to-back executes, per-matmul
        let mut ts = Vec::new();
        for i in 0..30 {
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let rc = ge_run(x.as_ptr() as *mut (), w1.as_ptr() as *mut (), w2.as_ptr() as *mut (),
                                y.as_ptr() as *mut (), stream.handle() as *mut ());
                assert_eq!(rc, 0);
            }
            drop(stream.synchronize());
            if i >= 3 {
                ts.push(t0.elapsed().as_secs_f64() * 1000.0 / (10.0 * pairs as f64 * 2.0));
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ts[ts.len() / 2];
        println!(
            "GE static OM: {med:.4} ms/matmul (m={m} n={n}) -- aclnn+ACLGraph reference: 5.86 ms"
        );
        let rc = ge_fini();
        assert_eq!(rc, 0);
    }
    println!("GE_POC_OK");
}
