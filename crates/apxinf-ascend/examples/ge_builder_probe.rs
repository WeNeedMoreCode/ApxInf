//! C1-① 收尾探针：模型定义全在 Rust，经 ge_builder FFI（C++ 薄层）构图。
//! 复刻 ge_poc_probe 的链（8 对 MatMulV2，prefix gate_up 形状），证明
//! 泛化 FFI 与专用 POC shim 同分：
//!   (1) GE OM vs aclnn 组合同输入对拍
//!   (2) 背靠背执行的 ms/matmul（零调度税判定复验）
//! APXINF_GE_BUILDER_LIB 可覆盖 .so 路径。
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ge_builder_probe --release -p apxinf-ascend
use apxinf_core::Backend as _;
use apxinf_ascend::ge_builder::{self, Dtype, GeGraph};
use half::f16;

fn main() {
    // GE 会话先于 ACL runtime 初始化（ge_poc 实测反序 GRAPH_FAILED）
    ge_builder::init("Ascend310P3").expect("geb init");

    let be = apxinf_ascend::AscendBackend::new(0).expect("be");
    let ctx = be.ctx();
    let stream = be.stream();

    // prefix gate_up family: x [832,2048], w1 [2048,32768], w2 [32768,2048]
    //（env 可覆盖用于二分定位）
    let envi = |k: &str, d: i64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let m = envi("GEB_M", 832);
    let k = envi("GEB_K", 2048);
    let n = envi("GEB_N", 32768);
    let pairs = envi("GEB_PAIRS", 8); // 16 matmuls per execute

    // ---- 模型定义（Rust）：x -> (mm1 -> mm2) x pairs ----
    let g = GeGraph::begin("ge_builder_probe").expect("begin");
    g.add_data("x", 0, &[m, k], Dtype::Fp16).unwrap();
    g.add_data("w1", 1, &[k, n], Dtype::Fp16).unwrap();
    g.add_data("w2", 2, &[n, k], Dtype::Fp16).unwrap();
    let mut prev = "x".to_string();
    for i in 0..pairs {
        let mm1 = format!("mm1_{i}");
        let mm2 = format!("mm2_{i}");
        g.add_op(&mm1, "MatMulV2").unwrap();
        g.set_input_desc(&mm1, "x1", &[m, k], Dtype::Fp16).unwrap();
        g.set_input_desc(&mm1, "x2", &[k, n], Dtype::Fp16).unwrap();
        g.set_output_desc(&mm1, "y", &[m, n], Dtype::Fp16).unwrap();
        g.set_attr_bool(&mm1, "transpose_x1", false).unwrap();
        g.set_attr_bool(&mm1, "transpose_x2", false).unwrap();
        g.link(&mm1, "x1", &prev).unwrap();
        g.link(&mm1, "x2", "w1").unwrap();

        g.add_op(&mm2, "MatMulV2").unwrap();
        g.set_input_desc(&mm2, "x1", &[m, n], Dtype::Fp16).unwrap();
        g.set_input_desc(&mm2, "x2", &[n, k], Dtype::Fp16).unwrap();
        g.set_output_desc(&mm2, "y", &[m, k], Dtype::Fp16).unwrap();
        g.set_attr_bool(&mm2, "transpose_x1", false).unwrap();
        g.set_attr_bool(&mm2, "transpose_x2", false).unwrap();
        g.link(&mm2, "x1", &mm1).unwrap();
        g.link(&mm2, "x2", "w2").unwrap();
        prev = mm2;
    }
    g.graph_inputs(&["x", "w1", "w2"]).unwrap();
    g.graph_outputs(&[&prev]).unwrap();
    g.set_nd_input_shape(&[("x", &[m, k]), ("w1", &[k, n]), ("w2", &[n, k])]).unwrap();
    g.build().expect("build");
    println!(
        "model io: n_in={} n_out={} in0={:?} ({} B) out0={:?} ({} B)",
        g.num_inputs().unwrap(),
        g.num_outputs().unwrap(),
        g.input_dims(0).unwrap(),
        g.input_size(0).unwrap(),
        g.output_dims(0).unwrap(),
        g.output_size(0).unwrap(),
    );

    // ---- 数据 ----
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
    // 权重幅度 ~±0.017（std≈0.0096）：每对 matmul 增益 = std_w²·√(k·n) ≈0.76，
    // 8 对链全程衰减不饱和。更大的权重（/2000→增益 6.8/对）第 6 对起 fp16
    // 饱和，双路径在 ±65504 处异号——对拍 1616 的假阳性来源（pairs≤2 可证
    // 数值本身正确：0.002/0.016）
    let x_h: Vec<f16> = (0..m as usize * k as usize).map(|_| next(400.0)).collect();
    let w1_h: Vec<f16> = (0..k as usize * n as usize).map(|_| next(6000.0)).collect();
    let w2_h: Vec<f16> = (0..n as usize * k as usize).map(|_| next(6000.0)).collect();
    let x = upload(&x_h);
    let w1 = upload(&w1_h);
    let w2 = upload(&w2_h);
    let y = ctx.malloc((m * k * 2) as usize).expect("y");
    drop(stream.synchronize());

    // parity: GE OM vs aclnn composition (same data, same order)
    g.run(&[&x, &w1, &w2], &[&y], stream).expect("run");
    drop(stream.synchronize());
    let mut back = vec![0u8; (m * k * 2) as usize];
    ctx.copy_d2h(&y, &mut back).expect("d2h");
    let y_h: Vec<f16> = bytemuck::cast_slice(&back).to_vec();

    let mut cur = upload(&x_h);
    for _ in 0..pairs {
        let a = apxinf_ascend::ops::matmul_fp16(ctx, stream, &cur, [m, k], &w1, [k, n]).expect("mm1");
        cur = apxinf_ascend::ops::matmul_fp16(ctx, stream, &a, [m, n], &w2, [n, k]).expect("mm2");
    }
    drop(stream.synchronize());
    let mut back2 = vec![0u8; (m * k * 2) as usize];
    ctx.copy_d2h(&cur, &mut back2).expect("d2h2");
    let ref_h: Vec<f16> = bytemuck::cast_slice(&back2).to_vec();
    let max_diff = y_h
        .iter()
        .zip(&ref_h)
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    println!("parity GE-vs-aclnn over {pairs} matmul pairs: max_diff={max_diff:.5}");
    assert!(max_diff < 0.05, "GE chain diverged: {max_diff}");

    // timing: back-to-back executes, per-matmul
    let mut ts = Vec::new();
    for i in 0..30 {
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            g.run(&[&x, &w1, &w2], &[&y], stream).expect("run");
        }
        drop(stream.synchronize());
        if i >= 3 {
            ts.push(t0.elapsed().as_secs_f64() * 1000.0 / (10.0 * pairs as f64 * 2.0));
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ts[ts.len() / 2];
    println!(
        "GE static OM (via ge_builder): {med:.4} ms/matmul (m={m} n={n}) -- refs: GE-POC 4.745 / aclnn+ACLGraph 5.812"
    );

    ge_builder::fini().expect("fini");
    println!("GE_BUILDER_PROBE_OK");
}
