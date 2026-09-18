//! C1-④ 单层 language layer GE 化探针：把一个 GEMMA_2B language layer 的
//! attention+MLP 全序声明成 GE 静态 OM（ge_builder FFI），与 eager aclnn
//! 同输入同序列对拍，再背靠背 bench（GE OM vs eager 逐算子）。
//!
//! 图内序列（roped q/k 走外部输入——ARPE 单算子入图是 v2，环境变量
//! GEB_ARPE=1 时启用）：
//!   PFA(BSH, GQA 8:1, d=256) → proj mm → +x → RmsNorm → gate/up 双 mm
//!   → GeluV2(tanh)·up → down mm → +res
//! GE IR 契约来源（310P opp tbe impl 源码，2026-09-19 取证）：
//!   PromptFlashAttention: query/key/value/attention_out +
//!     num_heads/scale_value/pre_tokens/next_tokens/input_layout/
//!     num_key_value_heads/sparse_mode/inner_precise
//!   RmsNorm: x/gamma → y/rstd, epsilon
//!   GeluV2: x → y, approximate="tanh"
//!   Add/Mul: x1/x2 → y
//!   ApplyRotaryPosEmb: query/key/cos/sin → query/key, layout(int)/
//!     rotary_mode("half")
//! 运行：
//!   source /data/apxinf/rust_env.sh
//!   GEB_TOKENS=8  [GEB_ARPE=1]  cargo run --example ge_layer_probe --features ascend --release -p apxinf-model
//!   GEB_BENCH=1 GEB_TOKENS=832 cargo run ... （bench 模式）
use half::f16;

use apxinf_ascend::ge_builder::{self, Dtype, GeGraph};
use apxinf_ascend::{ops as aops, AscendBackend, AscendContext, DeviceBuffer, AscendStream};

// GEMMA_2B language dims (config.rs)
const WIDTH: i64 = 2048;
const HEADS: i64 = 8;
const KV_HEADS: i64 = 1;
const HEAD_DIM: i64 = 256;
const INTER: i64 = 16384;
const ROPE_THETA: f64 = 10000.0;
const RMS_EPS: f64 = 1e-6;

fn envi(k: &str, d: i64) -> i64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn rand_f16(n: usize, seed: &mut u32, div: f32) -> Vec<f16> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push(f16::from_f32(((*seed >> 16) as i32 % 200 - 100) as f32 / div));
    }
    v
}

/// host rotate-half rope（eager 表语义：cos 半维重复、sin 符号折叠）。
/// 输入 [tokens, heads*d] 行主序；位置 = 行号。
fn rope_host(q: &[f16], tokens: usize, heads: usize) -> Vec<f16> {
    let d = HEAD_DIM as usize;
    let half = d / 2;
    let mut out = vec![f16::from_f32(0.0); q.len()];
    for t in 0..tokens {
        for h in 0..heads {
            let base = (h * d) as usize;
            for i in 0..half {
                let freq = (t as f64) * ROPE_THETA.powf(-(2.0 * i as f64) / d as f64);
                let (c, s) = (freq.cos() as f32, freq.sin() as f32);
                let a = q[t * heads * d + base + i].to_f32();
                let b = q[t * heads * d + base + half + i].to_f32();
                out[t * heads * d + base + i] = f16::from_f32(a * c - b * s);
                out[t * heads * d + base + half + i] = f16::from_f32(b * c + a * s);
            }
        }
    }
    out
}

/// cos/sin 表 [tokens, d]（半维重复/符号折叠，ARPE "half" 模式契约）。
fn rope_tables(tokens: usize) -> (Vec<f16>, Vec<f16>) {
    let d = HEAD_DIM as usize;
    let half = d / 2;
    let mut cos = vec![f16::from_f32(0.0); tokens * d];
    let mut sin = vec![f16::from_f32(0.0); tokens * d];
    for t in 0..tokens {
        for i in 0..half {
            let freq = (t as f64) * ROPE_THETA.powf(-(2.0 * i as f64) / d as f64);
            let (c, s) = (freq.cos() as f32, freq.sin() as f32);
            cos[t * d + i] = f16::from_f32(c);
            cos[t * d + half + i] = f16::from_f32(c);
            sin[t * d + i] = f16::from_f32(s);
            sin[t * d + half + i] = f16::from_f32(-s);
        }
    }
    (cos, sin)
}

fn upload(ctx: &AscendContext, vals: &[f16]) -> DeviceBuffer {
    let bytes =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 2) };
    let buf = ctx.malloc(bytes.len()).expect("malloc");
    ctx.copy_h2d(&buf, bytes).expect("h2d");
    buf
}

/// fp16 device buffer -> host Vec<f16>（d2h 字节流按 LE u16 重组）
fn download_f16(ctx: &AscendContext, buf: &DeviceBuffer, n: usize) -> Vec<f16> {
    let mut back = vec![0u8; n * 2];
    ctx.copy_d2h(buf, &mut back).expect("d2h");
    back.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// eager 参考序列（与 GE 图同序，aclnn 逐算子）。
fn eager_layer(
    ctx: &AscendContext, stream: &AscendStream, m: i64,
    q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, x: &DeviceBuffer,
    w_out: &DeviceBuffer, g2: &DeviceBuffer, zeros: &DeviceBuffer,
    w_gate: &DeviceBuffer, w_up: &DeviceBuffer, w_down: &DeviceBuffer,
) -> DeviceBuffer {
    let qd = HEADS * HEAD_DIM;
    let attn = aops::prompt_flash_attention_bsh_fp16(
        ctx, stream, q, k, v, m, HEADS, KV_HEADS, HEAD_DIM, None,
    ).expect("pfa");
    let proj = aops::matmul_fp16(ctx, stream, &attn, [m, qd], w_out, [qd, WIDTH]).expect("proj");
    let res = aops::add_fp16(ctx, stream, &proj, x, &[m, WIDTH]).expect("res");
    let norm = aops::add_rms_norm_fp16(ctx, stream, &res, zeros, g2, &[m, WIDTH], RMS_EPS)
        .expect("rms").0;
    let gate = aops::matmul_fp16(ctx, stream, &norm, [m, WIDTH], w_gate, [WIDTH, INTER]).expect("gate");
    let up = aops::matmul_fp16(ctx, stream, &norm, [m, WIDTH], w_up, [WIDTH, INTER]).expect("up");
    let g = aops::gelu_fp16(ctx, stream, &gate, &[m, INTER], true).expect("gelu");
    let act = aops::mul_fp16(ctx, stream, &g, &up, &[m, INTER]).expect("mul");
    let down = aops::matmul_fp16(ctx, stream, &act, [m, INTER], w_down, [INTER, WIDTH]).expect("down");
    aops::add_fp16(ctx, stream, &down, &res, &[m, WIDTH]).expect("res2")
}

/// GE 图：同序声明。roped q/k 直接入图（GEB_ARPE=1 时改为原始 q/k +
/// cos/sin 表走 ApplyRotaryPosEmb）。
/// GEB_SUB 编译冒烟模式：1=rms 2=gelu 3=add/mul 4=pfa（定位编译拒绝点，
/// 跳过数值对拍）
fn build_ge_layer(m: i64, use_arpe: bool) -> (GeGraph, usize) {
    let sub = envi("GEB_SUB", 0);
    if (sub != 0) && (sub != 7) {
        let g = GeGraph::begin("ge_op_smoke").expect("begin");
        let qd = HEADS * HEAD_DIM;
        let kvd = KV_HEADS * HEAD_DIM;
        match sub {
            1 => {
                // RmsNorm
                g.add_data("x", 0, &[m, WIDTH], Dtype::Fp16).unwrap();
                g.add_data("gamma", 1, &[WIDTH], Dtype::Fp16).unwrap();
                g.add_op("norm", "RmsNorm").unwrap();
                g.set_input_desc("norm", "x", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_input_desc("norm", "gamma", &[WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("norm", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_attr_float("norm", "epsilon", RMS_EPS).unwrap();
                g.link("norm", "x", "x").unwrap();
                g.link("norm", "gamma", "gamma").unwrap();
                g.graph_inputs(&["x", "gamma"]).unwrap();
                g.graph_outputs(&["norm"]).unwrap();
                g.set_nd_input_shape(&[("x", &[m, WIDTH]), ("gamma", &[WIDTH])]).unwrap();
            }
            2 => {
                // GeluV2(tanh)
                g.add_data("a", 0, &[m, INTER], Dtype::Fp16).unwrap();
                g.add_op("g", "GeluV2").unwrap();
                g.set_input_desc("g", "x", &[m, INTER], Dtype::Fp16).unwrap();
                g.set_output_desc("g", "y", &[m, INTER], Dtype::Fp16).unwrap();
                g.set_attr_str("g", "approximate", "tanh").unwrap();
                g.link("g", "x", "a").unwrap();
                g.graph_inputs(&["a"]).unwrap();
                g.graph_outputs(&["g"]).unwrap();
                g.set_nd_input_shape(&[("a", &[m, INTER])]).unwrap();
            }
            3 => {
                // Add / Mul 双联
                g.add_data("a", 0, &[m, WIDTH], Dtype::Fp16).unwrap();
                g.add_data("b", 1, &[m, WIDTH], Dtype::Fp16).unwrap();
                g.add_op("s", "Add").unwrap();
                g.set_input_desc("s", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_input_desc("s", "x2", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("s", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.link("s", "x1", "a").unwrap();
                g.link("s", "x2", "b").unwrap();
                g.add_op("p", "Mul").unwrap();
                g.set_input_desc("p", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_input_desc("p", "x2", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("p", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.link("p", "x1", "s").unwrap();
                g.link("p", "x2", "a").unwrap();
                g.graph_inputs(&["a", "b"]).unwrap();
                g.graph_outputs(&["p"]).unwrap();
                g.set_nd_input_shape(&[("a", &[m, WIDTH]), ("b", &[m, WIDTH])]).unwrap();
            }
            4 => {
                // PFA 单算子（rank-3 [1,m,*]——eager aclnn 的 BSH 视图同构；
                // 2-D desc 已被拒）
                g.add_data("q", 0, &[1, m, qd], Dtype::Fp16).unwrap();
                g.add_data("k", 1, &[1, m, kvd], Dtype::Fp16).unwrap();
                g.add_data("v", 2, &[1, m, kvd], Dtype::Fp16).unwrap();
                g.add_op("pfa", "PromptFlashAttention").unwrap();
                g.set_input_desc("pfa", "query", &[1, m, qd], Dtype::Fp16).unwrap();
                g.set_input_desc("pfa", "key", &[1, m, kvd], Dtype::Fp16).unwrap();
                g.set_input_desc("pfa", "value", &[1, m, kvd], Dtype::Fp16).unwrap();
                g.set_output_desc("pfa", "attention_out", &[1, m, qd], Dtype::Fp16).unwrap();
                g.set_attr_int("pfa", "num_heads", HEADS).unwrap();
                g.set_attr_float("pfa", "scale_value", 1.0 / (HEAD_DIM as f64).sqrt()).unwrap();
                g.set_attr_int("pfa", "pre_tokens", 2147483647).unwrap();
                g.set_attr_int("pfa", "next_tokens", 0).unwrap();
                g.set_attr_str("pfa", "input_layout", "BSH").unwrap();
                g.set_attr_int("pfa", "num_key_value_heads", KV_HEADS).unwrap();
                g.set_attr_int("pfa", "sparse_mode", 0).unwrap();
                g.set_attr_int("pfa", "inner_precise", 0).unwrap();
                g.link("pfa", "query", "q").unwrap();
                g.link("pfa", "key", "k").unwrap();
                g.link("pfa", "value", "v").unwrap();
                g.graph_inputs(&["q", "k", "v"]).unwrap();
                g.graph_outputs(&["pfa"]).unwrap();
                g.set_nd_input_shape(&[("q", &[1, m, qd]), ("k", &[1, m, kvd]), ("v", &[1, m, kvd])])
                    .unwrap();
            }
            5 => {
                // PFA→Squeeze→MatMulV2 桥接（rank-3→2-D 适配的最小组合）
                g.add_data("q", 0, &[1, m, qd], Dtype::Fp16).unwrap();
                g.add_data("k", 1, &[1, m, kvd], Dtype::Fp16).unwrap();
                g.add_data("v", 2, &[1, m, kvd], Dtype::Fp16).unwrap();
                g.add_data("w", 3, &[qd, WIDTH], Dtype::Fp16).unwrap();
                g.add_op("pfa", "PromptFlashAttention").unwrap();
                g.set_input_desc("pfa", "query", &[1, m, qd], Dtype::Fp16).unwrap();
                g.set_input_desc("pfa", "key", &[1, m, kvd], Dtype::Fp16).unwrap();
                g.set_input_desc("pfa", "value", &[1, m, kvd], Dtype::Fp16).unwrap();
                g.set_output_desc("pfa", "attention_out", &[1, m, qd], Dtype::Fp16).unwrap();
                g.set_attr_int("pfa", "num_heads", HEADS).unwrap();
                g.set_attr_float("pfa", "scale_value", 1.0 / (HEAD_DIM as f64).sqrt()).unwrap();
                g.set_attr_int("pfa", "pre_tokens", 2147483647).unwrap();
                g.set_attr_int("pfa", "next_tokens", 0).unwrap();
                g.set_attr_str("pfa", "input_layout", "BSH").unwrap();
                g.set_attr_int("pfa", "num_key_value_heads", KV_HEADS).unwrap();
                g.set_attr_int("pfa", "sparse_mode", 0).unwrap();
                g.set_attr_int("pfa", "inner_precise", 0).unwrap();
                g.link("pfa", "query", "q").unwrap();
                g.link("pfa", "key", "k").unwrap();
                g.link("pfa", "value", "v").unwrap();
                g.add_op("sq", "Squeeze").unwrap();
                g.set_input_desc("sq", "x", &[1, m, qd], Dtype::Fp16).unwrap();
                g.set_output_desc("sq", "y", &[m, qd], Dtype::Fp16).unwrap();
                g.set_attr_int_list("sq", "axis", &[0]).unwrap();
                g.link("sq", "x", "pfa").unwrap();
                g.add_op("mm", "MatMulV2").unwrap();
                g.set_input_desc("mm", "x1", &[m, qd], Dtype::Fp16).unwrap();
                g.set_input_desc("mm", "x2", &[qd, WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("mm", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_attr_bool("mm", "transpose_x1", false).unwrap();
                g.set_attr_bool("mm", "transpose_x2", false).unwrap();
                g.link("mm", "x1", "sq").unwrap();
                g.link("mm", "x2", "w").unwrap();
                g.graph_inputs(&["q", "k", "v", "w"]).unwrap();
                g.graph_outputs(&["mm"]).unwrap();
                g.set_nd_input_shape(&[("q", &[1, m, qd]), ("k", &[1, m, kvd]), ("v", &[1, m, kvd]), ("w", &[qd, WIDTH])])
                    .unwrap();
            }
            6 => {
                // sub5 桥 + res Add + RmsNorm（增量定位全层失败点）
                g.add_data("q", 0, &[1, m, qd], Dtype::Fp16).unwrap();
                g.add_data("k", 1, &[1, m, kvd], Dtype::Fp16).unwrap();
                g.add_data("v", 2, &[1, m, kvd], Dtype::Fp16).unwrap();
                g.add_data("w", 3, &[qd, WIDTH], Dtype::Fp16).unwrap();
                g.add_data("x", 4, &[m, WIDTH], Dtype::Fp16).unwrap();
                g.add_data("g2", 5, &[WIDTH], Dtype::Fp16).unwrap();
                g.add_op("pfa", "PromptFlashAttention").unwrap();
                g.set_input_desc("pfa", "query", &[1, m, qd], Dtype::Fp16).unwrap();
                g.set_input_desc("pfa", "key", &[1, m, kvd], Dtype::Fp16).unwrap();
                g.set_input_desc("pfa", "value", &[1, m, kvd], Dtype::Fp16).unwrap();
                g.set_output_desc("pfa", "attention_out", &[1, m, qd], Dtype::Fp16).unwrap();
                g.set_attr_int("pfa", "num_heads", HEADS).unwrap();
                g.set_attr_float("pfa", "scale_value", 1.0 / (HEAD_DIM as f64).sqrt()).unwrap();
                g.set_attr_int("pfa", "pre_tokens", 2147483647).unwrap();
                g.set_attr_int("pfa", "next_tokens", 0).unwrap();
                g.set_attr_str("pfa", "input_layout", "BSH").unwrap();
                g.set_attr_int("pfa", "num_key_value_heads", KV_HEADS).unwrap();
                g.set_attr_int("pfa", "sparse_mode", 0).unwrap();
                g.set_attr_int("pfa", "inner_precise", 0).unwrap();
                g.link("pfa", "query", "q").unwrap();
                g.link("pfa", "key", "k").unwrap();
                g.link("pfa", "value", "v").unwrap();
                g.add_op("sq", "Squeeze").unwrap();
                g.set_input_desc("sq", "x", &[1, m, qd], Dtype::Fp16).unwrap();
                g.set_output_desc("sq", "y", &[m, qd], Dtype::Fp16).unwrap();
                g.set_attr_int_list("sq", "axis", &[0]).unwrap();
                g.link("sq", "x", "pfa").unwrap();
                g.add_op("mm", "MatMulV2").unwrap();
                g.set_input_desc("mm", "x1", &[m, qd], Dtype::Fp16).unwrap();
                g.set_input_desc("mm", "x2", &[qd, WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("mm", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_attr_bool("mm", "transpose_x1", false).unwrap();
                g.set_attr_bool("mm", "transpose_x2", false).unwrap();
                g.link("mm", "x1", "sq").unwrap();
                g.link("mm", "x2", "w").unwrap();
                g.add_op("res", "Add").unwrap();
                g.set_input_desc("res", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_input_desc("res", "x2", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("res", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.link("res", "x1", "mm").unwrap();
                g.link("res", "x2", "x").unwrap();
                g.add_op("norm2", "RmsNorm").unwrap();
                g.set_input_desc("norm2", "x", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_input_desc("norm2", "gamma", &[WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("norm2", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_attr_float("norm2", "epsilon", RMS_EPS).unwrap();
                g.link("norm2", "x", "res").unwrap();
                g.link("norm2", "gamma", "g2").unwrap();
                g.graph_inputs(&["q", "k", "v", "w", "x", "g2"]).unwrap();
                g.graph_outputs(&["norm2"]).unwrap();
                g.set_nd_input_shape(&[("q", &[1, m, qd]), ("k", &[1, m, kvd]), ("v", &[1, m, kvd]), ("w", &[qd, WIDTH]), ("x", &[m, WIDTH]), ("g2", &[WIDTH])])
                    .unwrap();
            }
            9 => {
                // norm 作中间节点的对照：裸 RmsNorm 只有 dynamic 变体（静态图
                // 中下游 mm 拒绝），AddRmsNorm 是 eager 同款（aclnnAddRmsNorm）
                g.add_data("x", 0, &[m, WIDTH], Dtype::Fp16).unwrap();
                g.add_data("zeros", 1, &[m, WIDTH], Dtype::Fp16).unwrap();
                g.add_data("gamma", 2, &[WIDTH], Dtype::Fp16).unwrap();
                g.add_data("w", 3, &[WIDTH, WIDTH], Dtype::Fp16).unwrap();
                g.add_op("norm", "AddRmsNorm").unwrap();
                g.set_input_desc("norm", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_input_desc("norm", "x2", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_input_desc("norm", "gamma", &[WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("norm", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_attr_float("norm", "epsilon", RMS_EPS).unwrap();
                g.link("norm", "x1", "x").unwrap();
                g.link("norm", "x2", "zeros").unwrap();
                g.link("norm", "gamma", "gamma").unwrap();
                g.add_op("mm", "MatMul").unwrap();
                g.set_input_desc("mm", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_input_desc("mm", "x2", &[WIDTH, WIDTH], Dtype::Fp16).unwrap();
                g.set_output_desc("mm", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
                g.set_attr_bool("mm", "transpose_a", false).unwrap();
                g.set_attr_bool("mm", "transpose_b", false).unwrap();
                g.link_out("mm", "x1", "norm", "y").unwrap();
                g.link("mm", "x2", "w").unwrap();
                g.graph_inputs(&["x", "zeros", "gamma", "w"]).unwrap();
                g.graph_outputs_idx(&["mm", "norm"], &[0, 1]).unwrap();
                g.set_nd_input_shape(&[("x", &[m, WIDTH]), ("zeros", &[m, WIDTH]), ("gamma", &[WIDTH]), ("w", &[WIDTH, WIDTH])])
                    .unwrap();
            }
            _ => panic!("GEB_SUB: 1=rms 2=gelu 3=add/mul 4=pfa 5=pfa+sq+mm 6=+add+rms 9=rms-mm"),
        }
        g.build().expect("sub build");
        return (g, 0);
    }
    build_full_layer(m, use_arpe)
}

/// 全层图本体（GEB_SUB=7 等价直达）。GEB_INTER 可缩 MLP 宽度定位编译问题。
fn build_full_layer(m: i64, use_arpe: bool) -> (GeGraph, usize) {
    let inter = envi("GEB_INTER", INTER);
    let qd = HEADS * HEAD_DIM;
    let kvd = KV_HEADS * HEAD_DIM;
    let g = GeGraph::begin("ge_layer").expect("begin");
    let mut idx = 0i64;
    let mut names: Vec<String> = Vec::new();
    let mut add_data = |g: &GeGraph, names: &mut Vec<String>, name: &str, dims: &[i64]| {
        g.add_data(name, idx, dims, Dtype::Fp16).unwrap();
        idx += 1;
        names.push(name.to_string());
    };

    if use_arpe {
        // v2：原始 q/k + cos/sin 表，图内 ARPE 单算子（layout=0 BSH 猜测，
        // rotary_mode="half" 与 eager 表语义一致）
        add_data(&g, &mut names, "q", &[1, m, qd]);
        add_data(&g, &mut names, "k", &[1, m, kvd]);
        add_data(&g, &mut names, "v", &[1, m, kvd]);
        add_data(&g, &mut names, "cos", &[1, m, HEAD_DIM]);
        add_data(&g, &mut names, "sin", &[1, m, HEAD_DIM]);
        g.add_op("arpe", "ApplyRotaryPosEmb").unwrap();
        g.set_input_desc("arpe", "query", &[1, m, qd], Dtype::Fp16).unwrap();
        g.set_input_desc("arpe", "key", &[1, m, kvd], Dtype::Fp16).unwrap();
        g.set_input_desc("arpe", "cos", &[1, m, HEAD_DIM], Dtype::Fp16).unwrap();
        g.set_input_desc("arpe", "sin", &[1, m, HEAD_DIM], Dtype::Fp16).unwrap();
        g.set_output_desc("arpe", "query", &[1, m, qd], Dtype::Fp16).unwrap();
        g.set_output_desc("arpe", "key", &[1, m, kvd], Dtype::Fp16).unwrap();
        g.set_attr_int("arpe", "layout", 0).unwrap();
        g.set_attr_str("arpe", "rotary_mode", "half").unwrap();
        g.link("arpe", "query", "q").unwrap();
        g.link("arpe", "key", "k").unwrap();
        g.link("arpe", "cos", "cos").unwrap();
        g.link("arpe", "sin", "sin").unwrap();
    } else {
        add_data(&g, &mut names, "q", &[1, m, qd]);
        add_data(&g, &mut names, "k", &[1, m, kvd]);
        add_data(&g, &mut names, "v", &[1, m, kvd]);
    }
    add_data(&g, &mut names, "x", &[m, WIDTH]);
    add_data(&g, &mut names, "w_out", &[qd, WIDTH]);
    add_data(&g, &mut names, "g2", &[WIDTH]);
    add_data(&g, &mut names, "zeros", &[m, WIDTH]);
    add_data(&g, &mut names, "w_gate", &[WIDTH, inter]);
    add_data(&g, &mut names, "w_up", &[WIDTH, inter]);
    add_data(&g, &mut names, "w_down", &[inter, WIDTH]);

    // PFA：BSH rank-3 desc（[1,m,*]——2-D 被静默拒；eager aclnn 的
    // [1,tokens,qd] 视图同构）。下游 mm 保持 2-D desc，FormatAndShape
    // 自动适配。q/k Data 也必须 rank-3（链接同 rank）。
    let (q_src, k_src) = if use_arpe { ("arpe", "arpe") } else { ("q", "k") };
    g.add_op("pfa", "PromptFlashAttention").unwrap();
    g.set_input_desc("pfa", "query", &[1, m, qd], Dtype::Fp16).unwrap();
    g.set_input_desc("pfa", "key", &[1, m, kvd], Dtype::Fp16).unwrap();
    g.set_input_desc("pfa", "value", &[1, m, kvd], Dtype::Fp16).unwrap();
    g.set_output_desc("pfa", "attention_out", &[1, m, qd], Dtype::Fp16).unwrap();
    g.set_attr_int("pfa", "num_heads", HEADS).unwrap();
    g.set_attr_float("pfa", "scale_value", 1.0 / (HEAD_DIM as f64).sqrt()).unwrap();
    g.set_attr_int("pfa", "pre_tokens", 2147483647).unwrap();
    g.set_attr_int("pfa", "next_tokens", 0).unwrap();
    g.set_attr_str("pfa", "input_layout", "BSH").unwrap();
    g.set_attr_int("pfa", "num_key_value_heads", KV_HEADS).unwrap();
    g.set_attr_int("pfa", "sparse_mode", 0).unwrap();
    g.set_attr_int("pfa", "inner_precise", 0).unwrap();
    g.link("pfa", "query", q_src).unwrap();
    g.link("pfa", "key", k_src).unwrap();
    g.link("pfa", "value", "v").unwrap();

    let mut mm = |name: &str, a: &str, a_dims: &[i64], w: &str, w_dims: &[i64], o: &[i64]| {
        g.add_op(name, "MatMulV2").unwrap();
        g.set_input_desc(name, "x1", a_dims, Dtype::Fp16).unwrap();
        g.set_input_desc(name, "x2", w_dims, Dtype::Fp16).unwrap();
        g.set_output_desc(name, "y", o, Dtype::Fp16).unwrap();
        g.set_attr_bool(name, "transpose_x1", false).unwrap();
        g.set_attr_bool(name, "transpose_x2", false).unwrap();
        g.link(name, "x1", a).unwrap();
        g.link(name, "x2", w).unwrap();
    };
    // PFA 出 rank-3 [1,m,qd]，MatMulV2 有 rank∈{2,4} 门槛——Squeeze
    // （axis attr，无张量输入）压回 2-D 再进 matmul
    g.add_op("attn_sq", "Squeeze").unwrap();
    g.set_input_desc("attn_sq", "x", &[1, m, qd], Dtype::Fp16).unwrap();
    g.set_output_desc("attn_sq", "y", &[m, qd], Dtype::Fp16).unwrap();
    g.set_attr_int_list("attn_sq", "axis", &[0]).unwrap();
    g.link("attn_sq", "x", "pfa").unwrap();
    mm("mm_out", "attn_sq", &[m, qd], "w_out", &[qd, WIDTH], &[m, WIDTH]);

    g.add_op("res", "Add").unwrap();
    g.set_input_desc("res", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_input_desc("res", "x2", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_output_desc("res", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.link("res", "x1", "mm_out").unwrap();
    g.link("res", "x2", "x").unwrap();

    // norm2 走 AddRmsNorm（eager 同款；裸 RmsNorm 是 dynamic 变体，静态图
    // 下游 mm 拒绝）+ zeros 当 x2；多输出算子的出边必须 link_out 显式端口
    g.add_op("norm2", "AddRmsNorm").unwrap();
    g.set_input_desc("norm2", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_input_desc("norm2", "x2", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_input_desc("norm2", "gamma", &[WIDTH], Dtype::Fp16).unwrap();
    g.set_output_desc("norm2", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_attr_float("norm2", "epsilon", RMS_EPS).unwrap();
    g.link("norm2", "x1", "res").unwrap();
    g.link("norm2", "x2", "zeros").unwrap();
    g.link("norm2", "gamma", "g2").unwrap();

    // norm2 → 双 mm 的 fan-out 出边用显式端口（默认输出解析对多输出算子失效）
    g.add_op("mm_gate", "MatMulV2").unwrap();
    g.set_input_desc("mm_gate", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_input_desc("mm_gate", "x2", &[WIDTH, inter], Dtype::Fp16).unwrap();
    g.set_output_desc("mm_gate", "y", &[m, inter], Dtype::Fp16).unwrap();
    g.set_attr_bool("mm_gate", "transpose_x1", false).unwrap();
    g.set_attr_bool("mm_gate", "transpose_x2", false).unwrap();
    g.link_out("mm_gate", "x1", "norm2", "y").unwrap();
    g.link("mm_gate", "x2", "w_gate").unwrap();
    g.add_op("mm_up", "MatMulV2").unwrap();
    g.set_input_desc("mm_up", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_input_desc("mm_up", "x2", &[WIDTH, inter], Dtype::Fp16).unwrap();
    g.set_output_desc("mm_up", "y", &[m, inter], Dtype::Fp16).unwrap();
    g.set_attr_bool("mm_up", "transpose_x1", false).unwrap();
    g.set_attr_bool("mm_up", "transpose_x2", false).unwrap();
    g.link_out("mm_up", "x1", "norm2", "y").unwrap();
    g.link("mm_up", "x2", "w_up").unwrap();

    g.add_op("gact", "GeluV2").unwrap();
    g.set_input_desc("gact", "x", &[m, inter], Dtype::Fp16).unwrap();
    g.set_output_desc("gact", "y", &[m, inter], Dtype::Fp16).unwrap();
    g.set_attr_str("gact", "approximate", "tanh").unwrap();
    g.link("gact", "x", "mm_gate").unwrap();

    g.add_op("act", "Mul").unwrap();
    g.set_input_desc("act", "x1", &[m, inter], Dtype::Fp16).unwrap();
    g.set_input_desc("act", "x2", &[m, inter], Dtype::Fp16).unwrap();
    g.set_output_desc("act", "y", &[m, inter], Dtype::Fp16).unwrap();
    g.link("act", "x1", "gact").unwrap();
    g.link("act", "x2", "mm_up").unwrap();

    mm("mm_down", "act", &[m, inter], "w_down", &[inter, WIDTH], &[m, WIDTH]);

    g.add_op("out", "Add").unwrap();
    g.set_input_desc("out", "x1", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_input_desc("out", "x2", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.set_output_desc("out", "y", &[m, WIDTH], Dtype::Fp16).unwrap();
    g.link("out", "x1", "mm_down").unwrap();
    g.link("out", "x2", "res").unwrap();

    let input_names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let shapes: Vec<(&str, Vec<i64>)> = names
        .iter()
        .map(|n| {
            let dims = match n.as_str() {
                "q" => vec![1, m, qd],
                "k" | "v" => vec![1, m, kvd],
                "cos" | "sin" => vec![1, m, HEAD_DIM],
                "x" => vec![m, WIDTH],
                "w_out" => vec![qd, WIDTH],
                "g2" => vec![WIDTH],
                "zeros" => vec![m, WIDTH],
                "w_gate" | "w_up" => vec![WIDTH, inter],
                "w_down" => vec![inter, WIDTH],
                _ => unreachable!(),
            };
            (n.as_str(), dims)
        })
        .collect();
    let shape_refs: Vec<(&str, &[i64])> =
        shapes.iter().map(|(n, d)| (*n, d.as_slice())).collect();
    g.graph_inputs(&input_names).unwrap();
    // rstd 是 RmsNorm 的 REQUIRED 输出——绑成第二个图输出（死端会静默
    // 杀编译），运行时多绑一个 fp32 缓冲
    g.graph_outputs_idx(&["out", "norm2"], &[0, 1]).unwrap();
    g.set_nd_input_shape(&shape_refs).unwrap();
    g.build().expect("build");
    (g, input_names.len())
}

fn main() {
    let m = envi("GEB_TOKENS", 8);
    let bench = std::env::var("GEB_BENCH").is_ok();
    let use_arpe = std::env::var("GEB_ARPE").is_ok();

    ge_builder::init("Ascend310P3").expect("geb init (before acl runtime)");
    let be = AscendBackend::new(0).expect("be");
    let ctx = be.ctx();
    let stream = be.stream();

    // 编译冒烟模式：只建子图，跳过数据/对拍
    if envi("GEB_SUB", 0) != 0 {
        let _ = build_ge_layer(m, false);
        println!("GE_OP_SMOKE_OK sub={}", envi("GEB_SUB", 0));
        ge_builder::fini().expect("fini");
        return;
    }

    // ---- 数据（单层幅度安全：±0.25 权重经 2048/16384 维点积后 std ~9/26）----
    let qd = (HEADS * HEAD_DIM) as usize;
    let kvd = (KV_HEADS * HEAD_DIM) as usize;
    let mus = m as usize;
    let mut seed = 0x1234u32;
    let rand = |n: usize, s: &mut u32| rand_f16(n, s, 400.0);

    let q_raw = rand(mus * qd, &mut seed);
    let k_raw = rand(mus * kvd, &mut seed);
    let v_h = rand(mus * kvd, &mut seed);
    let q_h = rope_host(&q_raw, mus, HEADS as usize);
    let k_h = rope_host(&k_raw, mus, KV_HEADS as usize);
    let (cos_h, sin_h) = rope_tables(mus);
    let x_h = rand(mus * WIDTH as usize, &mut seed);
    let w_out_h = rand((qd * WIDTH as usize), &mut seed);
    let g2_h = rand(WIDTH as usize, &mut seed);
    let w_gate_h = rand((WIDTH * INTER) as usize, &mut seed);
    let w_up_h = rand((WIDTH * INTER) as usize, &mut seed);
    let w_down_h = rand((INTER * WIDTH) as usize, &mut seed);

    let q_raw_b = upload(ctx, &q_raw);
    let k_raw_b = upload(ctx, &k_raw);
    let q_b = upload(ctx, &q_h);
    let k_b = upload(ctx, &k_h);
    let v_b = upload(ctx, &v_h);
    let (cos_b, sin_b) = (upload(ctx, &cos_h), upload(ctx, &sin_h));
    let x_b = upload(ctx, &x_h);
    let w_out_b = upload(ctx, &w_out_h);
    let g2_b = upload(ctx, &g2_h);
    let w_gate_b = upload(ctx, &w_gate_h);
    let w_up_b = upload(ctx, &w_up_h);
    let w_down_b = upload(ctx, &w_down_h);
    let zeros = {
        let z = vec![f16::from_f32(0.0); mus * WIDTH as usize];
        upload(ctx, &z)
    };
    drop(stream.synchronize());

    // ---- eager 参考（ARPE 模式下同样用 host-roped q/k——同一数理）----
    let ref_out = eager_layer(ctx, stream, m, &q_b, &k_b, &v_b, &x_b, &w_out_b, &g2_b, &zeros,
                              &w_gate_b, &w_up_b, &w_down_b);
    drop(stream.synchronize());

    // ---- GE 图 ----
    let (g, n_in) = build_ge_layer(m, use_arpe);
    println!("ge layer built: n_in={}", g.num_inputs().unwrap());

    let y = ctx.malloc((mus * WIDTH as usize) * 2).expect("y");
    // 第二输出 = RmsNorm rstd（fp32 [m,1]，按模型报告大小分配）
    let rstd = ctx.malloc(g.output_size(1).unwrap().max(16)).expect("rstd");
    let mut ins: Vec<&DeviceBuffer> = Vec::new();
    if use_arpe {
        ins.extend_from_slice(&[&q_raw_b, &k_raw_b, &v_b, &cos_b, &sin_b]);
    } else {
        ins.extend_from_slice(&[&q_b, &k_b, &v_b]);
    }
    ins.extend_from_slice(&[&x_b, &w_out_b, &g2_b, &zeros, &w_gate_b, &w_up_b, &w_down_b]);
    assert_eq!(ins.len(), n_in, "input binding count");

    g.run(&ins, &[&y, &rstd], stream).expect("run");
    drop(stream.synchronize());

    let elems = mus * WIDTH as usize;
    let ge_h = download_f16(ctx, &y, elems);
    let ref_h = download_f16(ctx, &ref_out, elems);

    let mut max_diff = 0f32;
    let mut max_at = 0usize;
    for (i, (a, b)) in ge_h.iter().zip(&ref_h).enumerate() {
        let d = (a.to_f32() - b.to_f32()).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    let ref_max = ref_h.iter().fold(0f32, |mx, v| mx.max(v.to_f32().abs()));
    println!(
        "parity GE-OM vs eager aclnn layer (m={m}): max_diff={max_diff:.5} @ {max_at} (ref |max|={ref_max:.3})"
    );
    println!("  ge[{max_at}]={:.4} ref[{max_at}]={:.4}", ge_h[max_at].to_f32(), ref_h[max_at].to_f32());
    assert!(max_diff < 0.05, "GE layer diverged: {max_diff}");

    // ---- bench ----
    if bench {
        for _ in 0..3 {
            g.run(&ins, &[&y, &rstd], stream).unwrap();
        }
        drop(stream.synchronize());
        let mut ge_ts = Vec::new();
        for r in 0..30 {
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                g.run(&ins, &[&y, &rstd], stream).unwrap();
            }
            drop(stream.synchronize());
            if r >= 3 {
                ge_ts.push(t0.elapsed().as_secs_f64() * 1000.0 / 10.0);
            }
        }
        ge_ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("GE OM layer: {:.4} ms/layer (median, m={m})", ge_ts[ge_ts.len() / 2]);

        // eager 同循环对照
        for _ in 0..3 {
            std::mem::drop(eager_layer(ctx, stream, m, &q_b, &k_b, &v_b, &x_b, &w_out_b, &g2_b,
                                       &zeros, &w_gate_b, &w_up_b, &w_down_b));
        }
        drop(stream.synchronize());
        let mut eg_ts = Vec::new();
        for r in 0..30 {
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                std::mem::drop(eager_layer(ctx, stream, m, &q_b, &k_b, &v_b, &x_b, &w_out_b, &g2_b,
                                           &zeros, &w_gate_b, &w_up_b, &w_down_b));
            }
            drop(stream.synchronize());
            if r >= 3 {
                eg_ts.push(t0.elapsed().as_secs_f64() * 1000.0 / 10.0);
            }
        }
        eg_ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("eager aclnn layer: {:.4} ms/layer (median, m={m})", eg_ts[eg_ts.len() / 2]);
    }

    ge_builder::fini().expect("fini");
    println!("GE_LAYER_PROBE_OK");
}
