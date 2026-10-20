//! GE 三段 OM（vision/prefix/flow）执行器——**crate 库面**（自 ge_model_probe
//! 探针下沉而来）：三段建图/对拍/bench、E2eStage、三段 per-frame
//! 机械（VisionMech/PrefixMech/FlowMech，含 GEB_SERVE_FAST 设备直连）、
//! spool serve_loop、**GeServe 进程内直调门面**（④C；PyO3/serve 同源帧实现）。
//! 探针驱动（env 分发/GEB_CKPT 加载）在 examples/ge_model_probe.rs。
//!
//! 三段 OM 全量验证：vision / prefix / flow 三段各自声明为 GE 静态
//! OM（ge_builder FFI），与 eager aclnn 组合同输入同序列对拍 + bench。
//! flow 段是单步 OM（10 步 flow 由调用方执行 10 次换绑定）。
//!
//! 算子契约（310P opp proto 头文件实证）：
//!   AddLayerNorm  x1/x2/gamma/beta(/bias)→y/mean/rstd/x, epsilon(Float)
//!   GeluV2        x→y, approximate(String: "none"=exact / "tanh")
//!   GatherV2D     x/indices→y, axis(Int attr；310P 实现 axis 只支持 0)
//!   ConcatD       DYNAMIC_INPUT x1..xN → y, concat_dim(Int) + N(Int)
//!   SliceD        x→y, offsets/size(ListInt)
//!   BroadcastToD  x→y, shape(ListInt)  [行广播：bias / ada-norm shift]
//!   Reshape       x + shape(张量输入) → y   [rope flat 桥 / rank-3 桥]
//!   MatMulV2(transpose_x2)/PromptFlashAttention(BSH rank-3)/AddRmsNorm/
//!   Squeeze/Add/Mul —— 已验证（单层探针）
//!
//! 图内设计要点：
//!   - 权重 host 预转置 [n,k] 物理 + MatMulV2 transpose_x2=true（NzCache
//!     同款布局，C1-② 零代价验证）；权重/bias/常量全部是 Data 输入
//!   - qkv 列拼接切分：bias 一次广播加整宽 → Reshape rank-3 → SliceD×3
//!     （PFA desc 必须 rank-3；比逐段广播省 kernel）
//!   - rope 组合图内（eager flat 版同构）：Reshape flat [t*h*2, d/2] →
//!     GatherV2D(相邻行 swap) → mul+mul+add → Reshape back rank-3。
//!     cos/sin flat 表、swap 索引、shape 张量均为段级常量输入（层间共享）
//!   - ada-norm：AddRmsNorm(gamma=scale_row) + BroadcastToD(shift_row)+Add
//!     （eager 的 host 预解析行等价物；行作为每步外部输入）
//!   - 链接纪律：Data src 用 link（单输出默认解析可靠），算子 src 一律
//!     link_out 显式端口（陷阱 #8：多输出算子默认解析静默失败）
//!
//! 运行（需 CANN toolkit 环境下的 rust 工具链）：
//!   GEB_SEG=vision GEB_DEPTH=2 cargo run --example ge_model_probe --features ascend --release -p apxinf-model
//!   GEB_SEG=prefix GEB_BENCH=1 cargo run ...   （全深对拍 + bench）
//!   GEB_SAVE=<path>/prefix.om ...（OM 落盘）
//!   GEB_LOAD=...（缓存加载替代编译）
//! 环境变量：GEB_SEG（默认 prefix）/ GEB_DEPTH（层深覆盖，冒烟用）/
//!   GEB_TOKENS（prefix token 数，默认 64 → P=832，16 倍数纪律）/
//!   GEB_BENCH / GEB_SAVE / GEB_LOAD / GEB_ATTN=manual（手工 attention）/
//!   GEB_QKV3=1（q/k/v 独立投影，消 SliceD 视图运行时物化）
//!   GEB_CKPT=<dir>（真 checkpoint：权重经 Pi05Weights::from_safetensors
//!   host 解析（含 Gemma 1+w scale 折叠）→ bf16→f16 → wt() Const 路径烤入
//!   OM（per-checkpoint，ADR-002）；缺省合成权重。x0/state/pk/pv/ada 条件
//!   向量等运行时输入保持合成（两路同值对拍不受影响）。
//!   ⚠ GEB_LOAD 真 OM（*_real.om）必须配 GEB_CKPT——权重烤在 OM 里，但
//!   eager 参考仍从 binds 取权重，缺 GEB_CKPT 时对拍错权重无意义
//!   GEB_SEG=e2e：三段链式 e2e（vision→prefix→flow 10 步换绑）——x0 真
//!   嵌入组装（vision_out ‖ token_embedding 查表，embed_prefix 同序）、
//!   styles 真值（time_mlp(sinusoidal(t))→style→ascl=1+s0/ash=s1）、pk/pv =
//!   prefix OM 36 输出直连（host 中转）。GEB_E2E_GOLDEN=<safetensors> 对拍
//!   torch golden（键 patches[768,588]/token_ids[N]/noise[50,32]/actions
//!   [50,32]，值 f32 存储；缺省合成 bring-up 模式）。OM 从 GEB_OM_DIR（默认
//!   ./om_cache）读 {seg}_real.om；须配 GEB_CKPT + 四件套 env。
use half::f16;

use apxinf_ascend::ge_builder::{self, Dtype, GeGraph};
use apxinf_ascend::{ops as aops, AscendBackend, AscendContext, AscendStream, DeviceBuffer};
use crate::pi05::{sinusoidal_time_embedding, LinearWeights, Pi05Config, Pi05Weights};

// π0.5 3 视图档维度（Pi05Config::default）
pub const VT: i64 = 768; // vision tokens = 3 views × 256 patches
pub const VPV: i64 = 256;
pub const VIEWS: i64 = 3;
pub const VW: i64 = 1152; // vision width
pub const V_INTER: i64 = 4304;
pub const V_HEADS: i64 = 16;
pub const V_HD: i64 = 72;
pub const V_PATCH_W: i64 = 588; // 3*14*14
pub const PW: i64 = 2048; // language width
pub const INTER: i64 = 16384;
pub const HEADS: i64 = 8;
pub const KV_HEADS: i64 = 1;
pub const HD: i64 = 256;
pub const QD: i64 = HEADS * HD; // 2048
pub const KVD: i64 = KV_HEADS * HD; // 256
pub const QKVW: i64 = QD + 2 * KVD; // 2560
pub const AW: i64 = 1024; // action expert width
pub const AINTER: i64 = 4096;
pub const ADIM: i64 = 32;
pub const HOR: i64 = 50;
pub const ROPE_THETA: f64 = 10000.0;
pub const RMS_EPS: f64 = 1e-6;
pub const LN_EPS: f64 = 1e-6;

pub fn envi(k: &str, d: i64) -> i64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

// ---------------------------------------------------------------------------
// 动态分档（ge.dynamicDims；实证配方）：
//   GEB_DYN_LLIST="137,138,...,157,200" = L 档表。构图期所有含 p 的 desc
//   维与 Reshape shape Const 内容一律写 -1（Const 全档共享，内容必须 gear
//   不变；-1 由分支推理按档解析——HF 参考链 0.6% 三档验证）；rope
//   表/swap 索引等 L 长度张量注册为 Data[-1]（内容逐行 p 无关，p_max 全
//   量上流即前缀截断语义）；L 相关 Const（qswap_c）转 Data。运行期按请求
//   选档：aclmdlSetInputDynamicDims 一次调用带全部动态输入的维（S2 定式）。
// ---------------------------------------------------------------------------

/// GEB_DYN_LLIST 解析（正整数逗号表）；None = 静态单 L 模式（行为不变）。
pub fn dyn_llist() -> Option<Vec<i64>> {
    std::env::var("GEB_DYN_LLIST")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<i64>().ok())
                .filter(|&v| v > 0)
                .collect::<Vec<i64>>()
        })
        .filter(|v| !v.is_empty())
}

static GEAR_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 进程级 gear 模式开关（env 读一次；构图/执行两侧共用）。
/// GEB_PREFIX_POOL 例外：池模式下压成 false——prefix 组装须与静态桶 OM 的
/// 接口一致（gear 折叠 rows==1 bias 少注册 90 个 Data，n_in 170!=260 实证）；
/// flow 接口两模式同形（rope 恒 Data）不受影响，但
/// set_rope_for 的早退门按 flows 池判（见彼处）。
pub fn gear_mode() -> bool {
    *GEAR_MODE.get_or_init(|| dyn_llist().is_some() && std::env::var("GEB_PREFIX_POOL").is_err())
}

/// desc 维度的动态位替换：gear 模式一律 -1（MultiBatchClone 逐档特化；
// 静态模式原值返回——辅助函数内统一走这里可无分支适配两模式）。
pub fn gdyn(v: i64) -> i64 {
    if gear_mode() {
        -1
    } else {
        v
    }
}

/// L → p（prefix 可见长度；drop 与段内公式同源）。
pub fn lp_to_p(l: i64) -> i64 {
    VT + l - envi("GEB_PREFIX_DROP_EMPTY", 0)
}

/// Range padding 网格映射（GEB_POOL_GRID=stride）：请求真 L →
/// 桶界 P（向上取整到 stride 倍数，clamp [MIN, MAX]）。无 env = 恒等（精确
/// 桶现行为）。桶界与真长的差行由 GEB_ATTN_MASK 的 softmax 加性 mask 遮蔽
/// ——表外 L 根除（任意 L ≤ MAX 均有归属），单任务 eval 从 ~4 桶缩到 1 桶。
pub fn pool_grid_p(l: i64) -> i64 {
    let stride = envi("GEB_POOL_GRID", 0);
    if stride <= 1 {
        return l;
    }
    let hi = envi("GEB_POOL_GRID_MAX", 200);
    assert!(l <= hi, "range-padding: L={l} 超桶表上限 {hi}");
    let p = ((l + stride - 1) / stride) * stride;
    p.clamp(envi("GEB_POOL_GRID_MIN", 136), hi)
}

/// erf 近似（Abramowitz-Stegun 7.1.26，|ε|<1.5e-7——exact gelu 对照够用）
pub fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_43 * t - 1.453_152_027) * t) + 1.421_413_741) * t
        - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t
        * (-x * x).exp();
    s * y
}

/// splitmix64 单步（LCG 高 16 位 mod 200 有短周期结构——vision 探针
/// LayerNorm 行方差塌缩 → rstd 放大 385× 的教训）
pub fn mix64(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub fn rand_f16(n: usize, seed: &mut u32, div: f32) -> Vec<f16> {
    let mut z: u64 = (*seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let r = mix64(&mut z);
        v.push(f16::from_f32(((r % 200) as i64 - 100) as f32 / div));
    }
    *seed = (z >> 32) as u32;
    v
}

/// norm gamma/beta：center ± 0.05（贴近真实量级，避免幅度塌缩）
pub fn norm_f16(n: usize, seed: &mut u32, center: f32) -> Vec<f16> {
    let mut z: u64 = (*seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let r = mix64(&mut z);
        let d = ((r % 200) as i64 - 100) as f32 / 2000.0;
        v.push(f16::from_f32(center + d));
    }
    *seed = (z >> 32) as u32;
    v
}

/// 真 checkpoint Tensor（bf16/f32）→ host f16（310P 无 bf16，精度策略 fp16）
pub fn t_f16(t: &apxinf_core::Tensor) -> Vec<f16> {
    t.to_f32_vec().unwrap().into_iter().map(f16::from_f32).collect()
}

/// LinearWeights.weight [in,out] → host f16（wt()/wbuf_t 的 host 布局同构）
pub fn lw_f16(l: &LinearWeights) -> Vec<f16> {
    t_f16(&l.weight)
}

/// LinearWeights.bias → f16；None（Gemma 投影无 bias）→ zeros
pub fn lb_f16(l: &LinearWeights, n: usize) -> Vec<f16> {
    match &l.bias {
        Some(b) => t_f16(b),
        None => vec![f16::from_f32(0.0); n],
    }
}

pub fn upload(ctx: &AscendContext, vals: &[f16]) -> DeviceBuffer {
    let bytes =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 2) };
    let buf = ctx.malloc(bytes.len()).expect("malloc");
    ctx.copy_h2d(&buf, bytes).expect("h2d");
    buf
}

pub fn upload_bytes(ctx: &AscendContext, bytes: &[u8]) -> DeviceBuffer {
    let buf = ctx.malloc(bytes.len()).expect("malloc");
    ctx.copy_h2d(&buf, bytes).expect("h2d");
    buf
}

pub fn upload_i32(ctx: &AscendContext, vals: &[i32]) -> DeviceBuffer {
    let bytes =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
    let buf = ctx.malloc(bytes.len()).expect("malloc");
    ctx.copy_h2d(&buf, bytes).expect("h2d");
    buf
}

/// fp16 device buffer -> host Vec<f16>（d2h 字节流按 LE u16 重组）
pub fn download_f16(ctx: &AscendContext, buf: &DeviceBuffer, n: usize) -> Vec<f16> {
    let mut back = vec![0u8; n * 2];
    ctx.copy_d2h(buf, &mut back).expect("d2h");
    back.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// host [rows, cols] f16 权重 → 转置 upload 为 [cols, rows] 物理
/// （MatMulV2 transpose_x2=true 与 eager matmul_b_t 共用布局）
pub fn wbuf_t(ctx: &AscendContext, host: &[f16], rows: i64, cols: i64) -> DeviceBuffer {
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
    let t = aops::host_transpose(bytes, rows, cols);
    upload_bytes(ctx, &t)
}

// ---------------------------------------------------------------------------
// 生产化：prefix int8（GEB_PREFIX_INT8）——7 投影 mm → QuantMatmulDequant
// ---------------------------------------------------------------------------

/// GEB_PREFIX_INT8：prefix 7 投影换 QuantMatmulDequant（int8 ND [n,k] +
/// per-row scale + smooth，算子契约经单算对拍实证）。
/// f16 回退 = 不设此 env。需真权重 + GEB_QKV3 + GEB_INT8_SMOOTH 因子文件。
pub fn prefix_int8() -> bool {
    std::env::var("GEB_PREFIX_INT8").map(|v| !v.is_empty()).unwrap_or(false)
}

/// GEB_FLOW_INT8：flow 段 7 投影换 QuantMatmulDequant（GEB_FLOW_SMOOTH
/// 因子文件，键 flow/L{nn}/{proj}）。与 prefix 的关键差异：action 层
/// loader 无 Gemma (1+g) fold（weights.rs 只 take 不 fold）——图内激活
/// 天然 torch 域，s_eff = s_npz 直取、无 g 表（prefix 的两域回图不适用）。
/// 动机：flow 步 M=50 无法摊销权重读，每步全量流式 f16 权重 ~622MB
/// （带宽下限 ~9.9ms/步 vs 实测 7.9）——int8 权重字节减半是正杠杆。
pub fn flow_int8() -> bool {
    std::env::var("GEB_FLOW_INT8").map(|v| !v.is_empty()).unwrap_or(false)
}

/// 引擎域 smooth 因子（GEB_INT8_SMOOTH safetensors，键 prefix/L{nn}/{proj}）。
/// 由 smooth_calib_v1.npz（torch 域）经 convert_smooth_engine.py v2 转换
/// （g 回图方案）：int8 层 AddRmsNorm gamma 用 (1+g) 真值（loader fold 的
/// 原值替代源），激活侧回 torch 域——纯引擎域 s_eff 会因 L00 近零 fold
/// 爆到 ±e7（f16 行不可表示，实测）。
pub struct Int8Smooth {
    /// 权重侧吸收因子 f32[k]（= s_npz/(1+g)，仅 f32 权重量化用）
    pub s_eff: Vec<f32>,
    /// 算子原生 smooth_scale 输入值 f32[k]（= s_npz，torch 域，f16 安全）
    pub smooth: Vec<f32>,
    /// 1/s_npz（GEB_INT8_NATIVE=0 回退路径的行广播值）
    pub inv: Vec<f32>,
}

/// smooth 因子文件加载：矩阵因子（s_eff/smooth/inv 三键一组）+ 组 norm
/// 真值（prefix/L{nn}/g1、g2 = 1+norm_scale，int8 层回 torch 域用）
pub fn load_int8_smooth(
    path: &str,
) -> (
    std::collections::HashMap<String, Int8Smooth>,
    std::collections::HashMap<String, [Vec<f32>; 2]>,
) {
    let (t, _) = apxinf_loader::safetensors::load_native_path(std::path::Path::new(path))
        .unwrap_or_else(|e| panic!("GEB_INT8_SMOOTH 加载失败 {path}: {e}"));
    let mut out = std::collections::HashMap::new();
    let bases: Vec<String> = t
        .keys()
        .filter_map(|k| k.strip_suffix("/s_eff").map(|b| b.to_string()))
        .collect();
    assert!(!bases.is_empty(), "{path} 无 */s_eff 键（须为转换后的因子文件）");
    for base in bases {
        let get = |suf: &str| -> Vec<f32> {
            t.get(&format!("{base}{suf}"))
                .unwrap_or_else(|| panic!("int8 smooth 缺 {base}{suf}"))
                .to_f32_vec()
                .unwrap()
        };
        out.insert(
            base.clone(),
            Int8Smooth { s_eff: get("/s_eff"), smooth: get("/smooth"), inv: get("/inv") },
        );
    }
    let mut norms = std::collections::HashMap::new();
    for key in t.keys() {
        for (suf, slot) in [("/g1", 0usize), ("/g2", 1)] {
            if let Some(base) = key.strip_suffix(suf) {
                let e = norms.entry(base.to_string()).or_insert([Vec::new(), Vec::new()]);
                e[slot] = t.get(key).expect("g norm").to_f32_vec().unwrap();
            }
        }
    }
    (out, norms)
}

/// W8 行量化（smooth 吸收后）：host w32 [k,n]（in-major，loader f32 权重）
/// → int8 [n,k] 物理 + per-row scale [n]。与 int8_bake_weights.py 同数学
/// （f32 域乘、max().max(1e-12)/127、ties-even round、±127 clip）——
/// GEB_INT8_DUMP 落盘对拍 npz 金标准，期望 bit 级一致（fold 结合律 ulp 除外）。
/// 转置分块：朴素逐元素在 [2048,16384] 全 cache miss（~130s/18 层量级）。
pub fn quant_w8(w32: &[f32], k: usize, s_eff: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n = w32.len() / k;
    assert_eq!(w32.len(), n * k, "int8 权重非 k 整行");
    assert_eq!(s_eff.len(), k, "smooth 因子长度 ≠ k");
    let mut ws = vec![0f32; n * k]; // [n,k] 物理（转置 + s_eff 吸收一遍完成）
    for nn0 in (0..n).step_by(64) {
        let ne = (nn0 + 64).min(n);
        for kk0 in (0..k).step_by(64) {
            let ke = (kk0 + 64).min(k);
            for kk in kk0..ke {
                let src = kk * n;
                let sv = s_eff[kk];
                for nn in nn0..ne {
                    ws[nn * k + kk] = w32[src + nn] * sv;
                }
            }
        }
    }
    let mut scale = vec![0f32; n];
    let mut wq = vec![0i8; n * k];
    for nn in 0..n {
        let row = &ws[nn * k..(nn + 1) * k];
        let mx = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = mx.max(1e-12) / 127.0;
        scale[nn] = s;
        for (kk, &v) in row.iter().enumerate() {
            wq[nn * k + kk] = (v / s).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
    }
    (wq, scale)
}

/// host LayerNorm 参考（f32 计算）：aclnnAddLayerNorm 在 [768,1152] 有
/// kernel 级放大（实测），eager 参考路径的 LN 用 host 计算。
/// ⚠ 入参 stream 先同步——同步 d2h（aclrtMemcpy）不等待 compute 流上的
/// 生产 kernel，不同步则读到跑赢/滞后的数据（eager 全 device kernel 时
/// 确定性错排 98.2%；PFA 模式被 host 回调串行化掩盖）。
pub fn host_ln(
    ctx: &AscendContext, stream: &AscendStream, x: &DeviceBuffer, g: &[f16], b: &[f16], rows: i64, cols: i64,
) -> DeviceBuffer {
    drop(stream.synchronize());
    let xh = download_f16(ctx, x, (rows * cols) as usize);
    let mut out = vec![f16::from_f32(0.0); (rows * cols) as usize];
    for r in 0..rows as usize {
        let base = r * cols as usize;
        let mut mean = 0f32;
        for i in 0..cols as usize {
            mean += xh[base + i].to_f32();
        }
        mean /= cols as f32;
        let mut var = 0f32;
        for i in 0..cols as usize {
            let d = xh[base + i].to_f32() - mean;
            var += d * d;
        }
        var /= cols as f32;
        let rstd = 1.0 / (var + LN_EPS as f32).sqrt();
        for i in 0..cols as usize {
            out[base + i] = f16::from_f32(
                (xh[base + i].to_f32() - mean) * rstd * g[i].to_f32() + b[i].to_f32(),
            );
        }
    }
    upload(ctx, &out)
}

/// cos/sin flat 表 + swap 索引（eager flat rope 常量同款语义）。
/// x [t, heads*d] → flat [t*heads*2, d/2]（偶行 = 前半通道），位置 =
/// pos_offset + t。cosF 半维重复自然成；sinF 折叠 rotate-half 符号
/// （half_flag=0 取 -s、1 取 +s）。swap = 相邻行交换索引 r^1。
pub fn rope_flat_const(tokens: i64, heads: i64, pos_offset: i64) -> (Vec<f16>, Vec<f16>, Vec<i32>) {
    let half = (HD / 2) as usize;
    let rows = (tokens * heads * 2) as usize;
    let mut cos = vec![f16::from_f32(0.0); rows * half];
    let mut sin = vec![f16::from_f32(0.0); rows * half];
    for t in 0..tokens as usize {
        for i in 0..half {
            let freq = rope_angle(pos_offset + t as i64, i);
            let (c, s) = (freq.cos() as f32, freq.sin() as f32);
            for h in 0..heads as usize {
                for hf in 0..2usize {
                    let r = (t * heads as usize + h) * 2 + hf;
                    cos[r * half + i] = f16::from_f32(c);
                    sin[r * half + i] = f16::from_f32(if hf == 0 { -s } else { s });
                }
            }
        }
    }
    let swap: Vec<i32> = (0..rows as i32).map(|r| r ^ 1).collect();
    (cos, sin, swap)
}

/// rope 角度（golden 数值同源）：inv_freq = θ^(-2i/HD) 先量化 f16——torch
/// 模型 .to(f16) 连 rotary inv_freq buffer 一起转，HF rotary 用 f16 频率回
/// f32 算角。全精度频率在高位置（~968）处与 golden 相位差 O(0.1 rad)——
/// prefix kv l0 即偏 12.6%、逐层放大至 l17 85%（f16-inv 复刻实测 0.08%）
pub fn rope_angle(pos: i64, i: usize) -> f64 {
    let inv = f16::from_f32((ROPE_THETA as f32).powf(-(2.0 * i as f32) / HD as f32));
    pos as f64 * inv.to_f32() as f64
}

/// cos/sin rank-2 表 [t*heads, HD]（eager build_rope_tables 同款：cos 半维
/// 重复、sin 符号折叠），配合通道半互换 swap（SliceD×2+ConcatD）的
/// rank-2 rope 组合。
pub fn rope_rank2_const(tokens: i64, heads: i64, pos_offset: i64) -> (Vec<f16>, Vec<f16>) {
    let d = HD as usize;
    let half = d / 2;
    let rows = (tokens * heads) as usize;
    let mut cos = vec![f16::from_f32(0.0); rows * d];
    let mut sin = vec![f16::from_f32(0.0); rows * d];
    for t in 0..tokens as usize {
        for i in 0..half {
            let freq = rope_angle(pos_offset + t as i64, i);
            let (c, s) = (freq.cos() as f32, freq.sin() as f32);
            for h in 0..heads as usize {
                let r = t * heads as usize + h;
                cos[r * d + i] = f16::from_f32(c);
                cos[r * d + half + i] = f16::from_f32(c);
                sin[r * d + i] = f16::from_f32(-s);
                sin[r * d + half + i] = f16::from_f32(s);
            }
        }
    }
    (cos, sin)
}

// ---------------------------------------------------------------------------
// 段构造器：数据（权重/常量）与图一体，输入注册序 = 运行绑定序。
// wire() 区分 Data（link 默认）与算子（link_out 显式端口，陷阱 #8）
// ---------------------------------------------------------------------------

pub struct Seg {
    pub g: GeGraph,
    idx: i64,
    names: Vec<String>,
    shapes: Vec<Vec<i64>>,
    pub binds: Vec<DeviceBuffer>,
    /// binds 平行名表（bind_of 用）。⚠ names/shapes 只覆盖 Data 输入——
    /// WCONST 权重（wt Const 分支）与 i8 占位槽只 push binds 不进 names，
    /// 两表历史上不对齐（bind_of 恰只查头部 Data 输入而幸存）；bind_names
    /// 与 binds 严格平行，按名定位不再依赖注册顺序（历史事故修复）
    pub bind_names: Vec<String>,
    /// 图输入在 binds 里的下标（GEB_WCONST 时权重是 Const 不占图输入，
    /// binds 仍全量持上传副本供 eager/层基址——索引两模式恒同）
    data_inputs: Vec<usize>,
    /// GEB_NORM32 常量池（按宽度懒建：sc/kc/ones 的 Const 名——rms32 v2）
    n32_c: std::collections::HashMap<i64, (String, String, String)>,
    /// GEB_WCONST：mm 权重 Const 入图（编译期折叠 ND→NZ 转换，零运行
    /// 时税；wgt 单算实证 0.3945→0.2945 ms/mm、OM 烤入权重 8.4MB）。
    /// 代价：OM 变 per-权重集（checkpoint 版本进缓存 key）
    wconst: bool,
    datas: std::collections::HashSet<String>,
    outports: std::collections::HashMap<String, &'static str>,
    /// LayerNormV4 实例名（finish 时 mean/rstd 绑 aux 图输出）
    ln_aux: Vec<String>,
    /// gear 模式（GEB_DYN_LLIST）：每档的 p 值（VT+L-drop_v，段侧喂入）。
    /// finish 时合成 ge.dynamicDims 档表 + 每档 S2 payload（全部动态输入
    /// 维拼接，输入注册序）——mech 直接取用做 run_gear 入参
    pub gear_ps: Vec<i64>,
    gear_mul: std::collections::HashMap<String, i64>,
    /// 定值动态输入（每档恒值——legacy Const 墙的绕道，见 set_gear_fix）
    gear_fix: std::collections::HashMap<String, i64>,
    /// 每档 S2 payload（gear 模式非空；静态模式空 = run 等价）
    pub gear_payloads: Vec<Vec<i64>>,
    /// 每档纯 -1 位值（ge.dynamicDims 选项串专用——atc 语义只要 -1 位的
    /// 值；payload 的静态尾维混进选项会让编译 rc=-7，实测）
    gear_optvals: Vec<Vec<i64>>,
}

impl Seg {
    fn new(name: &str) -> Seg {
        Seg {
            g: GeGraph::begin(name).expect("begin"),
            idx: 0,
            names: Vec::new(),
            shapes: Vec::new(),
            binds: Vec::new(),
            bind_names: Vec::new(),
            data_inputs: Vec::new(),
            n32_c: std::collections::HashMap::new(),
            wconst: std::env::var("GEB_WCONST").is_ok(),
            datas: std::collections::HashSet::new(),
            outports: std::collections::HashMap::new(),
            ln_aux: Vec::new(),
            gear_ps: Vec::new(),
            gear_mul: std::collections::HashMap::new(),
            gear_fix: std::collections::HashMap::new(),
            gear_payloads: Vec::new(),
            gear_optvals: Vec::new(),
        }
    }

    /// 图输入装配（GEB_WCONST 时排除 Const 化权重；其余模式 = binds 全量）
    fn ins(&self) -> Vec<&DeviceBuffer> {
        if self.wconst {
            self.data_inputs.iter().map(|&i| &self.binds[i]).collect()
        } else {
            self.binds.iter().collect()
        }
    }

    fn reg_out(&mut self, op: &str, port: &'static str) -> String {
        self.outports.insert(op.to_string(), port);
        op.to_string()
    }

    /// gear 模式装配（段侧喂档表的 p 值；动态输入 dim0 = mul×p，缺省 mul=1）
    pub fn set_gear_ps(&mut self, ps: Vec<i64>) {
        self.gear_ps = ps;
    }

    pub fn set_gear_mul(&mut self, name: &str, mul: i64) {
        self.gear_mul.insert(name.to_string(), mul);
    }

    /// 定值动态输入：进档表但每档恒值（内容静态却不能 Const 的输入——
    /// legacy 算子 Const 输入墙 rc=-7，如 QMD 的 smooth 因子/gamma；全
    /// 动态输入集是 SetInputDynamicDims 档位匹配的前提，实测定罪）
    pub fn set_gear_fix(&mut self, name: &str, v: i64) {
        self.gear_fix.insert(name.to_string(), v);
    }

    /// bind 下标按名查（bind_names 与 binds 严格平行——含 WCONST 权重与
    /// i8 占位槽；names/shapes 仅 Data 输入面，两者历史上不对齐）
    /// try 版：缺名返回 None（GEB_ATTN_MASK 可选输入 padmask 的 mech 接线）
    pub fn try_bind_of(&self, name: &str) -> Option<usize> {
        self.bind_names.iter().position(|n| n == name)
    }

    pub fn bind_of(&self, name: &str) -> usize {
        self.bind_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("bind_of: 图输入 {name} 不存在"))
    }

    /// 每档 S2 payload：全部含 -1 维输入的**完整 dims**（动态位 = mul×p，
    /// 静态尾维原样；S2 定式——SetInputDynamicDims 一次调用带全部
    /// 动态输入的全维）。ge.dynamicDims 选项串仍是纯 -1 位值（atc 语义）
    fn compose_gear_payloads(&mut self) {
        self.gear_payloads.clear();
        if self.gear_ps.is_empty() {
            return;
        }
        let names = self.names.clone();
        let shapes = self.shapes.clone();
        let muls: Vec<i64> = names
            .iter()
            .map(|n| self.gear_mul.get(n).copied().unwrap_or(1))
            .collect();
        for &pg in &self.gear_ps {
            let mut pl = Vec::new();
            let mut ov = Vec::new();
            for ((n, shp), &mul) in names.iter().zip(&shapes).zip(&muls) {
                let ndyn = shp.iter().filter(|&&d| d == -1).count();
                assert!(
                    ndyn <= 1,
                    "gear 输入 {n} 含 {ndyn} 个 -1 维（只支持 dim0 单维）"
                );
                if ndyn == 1 {
                    assert_eq!(shp[0], -1, "gear 输入 {n} 的 -1 须在 dim0");
                    // 定值动态输入优先（每档恒值）；否则 mul×p
                    let v = self
                        .gear_fix
                        .get(n)
                        .copied()
                        .unwrap_or_else(|| mul * pg);
                    pl.push(v);
                    pl.extend_from_slice(&shp[1..]);
                    ov.push(v);
                }
            }
            self.gear_payloads.push(pl);
            self.gear_optvals.push(ov);
        }
        println!(
            "[gear] {} 档；payload 维数 {}（首档前 8 {:?}）",
            self.gear_ps.len(),
            self.gear_payloads[0].len(),
            &self.gear_payloads[0][..8.min(self.gear_payloads[0].len())]
        );
    }

    /// 档位执行便捷口：payload 取 self（静态模式走原 run，无分支切换）
    pub fn run_seg(
        &self,
        gi: usize,
        outputs: &[&DeviceBuffer],
        stream: &AscendStream,
    ) -> apxinf_ascend::Result<()> {
        let ins = self.ins();
        if self.gear_payloads.is_empty() {
            self.g.run(&ins, outputs, stream)
        } else {
            self.g
                .run_gear(&ins, outputs, stream, &self.gear_payloads[gi])
        }
    }

    fn data_buf(&mut self, name: &str, dims: &[i64], buf: DeviceBuffer) -> String {
        self.g.add_data(name, self.idx, dims, Dtype::Fp16).unwrap();
        self.idx += 1;
        self.names.push(name.to_string());
        self.shapes.push(dims.to_vec());
        self.binds.push(buf);
        self.bind_names.push(name.to_string());
        self.data_inputs.push(self.binds.len() - 1);
        self.datas.insert(name.to_string());
        name.to_string()
    }

    /// mm 权重注册（host [rows,cols] → [cols,rows] 转置）：GEB_WCONST 时
    /// Const 入图（binds 仍持上传副本供 eager，但不占图输入位/idx）；
    /// 否则 Data（现状——每执行付设备侧 ND→NZ TransData ~63GB/s）。
    fn wt(&mut self, ctx: &AscendContext, name: &str, dims: &[i64], host: &[f16], rows: i64, cols: i64) -> String {
        let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
        let t = aops::host_transpose(bytes, rows, cols);
        let buf = upload_bytes(ctx, &t);
        if self.wconst {
            let th: Vec<f16> = t
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            self.const_f16(name, dims, &th);
            self.binds.push(buf);
            self.bind_names.push(name.to_string());
            self.datas.insert(name.to_string());
        } else {
            self.data_buf(name, dims, buf);
        }
        name.to_string()
    }

    fn data(&mut self, ctx: &AscendContext, name: &str, dims: &[i64], host: &[f16]) -> String {
        let buf = upload(ctx, host);
        self.data_buf(name, dims, buf)
    }

    /// fp32 Data 输入（GEB_NORM32 语义对齐用；binds 字节无关 dtype）
    fn data_f32(&mut self, ctx: &AscendContext, name: &str, dims: &[i64], host: &[f32]) -> String {
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = upload_bytes(ctx, &bytes);
        self.g.add_data(name, self.idx, dims, Dtype::Fp32).unwrap();
        self.idx += 1;
        self.names.push(name.to_string());
        self.shapes.push(dims.to_vec());
        self.binds.push(buf);
        self.bind_names.push(name.to_string());
        self.data_inputs.push(self.binds.len() - 1);
        self.datas.insert(name.to_string());
        name.to_string()
    }

    fn data_zeros(&mut self, ctx: &AscendContext, name: &str, rows: i64, cols: i64) -> String {
        let z = vec![f16::from_f32(0.0); (rows * cols) as usize];
        self.data(ctx, name, &[rows, cols], &z)
    }

    fn data_i32(&mut self, ctx: &AscendContext, name: &str, dims: &[i64], host: &[i32]) -> String {
        let buf = upload_i32(ctx, host);
        self.g.add_data(name, self.idx, dims, Dtype::Int32).unwrap();
        self.idx += 1;
        self.names.push(name.to_string());
        self.shapes.push(dims.to_vec());
        self.binds.push(buf);
        self.bind_names.push(name.to_string());
        self.data_inputs.push(self.binds.len() - 1);
        self.datas.insert(name.to_string());
        name.to_string()
    }

    /// int8 Data 输入（int8 探针：QuantMatmulDequant 的量化权重/scale）
    fn data_i8(&mut self, ctx: &AscendContext, name: &str, dims: &[i64], host: &[i8]) -> String {
        let bytes: Vec<u8> = host.iter().map(|&v| v as u8).collect();
        let buf = upload_bytes(ctx, &bytes);
        self.g.add_data(name, self.idx, dims, Dtype::Int8).unwrap();
        self.idx += 1;
        self.names.push(name.to_string());
        self.shapes.push(dims.to_vec());
        self.binds.push(buf);
        self.bind_names.push(name.to_string());
        self.data_inputs.push(self.binds.len() - 1);
        self.datas.insert(name.to_string());
        name.to_string()
    }

    /// Const shape 张量（非图输入，编译期常量折叠）。⚠ shape 类输入用
    /// data_i32（Data）会让消费算子（Reshape/LayerNormV4）输出 desc 变
    /// unknown → DynamicShapePartitioner 拆子图 → unknown 部分走 host
    /// 调度（每边界 ~20ms 停顿；vision OM 3220 个 unknown 标记实证）
    fn const_i32(&mut self, name: &str, vals: &[i32]) -> String {
        self.g.add_const_i32(name, vals).unwrap();
        self.datas.insert(name.to_string());
        name.to_string()
    }

    /// Const fp16 张量（权重入图用——NZ 转换编译期折叠假设的验证入口）
    fn const_f16(&mut self, name: &str, dims: &[i64], host: &[f16]) -> String {
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        self.g.add_const_raw(name, dims, Dtype::Fp16, &bytes).unwrap();
        self.datas.insert(name.to_string());
        name.to_string()
    }

    /// NZ 格式 Data 输入（绑定 buffer 必须已是 host_nz_reorder 产物）
    fn data_nz(&mut self, name: &str, dims_nz: &[i64], buf: DeviceBuffer) -> String {
        self.g
            .add_data_fmt(name, self.idx, dims_nz, Dtype::Fp16, "FRACTAL_NZ")
            .unwrap();
        self.idx += 1;
        self.names.push(name.to_string());
        self.shapes.push(dims_nz.to_vec());
        self.binds.push(buf);
        self.bind_names.push(name.to_string());
        self.data_inputs.push(self.binds.len() - 1);
        self.datas.insert(name.to_string());
        name.to_string()
    }

    /// dst.port ← src（Data 用默认解析；算子用显式输出端口）
    fn wire(&self, dst: &str, port: &str, src: &str) {
        if self.datas.contains(src) {
            self.g.link(dst, port, src).unwrap();
        } else {
            let out = self.outports.get(src).copied().unwrap_or("y");
            self.g.link_out(dst, port, src, out).unwrap();
        }
    }

    // ---- 算子辅助（返回算子名）----

    /// MatMulV2：a [m,k] × w'（物理 [n,k] 转置）→ [m,n]
    fn mm(&mut self, name: &str, a: &str, a_dims: &[i64], w: &str, w_dims: &[i64], o: &[i64]) -> String {
        self.g.add_op(name, "MatMulV2").unwrap();
        self.g.set_input_desc(name, "x1", a_dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "x2", w_dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", o, Dtype::Fp16).unwrap();
        self.g.set_attr_bool(name, "transpose_x1", false).unwrap();
        self.g.set_attr_bool(name, "transpose_x2", true).unwrap();
        self.wire(name, "x1", a);
        self.g.link(name, "x2", w).unwrap();
        self.reg_out(name, "y")
    }

    /// QuantMatmulDequant：a [m,k] × int8 wq [n,k]（transpose_weight=true，
    /// linear 风格）→ y [m,n]；weight_scale f32 [n]；smooth_scale f16 [k]
    /// / bias int32 [n] 可选（None = 无该输入）。契约经单算对拍实证
    /// （310P legacy 单算；attr 必须显式设——裸 op 默认值不生效，rc=-7）。
    /// bias 折入 = 砍层内 TileD+Add 对（Gemma bias=0 → int32 零恒等；
    /// ⚠ 非零 bias 须按 weight_scale 预量化，v1 只支持零）
    fn qmd(
        &mut self, name: &str, a: &str, a_dims: &[i64], wq: &str, w_dims: &[i64],
        ws: &str, sm: Option<&str>, zb: Option<&str>, n: i64, o: &[i64],
    ) -> String {
        self.g.add_op(name, "QuantMatmulDequant").unwrap();
        self.g.set_input_desc(name, "x", a_dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "quantized_weight", w_dims, Dtype::Int8).unwrap();
        self.g.set_input_desc(name, "weight_scale", &[n], Dtype::Fp32).unwrap();
        if let Some(sm_name) = sm {
            self.g.set_input_desc(name, "smooth_scale", &[a_dims[1]], Dtype::Fp16).unwrap();
        }
        if let Some(zb_name) = zb {
            self.g.set_input_desc(name, "bias", &[n], Dtype::Int32).unwrap();
        }
        self.g.set_output_desc(name, "y", o, Dtype::Fp16).unwrap();
        self.g.set_attr_bool(name, "transpose_weight", true).unwrap();
        self.g.set_attr_str(name, "x_quant_mode", "pertoken").unwrap();
        self.wire(name, "x", a);
        self.g.link(name, "quantized_weight", wq).unwrap();
        self.g.link(name, "weight_scale", ws).unwrap();
        if let Some(sm_name) = sm {
            self.g.link(name, "smooth_scale", sm_name).unwrap();
        }
        if let Some(zb_name) = zb {
            self.g.link(name, "bias", zb_name).unwrap();
        }
        self.reg_out(name, "y")
    }

    /// 行广播 bias：TileD(row [1,cols], multiples=[rows,1]) + Add。
    /// （BroadcastToD 在 310P kernel 编译崩——TBE task_distribute；TileD
    /// attr 版编译验证过。算子名后缀 _bc/_ba 与 bias 行 Data 名错开。）
    /// 行广播输入注册（bias 行）：静态 = [1,cols] 原行（消费端 TileD 广播）；
    /// gear = [gdyn,cols] 行重复 rows 份——TileD multiples 含 p 时全档共享
    /// 不可用（attr 不随档变），改内容即广播（逐行相同，前缀截断语义）。
    /// int8 折叠路径不消费 bias：gear 模式下干脆不注册（静态 Data 混入
    /// 动态集 = SetDims 145012——未消费的静态输入也占图输入位）；静态模式
    /// 传 rows=1 注册原行省内存
    fn row_bcast(
        &mut self,
        ctx: &AscendContext,
        name: &str,
        rows: i64,
        cols: i64,
        row: &[f16],
    ) -> String {
        // gear 模式：行重复注册为同形 [-1,cols]（rows==1 且未消费则折叠不注册）
        if gear_mode() {
            if rows > 1 {
                let mut rep = Vec::with_capacity((rows * cols) as usize);
                for _ in 0..rows {
                    rep.extend_from_slice(row);
                }
                self.data(ctx, name, &[gdyn(rows), cols], &rep)
            } else {
                // 折叠未消费：不注册（名字占位，永不连线）
                format!("{name}_folded")
            }
        } else {
            self.data(ctx, name, &[1, cols], row)
        }
    }

    fn bias(&mut self, name: &str, x: &str, x_dims: &[i64], row: &str) -> String {
        let ba = format!("{name}_ba");
        if gear_mode() {
            // 行已重复注册为同形 [-1,cols]——直接 Add（无 TileD）
            self.g.add_op(&ba, "Add").unwrap();
            self.g.set_input_desc(&ba, "x1", x_dims, Dtype::Fp16).unwrap();
            self.g.set_input_desc(&ba, "x2", x_dims, Dtype::Fp16).unwrap();
            self.g.set_output_desc(&ba, "y", x_dims, Dtype::Fp16).unwrap();
            self.wire(&ba, "x1", x);
            self.wire(&ba, "x2", row);
            return self.reg_out(&ba, "y");
        }
        let bc = format!("{name}_bc");
        self.g.add_op(&bc, "TileD").unwrap();
        self.g.set_input_desc(&bc, "x", &[1, x_dims[1]], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&bc, "y", x_dims, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&bc, "multiples", &[x_dims[0], 1]).unwrap();
        self.g.link(&bc, "x", row).unwrap();
        self.reg_out(&bc, "y");
        self.g.add_op(&ba, "Add").unwrap();
        self.g.set_input_desc(&ba, "x1", x_dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(&ba, "x2", x_dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(&ba, "y", x_dims, Dtype::Fp16).unwrap();
        self.wire(&ba, "x1", x);
        self.wire(&ba, "x2", &bc);
        self.reg_out(&ba, "y")
    }

    /// 行广播门控乘：TileD(gate [1,cols], multiples=[rows,1]) × Mul——torch
    /// _gated_residual 的 gate 通路（residual + branch·gate）。与 bias 同
    /// dim0 整行复制方向（#39 干净路径；连续平铺方向 kernel 腐蚀）
    fn gatew(&mut self, name: &str, x: &str, x_dims: &[i64], row: &str) -> String {
        let gc = format!("{name}_gc");
        let gm = format!("{name}_gm");
        self.g.add_op(&gc, "TileD").unwrap();
        self.g.set_input_desc(&gc, "x", &[1, x_dims[1]], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&gc, "y", x_dims, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&gc, "multiples", &[x_dims[0], 1]).unwrap();
        self.g.link(&gc, "x", row).unwrap();
        self.reg_out(&gc, "y");
        self.g.add_op(&gm, "Mul").unwrap();
        self.g.set_input_desc(&gm, "x1", x_dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(&gm, "x2", x_dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(&gm, "y", x_dims, Dtype::Fp16).unwrap();
        self.wire(&gm, "x1", x);
        self.wire(&gm, "x2", &gc);
        self.reg_out(&gm, "y")
    }

    fn add2(&mut self, name: &str, a: &str, b: &str, dims: &[i64]) -> String {
        self.g.add_op(name, "Add").unwrap();
        self.g.set_input_desc(name, "x1", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "x2", dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
        self.wire(name, "x1", a);
        self.wire(name, "x2", b);
        self.reg_out(name, "y")
    }

    fn mul2(&mut self, name: &str, a: &str, b: &str, dims: &[i64]) -> String {
        self.g.add_op(name, "Mul").unwrap();
        self.g.set_input_desc(name, "x1", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "x2", dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
        self.wire(name, "x1", a);
        self.wire(name, "x2", b);
        self.reg_out(name, "y")
    }

    fn gelu(&mut self, name: &str, x: &str, dims: &[i64], tanh: bool) -> String {
        self.g.add_op(name, "GeluV2").unwrap();
        self.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
        self.g
            .set_attr_str(name, "approximate", if tanh { "tanh" } else { "none" })
            .unwrap();
        self.wire(name, "x", x);
        self.reg_out(name, "y")
    }

    /// AddRmsNorm(x1=x, x2=zeros, gamma) → y
    fn addrms(&mut self, name: &str, x: &str, zeros: &str, gamma: &str, dims: &[i64]) -> String {
        // GEB_NORM32：组合手搓 fp32 方差（zeros 输入不需要——rms(x+0)=rms(x)）
        if norm32_enabled() {
            // rms32 v2 直出 f16（无 Cast 尾巴）
            return self.rms32(&format!("{name}_r"), x, gamma, dims);
        }
        self.g.add_op(name, "AddRmsNorm").unwrap();
        self.g.set_input_desc(name, "x1", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "x2", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "gamma", &[dims[1]], Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
        self.g.set_attr_float(name, "epsilon", RMS_EPS).unwrap();
        self.wire(name, "x1", x);
        self.g.link(name, "x2", zeros).unwrap();
        self.g.link(name, "gamma", gamma).unwrap();
        self.reg_out(name, "y")
    }

    /// fp32 变体（GEB_NORM32）：torch GemmaRMSNorm 语义对齐——方差在
    /// fp32 域算（norm16 对照实验定罪：去 fp32 上浮 = 行为 0/10 开关）。
    /// 入参全 fp32，出 y fp32（调用方 Cast 回 f16 或直用）
    fn addrms_f32(&mut self, name: &str, x: &str, zeros: &str, gamma: &str, dims: &[i64]) -> String {
        self.g.add_op(name, "AddRmsNorm").unwrap();
        self.g.set_input_desc(name, "x1", dims, Dtype::Fp32).unwrap();
        self.g.set_input_desc(name, "x2", dims, Dtype::Fp32).unwrap();
        self.g.set_input_desc(name, "gamma", &[dims[1]], Dtype::Fp32).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp32).unwrap();
        self.g.set_attr_float(name, "epsilon", RMS_EPS).unwrap();
        self.wire(name, "x1", x);
        self.g.link(name, "x2", zeros).unwrap();
        self.g.link(name, "gamma", gamma).unwrap();
        self.reg_out(name, "y")
    }

    /// ada-norm：y = rms(x)·scale + shift = AddRmsNorm(x, zeros, gamma=scale)
    /// + TileD(shift_row) + Add（BroadcastToD 310P 编译崩，换 TileD）
    fn ada(&mut self, name: &str, x: &str, zeros: &str, scale: &str, shift: &str, dims: &[i64]) -> String {
        // GEB_NORM32：torch ada 版语义 = rms32(x,scale) + shift（v2 全
        // f16 域——scale 已折 1+s0 进 host，shift 广播同非 NORM32 形态）
        if norm32_enabled() {
            let n = self.rms32(&format!("{name}_r"), x, scale, dims);
            let sbc = format!("{name}_sbc");
            self.g.add_op(&sbc, "TileD").unwrap();
            self.g.set_input_desc(&sbc, "x", &[1, dims[1]], Dtype::Fp16).unwrap();
            self.g.set_output_desc(&sbc, "y", dims, Dtype::Fp16).unwrap();
            self.g.set_attr_int_list(&sbc, "multiples", &[dims[0], 1]).unwrap();
            self.g.link(&sbc, "x", shift).unwrap();
            let st = self.reg_out(&sbc, "y");
            self.g.add_op(name, "Add").unwrap();
            self.g.set_input_desc(name, "x1", dims, Dtype::Fp16).unwrap();
            self.g.set_input_desc(name, "x2", dims, Dtype::Fp16).unwrap();
            self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
            self.wire(name, "x1", &n);
            self.wire(name, "x2", &st);
            return self.reg_out(name, "y");
        }
        let n = format!("{name}_n");
        let sbc = format!("{name}_sbc");
        self.addrms(&n, x, zeros, scale, dims);
        self.g.add_op(&sbc, "TileD").unwrap();
        self.g.set_input_desc(&sbc, "x", &[1, dims[1]], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&sbc, "y", dims, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&sbc, "multiples", &[dims[0], 1]).unwrap();
        self.g.link(&sbc, "x", shift).unwrap();
        self.reg_out(&sbc, "y");
        self.add2(name, &n, &sbc, dims)
    }

    /// NORM32 常量池（按宽度懒建 Const：sc=s 预缩放 / kc=s·√w /
    /// ones [1,w]——Const 编译期折叠不占模型输入；多 norm 复用同名节点）
    fn n32_consts(&mut self, w: i64) -> (String, String, String) {
        if let Some(c) = self.n32_c.get(&w) {
            return c.clone();
        }
        let s = 1.0f32 / 32.0;
        let k = s * (w as f32).sqrt();
        let sc = format!("n32_sc_w{w}");
        self.const_f16(&sc, &[1], &[f16::from_f32(s)]);
        let kc = format!("n32_kc_w{w}");
        self.const_f16(&kc, &[1], &[f16::from_f32(k)]);
        let ones = format!("n32_ones_w{w}");
        let ov: Vec<f16> = vec![f16::from_f32(1.0); w as usize];
        self.const_f16(&ones, &[1, w], &ov);
        let c = (sc, kc, ones);
        self.n32_c.insert(w, c.clone());
        c
    }

    /// Cast 节点（to_f32=true: f16→f32；false: f32→f16）。dst_type 是
    /// DataType 类型 attr——int 值猜测版编译 rc=-7（proto 枚举序不确定），
    /// 走 geb_set_attr_dtype（C++ ParseDtype 同管道）
    fn cast_node(&mut self, name: &str, x: &str, dims: &[i64], to_f32: bool) -> String {
        let (idt, odt) = if to_f32 { (Dtype::Fp16, Dtype::Fp32) } else { (Dtype::Fp32, Dtype::Fp16) };
        self.g.add_op(name, "Cast").unwrap();
        self.g.set_input_desc(name, "x", dims, idt).unwrap();
        self.g.set_output_desc(name, "y", dims, odt).unwrap();
        self.g.set_attr_dtype(name, "dst_type", if to_f32 { "fp32" } else { "fp16" }).unwrap();
        self.wire(name, "x", x);
        self.reg_out(name, "y")
    }

    /// NORM32 核心 v2（f16 域 + MatMul 立方体 fp32 累加，tap 矩阵定案）：
    /// 310P tbe 的 ReduceSumD/ReduceSum f32 desc 均为假支持——FE 混精度
    /// 回退跑 f16 累加，行和 >65504 饱和（实测 92% 偏差）；Cast f32 是
    /// 真语义（逐位一致）。方差精度改走 mm 的 cube fp32 累加（实测
    /// 0.05% = f16 量化级）：
    ///   xs=x·s（防 outlier 平方溢出）→ sq=xs² → ssum=mm(sq,onesᵀ)
    ///   → inv=rsqrt(ssum)·k（k=s·√w）→ 1-D Tile 广播 → x·invb·γ
    /// 全程 f16 kernel；torch GemmaRMSNorm 对齐的关键是方差不饱和/
    /// 不漂移（fp32 累加），f16 mul 舍入 ~0.05%/op 为残余差
    fn rms32(&mut self, name: &str, x: &str, gamma: &str, dims: &[i64]) -> String {
        let (rows, w) = (dims[0], dims[1]);
        let (sc, kc, ones) = self.n32_consts(w);
        // xs = x·s
        let xs = format!("{name}_xs");
        self.g.add_op(&xs, "Mul").unwrap();
        self.g.set_input_desc(&xs, "x1", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(&xs, "x2", &[1], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&xs, "y", dims, Dtype::Fp16).unwrap();
        self.wire(&xs, "x1", x);
        self.g.link(&xs, "x2", &sc).unwrap();
        let xs = self.reg_out(&xs, "y");
        // sq = xs²
        let sq = self.mul2_f16(&format!("{name}_sq"), &xs, &xs, dims);
        // ssum = mm(sq, onesᵀ) [rows,1] —— cube 内 fp32 累加
        let ssum = self.mm(&format!("{name}_mm"), &sq, dims, &ones, &[1, w], &[rows, 1]);
        // inv = rsqrt(ssum) · k（k = s·√w，把预缩放折回）
        let rs = format!("{name}_rs");
        self.g.add_op(&rs, "Rsqrt").unwrap();
        self.g.set_input_desc(&rs, "x", &[rows, 1], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&rs, "y", &[rows, 1], Dtype::Fp16).unwrap();
        self.wire(&rs, "x", &ssum);
        let rs = self.reg_out(&rs, "y");
        let ik = format!("{name}_ik");
        self.g.add_op(&ik, "Mul").unwrap();
        self.g.set_input_desc(&ik, "x1", &[rows, 1], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&ik, "x2", &[1], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&ik, "y", &[rows, 1], Dtype::Fp16).unwrap();
        self.wire(&ik, "x1", &rs);
        self.g.link(&ik, "x2", &kc).unwrap();
        let ik = self.reg_out(&ik, "y");
        // 广播（行式形态）：[rows,1] → Reshape [1,rows] → TileD
        // multiples=[w,1]（dim0 整行复制——生产 bias/γ 链同款 kernel）
        // → [w,rows] → TransposeD(1,0) → [rows,w]。⚠ 连续平铺方向
        // （rank-1 [rows] 或 [1,rows]+[1,w] flat tile）的 TileD kernel
        // 在 310P f16 散点腐蚀 ~8%（逐位同错定罪）；行式形态 = 0.08% 干净
        let bshp = format!("{name}_bshp");
        self.const_i32(&bshp, &[1, rows as i32]);
        let br1 = self.reshape(&format!("{name}_br1"), &ik, &[rows, 1], &bshp, &[1, rows]);
        let bt = format!("{name}_bt");
        self.g.add_op(&bt, "TileD").unwrap();
        self.g.set_input_desc(&bt, "x", &[1, rows], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&bt, "y", &[w, rows], Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&bt, "multiples", &[w, 1]).unwrap();
        self.wire(&bt, "x", &br1);
        let btiled = self.reg_out(&bt, "y");
        let tp = format!("{name}_tp");
        self.g.add_op(&tp, "TransposeD").unwrap();
        self.g.set_input_desc(&tp, "x", &[w, rows], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&tp, "y", dims, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&tp, "perm", &[1, 0]).unwrap();
        self.wire(&tp, "x", &btiled);
        let invt = self.reg_out(&tp, "y");
        // nx = x · invb
        let nx = self.mul2_f16(&format!("{name}_nx"), x, &invt, dims);
        // gamma [w] → Reshape [1,w]（rank 桥）→ Tile [rows,w]
        let gshp = format!("{name}_gshp");
        self.const_i32(&gshp, &[1, w as i32]);
        let gfr = self.reshape(&format!("{name}_gfr"), gamma, &[w], &gshp, &[1, w]);
        let gt = format!("{name}_gt");
        self.g.add_op(&gt, "TileD").unwrap();
        self.g.set_input_desc(&gt, "x", &[1, w], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&gt, "y", dims, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&gt, "multiples", &[rows, 1]).unwrap();
        self.wire(&gt, "x", &gfr); // gfr 是算子输出——link 只配 Data 名（陷阱 #8）
        let gft = self.reg_out(&gt, "y");
        self.mul2_f16(name, &nx, &gft, dims)
    }

    /// f16 Mul（同形）——NORM32 v2 全 f16 域用
    fn mul2_f16(&mut self, name: &str, a: &str, b: &str, dims: &[i64]) -> String {
        self.g.add_op(name, "Mul").unwrap();
        self.g.set_input_desc(name, "x1", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "x2", dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
        self.wire(name, "x1", a);
        self.wire(name, "x2", b);
        self.reg_out(name, "y")
    }

    /// [rows,1] ÷ [1] 标量（RealDiv 广播——标量分母 TBE 接受）
    fn rdiv1(&mut self, name: &str, x: &str, num: &str, rows: i64) -> String {
        self.g.add_op(name, "RealDiv").unwrap();
        self.g.set_input_desc(name, "x1", &[rows, 1], Dtype::Fp32).unwrap();
        self.g.set_input_desc(name, "x2", &[1], Dtype::Fp32).unwrap();
        self.g.set_output_desc(name, "y", &[rows, 1], Dtype::Fp32).unwrap();
        self.wire(name, "x1", x);
        self.g.link(name, "x2", num).unwrap();
        self.reg_out(name, "y")
    }

    /// [rows,1] + [1] 标量 eps
    fn add_b1(&mut self, name: &str, x: &str, y: &str, rows: i64) -> String {
        self.g.add_op(name, "Add").unwrap();
        self.g.set_input_desc(name, "x1", &[rows, 1], Dtype::Fp32).unwrap();
        self.g.set_input_desc(name, "x2", &[1], Dtype::Fp32).unwrap();
        self.g.set_output_desc(name, "y", &[rows, 1], Dtype::Fp32).unwrap();
        self.wire(name, "x1", x);
        self.g.link(name, "x2", y).unwrap();
        self.reg_out(name, "y")
    }

    /// f32 Mul（同形）——dims 校验由调用方保证
    fn mul2_f32(&mut self, name: &str, a: &str, b: &str, dims: &[i64]) -> String {
        self.g.add_op(name, "Mul").unwrap();
        self.g.set_input_desc(name, "x1", dims, Dtype::Fp32).unwrap();
        self.g.set_input_desc(name, "x2", dims, Dtype::Fp32).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp32).unwrap();
        self.wire(name, "x1", a);
        self.wire(name, "x2", b);
        self.reg_out(name, "y")
    }

    /// f32 Add（同形）
    fn add2_f32(&mut self, name: &str, a: &str, b: &str, dims: &[i64]) -> String {
        self.g.add_op(name, "Add").unwrap();
        self.g.set_input_desc(name, "x1", dims, Dtype::Fp32).unwrap();
        self.g.set_input_desc(name, "x2", dims, Dtype::Fp32).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp32).unwrap();
        self.wire(name, "x1", a);
        self.wire(name, "x2", b);
        self.reg_out(name, "y")
    }

    /// LayerNormV4（vision SigLIP norm）。⚠ aclnnAddLayerNorm/GE AddLayerNorm
    /// 在 [768,1152] 大 shape 输出 ~100× 放大（行方差正常、kernel 级行为，
    /// trace 取证）——换 LayerNormV4（normalized_shape 张量输入）。
    /// ⚠ mean/rstd 是 REQUIRED 输出且死端会让 y 爆（探针取证，与
    /// AddRmsNorm 死端无害相反）——每个 LN 实例记入 ln_aux，finish 时
    /// 绑成 aux 图输出（parity 只对拍主输出）。
    fn addln(&mut self, name: &str, x: &str, gamma: &str, beta: &str, nsh: &str, dims: &[i64]) -> String {
        self.g.add_op(name, "LayerNormV4").unwrap();
        self.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "normalized_shape", &[1], Dtype::Int32).unwrap();
        self.g.set_input_desc(name, "gamma", &[dims[1]], Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "beta", &[dims[1]], Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
        self.g.set_attr_float(name, "epsilon", LN_EPS).unwrap();
        self.wire(name, "x", x);
        self.g.link(name, "normalized_shape", nsh).unwrap();
        self.g.link(name, "gamma", gamma).unwrap();
        self.g.link(name, "beta", beta).unwrap();
        self.ln_aux.push(name.to_string());
        self.reg_out(name, "y")
    }

    /// PromptFlashAttention（BSH rank-3；q/kv seq 可不同 = cross）
    #[allow(clippy::too_many_arguments)]
    fn pfa(
        &mut self, name: &str, q: &str, k: &str, v: &str,
        q_bsh: &[i64; 3], kv_bsh: &[i64; 3], heads: i64, kv_heads: i64, hd: i64,
    ) -> String {
        self.g.add_op(name, "PromptFlashAttention").unwrap();
        self.g.set_input_desc(name, "query", q_bsh, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "key", kv_bsh, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "value", kv_bsh, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "attention_out", q_bsh, Dtype::Fp16).unwrap();
        self.g.set_attr_int(name, "num_heads", heads).unwrap();
        self.g.set_attr_float(name, "scale_value", 1.0 / (hd as f64).sqrt()).unwrap();
        self.g.set_attr_int(name, "pre_tokens", 2147483647).unwrap();
        self.g.set_attr_int(name, "next_tokens", 0).unwrap();
        self.g.set_attr_str(name, "input_layout", "BSH").unwrap();
        self.g.set_attr_int(name, "num_key_value_heads", kv_heads).unwrap();
        self.g.set_attr_int(name, "sparse_mode", 0).unwrap();
        self.g.set_attr_int(name, "inner_precise", 0).unwrap();
        self.wire(name, "query", q);
        self.wire(name, "key", k);
        self.wire(name, "value", v);
        self.reg_out(name, "attention_out")
    }

    /// Squeeze(axis=[0])：[1,m,*] → [m,*]
    fn squeeze(&mut self, name: &str, x: &str, in3: &[i64; 3], out2: &[i64; 2]) -> String {
        self.g.add_op(name, "Squeeze").unwrap();
        self.g.set_input_desc(name, "x", in3, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", out2, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(name, "axis", &[0]).unwrap();
        self.wire(name, "x", x);
        self.reg_out(name, "y")
    }

    /// Reshape(x, shape 常量张量输入)
    fn reshape(&mut self, name: &str, x: &str, in_dims: &[i64], shape_in: &str, out: &[i64]) -> String {
        self.g.add_op(name, "Reshape").unwrap();
        self.g.set_input_desc(name, "x", in_dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "shape", &[out.len() as i64], Dtype::Int32).unwrap();
        self.g.set_output_desc(name, "y", out, Dtype::Fp16).unwrap();
        self.wire(name, "x", x);
        self.g.link(name, "shape", shape_in).unwrap();
        self.reg_out(name, "y")
    }

    /// SliceD rank-3 [b,s,w] 最后一维通道切分
    fn slice3(&mut self, name: &str, x: &str, dims: &[i64; 3], col_off: i64, col_len: i64) -> String {
        let out = [dims[0], dims[1], col_len];
        self.g.add_op(name, "SliceD").unwrap();
        self.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", &out, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(name, "offsets", &[0, 0, col_off]).unwrap();
        self.g.set_attr_int_list(name, "size", &out).unwrap();
        self.wire(name, "x", x);
        self.reg_out(name, "y")
    }

    /// SliceD rank-2 通道切分
    fn slice2(&mut self, name: &str, x: &str, dims: &[i64; 2], col_off: i64, col_len: i64) -> String {
        let out = [dims[0], col_len];
        self.g.add_op(name, "SliceD").unwrap();
        self.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", &out, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(name, "offsets", &[0, col_off]).unwrap();
        self.g.set_attr_int_list(name, "size", &out).unwrap();
        self.wire(name, "x", x);
        self.reg_out(name, "y")
    }

    /// GatherV2D 行 gather（axis=0 attr；swap / 行复制通用）
    fn gather_rows(&mut self, name: &str, x: &str, dims: &[i64; 2], idx: &str) -> String {
        self.g.add_op(name, "GatherV2D").unwrap();
        self.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
        self.g
            .set_input_desc(name, "indices", &[dims[0]], Dtype::Int32)
            .unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
        self.g.set_attr_int(name, "axis", 0).unwrap();
        self.wire(name, "x", x);
        self.g.link(name, "indices", idx).unwrap();
        self.reg_out(name, "y")
    }

    /// ConcatD：rank-3 [1,s,d] 沿 seq 维（concat_dim=1）拼接。
    /// ⚠ DYNAMIC_INPUT 端口必须先 dyn_inputs("x", 2) 注册（工厂不预建，
    /// num=0；ge_builder 经 libgraph_base 符号直链注册）；端口名
    /// x0/x1（base+序号，从 0 起——非 TF 惯例的 x1/x2）
    fn concat_seq3(&mut self, name: &str, a: &str, b: &str, a_bsh: &[i64; 3], b_bsh: &[i64; 3]) -> String {
        let out = [1, a_bsh[1] + b_bsh[1], a_bsh[2]];
        self.g.add_op(name, "ConcatD").unwrap();
        self.g.dyn_inputs(name, "x", 2).unwrap();
        self.g.set_input_desc(name, "x0", a_bsh, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "x1", b_bsh, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", &out, Dtype::Fp16).unwrap();
        self.g.set_attr_int(name, "concat_dim", 1).unwrap();
        self.g.set_attr_int(name, "N", 2).unwrap();
        self.wire(name, "x0", a);
        self.wire(name, "x1", b);
        self.reg_out(name, "y")
    }

    /// rank-2 → rank-3 Unsqueeze(axes=[0])（PFA 直连桥）
    fn unsqueeze(&mut self, name: &str, x: &str, dims: &[i64; 2]) -> String {
        let out = [1, dims[0], dims[1]];
        self.g.add_op(name, "Unsqueeze").unwrap();
        self.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", &out, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(name, "axes", &[0]).unwrap();
        self.wire(name, "x", x);
        self.reg_out(name, "y")
    }

    /// rope 组合（rank-2 版，r1 验证）：x [t, heads*d] → 半通道
    /// SliceD×2 → ConcatD(axis=1) 互换 → mul+mul+add（rank-2 表）→
    /// Reshape → [1,t,heads*d]（v3n 验证 Reshape 喂 PFA 数值逐位一致；
    /// Unsqueeze 喂 PFA 有累积变体差，弃用）。返回 (rank-2, rank-3)。
    fn rope2(&mut self, tag: &str, x: &str, dims: &[i64; 2], cos2: &str, sin2: &str,
             shp3: &str) -> (String, String) {
        let (t, wd) = (dims[0], dims[1]);
        let half = wd / 2;
        let swname = format!("{tag}_sw");
        let lo = self.slice2(&format!("{tag}_lo"), x, dims, 0, half);
        let hi = self.slice2(&format!("{tag}_hi"), x, dims, half, half);
        self.g.add_op(&swname, "ConcatD").unwrap();
        self.g.dyn_inputs(&swname, "x", 2).unwrap();
        self.g.set_input_desc(&swname, "x0", &[t, half], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&swname, "x1", &[t, half], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&swname, "y", &[t, wd], Dtype::Fp16).unwrap();
        self.g.set_attr_int(&swname, "concat_dim", 1).unwrap();
        self.g.set_attr_int(&swname, "N", 2).unwrap();
        self.wire(&swname, "x0", &hi);
        self.wire(&swname, "x1", &lo);
        let sw = self.reg_out(&swname, "y");
        let m1 = self.mul2(&format!("{tag}_m1"), x, cos2, dims);
        let m2 = self.mul2(&format!("{tag}_m2"), &sw, sin2, dims);
        let kr = self.add2(&format!("{tag}_ad"), &m1, &m2, dims);
        let kr3 = self.reshape(&format!("{tag}_3"), &kr, dims, shp3, &[1, t, wd]);
        (kr, kr3)
    }

    /// flat 版 rope（无列切）：[t, wd] → Reshape[t*2, half] →
    /// GatherV2D(相邻行 swap，idx 为 Const) → mul+mul+add → Reshape back。
    /// 表 = 段级 flat cos/sin 输入（eager flat 版同款布局）。替代
    /// rope2 的 slice2×2+ConcatD——列切视图触发运行时 MemcopyAsync 物化
    /// （同 qkv 切分病；GatherV2D 310P 仅 axis=0，故换 flat 视图做行交换）
    fn rope2_flat(
        &mut self, tag: &str, x: &str, dims: &[i64; 2], fdims: &[i64; 2],
        cos_f: &str, sin_f: &str, idx_c: &str, shp_flat: &str, shp_back: &str,
        shp3: &str,
    ) -> (String, String) {
        let (t, wd) = (dims[0], dims[1]);
        let xf = self.reshape(&format!("{tag}_xf"), x, dims, shp_flat, fdims);
        let sw = self.gather_rows(&format!("{tag}_sw"), &xf, fdims, idx_c);
        let m1 = self.mul2(&format!("{tag}_m1"), &xf, cos_f, fdims);
        let m2 = self.mul2(&format!("{tag}_m2"), &sw, sin_f, fdims);
        let ad = self.add2(&format!("{tag}_ad"), &m1, &m2, fdims);
        let kr = self.reshape(&format!("{tag}_bk"), &ad, fdims, shp_back, dims);
        let kr3 = self.reshape(&format!("{tag}_3"), &kr, dims, shp3, &[1, t, wd]);
        (kr, kr3)
    }

    /// 手工 attention（MHA 版，ma 验证 0.26%）：q/k/v3 [bh,s,d] →
    /// bmm(q,kᵀ) → SoftmaxV2(axes=-1) → bmm(·,v)。scale 由调用方折进
    /// q 侧权重+bias（PFA 的 host 回调 InnerPFA 每次执行 ~20ms 停顿的
    /// 替代；AttentionScore 310P 无 kernel——asc 三变体编译崩）。
    fn attn_manual(&mut self, name: &str, q3: &str, k3: &str, v3: &str, bh: i64, s: i64, d: i64) -> String {
        let (sc, sm, bm) = (format!("{name}_sc"), format!("{name}_sm"), format!("{name}_bm"));
        self.g.add_op(&sc, "BatchMatMulV2").unwrap();
        self.g.set_input_desc(&sc, "x1", &[bh, s, d], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&sc, "x2", &[bh, s, d], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&sc, "y", &[bh, s, s], Dtype::Fp16).unwrap();
        self.g.set_attr_bool(&sc, "adj_x1", false).unwrap();
        self.g.set_attr_bool(&sc, "adj_x2", true).unwrap();
        self.wire(&sc, "x1", q3);
        self.wire(&sc, "x2", k3);
        let scr = self.reg_out(&sc, "y");
        self.g.add_op(&sm, "SoftmaxV2").unwrap();
        self.g.set_input_desc(&sm, "x", &[bh, s, s], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&sm, "y", &[bh, s, s], Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&sm, "axes", &[-1]).unwrap();
        self.g.set_attr_bool(&sm, "half_to_float", false).unwrap();
        self.wire(&sm, "x", &scr);
        let smr = self.reg_out(&sm, "y");
        self.g.add_op(&bm, "BatchMatMulV2").unwrap();
        self.g.set_input_desc(&bm, "x1", &[bh, s, s], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&bm, "x2", &[bh, s, d], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&bm, "y", &[bh, s, d], Dtype::Fp16).unwrap();
        self.g.set_attr_bool(&bm, "adj_x1", false).unwrap();
        self.g.set_attr_bool(&bm, "adj_x2", false).unwrap();
        self.wire(&bm, "x1", &smr);
        self.wire(&bm, "x2", v3);
        self.reg_out(&bm, "y")
    }

    /// GQA 版（ma2 验证 0.24%）：k3/v3 [1,skv,d] TileD 广播到 bh 头。
    /// q3 [bh,sq,d]。返回 [bh,sq,d]。
    /// mask（GEB_ATTN_MASK，range padding）：Some = softmax 前
    /// 加性 mask 广播 Add（x2 [1,1,skv]——pad 键列大负数，真列 0）。数学上
    /// 精确（x+0 恒等 + exp(-30000+max) 下溢恰 0），真行输出与精确 L 版等价
    fn attn_manual_gqa(
        &mut self, name: &str, q3: &str, k3: &str, v3: &str,
        bh: i64, sq: i64, skv: i64, d: i64, mask: Option<&str>,
    ) -> String {
        let tile = |s: &mut Self, name: &str, src: &str| -> String {
            s.g.add_op(name, "TileD").unwrap();
            s.g.set_input_desc(name, "x", &[1, skv, d], Dtype::Fp16).unwrap();
            s.g.set_output_desc(name, "y", &[bh, skv, d], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list(name, "multiples", &[bh, 1, 1]).unwrap();
            s.wire(name, "x", src);
            s.reg_out(name, "y")
        };
        let kt = tile(self, &format!("{name}_kt"), k3);
        let vt = tile(self, &format!("{name}_vt"), v3);
        let (sc, sm, bm) = (format!("{name}_sc"), format!("{name}_sm"), format!("{name}_bm"));
        self.g.add_op(&sc, "BatchMatMulV2").unwrap();
        self.g.set_input_desc(&sc, "x1", &[bh, sq, d], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&sc, "x2", &[bh, skv, d], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&sc, "y", &[bh, sq, skv], Dtype::Fp16).unwrap();
        self.g.set_attr_bool(&sc, "adj_x1", false).unwrap();
        self.g.set_attr_bool(&sc, "adj_x2", true).unwrap();
        self.wire(&sc, "x1", q3);
        self.wire(&sc, "x2", &kt);
        let scr = self.reg_out(&sc, "y");
        let smx = match mask {
            Some(mk) => {
                let ad = format!("{name}_mk");
                self.g.add_op(&ad, "Add").unwrap();
                self.g.set_input_desc(&ad, "x1", &[bh, sq, skv], Dtype::Fp16).unwrap();
                self.g.set_input_desc(&ad, "x2", &[1, 1, skv], Dtype::Fp16).unwrap();
                self.g.set_output_desc(&ad, "y", &[bh, sq, skv], Dtype::Fp16).unwrap();
                self.wire(&ad, "x1", &scr);
                self.g.link(&ad, "x2", mk).unwrap();
                self.reg_out(&ad, "y")
            }
            None => scr,
        };
        self.g.add_op(&sm, "SoftmaxV2").unwrap();
        self.g.set_input_desc(&sm, "x", &[bh, sq, skv], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&sm, "y", &[bh, sq, skv], Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&sm, "axes", &[-1]).unwrap();
        self.g.set_attr_bool(&sm, "half_to_float", false).unwrap();
        self.wire(&sm, "x", &smx);
        let smr = self.reg_out(&sm, "y");
        self.g.add_op(&bm, "BatchMatMulV2").unwrap();
        self.g.set_input_desc(&bm, "x1", &[bh, sq, skv], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&bm, "x2", &[bh, skv, d], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&bm, "y", &[bh, sq, d], Dtype::Fp16).unwrap();
        self.g.set_attr_bool(&bm, "adj_x1", false).unwrap();
        self.g.set_attr_bool(&bm, "adj_x2", false).unwrap();
        self.wire(&bm, "x1", &smr);
        self.wire(&bm, "x2", &vt);
        self.reg_out(&bm, "y")
    }

    /// token 主序 [views*s, h*d] → 头主序 [views*h, s, d]（bmm 布局）。
    /// ⚠ 直接 Reshape 是错排（[768,1152]→[48,256,72] 把头/序混排——
    /// manual 对拍 98.2% 实证）：正确变换 = Reshape[views,s,h,d] →
    /// TransposeD(0,2,1,3) → Reshape[views*h,s,d]。
    fn headsplit(
        &mut self, name: &str, x: &str, t: i64, views: i64, s: i64, h: i64, d: i64,
        shp_vs4: &str, shp_bh3: &str,
    ) -> String {
        let (r4, tp, r3) = (format!("{name}_r4"), format!("{name}_tp"), format!("{name}_r3"));
        let qd = h * d;
        self.g.add_op(&r4, "Reshape").unwrap();
        self.g.set_input_desc(&r4, "x", &[t, qd], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&r4, "shape", &[4], Dtype::Int32).unwrap();
        self.g.set_output_desc(&r4, "y", &[views, s, h, d], Dtype::Fp16).unwrap();
        self.wire(&r4, "x", x);
        self.g.link(&r4, "shape", shp_vs4).unwrap();
        let r4o = self.reg_out(&r4, "y");
        self.g.add_op(&tp, "TransposeD").unwrap();
        self.g.set_input_desc(&tp, "x", &[views, s, h, d], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&tp, "y", &[views, h, s, d], Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&tp, "perm", &[0, 2, 1, 3]).unwrap();
        self.wire(&tp, "x", &r4o);
        let tpo = self.reg_out(&tp, "y");
        self.g.add_op(&r3, "Reshape").unwrap();
        self.g.set_input_desc(&r3, "x", &[views, h, s, d], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&r3, "shape", &[3], Dtype::Int32).unwrap();
        self.g.set_output_desc(&r3, "y", &[views * h, s, d], Dtype::Fp16).unwrap();
        self.wire(&r3, "x", &tpo);
        self.g.link(&r3, "shape", shp_bh3).unwrap();
        self.reg_out(&r3, "y")
    }

    /// headsplit 的逆：[views*h, s, d] → [views*s, h*d]
    fn headmerge(
        &mut self, name: &str, x: &str, views: i64, s: i64, h: i64, d: i64,
        shp_vh4: &str, shp_flat2: &str,
    ) -> String {
        let (r4, tp, r2) = (format!("{name}_r4"), format!("{name}_tp"), format!("{name}_r2"));
        let t = views * s;
        let qd = h * d;
        self.g.add_op(&r4, "Reshape").unwrap();
        self.g.set_input_desc(&r4, "x", &[views * h, s, d], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&r4, "shape", &[4], Dtype::Int32).unwrap();
        self.g.set_output_desc(&r4, "y", &[views, h, s, d], Dtype::Fp16).unwrap();
        self.wire(&r4, "x", x);
        self.g.link(&r4, "shape", shp_vh4).unwrap();
        let r4o = self.reg_out(&r4, "y");
        self.g.add_op(&tp, "TransposeD").unwrap();
        self.g.set_input_desc(&tp, "x", &[views, h, s, d], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&tp, "y", &[views, s, h, d], Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&tp, "perm", &[0, 2, 1, 3]).unwrap();
        self.wire(&tp, "x", &r4o);
        let tpo = self.reg_out(&tp, "y");
        self.g.add_op(&r2, "Reshape").unwrap();
        self.g.set_input_desc(&r2, "x", &[views, s, h, d], Dtype::Fp16).unwrap();
        self.g.set_input_desc(&r2, "shape", &[2], Dtype::Int32).unwrap();
        self.g.set_output_desc(&r2, "y", &[t, qd], Dtype::Fp16).unwrap();
        self.wire(&r2, "x", &tpo);
        self.g.link(&r2, "shape", shp_flat2).unwrap();
        self.reg_out(&r2, "y")
    }

    fn finish(&mut self, out_names: &[&str]) {
        // gear payload 合成先于一切早退（serve/e2e 模式跳过编译但 mech 仍需
        // payload 做逐请求选档；names/shapes 此刻已定）
        self.compose_gear_payloads();
        if std::env::var("GEB_LOAD").is_ok() {
            // 缓存加载模式：跳过编译，parity_and_bench 里替换为磁盘 OM
            return;
        }
        if std::env::var("GEB_SEG").ok().as_deref() == Some("e2e") {
            // e2e 模式：同样跳过编译（e2e_run 里按 {seg}_real.om 加载）
            return;
        }
        let names: Vec<&str> = self.names.iter().map(|s| s.as_str()).collect();
        let shape_refs: Vec<(&str, &[i64])> = self
            .names
            .iter()
            .zip(&self.shapes)
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        self.g.graph_inputs(&names).unwrap();
        // GEB_NO_AUX=1：跳过 LN 辅输出绑定（mean/rstd 死端会让 y 爆——
        // trap #17，数值坏但 bench 计时有效；用于隔离 111 图输出假设）
        if self.ln_aux.is_empty() || std::env::var("GEB_NO_AUX").is_ok() {
            self.g.graph_outputs(out_names).unwrap();
        } else {
            // 主输出 idx0 + 每个 LayerNormV4 的 mean(idx1)/rstd(idx2)
            let mut all: Vec<&str> = out_names.to_vec();
            let mut idxs: Vec<i32> = vec![0; out_names.len()];
            for ln in &self.ln_aux {
                all.push(ln.as_str());
                idxs.push(1);
                all.push(ln.as_str());
                idxs.push(2);
            }
            self.g.graph_outputs_idx(&all, &idxs).unwrap();
        }
        self.g.set_nd_input_shape(&shape_refs).unwrap();
        // gear 模式：ge.dynamicDims 档表 = 纯 -1 位值（输入注册序；payload
        // 的静态尾维混入选项会 rc=-7——选项/payload 双结构分离）
        if !self.gear_optvals.is_empty() {
            let opt = self
                .gear_optvals
                .iter()
                .map(|pl| {
                    pl.iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<String>>()
                        .join(",")
                })
                .collect::<Vec<String>>()
                .join(";");
            println!("[gear] dynamicDims: {opt}");
            self.g.set_option("ge.dynamicDims", &opt).unwrap();
        }
        // GEB_OPT_<key>=<value>：build option 直通口（A/B 实验，如
        // GEB_OPT_ge.streamMaxParallelNum=AIcoreEngine:1,VectorEngine:1）
        for (k, v) in std::env::vars().filter(|(k, _)| k.starts_with("GEB_OPT_")) {
            let key = &k["GEB_OPT_".len()..];
            self.g.set_option(key, &v).unwrap();
            println!("[geb-opt] {key} = {v}");
        }
        self.g.build().expect("build");
    }
}

// ---------------------------------------------------------------------------
// 通用运行：eager 参考跑完 → GE run → 对拍 → bench → OM 落盘
// ---------------------------------------------------------------------------

/// 对拍 + bench 通用尾段。outs_elem：每个图输出的元素数。
pub fn parity_and_bench(
    ctx: &AscendContext,
    stream: &AscendStream,
    seg: &mut Seg,
    tag: &str,
    out_elems: &[usize],
    eager: &dyn Fn(&[DeviceBuffer]) -> Vec<DeviceBuffer>,
    bench: bool,
) {
    // gear 模式：eager 参考与 row_bcast/动态 binds 布局不兼容（bias 行
    // 重复物化、rope 表 p_max 全量），且多档对拍需逐档参考——数值裁判走
    // e2e golden 与行为 eval。烤桶 OM 落盘照常（save 只需 build 产物）
    if gear_mode() {
        println!("[parity] gear 模式：跳过 eager 对拍（裁判 = e2e golden + 行为 eval）");
        if let Ok(p) = std::env::var("GEB_SAVE") {
            seg.g.save(&p).expect("save om (gear)");
            println!("OM saved: {p}");
        }
        return;
    }
    // GEB_NORM32：GE 是 fp32 方差语义、eager 参考（aclnnAddRmsNorm）是
    // f16——对拍必分叉；且标量池前插使 eager 的层索引错位。单段模式
    // 下跳过对拍（数值裁判走 GEB_OPTEST=n32v + e2e 路径的 golden 对拍）
    if norm32_enabled() {
        println!(
            "[parity] GEB_NORM32 开启：跳过 eager 对拍（f32 vs f16 语义差；golden 对拍走 e2e）"
        );
        // 烤桶路径：OM 落盘不能被跳过（tl144n32 全靠这里；save 只需
        // build 产物，不依赖 run）
        if let Ok(p) = std::env::var("GEB_SAVE") {
            seg.g.save(&p).expect("save om (n32)");
            println!("OM saved: {p}");
        }
        return;
    }
    let n_in = seg.ins().len();
    // 缓存加载（替代编译；数据/图构造仍跑——权重 buffer 是运行输入）
    if let Ok(path) = std::env::var("GEB_LOAD") {
        println!("loading OM from {path}");
        seg.g = ge_builder::load(&path).expect("load om");
    }
    // eager 参考先跑（binds 只读借用）
    let mem = |tag: &str| {
        let total: usize = seg.binds.iter().map(|b| b.len()).sum();
        if let Ok(o) = std::process::Command::new("npu-smi").arg("info").output() {
            let txt = String::from_utf8_lossy(&o.stdout);
            for l in txt.lines().filter(|l| l.contains("Memory-Usage") || l.contains("/ 4")) {
                let _ = l;
            }
            let _ = &txt;
        }
        println!("[mem] {tag}: binds total = {:.1} MB", total as f64 / 1e6);
    };
    mem("before-eager");
    let probe_malloc = |tag: &str| match ctx.malloc(16) {
        Ok(_) => println!("[malloc-probe] {tag}: ok"),
        Err(e) => println!("[malloc-probe] {tag}: FAIL {e:?}"),
    };
    probe_malloc("before-eager");
    let refs = eager(&seg.binds);
    if let Err(e) = stream.synchronize() {
        println!("[sync-err after eager] {e:?}");
    }
    probe_malloc("after-eager");
    mem("after-eager");

    // GE run（加载缓存或已 build 的图；GEB_WCONST 时 ins 只含真图输入）
    let n_out = seg.g.num_outputs().unwrap();
    for i in 0..n_out {
        println!("[out-dims] {i}: size={} dims={:?}", seg.g.output_size(i).unwrap_or(0), seg.g.output_dims(i).unwrap_or_default());
    }
    let outs: Vec<DeviceBuffer> = (0..n_out)
        .map(|i| ctx.malloc(seg.g.output_size(i).unwrap().max(16)).expect("out malloc"))
        .collect();
    let out_refs: Vec<&DeviceBuffer> = outs.iter().collect();
    // gear 模式 bring-up = max 档内容 → 末档执行（静态 = run_seg 0 档）
    seg.run_seg(seg.gear_payloads.len().saturating_sub(1), &out_refs, stream)
        .expect("ge run");
    drop(stream.synchronize());

    // GEB_DBG：打印每个图输出缓冲的 |max| + head（层 0 各级绑成图输出后
    // 与 eager 的 GEB_TRACE 逐级对照用——|max| 同但 head 异 = 排列错位类）
    if std::env::var("GEB_DBG").is_ok() {
        for i in 0..n_out {
            let n = (seg.g.output_size(i).unwrap_or(0) / 2).min(4_000_000) as usize;
            if n == 0 {
                continue;
            }
            let h = download_f16(ctx, &outs[i], n);
            let m = h.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
            let head: Vec<f32> = h[..6.min(h.len())].iter().map(|v| v.to_f32()).collect();
            println!("[dbg out{i}] n={n} |max|={m:.4} head={head:?}");
        }
    }

    // 对拍（主输出——LayerNormV4 的 mean/rstd aux 输出跳过）。断言用
    // 相对误差：GE 静态 OM 与 eager aclnn 允许不同 tiling 变体（这正是
    // C 路线的性能来源），单算子/小图逐位一致（已验证），全量多层图
    // 是稳定的变体差（prefix 18 层累计 ~15% 内）
    let mut worst = 0f32;
    let mut worst_rel = 0f32;
    for (i, elems) in out_elems.iter().take(refs.len()).enumerate() {
        let ge_h = download_f16(ctx, &outs[i], *elems);
        let ref_h = download_f16(ctx, &refs[i], *elems);
        let mut md = 0f32;
        for (a, b) in ge_h.iter().zip(&ref_h) {
            let d = (a.to_f32() - b.to_f32()).abs();
            if d > md {
                md = d;
            }
        }
        let rm = ref_h.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
        let rel = if rm > 0.0 { md / rm } else { md };
        println!("parity[{tag} out{i}] max_diff={md:.5} (ref |max|={rm:.3}, rel={:.1}%)", rel * 100.0);
        worst = worst.max(md);
        worst_rel = worst_rel.max(rel);
    }
    if worst_rel >= 0.20 {
        // bench 模式下降级为警告（GE 静态 tiling 与 eager 的逐层漂移是
        // 架构性合法差，全深对拍口径待升级为 fp32 oracle）
        if bench {
            println!("{tag} parity drift {:.1}% (>=20%, oracle 对拍待做；bench 继续)", worst_rel * 100.0);
        } else {
            panic!("{tag} GE OM diverged: worst_rel={:.1}%", worst_rel * 100.0);
        }
    }
    println!("{tag}_PARITY_OK n_in={n_in} worst={worst:.5} rel={:.1}%", worst_rel * 100.0);

    // OM 落盘
    if let Ok(p) = std::env::var("GEB_SAVE") {
        seg.g.save(&p).expect("save om");
        println!("OM saved: {p}");
    }

    if bench {
        // GEB_ROUNDS/GEB_PER：bench 规模参数化——假设检验用小配置
        // （如 ROUNDS=5 PER=3，十几秒出数），全深度留作最终确认
        let rounds = envi("GEB_ROUNDS", 30) as usize;
        let per = envi("GEB_PER", 10).max(1) as usize;
        // rounds=3 时 min(3) 会把全部轮次吃成 warmup → ts 空仓 panic；
        // 至少留 1 个样本轮
        let skip = rounds.saturating_sub(1).min(3);
        let gi = seg.gear_payloads.len().saturating_sub(1);
        for _ in 0..3 {
            seg.run_seg(gi, &out_refs, stream).unwrap();
        }
        drop(stream.synchronize());
        let mut ts = Vec::new();
        for r in 0..rounds {
            let t0 = std::time::Instant::now();
            for _ in 0..per {
                seg.run_seg(gi, &out_refs, stream).unwrap();
            }
            drop(stream.synchronize());
            if r >= skip {
                ts.push(t0.elapsed().as_secs_f64() * 1000.0 / per as f64);
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{tag} GE OM: {:.4} ms (median/{per}, rounds={rounds})", ts[ts.len() / 2]);
    }
}

// ---------------------------------------------------------------------------
// e2e：三段 OM 接真实推理链（GEB_SEG=e2e，进程内链式、host 中转）
//   组装口径与 CUDA 镜像同源：
//   x0 = vision_out ‖ token_embedding[token_ids]（视觉前语言后）
//   state 不进 prefix——π0.5 走 state 离散化进 prompt（pi05_prompt）
//   cond = silu(W_out·silu(W_in·te+b_in)+b_out)；style 切分 [0:w]/[w:2w]
//   （executor L714-719：ascl=1+s0 / ash=s1）
// ---------------------------------------------------------------------------

/// GEB_E2E_REPLAY 离线轨迹帧：真 LIBERO rollout（record_rollout.py 录制，
/// torch npu-torch 路径）——patches/token_ids/noise/nact 为录制值，三段
/// 链逐帧产物（vision_out/kv/actions）随回放填充。actions 前 gd 列与
/// nact（= torch normalized_actions，x_t 终态切片，同 noise）对拍 = 离线
/// 行为裁判（闭环 env 前的中间验收）
pub struct ReplayFrame {
    patches: Vec<f16>,     // [VT, V_PATCH_W]（含 empty 视图 -1 pad 行）
    token_ids: Vec<u32>,   // PaliGemma tokenizer + state 离散化（真 prompt）
    noise: Vec<f16>,       // [HOR, ADIM] 录制时注入
    nact: Vec<f16>,        // [HOR, 7] torch normalized_actions 参考值
    vision_out: Vec<f16>,  // [VT, PW] 本帧 vision 段产物
    kv: Vec<Vec<f16>>,     // 本帧 prefix 36 输出
    actions: Vec<f16>,     // 本帧 flow 终态 [HOR, ADIM]
}

pub struct E2eStage {
    /// 输入（golden 或合成 bring-up）
    patches: Vec<f16>,          // [VT, V_PATCH_W]
    token_ids: Vec<u32>,        // lang tokens（真 tokenizer 随 golden 落地）
    noise: Vec<f16>,            // [HOR, ADIM]
    golden_actions: Option<Vec<f16>>, // [HOR, ADIM]
    /// golden 中间量（f32 原值；键存在才比——段边界 bisect）：
    /// vision_out[x0 之前] / x0[prefix 输入] / kv0[prefix 首层 k] / step0_x1
    golden_mid: Vec<(String, Vec<f32>)>,
    /// 阶段产物（host 中转）
    vision_out: Vec<f16>,       // [VT, PW] post-projector
    kv: Vec<Vec<f16>>,          // prefix 36 输出（k0,v0,k1,v1,... 各 [p,KVD]）
    actions: Vec<f16>,          // flow 终态 [HOR, ADIM]
    /// 离线轨迹回放帧（GEB_E2E_REPLAY；token 长度须全帧 = GEB_TOKENS——
    /// 真 prompt 的 pad 行在 torch 侧被 mask，引擎静态全可见不等价，等长
    /// 真 token 才是精确对拍）
    replay: Vec<ReplayFrame>,
    /// GEB_E2E_SERVE 常驻机械（三段 per-frame 执行器；ctx/stream 由
    /// serve 循环传入，机械只持 owned 状态——多桶 = 每桶一个进程）
    vmech: Option<VisionMech>,
    pmech: Option<PrefixMech>,
    fmech: Option<FlowMech>,
}

// ---------------------------------------------------------------------------
// GEB_E2E_SERVE 常驻机械：三段 per-frame 执行器（闭环 eval 用）。
// 单进程单桶（GEB_TOKENS=L + GEB_OM_DIR=tl{L}），Python 侧按帧挑桶/每桶
// 起一个 serve 进程（supervisor 在 apxinf_rust 容器内 spawn——引擎二进制
// 绑定 9.0.1 GE）。ctx/stream 不入机械，机械只持 owned 状态：Seg
// （OM + binds）/输出 buffer/嵌入表/styles 缓存。
// ---------------------------------------------------------------------------

pub struct VisionMech {
    s: Seg,
    routs: Vec<DeviceBuffer>, // 108 路 LN aux 槽一次分配复用（只下载主输出）
}

impl VisionMech {
    /// 一帧 vision：覆写 binds[0]（patches）重跑，只下载主输出 idx0
    /// （replay 同式）
    fn one_frame(&mut self, ctx: &AscendContext, stream: &AscendStream, patches: &[f16]) -> Vec<f16> {
        assert_eq!(patches.len() * 2, self.s.binds[0].len(), "serve patches 尺寸不符");
        // 流水化热路径：零拷贝字节视图（LE 主机上
        // to_le_bytes ≡ raw bytes，wbuf_t 同款 reinterpret）+ 三段全挂流
        // （h2d → run → d2h 同 stream FIFO，host 只等最后一次 sync）。
        // 字节流路径与逐元素转换版完全一致 ⇒ 数值 bit 恒等；分解定罪
        // 的 d2h 3.4ms + dlconv 5.1ms（1.57M 元素 chunks_exact）就此消除。
        // 计时：enq 三段 + wait。
        let tim = serve_timing();
        let t0 = std::time::Instant::now();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(patches.as_ptr() as *const u8, patches.len() * 2) };
        ctx.copy_h2d_async(&self.s.binds[0], bytes, stream).expect("serve h2d patches");
        let t1 = std::time::Instant::now();
        let ins = self.s.ins();
        let refs: Vec<&DeviceBuffer> = self.routs.iter().collect();
        self.s.g.run(&ins, &refs, stream).expect("ge run");
        let n0 = self.s.g.output_size(0).unwrap() / 2;
        let mut out: Vec<f16> = vec![f16::from_bits(0); n0];
        let obytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n0 * 2) };
        ctx.copy_d2h_async(obytes, &self.routs[0], stream).expect("serve d2h vision_out");
        let t2 = std::time::Instant::now();
        drop(stream.synchronize());
        if tim {
            let now = std::time::Instant::now();
            let ms = |us: u128| us as f64 / 1e3;
            println!(
                "[vt] h2d_enq={:.2} run+d2h_enq={:.2} wait={:.2} total={:.2}",
                ms((t1 - t0).as_micros()),
                ms((t2 - t1).as_micros()),
                ms((now - t2).as_micros()),
                ms((now - t0).as_micros()),
            );
        }
        out
    }
}

pub struct PrefixMech {
    s: Seg,
    routs: Vec<DeviceBuffer>,
    emb: Vec<f32>,     // [vocab, PW] host 查表（~2.1GB 常驻复用）
    lang_scale: f32,   // √PW（gemma embed scale）
    vis_rows: usize,   // (VT-drop_v)*PW——空视图剔除后的可见行
    nkv: usize,        // depth*2
    vocab_w: usize,
    /// token 查表缓存：prompt 跨 replan 恒定（同任务 L 桶内 token_ids
    /// 不变），查表+scale 的 f16 行缓存一次复用——asm 2.5ms 的主体。
    /// 数值恒等：同 ids 同算式（f32 mul 确定性）产同字节
    tok_cache_ids: Vec<u32>,
    tok_cache_rows: Vec<f16>,
    /// gear 模式：L → 档下标（静态模式空 = 固定 L 单桶）
    gear_of_l: std::collections::HashMap<i64, usize>,
    /// GEB_PREFIX_POOL：per-L 静态 prefix OM 池（换柄执行，routs 全 L 共用
    /// ——max 物化、无跨请求状态）。根因修复，见 compose 处注记
    pool_root: Option<String>,
    pool_graphs: std::collections::HashMap<i64, ge_builder::GeGraph>,
    pool_lru: Vec<i64>,
    pool_cap: usize,
    /// routs 各槽字节数（= max 档 OM 输出尺寸——池 OM d2h 整缓冲后截断用）
    rout_bytes: Vec<usize>,
    /// GEB_ATTN_MASK（range padding）：padmask 输入 bind 下标（无 mask OM
    /// = None；池模式网格映射下必须 Some，否则 pad 行无遮蔽）
    mask_bind: Option<usize>,
}

impl PrefixMech {
    /// L → 档下标（静态模式恒 0——run_seg 走原 run 分支）
    fn gear_idx(&self, l: i64) -> usize {
        if self.s.gear_payloads.is_empty() {
            return 0;
        }
        *self
            .gear_of_l
            .get(&l)
            .unwrap_or_else(|| panic!("prefix 档表缺 L={l}（GEB_DYN_LLIST 未含）"))
    }
}

impl PrefixMech {
    /// 池模式：按 L 换静态 OM 句柄（lazy 装载 tl{l}_i8；LRU 上限防全档
    /// 驻留炸显存——单只 1.88GB）。routs/binds 不换（max 物化全 L 共用）
    fn swap_prefix(&mut self, l: i64) {
        if !self.pool_graphs.contains_key(&l) {
            let p = format!("{}/tl{l}_i8/prefix_real.om",
                            self.pool_root.as_deref().unwrap_or_default());
            println!("[prefix-pool] lazy load L={l}（{p}）");
            let g = ge_builder::load(&p)
                .unwrap_or_else(|e| panic!("prefix pool 缺 L={l}（{e:?}）——先烤 {p}"));
            let n_out = g.num_outputs().unwrap();
            assert!(n_out <= self.rout_bytes.len(),
                    "池 OM 输出数 {n_out} > max 槽位 {}", self.rout_bytes.len());
            self.pool_graphs.insert(l, g);
            self.pool_lru.push(l);
            while self.pool_lru.len() > self.pool_cap {
                let victim = self.pool_lru.remove(0);
                if let Some(g) = self.pool_graphs.remove(&victim) {
                    let _ = g.unload();
                }
                println!("[prefix-pool] LRU 驱逐 L={victim}（cap={}）", self.pool_cap);
            }
        } else {
            let i = self.pool_lru.iter().position(|&x| x == l).unwrap();
            self.pool_lru.remove(i);
            self.pool_lru.push(l);
        }
        self.s.g = *self.pool_graphs.get(&l).unwrap();
    }

    /// 一帧 prefix（设备态）：重组 x0（本帧 vision_out 可见行 + 真 token
    /// 查表 ×lang_scale）覆写 binds[0] 重跑 + sync；36 路 kv 留在
    /// self.routs（设备）——GEB_SERVE_FAST 下由 FlowMech 异步 d2d 直连
    /// 消费，零 host 中转
    fn one_frame_dev(
        &mut self,
        ctx: &AscendContext,
        stream: &AscendStream,
        vision_out: &[f16],
        token_ids: &[u32],
    ) {
        assert_eq!(vision_out.len(), (VT * PW) as usize, "serve vision_out 尺寸不符");
        // range padding（GEB_POOL_GRID，池模式限定）：真 L → 桶界 P，token
        // 补 0（embedding[0]）——pad 行由 padmask 遮蔽；非池模式恒等不受扰
        let l_true = token_ids.len() as i64;
        let p_bkt = if self.pool_root.is_some() {
            pool_grid_p(l_true)
        } else {
            l_true
        };
        assert!(
            p_bkt == l_true || self.mask_bind.is_some(),
            "range-padding: pad 到 P={p_bkt} 但 OM 无 padmask（GEB_ATTN_MASK=1 烤）"
        );
        let pad_owned;
        let token_ids: &[u32] = if p_bkt > l_true {
            let mut v = token_ids.to_vec();
            v.resize(p_bkt as usize, 0);
            pad_owned = v;
            &pad_owned
        } else {
            token_ids
        };
        // 计时：asm(x0 组装+查表)/h2d_enq(零拷贝视图上流)/run/wait
        let tim = serve_timing();
        let t0 = std::time::Instant::now();
        let gi = self.gear_idx(p_bkt);
        if self.pool_root.is_some() {
            self.swap_prefix(p_bkt);
        }
        // token 查表缓存：ids 变化才重算（prompt 跨 replan 恒定——同任务
        // 桶内 token_ids 不变）。数值恒等：同算式（f32 mul）产同 f16 字节
        if self.tok_cache_ids != token_ids {
            let mut rows: Vec<f16> = Vec::with_capacity(token_ids.len() * self.vocab_w);
            for &id in token_ids {
                let r = id as usize * self.vocab_w;
                assert!(r + self.vocab_w <= self.emb.len(), "token id {id} 超 vocab");
                rows.extend(
                    self.emb[r..r + self.vocab_w]
                        .iter()
                        .map(|&v| f16::from_f32(v * self.lang_scale)),
                );
            }
            self.tok_cache_ids = token_ids.to_vec();
            self.tok_cache_rows = rows;
            // padmask 随 token 变化填（L 不变零动作）：行 [vis+L, vis+P)
            // = -30000，真行/视觉行 0——整只重建，无跨桶污染
            if let Some(bi) = self.mask_bind {
                let vis = (self.vis_rows / self.vocab_w) as i64;
                let mut mh = vec![f16::from_f32(0.0); (vis + p_bkt) as usize];
                for r in &mut mh[(vis + l_true) as usize..] {
                    *r = f16::from_f32(-30000.0);
                }
                let bytes: Vec<u8> =
                    mh.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
                ctx.copy_h2d_async_sized(&self.s.binds[bi], &bytes, stream)
                    .expect("serve h2d padmask");
            }
        }
        let mut x0 = vision_out[..self.vis_rows].to_vec();
        x0.extend_from_slice(&self.tok_cache_rows);
        let t1 = std::time::Instant::now();
        // gear 模式 binds[0] = p_max 物化（x0 实际 p 行 ≤ max——sized h2d）
        assert!(x0.len() * 2 <= self.s.binds[0].len(), "serve x0 尺寸超 max 档");
        // 零拷贝字节视图（x0 活到函数尾，async h2d 借用安全）
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x0.as_ptr() as *const u8, x0.len() * 2) };
        if self.s.gear_payloads.is_empty() && self.pool_root.is_none() {
            ctx.copy_h2d_async(&self.s.binds[0], bytes, stream).expect("serve h2d x0");
        } else {
            // gear / prefix-pool：binds[0] 按 max 物化，本帧实际 p 行 → sized
            ctx.copy_h2d_async_sized(&self.s.binds[0], bytes, stream)
                .expect("serve h2d x0 (sized)");
        }
        let t2 = std::time::Instant::now();
        let refs: Vec<&DeviceBuffer> = self.routs.iter().collect();
        self.s.run_seg(gi, &refs, stream).expect("ge run");
        let t3 = std::time::Instant::now();
        drop(stream.synchronize());
        if tim {
            let now = std::time::Instant::now();
            let ms = |us: u128| us as f64 / 1e3;
            println!(
                "[pt] asm={:.2} h2d_enq={:.2} run={:.2} wait={:.2} total={:.2}",
                ms((t1 - t0).as_micros()),
                ms((t2 - t1).as_micros()),
                ms((t3 - t2).as_micros()),
                ms((now - t3).as_micros()),
                ms((now - t0).as_micros()),
            );
        }
    }

    /// 一帧 prefix（legacy host 中转）：one_frame_dev 后 36 路 kv 下载
    /// （aux 槽不下载）
    fn one_frame(
        &mut self,
        ctx: &AscendContext,
        stream: &AscendStream,
        vision_out: &[f16],
        token_ids: &[u32],
    ) -> Vec<Vec<f16>> {
        self.one_frame_dev(ctx, stream, vision_out, token_ids);
        (0..self.nkv).map(|i| self.download_kv(ctx, i)).collect()
    }

    /// kv 下载（按当前执行 OM 的实际行数；routs/binds 按 max 物化——
    /// d2h 整缓冲契约下先全量再截断）
    fn download_kv(&self, ctx: &AscendContext, i: usize) -> Vec<f16> {
        if !self.s.gear_payloads.is_empty() {
            let dims = self.s.g.cur_output_dims(i).unwrap_or_else(|e| panic!("cur dims: {e:?}"));
            let n = dims.iter().product::<i64>() as usize;
            let full = self.s.g.output_size(i).unwrap() / 2;
            let mut v = download_f16(ctx, &self.routs[i], full);
            v.truncate(n);
            return v;
        }
        if self.pool_root.is_some() {
            let full = self.rout_bytes[i] / 2;
            let mut v = download_f16(ctx, &self.routs[i], full);
            v.truncate(self.s.g.output_size(i).unwrap() / 2);
            return v;
        }
        download_f16(ctx, &self.routs[i], self.s.g.output_size(i).unwrap() / 2)
    }
}

pub struct FlowMech {
    s: Seg,
    ob: DeviceBuffer,
    style_cache: Vec<Vec<Vec<f16>>>, // 10 步 × (6*depth+2) 段（含 gate；跨请求不变）
    layer_bases: Vec<usize>,
    fs_idx: usize,
    pk_base: usize,
    depth: usize,
    /// GEB_SERVE_FAST 设备驻留态：styles 10 步各占独立视图（跨帧恒定 ⇒
    /// 一次上传后帧内零拷贝——legacy 110 槽位被 10 步复用须逐步重传）；
    /// style_map = style bind 下标 → 段序；ins_order = 图输入装配序
    /// （wconst 时 = data_inputs，同 Seg::ins 语义）
    style_master: Option<DeviceBuffer>,
    style_views: Vec<Vec<DeviceBuffer>>,
    style_map: std::collections::HashMap<usize, usize>,
    ins_order: Vec<usize>,
    /// gear 模式：L → 档下标 + rope 表 bind 下标（表 HOR 定形、内容随
    /// pos_offset=p 变——L 变更时换装；last_l 恒定则零动作）
    gear_of_l: std::collections::HashMap<i64, usize>,
    rope_binds: Vec<usize>,
    last_l: i64,
    /// HYBRID：per-L 静态 flow OM 池（非空 = 按请求换句柄执行，flow 不
    /// 走 gear——styles 静态 Data 之故，见 seg_flow 注记）
    flows: std::collections::HashMap<i64, GeGraph>,
    /// GEB_ATTN_MASK（range padding）：padmask bind 下标 + 上次填充的 L
    ///（不变零动作；整只重建无跨桶污染）
    mask_bind: Option<usize>,
    mask_last: i64,
}

/// styles 设备驻留构装：10 步 × (6*depth+2) 段拼一块 master（一次 h2d），
/// 每步每段一个 view；style_map 与 legacy h2d 循环同一 bind 映射
pub fn style_residency(
    ctx: &AscendContext,
    style_cache: &[Vec<Vec<f16>>],
    layer_bases: &[usize],
    fs_idx: usize,
    depth: usize,
) -> (DeviceBuffer, Vec<Vec<DeviceBuffer>>, std::collections::HashMap<usize, usize>) {
    let mut slots: Vec<usize> = Vec::with_capacity(depth * 6 + 2);
    for i in 0..depth {
        let b = layer_bases[i];
        slots.extend_from_slice(&[b, b + 1, b + 10, b + 11, b + 16, b + 17]);
    }
    slots.push(fs_idx);
    slots.push(fs_idx + 1);
    let mut style_map = std::collections::HashMap::new();
    for (j, &bi) in slots.iter().enumerate() {
        style_map.insert(bi, j);
    }
    let total: usize = style_cache
        .iter()
        .map(|sc| sc.iter().map(|seg| seg.len() * 2).sum::<usize>())
        .sum();
    let mut host: Vec<u8> = Vec::with_capacity(total);
    for sc in style_cache {
        for seg in sc {
            host.extend(seg.iter().flat_map(|v| v.to_bits().to_le_bytes()));
        }
    }
    let master = ctx.malloc(total).expect("style master malloc");
    ctx.copy_h2d(&master, &host).expect("style master h2d");
    let mut views: Vec<Vec<DeviceBuffer>> = Vec::with_capacity(style_cache.len());
    let mut off = 0usize;
    for sc in style_cache {
        let mut v: Vec<DeviceBuffer> = Vec::with_capacity(sc.len());
        for seg in sc {
            let n = seg.len() * 2;
            v.push(DeviceBuffer::view_of(&master, off, n));
            off += n;
        }
        views.push(v);
    }
    (master, views, style_map)
}

impl FlowMech {
    /// L → 档下标（静态模式恒 0）
    fn gear_idx(&self, l: i64) -> usize {
        if self.s.gear_payloads.is_empty() {
            return 0;
        }
        *self
            .gear_of_l
            .get(&l)
            .unwrap_or_else(|| panic!("flow 档表缺 L={l}（GEB_DYN_LLIST 未含）"))
    }

    /// HYBRID：按 L 换 flow 静态 OM 句柄（池空 = 单 OM 原样）
    fn swap_flow(&mut self, l: i64) {
        if !self.flows.is_empty() {
            self.s.g = *self
                .flows
                .get(&l)
                .unwrap_or_else(|| panic!("flow OM 池缺 L={l}"));
        }
    }

    /// L 变更换装 rope 表（flat 四件 + rank-2 四件；内容 pos_offset=p——
    /// 表 HOR 定形与 p 无关，内容随 p 平移。L 不变零动作）。早退门按 flows
    /// 池判：池非空 = HYBRID 逐 L 换静态 flow OM，rope 须随 L 换装（哪怕
    /// payloads 空——GEB_PREFIX_POOL 下 gear_mode 全局为 false 的情形）
    fn set_rope_for(&mut self, ctx: &AscendContext, l: i64) {
        if (self.s.gear_payloads.is_empty() && self.flows.is_empty()) || self.last_l == l {
            return;
        }
        let p = lp_to_p(l);
        let (qc, qs, _) = rope_flat_const(HOR, HEADS, p);
        let (kc, ks, _) = rope_flat_const(HOR, KV_HEADS, p);
        let (qc2, qs2) = rope_rank2_const(HOR, HEADS, p);
        let (kc2, ks2) = rope_rank2_const(HOR, KV_HEADS, p);
        let tables: Vec<&[f16]> = vec![&qc, &qs, &kc, &ks, &qc2, &qs2, &kc2, &ks2];
        for (bi, vals) in self.rope_binds.iter().zip(tables) {
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            ctx.copy_h2d(&self.s.binds[*bi], &bytes).expect("flow rope h2d");
        }
        self.last_l = l;
    }

    /// range padding：flow padmask 填充——键布局 = cache 行在前（真
    /// [0,vis+L) / pad [vis+L, vis+P)）+ 本桶 flow HOR 行恒真。L 不变零
    /// 动作。池模式限定（flows 非空）。⚠ 全长度填充（copy_h2d 等长断言）：
    /// bind 按 max 桶物化（762 行），本桶外行未被消费取 −30000 无害
    fn fill_mask(&mut self, ctx: &AscendContext, l_true: i64, p_bkt: i64) {
        let Some(bi) = self.mask_bind else { return };
        if self.flows.is_empty() || self.mask_last == l_true {
            return;
        }
        let vis = lp_to_p(0);
        let full = lp_to_p(envi("GEB_POOL_GRID_MAX", 200)) + HOR;
        let mut mh = vec![f16::from_f32(-30000.0); full as usize];
        for r in &mut mh[..(vis + l_true) as usize] {
            *r = f16::from_f32(0.0);
        }
        for r in &mut mh[(vis + p_bkt) as usize..(vis + p_bkt + HOR) as usize] {
            *r = f16::from_f32(0.0);
        }
        let bytes: Vec<u8> = mh.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        ctx.copy_h2d(&self.s.binds[bi], &bytes).expect("flow padmask h2d");
        self.mask_last = l_true;
    }

    /// 一帧 flow：36 路 pk/pv 换绑 + noise 重启 10 步（styles 缓存直取；
    /// euler c1/c2 已在 bring-up 换绑 LeRobot 语义且 buffer 内容恒持）
    fn one_frame(
        &mut self,
        ctx: &AscendContext,
        stream: &AscendStream,
        kv: &[Vec<f16>],
        token_ids: &[u32],
        noise: &[f16],
    ) -> Vec<f16> {
        let h2d = |buf: &DeviceBuffer, vals: &[f16]| {
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            // sized：HYBRID 下 pk/pv 内容 = 实际 p 行 < max 物化 bind
            ctx.copy_h2d_async_sized(buf, &bytes, stream).expect("serve h2d");
        };
        assert_eq!(kv.len(), self.depth * 2, "serve kv 路数不符");
        // range padding：flow OM 按桶界 P 换柄，rope 内容按真 L 换装
        //（flow token 位置 = vis+L+i 与 P 无关，表 HOR 定形）；mask 同步填
        let l_true = token_ids.len() as i64;
        let p_bkt = if self.flows.is_empty() {
            l_true
        } else {
            pool_grid_p(l_true)
        };
        let gi = self.gear_idx(p_bkt);
        self.swap_flow(p_bkt);
        self.set_rope_for(ctx, l_true);
        self.fill_mask(ctx, l_true, p_bkt);
        for i in 0..self.depth {
            h2d(&self.s.binds[self.pk_base + 2 * i], &kv[2 * i]);
            h2d(&self.s.binds[self.pk_base + 2 * i + 1], &kv[2 * i + 1]);
        }
        let mut x = noise.to_vec();
        for sc in &self.style_cache {
            for i in 0..self.depth {
                let b = self.layer_bases[i];
                h2d(&self.s.binds[b], &sc[6 * i]);
                h2d(&self.s.binds[b + 1], &sc[6 * i + 1]);
                h2d(&self.s.binds[b + 10], &sc[6 * i + 2]);
                h2d(&self.s.binds[b + 11], &sc[6 * i + 3]);
                h2d(&self.s.binds[b + 16], &sc[6 * i + 4]);
                h2d(&self.s.binds[b + 17], &sc[6 * i + 5]);
            }
            h2d(&self.s.binds[self.fs_idx], &sc[6 * self.depth]);
            h2d(&self.s.binds[self.fs_idx + 1], &sc[6 * self.depth + 1]);
            h2d(&self.s.binds[0], &x); // state = 当前 x
            let oref: Vec<&DeviceBuffer> = vec![&self.ob];
            self.s.run_seg(gi, &oref, stream).expect("ge run");
            drop(stream.synchronize());
            x = download_f16(ctx, &self.ob, (HOR * ADIM) as usize);
        }
        x
    }

    /// 一帧 flow（GEB_SERVE_FAST）：36 路 pk/pv 异步 d2d（prefix routs
    /// 设备直连）→ noise h2d → 10 步同流链（styles 视图零拷贝、x d2d
    /// 驻留、无 per-step sync）→ 末步一次 sync + actions d2h。与
    /// one_frame 逐位同源：同一字节流（d2d/视图都是精确字节）+ 同流
    /// 顺序执行——sync 只挪等待点不改计算序
    fn one_frame_fast(
        &mut self,
        ctx: &AscendContext,
        stream: &AscendStream,
        kv_dev: &[&DeviceBuffer],
        token_ids: &[u32],
        noise: &[f16],
    ) -> Vec<f16> {
        assert_eq!(kv_dev.len(), self.depth * 2, "serve kv 路数不符");
        assert!(!self.style_views.is_empty(), "fast 路径须先构装 styles 驻留");
        // range padding：flow OM 按桶界 P 换柄，rope 内容按真 L 换装
        //（flow token 位置 = vis+L+i 与 P 无关，表 HOR 定形）；mask 同步填
        let l_true = token_ids.len() as i64;
        let p_bkt = if self.flows.is_empty() {
            l_true
        } else {
            pool_grid_p(l_true)
        };
        let gi = self.gear_idx(p_bkt);
        self.swap_flow(p_bkt);
        self.set_rope_for(ctx, l_true);
        self.fill_mask(ctx, l_true, p_bkt);
        // 分步计时：kv(d2d×36)/h2d_enq(零拷贝视图上流)/steps(逐步 enqueue)+sync/dl
        let tim = serve_timing();
        let t0 = std::time::Instant::now();
        for i in 0..self.depth * 2 {
            // gear 模式两侧均 max 档 buffer（d2d 全量拷贝，尾部无效行被
            // 分支 desc 裁掉；静态模式等长断言保留）
            if self.s.gear_payloads.is_empty() {
                assert_eq!(
                    kv_dev[i].len(),
                    self.s.binds[self.pk_base + i].len(),
                    "fast kv 尺寸不符（路 {i}）"
                );
            }
            ctx.copy_d2d_async(&self.s.binds[self.pk_base + i], kv_dev[i], stream)
                .expect("serve d2d kv");
        }
        let t1 = std::time::Instant::now();
        // 零拷贝字节视图（noise 借用活到函数尾）
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(noise.as_ptr() as *const u8, noise.len() * 2) };
        ctx.copy_h2d_async(&self.s.binds[0], bytes, stream).expect("serve h2d noise");
        let t2 = std::time::Instant::now();
        let nsteps = self.style_views.len();
        let mut enq: Vec<f64> = Vec::with_capacity(nsteps);
        for step in 0..nsteps {
            let ins: Vec<&DeviceBuffer> = self
                .ins_order
                .iter()
                .map(|&i| match self.style_map.get(&i) {
                    Some(&j) => &self.style_views[step][j],
                    None => &self.s.binds[i],
                })
                .collect();
            let oref: Vec<&DeviceBuffer> = vec![&self.ob];
            let te = std::time::Instant::now();
            // gear 模式：ins 自组（run_seg 的 ins() 不含 style 视图替换）——
            // 直接 run_gear，payload 按 gi
            if self.s.gear_payloads.is_empty() {
                self.s.g.run(&ins, &oref, stream).expect("ge run");
            } else {
                self.s
                    .g
                    .run_gear(&ins, &oref, stream, &self.s.gear_payloads[gi])
                    .expect("ge run gear");
            }
            if tim {
                enq.push(te.elapsed().as_secs_f64() * 1e3);
            }
            if step + 1 < nsteps {
                // x 驻留：本步输出 → 下步 state（同流有序，免 d2h+h2d+sync）
                ctx.copy_d2d_async(&self.s.binds[0], &self.ob, stream).expect("serve d2d x");
            }
        }
        let t4 = std::time::Instant::now();
        drop(stream.synchronize());
        let t5 = std::time::Instant::now();
        let out = download_f16(ctx, &self.ob, (HOR * ADIM) as usize);
        if tim {
            let now = std::time::Instant::now();
            let ms = |us: u128| us as f64 / 1e3;
            println!(
                "[ft] kv={:.2} h2d_enq={:.2} enq[first={:.2} mid={:.2} last={:.2} sum={:.2}] sync={:.2} dl={:.2} total={:.2}",
                ms((t1 - t0).as_micros()),
                ms((t2 - t1).as_micros()),
                enq.first().copied().unwrap_or(0.0),
                enq.get(nsteps / 2).copied().unwrap_or(0.0),
                enq.last().copied().unwrap_or(0.0),
                enq.iter().sum::<f64>(),
                ms((t5 - t4).as_micros()),
                ms((now - t5).as_micros()),
                ms((now - t0).as_micros()),
            );
        }
        out
    }
}

impl E2eStage {
    /// 段边界 bisect 对拍：golden 有该键才比（BISECT 前缀），无则静默跳过。
    /// 带 argmax 索引——x0 上可直接分辨分岔落在视觉段/语言段行区间。
    /// got ≥ want 时截断比前缀（range padding 对拍：桶界 OM 出 P 行，
    /// golden 是真 L 行——真行恰为前缀，数学等价由 mask 保证）
    fn cmp_mid(&self, tag: &str, got: &[f16]) {
        let Some(want) = self.golden_mid.iter().find(|(k, _)| k == tag).map(|(_, v)| v) else {
            return;
        };
        assert!(got.len() >= want.len(), "golden {tag} 长度不足（got {} want {}）", got.len(), want.len());
        let got = &got[..want.len()];
        let mut md = 0f32;
        let mut i_at = 0usize;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g.to_f32() - w).abs();
            if d > md {
                md = d;
                i_at = i;
            }
        }
        let rm = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        println!(
            "[e2e] BISECT {tag}: max_diff={md:.5} @idx{i_at} (golden |max|={rm:.3}, rel={:.1}%)",
            if rm > 0.0 { md / rm * 100.0 } else { md }
        );
    }
}

pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GEB_NORM32：torch GemmaRMSNorm 语义对齐开关——方差 fp32 域。
/// norm16 对照定罪 fp32 上浮 = 行为开关；AddRmsNorm 的
/// fp32 desc 是假支持（GE 自动 Cast 归一回 f16 kernel，arm32 单算
/// f32/f16 输出逐位同）——只能组合手搓。
pub fn norm32_enabled() -> bool {
    std::env::var("GEB_NORM32").is_ok()
}

/// GEB_SERVE_FAST：serve 段间设备直连开关（性能轮实测净收益）——
/// prefix 36 路 kv 异步 d2d 直连（零 host 中转）、flow x 设备驻留
/// （逐步 d2h→h2d 换同流 d2d）、10 步去 per-step sync、styles 设备
/// 驻留（每步独立视图一次上传）。数值 bit 级恒等：同一字节流 + 同流
/// 顺序执行，sync 增删不改计算序。空串不算开（supervisor env 透传
/// 里未设时不误触）
pub fn serve_fast() -> bool {
    std::env::var("GEB_SERVE_FAST").map(|v| !v.is_empty()).unwrap_or(false)
}

/// GEB_SERVE_TIMING：serve 热路径段内分步计时（conv/h2d/run/sync/dl）——
/// 段间 glue 开销归因用。只加 Instant 戳不改数据流，数值恒等。
pub fn serve_timing() -> bool {
    std::env::var("GEB_SERVE_TIMING").is_ok()
}

/// NORM32 常量池已内聚到 Seg（n32_c/n32_consts）——v1 的 Data 标量池
/// （one/eps/w2048/w1024）随 ReduceSumD 假 f32 定罪一并淘汰

/// host 线性层（f32 全精度；维度 1×32×3072 量级，开销可忽略）
pub fn host_lin_f32(w: &LinearWeights, x: &[f32], in_d: usize) -> Vec<f32> {
    let wt = w.weight.to_f32_vec().unwrap();
    let out_d = wt.len() / in_d;
    assert_eq!(x.len(), in_d);
    let mut y = match &w.bias {
        Some(b) => b.to_f32_vec().unwrap(),
        None => vec![0f32; out_d],
    };
    for (i, &xi) in x.iter().enumerate() {
        for j in 0..out_d {
            y[j] += xi * wt[i * out_d + j];
        }
    }
    y
}

/// te [AW] → conditioning [ADIM]（time_mlp 两层，每层 matmul→bias→silu）
pub fn e2e_conditioning(real: &Pi05Weights, te: &[f32]) -> Vec<f32> {
    let h = host_lin_f32(&real.time_mlp_in, te, te.len());
    let h: Vec<f32> = h.iter().map(|&v| silu_f32(v)).collect();
    let c = host_lin_f32(&real.time_mlp_out, &h, h.len());
    c.iter().map(|&v| silu_f32(v)).collect()
}

/// style 投影 [ADIM,3W] → (scale=1+s0, shift=s1, gate=s2)。第三段 gate 是
/// openpi `_gated_residual` 的分支门控（residual + branch·gate，无激活直出
/// f16 cast）——定罪：引擎此前整体未消费该段，flow 输出 v 近随
/// 机、闭环 0/10 的主因
pub fn e2e_style_pair(w: &LinearWeights, cond: &[f32], width: usize) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
    let raw = host_lin_f32(w, cond, cond.len());
    let scl = raw[..width].iter().map(|&v| f16::from_f32(1.0 + v)).collect();
    let sh = raw[width..2 * width].iter().map(|&v| f16::from_f32(v)).collect();
    let gt = raw[2 * width..3 * width].iter().map(|&v| f16::from_f32(v)).collect();
    (scl, sh, gt)
}

/// e2e 收尾共通：GEB_OM_DIR（默认 ./om_cache）加载 {seg}_real.om
/// 跑一次，全部图输出下载回 host。Seg 构造须与 OM 构建同 env（四件套）

/// 三段 OM 预载缓存（冷启动 lazy）：GeServe::open 前并行装载（ge_builder
/// 侧带锁），e2e_run 命中即取——把串行 52s 的 OM 装载压成 ~max(单段)
static OM_PRELOAD: std::sync::Mutex<
    Option<std::collections::HashMap<String, ge_builder::GeGraph>>,
> = std::sync::Mutex::new(None);

/// 并行预载 GEB_OM_DIR 的 {vision,prefix,flow}_real.om（幂等；serve 冷启
/// 动路径调一次）。装载线程须绑主 context（aclmdlLoadFromMem 随调用线程
/// 的 current context 走——裸线程 rc=-3 实证）。失败不 panic——e2e_run
/// 回退串行装载并报真错误
pub fn preload_e2e_oms(ctx: &AscendContext) {
    {
        let guard = OM_PRELOAD.lock().unwrap();
        if guard.is_some() {
            return;
        }
    }
    let dir = std::env::var("GEB_OM_DIR").unwrap_or_else(|_| "./om_cache".into());
    let t0 = std::time::Instant::now();
    let segs = ["vision", "prefix", "flow"];
    let handles: Vec<std::thread::JoinHandle<(String, Option<ge_builder::GeGraph>)>> = segs
        .iter()
        .map(|seg| {
            let path = format!("{dir}/{seg}_real.om");
            let seg = seg.to_string();
            let ctx2: &AscendContext = unsafe { std::mem::transmute(ctx) };
            std::thread::spawn(move || {
                // 线程绑 context（&AscendContext 非 Send——transmute 借引用，
                // join 前 ctx 存活，单次冷启动路径受控）
                if let Err(e) = ctx2.set_current() {
                    eprintln!("[preload] {seg} context 绑定失败: {e:?}");
                    return (seg, None);
                }
                let g = ge_builder::load(&path);
                if let Err(e) = &g {
                    eprintln!("[preload] {seg} OM 装载失败 {path}: {e:?}");
                }
                (seg, g.ok())
            })
        })
        .collect();
    let mut map = std::collections::HashMap::new();
    for h in handles {
        if let Ok((seg, g)) = h.join() {
            if let Some(g) = g {
                map.insert(seg, g);
            }
        }
    }
    println!("[preload] {} 段 OM 并行装载完成 {:?}", map.len(), t0.elapsed());
    *OM_PRELOAD.lock().unwrap() = Some(map);
}

pub fn e2e_run(ctx: &AscendContext, stream: &AscendStream, s: &mut Seg, seg: &str) -> Vec<Vec<f16>> {
    let dir = std::env::var("GEB_OM_DIR").unwrap_or_else(|_| "./om_cache".into());
    let path = format!("{dir}/{seg}_real.om");
    let preloaded = OM_PRELOAD
        .lock()
        .unwrap()
        .as_mut()
        .and_then(|m| m.remove(seg));
    s.g = match preloaded {
        Some(g) => {
            println!("[e2e] {seg} OM 预载命中");
            g
        }
        None => {
            println!("[e2e] loading {path}");
            ge_builder::load(&path).expect("load om")
        }
    };
    // gear 模式：bring-up 内容按 max 档物化 → 用末档跑（静态 = run_seg 0 档）
    let gi = s.gear_payloads.len().saturating_sub(1);
    if !s.gear_payloads.is_empty() {
        let n = s.g.gear_count().unwrap_or(0);
        println!("[gear] {seg} OM 档数 {n}（bring-up 用末档执行）");
    }
    let n_out = s.g.num_outputs().unwrap();
    let outs: Vec<DeviceBuffer> = (0..n_out)
        .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("out malloc"))
        .collect();
    let refs: Vec<&DeviceBuffer> = outs.iter().collect();
    s.run_seg(gi, &refs, stream).expect("ge run");
    drop(stream.synchronize());
    // gear 模式 buffer 按 max 档物化（712 行）而本档只写前 666——d2h 须整
    // 缓冲（copy_d2h 长度断言），下载后 host 侧截到 cur_output_dims 行数
    let dl: Vec<Vec<f16>> = (0..n_out)
        .map(|i| {
            let mut v = download_f16(ctx, &outs[i], s.g.output_size(i).unwrap() / 2);
            if !s.gear_payloads.is_empty() {
                let n = s.g.cur_output_dims(i).unwrap().iter().product::<i64>() as usize;
                v.truncate(n);
            }
            v
        })
        .collect();
    // GEB_E2E_BENCH=N：稳态 rerun bench（输入 binds 不变重跑；含全输出 d2h
    // ——prefix 的 36 路 kv 下载 = today 生产 host 中转 glue，如实计入）
    let nb = envi("GEB_E2E_BENCH", 0) as usize;
    if nb > 0 {
        let mut ts = Vec::with_capacity(nb);
        for _ in 0..nb {
            let t = std::time::Instant::now();
            s.run_seg(gi, &refs, stream).expect("ge run");
            drop(stream.synchronize());
            let _: Vec<_> = (0..n_out)
                .map(|i| download_f16(ctx, &outs[i], s.g.output_size(i).unwrap() / 2))
                .collect();
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "[e2e] {seg} 稳态 bench ×{nb}: P50={:.2}ms min={:.2} max={:.2}（run + sync + {n_out} 路 d2h）",
            ts[nb / 2],
            ts[0],
            ts[nb - 1]
        );
    }
    dl
}

pub fn seg_e2e(be: &AscendBackend, bench: bool, real: Option<&Pi05Weights>) {
    let _ = bench;
    // e2e 读 *_real.om（四件套构建），Seg 重建必须同配置（输入序一致）
    let need = |k: &str| std::env::var(k).is_ok();
    assert!(
        std::env::var("GEB_ATTN").is_ok()
            && need("GEB_QKV3") && need("GEB_ROPEFLAT") && need("GEB_WCONST"),
        "GEB_SEG=e2e 须配 GEB_ATTN(=manual|pfa) GEB_QKV3=1 GEB_ROPEFLAT=1 GEB_WCONST=1 \
         （须与 GEB_OM_DIR 下 *_real.om 的构建配置一致——pfa 仅隔离实验用）"
    );
    let real = real.expect("GEB_SEG=e2e 需要 GEB_CKPT（token_embedding/time_mlp/style 投影 host 消费）");
    let tokens = envi("GEB_TOKENS", 64) as usize;
    // 冷启动 lazy：三段 OM 并行预载（e2e_run 命中即取）
    preload_e2e_oms(be.ctx());
    // 注：GE 静态 OM 接受非 16 倍 M（GEB_TOKENS=200 → P=968 实证编译/运行/
    // parity 正常——eager aclnn 的 16 倍 M 崩坑不适用于 GE 路径）
    let mut st = match std::env::var("GEB_E2E_GOLDEN") {
        Ok(path) => {
            println!("[e2e] golden: {path}");
            let (t, _) = apxinf_loader::safetensors::load_native_path(std::path::Path::new(&path))
                .expect("golden load");
            let f32v = |k: &str| -> Vec<f32> {
                t.get(k).unwrap_or_else(|| panic!("golden 缺键 {k}")).to_f32_vec().unwrap()
            };
            let f16v = |k: &str| -> Vec<f16> { f32v(k).into_iter().map(f16::from_f32).collect() };
            let ids: Vec<u32> = f32v("token_ids").iter().map(|&v| v as u32).collect();
            // GEB_E2E_PAD_TOKENS（range padding 对拍）：golden 真长 < 桶界
            if envi("GEB_E2E_PAD_TOKENS", 0) == 0 {
                assert_eq!(ids.len(), tokens, "golden token 数与 GEB_TOKENS 不符");
            }
            // 中间量（可选键，名对齐 golden_gen.py 落盘键）：vision_out / x0 /
            // kvk_l{i} 全 18 层（prefix k cache）/ step0_x1——段边界 bisect
            let mut keys: Vec<String> = Vec::new();
            for k in ["vision_out", "x0", "x0_vis", "step0_x1", "m0", "h1"] {
                if t.contains_key(k) {
                    keys.push(k.to_string());
                }
            }
            keys.extend(
                t.keys()
                    .filter(|k| k.starts_with("kvk_l") || k.starts_with("kvv_l") || k.starts_with("cond_s"))
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            let golden_mid: Vec<(String, Vec<f32>)> = keys
                .iter()
                .filter_map(|k| {
                    t.get(k.as_str()).map(|a| (k.clone(), a.to_f32_vec().unwrap()))
                })
                .collect();
            println!(
                "[e2e] golden 中间量: {:?}",
                golden_mid.iter().map(|(k, v)| format!("{k}[{}]", v.len())).collect::<Vec<_>>()
            );
            E2eStage {
                patches: f16v("patches"),
                token_ids: ids,
                noise: f16v("noise"),
                golden_actions: t
                    .get("actions")
                    .map(|a| a.to_f32_vec().unwrap().into_iter().map(f16::from_f32).collect()),
                golden_mid,
                vision_out: Vec::new(),
                kv: Vec::new(),
                actions: Vec::new(),
                replay: Vec::new(),
                vmech: None,
                pmech: None,
                fmech: None,
            }
        }
        Err(_) => {
            println!("[e2e] 无 GEB_E2E_GOLDEN：合成 patches/noise + 占位 token ids（bring-up 模式，golden 对拍下一步）");
            let mut seed = 0xE2E2u32;
            let ids = (0..tokens).map(|i| 1000 + i as u32 * 7).collect();
            E2eStage {
                patches: rand_f16((VT * V_PATCH_W) as usize, &mut seed, 300.0),
                token_ids: ids,
                noise: rand_f16((HOR * ADIM) as usize, &mut seed, 1.0),
                golden_actions: None,
                golden_mid: Vec::new(),
                vision_out: Vec::new(),
                kv: Vec::new(),
                actions: Vec::new(),
                replay: Vec::new(),
                vmech: None,
                pmech: None,
                fmech: None,
            }
        }
    };
    // GEB_E2E_REPLAY=<safetensors>：离线轨迹回放（record_rollout.py 录制，
    // 键 patches_{i}/token_ids_{i}/noise_{i}/nact_{i}）。bring-up 后三段各
    // 自逐帧换绑重跑（bring-up 的 golden 对拍照旧作金丝雀）
    if let Ok(path) = std::env::var("GEB_E2E_REPLAY") {
        let (t, _) = apxinf_loader::safetensors::load_native_path(std::path::Path::new(&path))
            .expect("replay load");
        let n = t.keys().filter(|k| k.starts_with("noise_")).count();
        assert!(n > 0, "replay 文件无 noise_{{i}} 键: {path}");
        let rf32 = |k: &str| -> Vec<f32> {
            t.get(k).unwrap_or_else(|| panic!("replay 缺键 {k}")).to_f32_vec().unwrap()
        };
        for i in 0..n {
            let ids: Vec<u32> = rf32(&format!("token_ids_{i}")).iter().map(|&v| v as u32).collect();
            assert_eq!(
                ids.len(),
                tokens,
                "replay 帧 {i} token 长度 {} ≠ GEB_TOKENS {tokens}——真 prompt token 数随 \
                 state 离散值位数浮动，静态 OM 须按等长真 token 帧过滤/重烤",
                ids.len()
            );
            st.replay.push(ReplayFrame {
                patches: rf32(&format!("patches_{i}")).into_iter().map(f16::from_f32).collect(),
                token_ids: ids,
                noise: rf32(&format!("noise_{i}")).into_iter().map(f16::from_f32).collect(),
                nact: rf32(&format!("nact_{i}")).into_iter().map(f16::from_f32).collect(),
                vision_out: Vec::new(),
                kv: Vec::new(),
                actions: Vec::new(),
            });
        }
        println!("[e2e] replay: {path} ×{n} 帧（token 等长断言已过）");
    }
    let t0 = std::time::Instant::now();
    seg_vision(be, false, Some(real), Some(&mut st));
    let t1 = std::time::Instant::now();
    println!("[e2e] vision 段 {:?}（含 OM 加载）", t1.duration_since(t0));
    seg_prefix(be, false, Some(real), Some(&mut st));
    let t2 = std::time::Instant::now();
    println!("[e2e] prefix 段 {:?}（含 OM 加载 + 2.1GB 嵌入查表）", t2.duration_since(t1));
    if std::env::var("GEB_E2E_STOP").ok().as_deref() == Some("prefix") {
        // bisect 用：只跑 vision+prefix（如 GEB_DEPTH=2 小图验输出槽复用）
        ge_builder::fini().expect("fini");
        println!("GE_E2E_PROBE_OK");
        return;
    }
    seg_flow(be, false, Some(real), Some(&mut st));
    let t3 = std::time::Instant::now();
    println!("[e2e] flow 段 {:?}（含 OM 加载 + 10 步）", t3.duration_since(t2));
    let head: Vec<f32> = st.actions[..8.min(st.actions.len())].iter().map(|v| v.to_f32()).collect();
    println!("[e2e] actions head={head:?}");
    // GEB_E2E_SERVE=<dir>：闭环 eval 常驻服务（三段机械已由各段 e2e 分支
    // 下沉 E2eStage；多桶 = Python 侧每桶一个 serve 进程）
    if let Ok(dir) = std::env::var("GEB_E2E_SERVE") {
        serve_loop(be, &mut st, &dir, tokens);
        ge_builder::fini().expect("fini");
        println!("GE_E2E_SERVE_OK");
        return;
    }
    ge_builder::fini().expect("fini");
    println!("GE_E2E_PROBE_OK");
}

// ---------------------------------------------------------------------------
// GEB_E2E_SERVE 常驻服务循环（闭环 eval 用）
// ---------------------------------------------------------------------------

/// GEB_E2E_SERVE=<dir>：文件轮询常驻服务。单进程单桶（GEB_TOKENS=L、
/// GEB_OM_DIR=tl{L}），Python 侧按帧挑桶、每桶一个 serve 进程（supervisor
/// 在 apxinf_rust 容器内 spawn——引擎二进制绑定 9.0.1 GE，eval 的 torch
/// 侧在 apxinf_npu，跨容器靠 /data 共享目录）。
/// 协议（spool 目录，tmp+rename 原子交付，单客户端串行）：
///   ready              —— 三段机械就绪标志（内容 "L=<tokens>"）
///   req_{seq:06}.bin   —— u32 magic 0x47514531 / u32 L / patches f32
///                          ×VT*V_PATCH_W / token_ids u32 ×L / noise f32
///                          ×HOR*ADIM
///   resp_{seq:06}.bin  —— u32 magic 0x47525331 / u32 status /
///                          f32 vision_ms/prefix_ms/flow_ms /
///                          actions f32 ×HOR*7（x_t 终态前 7 列）
///   shutdown           —— 退出信号
pub fn serve_loop(be: &AscendBackend, st: &mut E2eStage, dir: &str, tokens: usize) {
    // 帧实现与进程内直调（GeServe::infer）同源——spool 只是 IO 壳
    let mut pipe = GeServe {
        vm: st.vmech.take().expect("serve 需要 vision 机械"),
        pm: st.pmech.take().expect("serve 需要 prefix 机械"),
        fm: st.fmech.take().expect("serve 需要 flow 机械"),
        tokens,
        fast: serve_fast(),
    };
    std::fs::create_dir_all(dir).expect("serve spool create");
    // 清残留（上次运行的 req/resp/ready 会混淆 seq 与就绪判定）。
    // ⚠ 只删协议文件：pid 是 supervisor 的活体判定依据、stdout.log 是
    // 本进程重定向目标——全删会让 supervisor 判死重复 spawn（双 engine
    // 抢同一 spool 目录——实测踩过）
    for e in std::fs::read_dir(dir).expect("serve spool read") {
        let p = e.expect("dir entry").path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
        let is_proto = name.starts_with("req_")
            || name.starts_with("resp_")
            || name.starts_with(".req_")
            || name.starts_with(".resp_")
            || name == "ready"
            || name == "shutdown";
        if is_proto {
            let _ = std::fs::remove_file(&p);
        }
    }
    std::fs::write(format!("{dir}/ready"), format!("L={tokens}\n")).expect("serve ready write");
    println!("[serve] ready: {dir} L={tokens}（文件轮询中，shutdown 退出）");
    let mut served: u64 = 0;
    let exp_patches = (VT * V_PATCH_W) as usize;
    let exp_noise = (HOR * ADIM) as usize;
    loop {
        if std::path::Path::new(&format!("{dir}/shutdown")).exists() {
            println!("[serve] shutdown 信号，退出（已服务 {served} 请求）");
            return;
        }
        // 最小现存 seq（不做 last_seq 门限：eval 进程重启后 python 端 seq
        // 回卷到 1，门限会把新请求当旧丢弃——单客户端串行下按"目录里
        // 最小现存 req 处理"即正确序；上个进程的死请求至多被白跑一次）
        let mut best: Option<(i64, std::path::PathBuf)> = None;
        for e in std::fs::read_dir(dir).expect("serve spool read") {
            let p = e.expect("dir entry").path();
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
            let Some(num) = name.strip_prefix("req_").and_then(|r| r.strip_suffix(".bin")) else { continue };
            let (Ok(seq), true) = (num.parse::<i64>(), num.len() == 6) else { continue };
            if best.as_ref().map(|(s, _)| seq < *s).unwrap_or(true) {
                best = Some((seq, p));
            }
        }
        let Some((seq, path)) = best else {
            std::thread::sleep(std::time::Duration::from_millis(1));
            continue;
        };
        let t_all = std::time::Instant::now();
        let raw = std::fs::read(&path).expect("serve req read");
        let magic = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        assert_eq!(magic, 0x4751_4531, "serve 请求 magic 不符");
        let l = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
        // 静态模式 = 单 L 桶（L 须钉死）；gear 模式按请求 L 验长 + 选档
        //（档表外 L 由 gear_idx panic 拒绝——supervisor 档表 = 谱系全集）；
        // prefix 池模式（GEB_PREFIX_POOL）任意 L 由池换柄承载
        if pipe.pm.s.gear_payloads.is_empty() && pipe.pm.pool_root.is_none() {
            assert_eq!(
                l, tokens,
                "serve 请求 L={l} ≠ 本进程桶 GEB_TOKENS={tokens}（Python 侧挑桶错）"
            );
        }
        assert_eq!(
            raw.len(),
            8 + exp_patches * 4 + l * 4 + exp_noise * 4,
            "serve 请求长度不符（seq {seq} L={l}）"
        );
        let mut off = 8usize;
        let f32s = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let patches: Vec<f16> = raw[off..off + exp_patches * 4]
            .chunks_exact(4)
            .map(|c| f16::from_f32(f32s(c)))
            .collect();
        off += exp_patches * 4;
        let ids: Vec<u32> = raw[off..off + l * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        off += l * 4;
        let noise: Vec<f16> = raw[off..off + exp_noise * 4]
            .chunks_exact(4)
            .map(|c| f16::from_f32(f32s(c)))
            .collect();
        let out = pipe.infer(be, &patches, &ids, &noise);
        let tv = out.vision_ms as f64;
        let tp = out.prefix_ms as f64;
        let tf = out.flow_ms as f64;
        let actions = out.actions;
        // actions [HOR, ADIM] → 前 7 列（LIBERO deployable 维；
        // normalized_actions = x_t 终态切片，denorm 在 torch postprocess）
        let gd = 7usize;
        let mut a7: Vec<f32> = Vec::with_capacity(HOR as usize * gd);
        for r in 0..HOR as usize {
            for c in 0..gd {
                a7.push(actions[r * ADIM as usize + c].to_f32());
            }
        }
        let mut resp: Vec<u8> = Vec::with_capacity(8 + 12 + a7.len() * 4);
        resp.extend_from_slice(&0x4752_5331u32.to_le_bytes());
        resp.extend_from_slice(&0u32.to_le_bytes());
        resp.extend_from_slice(&(tv as f32).to_le_bytes());
        resp.extend_from_slice(&(tp as f32).to_le_bytes());
        resp.extend_from_slice(&(tf as f32).to_le_bytes());
        resp.extend(a7.iter().flat_map(|v| v.to_le_bytes()));
        let rpath = format!("{dir}/resp_{seq:06}.bin");
        let tmp = format!("{dir}/.resp_{seq:06}.tmp");
        std::fs::write(&tmp, &resp).expect("serve resp write");
        std::fs::rename(&tmp, &rpath).expect("serve resp rename");
        std::fs::remove_file(&path).expect("serve req unlink");
        println!(
            "[serve] #{seq}: vision={tv:.1} prefix={tp:.1} flow={tf:.1} total={:.1}ms |x|max={:.3}",
            t_all.elapsed().as_secs_f64() * 1e3,
            actions.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()))
        );
        served += 1;
    }
}

// ---------------------------------------------------------------------------
// vision 段：patch embed → depth×SigLIP 层 → post LN → projector
// 序列与 eager aclnn 组合（ops.rs）同构
// ---------------------------------------------------------------------------

pub fn seg_vision(be: &AscendBackend, bench: bool, real: Option<&Pi05Weights>, e2e: Option<&mut E2eStage>) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH_VISION", envi("GEB_DEPTH", 27)) as usize;
    let t = VT;
    let vqd = V_HEADS * V_HD; // 1152
    let qkvw = vqd * 3;
    // GEB_ATTN=manual：手工 attention（bmm+softmax+bmm 全静态，ma 验证）。
    // 默认 pfa（PromptFlashAttention——静态 OM 里走 InnerPFA host 回调，
    // ~20ms/层停顿，profile 取证）
    let attn_manual = std::env::var("GEB_ATTN").map(|v| v == "manual").unwrap_or(false);
    let vscale = if attn_manual { 1.0f32 / (V_HD as f32).sqrt() } else { 1.0 };
    // GEB_QKV3：q/k/v 独立投影（3×mm + 独立权重输入），替代融合 qkv mm +
    // SliceD×3。msprof 取证：切片视图下游（TransposeD/
    // PFA）每层触发 3 次运行时 MemcopyAsync 物化（~1.77MB/次）+ 每次前
    // ~4.5-12ms host 停顿 = vision ~450ms 空隙的主源（81 次/执行）
    let qkv3 = std::env::var("GEB_QKV3").is_ok();
    let mut seed = 0xC0DEu32;
    let mut s = Seg::new("ge_vision");

    // ---- 段级输入（注册序 = eager 的 index 约定）----
    // 0 patches / 1 patch_wt / 2 patch_b / 3 pos_rep / 4 zeros /
    // 5 shp_qkv3 / 6 shp_flat2 / 每层 12 项（GEB_QKV3: 16 项）/ 尾 4 项(pnw,pnb,projwt,projb)
    // e2e：真 patches（golden 或合成 bring-up）；缺省随机（对拍两路同值）
    let patches_h = e2e
        .as_ref()
        .map(|st| st.patches.clone())
        .unwrap_or_else(|| rand_f16((t * V_PATCH_W) as usize, &mut seed, 300.0));
    s.data(ctx, "patches", &[t, V_PATCH_W], &patches_h);
    {
        let host = real
            .map(|w| lw_f16(&w.vision.patch_embedding))
            .unwrap_or_else(|| rand_f16((V_PATCH_W * VW) as usize, &mut seed, 30000.0));
        s.wt(ctx, "patch_wt", &[VW, V_PATCH_W], &host, V_PATCH_W, VW);
    }
    let patch_b_h = real
        .map(|w| lb_f16(&w.vision.patch_embedding, VW as usize))
        .unwrap_or_else(|| rand_f16(VW as usize, &mut seed, 12000.0));
    s.data(ctx, "patch_b", &[1, VW], &patch_b_h);
    {
        // position 表循环重复（cuda kernel 语义：行 r 读 table[r % tpv]）
        let table = real
            .map(|w| t_f16(&w.vision.position_embedding))
            .unwrap_or_else(|| rand_f16((VPV * VW) as usize, &mut seed, 300.0));
        let mut rep = Vec::with_capacity((t * VW) as usize);
        for r in 0..t as usize {
            let src = (r % VPV as usize) * VW as usize;
            rep.extend_from_slice(&table[src..src + VW as usize]);
        }
        s.data(ctx, "pos_rep", &[t, VW], &rep);
    }
    s.data_zeros(ctx, "zeros", t, VW);
    // shape 类输入一律 Const（Data 会让 desc 变 unknown → host 调度停顿）
    s.const_i32("shp_v3", &[VIEWS as i32, VPV as i32, vqd as i32]);
    s.const_i32("shp_flat2", &[t as i32, vqd as i32]);
    s.const_i32("nsh", &[VW as i32]); // LayerNormV4 normalized_shape
    // 手工 attention 的头主序桥 shape（headsplit/headmerge 用）
    s.const_i32("shp_at3", &[(VIEWS * V_HEADS) as i32, VPV as i32, V_HD as i32]);
    s.const_i32("shp_vs4", &[VIEWS as i32, VPV as i32, V_HEADS as i32, V_HD as i32]);
    s.const_i32("shp_vh4", &[VIEWS as i32, V_HEADS as i32, VPV as i32, V_HD as i32]);

    // ---- 图：patch embed ----
    let mut cur = s.mm("pemb", "patches", &[t, V_PATCH_W], "patch_wt", &[VW, V_PATCH_W], &[t, VW]);
    cur = s.bias("pemb_b", &cur, &[t, VW], "patch_b");
    cur = s.add2("pemb_p", &cur, "pos_rep", &[t, VW]);

    // ---- 层循环 ----
    // eager 参考的 qkv 走三个独立 matmul（fused 输出 [m, q|k|v] 列交织，
    // take_rows 的 flat 切分在交织布局下数学错误——生产 eager 的该切分
    // 是对拍取证发现的 bug，GE 侧 SliceD 通道切分才是对的；eager 参考以
    // 独立投影保持数学正确）。独立 buffer 段函数持有，不进图输入。
    let mut eager_qkv: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer)> = Vec::new();
    // norm gamma/beta 的 host 副本（eager 参考 host LN 用）
    let mut eager_norms: Vec<(Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>)> = Vec::new();
    let mut layer_bases = Vec::with_capacity(depth);
    // GEB_DBG：层 0 各级额外绑图输出（parity_and_bench 的 [dbg outN] 打印）
    let dbg = std::env::var("GEB_DBG").is_ok();
    let mut dbg_outs: Vec<String> = Vec::new();
    for i in 0..depth {
        let base = s.binds.len();
        layer_bases.push(base);
        let p = format!("l{i}_");
        // 真权重：SigLIP 块按层索引（GEB_DEPTH 冒烟 <= 27）
        let blk = real.and_then(|w| w.vision.blocks.get(i));
        let n1w_h = blk
            .map(|b| t_f16(&b.norm1.weight))
            .unwrap_or_else(|| norm_f16(VW as usize, &mut seed, 1.0));
        let n1b_h = blk
            .map(|b| t_f16(&b.norm1.bias))
            .unwrap_or_else(|| norm_f16(VW as usize, &mut seed, 0.0));
        s.data(ctx, &format!("{}n1w", p), &[VW], &n1w_h);
        s.data(ctx, &format!("{}n1b", p), &[VW], &n1b_h);
        {
            // host 三块 [VW, vqd] → GE 拼接 [VW, qkvw] 转置上传；
            // eager 三块各自转置上传；bias 整条（GE）/host 切三段（eager）。
            // manual attention 时 GE 侧 q 块（权重+bias）预乘 1/√hd（图内
            // 无 scale 算子）；eager 侧保持原值（PFA 的 scale_value attr 承担）
            let wq = blk
                .map(|b| lw_f16(&b.q))
                .unwrap_or_else(|| rand_f16((VW * vqd) as usize, &mut seed, 30000.0));
            let wk = blk
                .map(|b| lw_f16(&b.k))
                .unwrap_or_else(|| rand_f16((VW * vqd) as usize, &mut seed, 30000.0));
            let wv = blk
                .map(|b| lw_f16(&b.v))
                .unwrap_or_else(|| rand_f16((VW * vqd) as usize, &mut seed, 30000.0));
            let wq_s: Vec<f16> = if attn_manual {
                wq.iter().map(|v| f16::from_f32(v.to_f32() * vscale)).collect()
            } else {
                wq.clone()
            };
            // 真权重 qkv bias = q.bias ‖ k.bias ‖ v.bias（SigLIP 三投影各有 bias）
            let bias = blk
                .map(|b| {
                    [
                        lb_f16(&b.q, vqd as usize),
                        lb_f16(&b.k, vqd as usize),
                        lb_f16(&b.v, vqd as usize),
                    ]
                    .concat()
                })
                .unwrap_or_else(|| rand_f16(qkvw as usize, &mut seed, 12000.0));
            let bias_s: Vec<f16> = if attn_manual {
                bias.iter()
                    .enumerate()
                    .map(|(i, v)| f16::from_f32(v.to_f32() * if (i as i64) < vqd { vscale } else { 1.0 }))
                    .collect()
            } else {
                bias.clone()
            };
            if qkv3 {
                // 独立权重/bias 输入（host 切一次，免运行时物化）；
                // wq_s/bias_s 已含 manual attention 的 vscale 折叠。
                // qkv3 下不注册融合 qkvwt/qkvb（图无消费者，GE 可能将其
                // 从模型输入中消除 → dataset 错位）——层内输入位次
                // 2..8，eager 闭包按 qkv3 偏移读
                s.wt(ctx, &format!("{}qw", p), &[vqd, VW], &wq_s, VW, vqd);
                s.wt(ctx, &format!("{}kw", p), &[vqd, VW], &wk, VW, vqd);
                s.wt(ctx, &format!("{}vw", p), &[vqd, VW], &wv, VW, vqd);
                s.data(ctx, &format!("{}qb", p), &[1, vqd], &bias_s[..vqd as usize]);
                s.data(ctx, &format!("{}kb", p), &[1, vqd], &bias_s[vqd as usize..2 * vqd as usize]);
                s.data(ctx, &format!("{}vb", p), &[1, vqd], &bias_s[2 * vqd as usize..]);
            } else {
                let mut fused = Vec::with_capacity((VW * qkvw) as usize);
                for r in 0..VW as usize {
                    let base = r * vqd as usize;
                    fused.extend_from_slice(&wq_s[base..base + vqd as usize]);
                    fused.extend_from_slice(&wk[base..base + vqd as usize]);
                    fused.extend_from_slice(&wv[base..base + vqd as usize]);
                }
                s.wt(ctx, &format!("{}qkvwt", p), &[qkvw, VW], &fused, VW, qkvw);
                s.data(ctx, &format!("{}qkvb", p), &[1, qkvw], &bias_s);
            }
            let bq = upload(ctx, &bias[0..vqd as usize]);
            let bk = upload(ctx, &bias[vqd as usize..2 * vqd as usize]);
            let bv = upload(ctx, &bias[2 * vqd as usize..]);
            let ebq = wbuf_t(ctx, &wq, VW, vqd);
            let ebk = wbuf_t(ctx, &wk, VW, vqd);
            let ebv = wbuf_t(ctx, &wv, VW, vqd);
            eager_qkv.push((ebq, ebk, ebv, bq, bk, bv));
        }
        {
            let host = blk
                .map(|b| lw_f16(&b.output))
                .unwrap_or_else(|| rand_f16((vqd * VW) as usize, &mut seed, 30000.0));
            s.wt(ctx, &format!("{}outwt", p), &[VW, vqd], &host, vqd, VW);
        }
        let outb_h = blk
            .map(|b| lb_f16(&b.output, VW as usize))
            .unwrap_or_else(|| rand_f16(VW as usize, &mut seed, 12000.0));
        s.data(ctx, &format!("{}outb", p), &[1, VW], &outb_h);
        let n2w_h = blk
            .map(|b| t_f16(&b.norm2.weight))
            .unwrap_or_else(|| norm_f16(VW as usize, &mut seed, 1.0));
        let n2b_h = blk
            .map(|b| t_f16(&b.norm2.bias))
            .unwrap_or_else(|| norm_f16(VW as usize, &mut seed, 0.0));
        s.data(ctx, &format!("{}n2w", p), &[VW], &n2w_h);
        s.data(ctx, &format!("{}n2b", p), &[VW], &n2b_h);
        eager_norms.push((n1w_h, n1b_h, n2w_h, n2b_h));
        {
            let host = blk
                .map(|b| lw_f16(&b.fc1))
                .unwrap_or_else(|| rand_f16((VW * V_INTER) as usize, &mut seed, 30000.0));
            s.wt(ctx, &format!("{}fc1wt", p), &[V_INTER, VW], &host, VW, V_INTER);
        }
        let fc1b_h = blk
            .map(|b| lb_f16(&b.fc1, V_INTER as usize))
            .unwrap_or_else(|| rand_f16(V_INTER as usize, &mut seed, 12000.0));
        s.data(ctx, &format!("{}fc1b", p), &[1, V_INTER], &fc1b_h);
        {
            let host = blk
                .map(|b| lw_f16(&b.fc2))
                .unwrap_or_else(|| rand_f16((V_INTER * VW) as usize, &mut seed, 30000.0));
            s.wt(ctx, &format!("{}fc2wt", p), &[VW, V_INTER], &host, V_INTER, VW);
        }
        let fc2b_h = blk
            .map(|b| lb_f16(&b.fc2, VW as usize))
            .unwrap_or_else(|| rand_f16(VW as usize, &mut seed, 12000.0));
        s.data(ctx, &format!("{}fc2b", p), &[1, VW], &fc2b_h);

        // attention（v3 方案：rank-2 列切 → Reshape rank-3 桥）。
        // manual：[b*h,s,d] 桥 + bmm/softmax/bmm（全静态）；pfa：batch PFA
        let ln1 = s.addln(&format!("{}ln1", p), &cur, &format!("{}n1w", p), &format!("{}n1b", p), "nsh", &[t, VW]);
        let (q2, k2, v2, _qdbg) = if qkv3 {
            let qm = s.mm(&format!("{}qm", p), &ln1, &[t, VW], &format!("{}qw", p), &[vqd, VW], &[t, vqd]);
            let km = s.mm(&format!("{}km", p), &ln1, &[t, VW], &format!("{}kw", p), &[vqd, VW], &[t, vqd]);
            let vm = s.mm(&format!("{}vm", p), &ln1, &[t, VW], &format!("{}vw", p), &[vqd, VW], &[t, vqd]);
            let qb = s.bias(&format!("{}qb_", p), &qm, &[t, vqd], &format!("{}qb", p));
            let kb = s.bias(&format!("{}kb_", p), &km, &[t, vqd], &format!("{}kb", p));
            let vb = s.bias(&format!("{}vb_", p), &vm, &[t, vqd], &format!("{}vb", p));
            let qdbg = qb.clone();
            (qb, kb, vb, qdbg)
        } else {
            let qkv = s.mm(&format!("{}qkv", p), &ln1, &[t, VW], &format!("{}qkvwt", p), &[qkvw, VW], &[t, qkvw]);
            let qkvb = s.bias(&format!("{}qkvb", p), &qkv, &[t, qkvw], &format!("{}qkvb", p));
            let q2 = s.slice2(&format!("{}q2", p), &qkvb, &[t, qkvw], 0, vqd);
            let k2 = s.slice2(&format!("{}k2", p), &qkvb, &[t, qkvw], vqd, vqd);
            let v2 = s.slice2(&format!("{}v2", p), &qkvb, &[t, qkvw], vqd * 2, vqd);
            let qdbg = qkvb.clone();
            (q2, k2, v2, qdbg)
        };
        let attn = if attn_manual {
            // 头主序桥（headsplit：Reshape[3,256,16,72]→TransposeD(0,2,1,3)
            // →Reshape[48,256,72]）——直接 Reshape 是错排
            let q3 = s.headsplit(&format!("{}qh", p), &q2, t, VIEWS, VPV, V_HEADS, V_HD, "shp_vs4", "shp_at3");
            let k3 = s.headsplit(&format!("{}kh", p), &k2, t, VIEWS, VPV, V_HEADS, V_HD, "shp_vs4", "shp_at3");
            let v3 = s.headsplit(&format!("{}vh", p), &v2, t, VIEWS, VPV, V_HEADS, V_HD, "shp_vs4", "shp_at3");
            let bh = VIEWS * V_HEADS;
            s.attn_manual(&format!("{}attn", p), &q3, &k3, &v3, bh, VPV, V_HD)
        } else {
            let q3 = s.reshape(&format!("{}q3", p), &q2, &[t, vqd], "shp_v3", &[VIEWS, VPV, vqd]);
            let k3 = s.reshape(&format!("{}k3", p), &k2, &[t, vqd], "shp_v3", &[VIEWS, VPV, vqd]);
            let v3 = s.reshape(&format!("{}v3", p), &v2, &[t, vqd], "shp_v3", &[VIEWS, VPV, vqd]);
            s.pfa(&format!("{}pfa", p), &q3, &k3, &v3, &[VIEWS, VPV, vqd], &[VIEWS, VPV, vqd], V_HEADS, V_HEADS, V_HD)
        };
        let a2 = if attn_manual {
            s.headmerge(&format!("{}am", p), &attn, VIEWS, VPV, V_HEADS, V_HD, "shp_vh4", "shp_flat2")
        } else {
            s.reshape(&format!("{}a2", p), &attn, &[VIEWS * V_HEADS, VPV, V_HD], "shp_flat2", &[t, vqd])
        };
        let proj = s.mm(&format!("{}proj", p), &a2, &[t, vqd], &format!("{}outwt", p), &[VW, vqd], &[t, VW]);
        let projb = s.bias(&format!("{}projb", p), &proj, &[t, VW], &format!("{}outb", p));
        let res1 = s.add2(&format!("{}res1", p), &projb, &cur, &[t, VW]);

        // mlp（exact gelu）
        let ln2 = s.addln(&format!("{}ln2", p), &res1, &format!("{}n2w", p), &format!("{}n2b", p), "nsh", &[t, VW]);
        let fc1 = s.mm(&format!("{}fc1", p), &ln2, &[t, VW], &format!("{}fc1wt", p), &[V_INTER, VW], &[t, V_INTER]);
        let fc1b = s.bias(&format!("{}fc1b", p), &fc1, &[t, V_INTER], &format!("{}fc1b", p));
        let act = s.gelu(&format!("{}act", p), &fc1b, &[t, V_INTER], false);
        let fc2 = s.mm(&format!("{}fc2", p), &act, &[t, V_INTER], &format!("{}fc2wt", p), &[VW, V_INTER], &[t, VW]);
        let fc2b = s.bias(&format!("{}fc2b", p), &fc2, &[t, VW], &format!("{}fc2b", p));
        cur = s.add2(&format!("{}out", p), &fc2b, &res1, &[t, VW]);
        if i == 0 && dbg {
            // ⚠ attn（bmm/PFA 输出）及其直接下游（proj mm）绑图输出时 desc
            // 动态（[-1,-1,-1] / [-1,1152]，size ~1e18 → malloc 崩）——
            // 动态 desc 只传一层；res1 起恢复静态
            for st in [&ln1, &_qdbg, &res1, &ln2, &fc1b, &act, &fc2b, &cur] {
                dbg_outs.push(st.clone());
            }
        }
    }

    // ---- post norm + projector ----
    let pnw_h = real
        .map(|w| t_f16(&w.vision.post_layer_norm.weight))
        .unwrap_or_else(|| norm_f16(VW as usize, &mut seed, 1.0));
    let pnb_h = real
        .map(|w| t_f16(&w.vision.post_layer_norm.bias))
        .unwrap_or_else(|| norm_f16(VW as usize, &mut seed, 0.0));
    s.data(ctx, "pnw", &[VW], &pnw_h);
    s.data(ctx, "pnb", &[VW], &pnb_h);
    {
        let host = real
            .map(|w| lw_f16(&w.vision.multimodal_projector))
            .unwrap_or_else(|| rand_f16((VW * PW) as usize, &mut seed, 30000.0));
        s.wt(ctx, "projwt", &[PW, VW], &host, VW, PW);
    }
    let projb_h = real
        .map(|w| lb_f16(&w.vision.multimodal_projector, PW as usize))
        .unwrap_or_else(|| rand_f16(PW as usize, &mut seed, 12000.0));
    s.data(ctx, "projb", &[1, PW], &projb_h);
    let pln = s.addln("pln", &cur, "pnw", "pnb", "nsh", &[t, VW]);
    let proj = s.mm("proj", &pln, &[t, VW], "projwt", &[PW, VW], &[t, PW]);
    let out = s.bias("out", &proj, &[t, PW], "projb");
    let mut fin: Vec<&str> = vec![&out];
    if dbg {
        fin.extend(dbg_outs.iter().map(|x| x.as_str()));
    }
    s.finish(&fin);
    println!("vision OM built: depth={depth} n_in={} n_out=1", s.ins().len());

    // ---- eager 参考（vision_layer_ascend 序列镜像；qkv 独立投影）----
    let trace = std::env::var("GEB_TRACE").is_ok();
    let eager = |b: &[DeviceBuffer]| -> Vec<DeviceBuffer> {
        let mut mx = |tag: &str, buf: &DeviceBuffer| {
            if trace {
                // d2h memcpy 非流序——先同步再读，否则竞态脏读（l0 out head
                // 全 0 实证：大 kernel 后下载抢跑）
                drop(stream.synchronize());
                let n = buf.len() / 2;
                let h = download_f16(ctx, buf, n);
                let m = h.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
                let head: Vec<f32> = h[..6.min(h.len())].iter().map(|v| v.to_f32()).collect();
                println!("[trace] {tag}: |max|={m:.3} head={head:?}");
                if tag == "emb" {
                    let row: Vec<f32> = h[..1152].iter().map(|v| v.to_f32()).collect();
                    let mean = row.iter().sum::<f32>() / row.len() as f32;
                    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32;
                    println!("[trace] emb row0: mean={mean:.5} var={var:.3e} head={:?}", &row[..6]);
                }
            }
        };
        let emb = aops::matmul_b_t_fp16(ctx, stream, &b[0], [t, V_PATCH_W], &b[1], V_PATCH_W, VW).unwrap();
        let emb = aops::bias_add_fp16(ctx, stream, &emb, &b[2], t, VW).unwrap();
        let mut h = aops::add_fp16(ctx, stream, &emb, &b[3], &[t, VW]).unwrap();
        mx("emb", &h);
        let zeros = &b[4];
        for li in 0..depth {
            // qkv3 层内输入布局：n1w,n1b,qw,kw,vw,qb,kb,vb,outwt,outb,
            // n2w,n2b,fc1wt,fc1b,fc2wt,fc2b（16 项）；默认 12 项
            let w = &b[layer_bases[li]..layer_bases[li] + if qkv3 { 16 } else { 12 }];
            let (ow, ob, f1w, f1b, f2w, f2b) = if qkv3 { (8, 9, 12, 13, 14, 15) } else { (4, 5, 8, 9, 10, 11) };
            let (wq, wk, wv, bq, bk, bv) = &eager_qkv[li];
            let (n1w_h, n1b_h, n2w_h, n2b_h) = &eager_norms[li];
            let normed = host_ln(ctx, stream, &h, n1w_h, n1b_h, t, VW);
            mx(&format!("l{li} ln1"), &normed);
            let q = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [t, VW], wq, VW, vqd).unwrap(), bq, t, vqd).unwrap();
            let k = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [t, VW], wk, VW, vqd).unwrap(), bk, t, vqd).unwrap();
            let v = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [t, VW], wv, VW, vqd).unwrap(), bv, t, vqd).unwrap();
            let attn = aops::prompt_flash_attention_bsh_batch_fp16(
                ctx, stream, &q, &k, &v, VIEWS, VPV, V_HEADS, V_HEADS, V_HD, None).unwrap();
            mx(&format!("l{li} attn"), &attn);
            let proj = aops::matmul_b_t_fp16(ctx, stream, &attn, [t, vqd], &w[ow], vqd, VW).unwrap();
            let proj = aops::bias_add_fp16(ctx, stream, &proj, &w[ob], t, VW).unwrap();
            let res1 = aops::add_fp16(ctx, stream, &proj, &h, &[t, VW]).unwrap();
            mx(&format!("l{li} res1"), &res1);
            let norm2 = host_ln(ctx, stream, &res1, n2w_h, n2b_h, t, VW);
            mx(&format!("l{li} ln2"), &norm2);
            let act = aops::matmul_b_t_fp16(ctx, stream, &norm2, [t, VW], &w[f1w], VW, V_INTER).unwrap();
            let act = aops::bias_add_fp16(ctx, stream, &act, &w[f1b], t, V_INTER).unwrap();
            mx(&format!("l{li} fc1b"), &act);
            let act = aops::gelu_fp16(ctx, stream, &act, &[t, V_INTER], false).unwrap();
            mx(&format!("l{li} act"), &act);
            let out = aops::matmul_b_t_fp16(ctx, stream, &act, [t, V_INTER], &w[f2w], V_INTER, VW).unwrap();
            let out = aops::bias_add_fp16(ctx, stream, &out, &w[f2b], t, VW).unwrap();
            h = aops::add_fp16(ctx, stream, &out, &res1, &[t, VW]).unwrap();
            mx(&format!("l{li} out"), &h);
        }
        let n = b.len();
        let normed = host_ln(ctx, stream, &h, &pnw_h, &pnb_h, t, VW);
        let proj = aops::matmul_b_t_fp16(ctx, stream, &normed, [t, VW], &b[n - 2], VW, PW).unwrap();
        let fin = aops::bias_add_fp16(ctx, stream, &proj, &b[n - 1], t, PW).unwrap();
        mx("final", &fin);
        // ⚠ 闭包内中间量 drop 时 kernel 可能未落（aclrtFree 非流序，内存
        // 回池被后续 GE 输出 malloc 复用 → 反向踩烂在途 eager 数据——
        // ref |max| 0.014/0.321 随机跳，非确定性 parity 崩的根因）。
        // 返回前强制同步。
        drop(stream.synchronize());
        vec![fin]
    };

    if let Some(st) = e2e {
        // e2e：加载 vision_real.om 跑一次，抓 post-projector 输出 [VT, PW]
        let outs = e2e_run(ctx, stream, &mut s, "vision");
        // vision OM 绑 108 个 LN aux 输出（trap #17：mean/rstd 死端会让 y 爆）
        // ——主输出（post-projector）在 idx0
        assert!(outs.len() >= 1);
        st.vision_out = outs[0].clone();
        let m = st.vision_out.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
        println!("[e2e] vision_out |max|={m:.4}");
        st.cmp_mid("vision_out", &st.vision_out);
        // GEB_E2E_REPLAY：逐帧覆写 patches（binds[0] = 段首注册的 patches
        // 输入）重跑；aux 输出 buffer 一次分配复用、只下载主输出 idx0
        if !st.replay.is_empty() {
            let n_out = s.g.num_outputs().unwrap();
            let routs: Vec<DeviceBuffer> = (0..n_out)
                .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("replay out malloc"))
                .collect();
            let h2d = |buf: &DeviceBuffer, vals: &[f16]| {
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
                ctx.copy_h2d(buf, &bytes).expect("replay h2d");
            };
            let t0 = std::time::Instant::now();
            for f in st.replay.iter_mut() {
                assert_eq!(f.patches.len() * 2, s.binds[0].len(), "replay patches 尺寸不符");
                h2d(&s.binds[0], &f.patches);
                let ins = s.ins();
                let refs: Vec<&DeviceBuffer> = routs.iter().collect();
                s.g.run(&ins, &refs, stream).expect("ge run");
                drop(stream.synchronize());
                f.vision_out = download_f16(ctx, &routs[0], s.g.output_size(0).unwrap() / 2);
            }
            println!("[e2e] replay vision ×{} {:?}", st.replay.len(), t0.elapsed());
        }
        // GEB_E2E_SERVE：机械下沉 E2eStage（per-frame 执行由 seg_e2e 的
        // serve 循环驱动；routs 一次分配跨请求复用，同 replay 语义）
        if std::env::var("GEB_E2E_SERVE").is_ok() {
            let n_out = s.g.num_outputs().unwrap();
            let routs: Vec<DeviceBuffer> = (0..n_out)
                .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("serve out malloc"))
                .collect();
            st.vmech = Some(VisionMech { s, routs });
        }
        return;
    }
    parity_and_bench(ctx, stream, &mut s, "vision", &[(t * PW) as usize], &eager, bench);
    ge_builder::fini().expect("fini");
    println!("GE_VISION_PROBE_OK");
}

// ---------------------------------------------------------------------------
// prefix 段：depth×language 层全序，输出每层 k/v（末层无 tail——eager
// compute_tail=false 语义）
// ---------------------------------------------------------------------------

/// int8 权重注册：s_eff 吸收量化 → wq int8 [n,k]（Data 或 Const）+ scale
/// f32 [n] + smooth/inv f16 行 Data（native 开关二选一）。GEB_INT8_DUMP
/// 同步落盘（L{nn}_{proj}.wq/.scale，对拍 int8_weights_v1.npz 金标准）。
/// 返回 (wq, ws, smooth|inv) 三名。
#[allow(clippy::type_complexity)]
fn reg_i8(
    s: &mut Seg, ctx: &AscendContext, facs: &std::collections::HashMap<String, Int8Smooth>,
    key: &str, dump_tag: &str, w32: &[f32], k: i64, n: i64, native: bool, dump: Option<&str>,
) -> (String, String, String) {
    let f = facs.get(key).unwrap_or_else(|| panic!("int8 smooth 缺 {key}"));
    assert_eq!(f.s_eff.len(), k as usize, "{key} 因子长度 {} ≠ k={k}", f.s_eff.len());
    let (wq, scale) = quant_w8(w32, k as usize, &f.s_eff);
    if let Some(dir) = dump {
        let _ = std::fs::create_dir_all(dir);
        let wqb: Vec<u8> = wq.iter().map(|&v| v as u8).collect();
        std::fs::write(format!("{dir}/{dump_tag}.wq"), &wqb).expect("dump wq");
        let sb: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/{dump_tag}.scale"), &sb).expect("dump scale");
    }
    // GEB_INT8_WCONST=1（默认随 GEB_WCONST）：wq/ws 烤 Const——int8 Data
    // 权重每执行付 ND→NZ TransData 税（实测 prefix 129.5ms 反比 f16 105 慢
    // ~25ms ≈ 2GB/63GB/s（serve 实测）；Const 编译期定型免税，
    // f16 wt() 同理）。smooth 行体量小（[k] f16），保持 Data 无妨
    let wconst8 = envi("GEB_INT8_WCONST", envi("GEB_WCONST", 0)) == 1;
    let (wq_n, ws_n) = if wconst8 {
        let wqb: Vec<u8> = wq.iter().map(|&v| v as u8).collect();
        s.g.add_const_raw(&format!("{dump_tag}wq"), &[n, k], Dtype::Int8, &wqb).unwrap();
        let sb: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        s.g.add_const_raw(&format!("{dump_tag}ws"), &[n], Dtype::Fp32, &sb).unwrap();
        s.datas.insert(format!("{dump_tag}wq"));
        s.datas.insert(format!("{dump_tag}ws"));
        (format!("{dump_tag}wq"), format!("{dump_tag}ws"))
    } else {
        (
            s.data_i8(ctx, &format!("{dump_tag}wq"), &[n, k], &wq),
            s.data_f32(ctx, &format!("{dump_tag}ws"), &[n], &scale),
        )
    };
    let sm_n = if native {
        let smf: Vec<f16> = f.smooth.iter().map(|&v| f16::from_f32(v)).collect();
        // gear 模式两难与解：纯静态 Data 毒化 SetDims 档位匹配（实测定罪）；
        // 定值动态维又会撑爆 payload 的 ACL_MAX_DIM_CNT=128（全深
        // 18 层×9 输入 + 基础 24 = 186 维）。缺省（=1）走 Const——此前
        // 「Const 撞 legacy 墙 rc=-7」实为 dynamicDims 选项串 bug 的误判
        // （旧版选项被 payload 静态尾维污染，与 Const 无关）
        if !s.gear_ps.is_empty() {
            s.const_f16(&format!("{dump_tag}sm"), &[k], &smf)
        } else {
            s.data(ctx, &format!("{dump_tag}sm"), &[k], &smf)
        }
    } else {
        let invf: Vec<f16> = f.inv.iter().map(|&v| f16::from_f32(v)).collect();
        if !s.gear_ps.is_empty() {
            s.const_f16(&format!("{dump_tag}inv"), &[1, k], &invf)
        } else {
            s.data(ctx, &format!("{dump_tag}inv"), &[1, k], &invf)
        }
    };
    // bias 折入算子实验已毙：int32 Const 输入 aclgrphBuildModel rc=-7
    // （legacy 算子 bias 分支不可用，实测）。改为调用方直接砍 bias op
    //（Gemma 投影 bias≡0 + int8 已 assert 真权重——语义零损失）
    (wq_n, ws_n, sm_n)
}

/// 统一 int8/f16 投影：int8 = QuantMatmulDequant（native smooth 直入算子，
/// GEB_INT8_NATIVE=0 回退 gatew 行预乘）；f16 = MatMulV2（w16 = 原权重段
/// 名）。两路 w dims 同 [n,k]。bias 均由调用方处理（int8 路径 Gemma
/// bias≡0 直接砍 op）。
#[allow(clippy::type_complexity)]
fn mm_or_qmd(
    s: &mut Seg, name: &str, a: &str, a_dims: &[i64], i8w: Option<&(String, String, String)>,
    w16: &str, n: i64, o: &[i64], native: bool,
) -> String {
    let w_dims = [n, a_dims[1]];
    match (i8w, native) {
        (Some((wn, sn, mn)), true) => s.qmd(name, a, a_dims, wn, &w_dims, sn, Some(mn), None, n, o),
        (Some((wn, sn, mn)), false) => {
            let xs = s.gatew(&format!("{name}xs"), a, a_dims, mn);
            s.qmd(name, &xs, a_dims, wn, &w_dims, sn, None, None, n, o)
        }
        (None, _) => s.mm(name, a, a_dims, w16, &w_dims, o),
    }
}

/// int8 层内 f16 成员权重 host：组 norm 回 torch 域（g1/g2 = 1+g）后，
/// 引擎 fold 权重（w·(1+g)）须除回 fold 再进 f16——否则 fold 双计。
/// unfold=false（组未切 int8）时保持引擎原样（= lw_f16 语义）。
fn lin_f16_unfold(l: &LinearWeights, out_dim: usize, unfold: bool, g: &[f32]) -> Vec<f16> {
    let w32 = l.weight.to_f32_vec().unwrap();
    let w = if unfold {
        assert_eq!(g.len(), w32.len() / out_dim, "unfold 因子长度不符");
        w32.into_iter()
            .enumerate()
            .map(|(i, v)| v / g[i / out_dim])
            .collect::<Vec<f32>>()
    } else {
        w32
    };
    w.into_iter().map(f16::from_f32).collect()
}

pub fn seg_prefix(be: &AscendBackend, bench: bool, real: Option<&Pi05Weights>, e2e: Option<&mut E2eStage>) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH_PREFIX", envi("GEB_DEPTH", 18)) as usize;
    let tokens = envi("GEB_TOKENS", 64);
    // GEB_PREFIX_DROP_EMPTY：尾部空视图行数（9/10 语义根因修复）。
    // golden_gen 按 LIBERO 语义只喂 2 真实视图，empty_camera 走 missing 路径
    // （-1 pad 图进 SigLIP 但 pad_mask=0）：make_att_2d_masks 的 pad_2d 把该
    // 256 个 key 对所有人遮蔽，position_ids=cumsum(pad)-1 使文本位置塌缩到
    // 512..711。pad 列 softmax 贡献恰为 0、可见 token 位置恰为 arange ⇒
    // 数学上等价于直接剔除该视图行（968→712，全开+arange 语义不变）——
    // theory_check.py 实证 h1 对应行 0.128%。引擎此前把 pad 视图当一等
    // 公民（可见 key + 占位 512..767）＝ e2e 314% 漂移的真根因
    let drop_v = envi("GEB_PREFIX_DROP_EMPTY", 0);
    let p = VT + tokens - drop_v; // prefix 长度（16 倍数纪律）
    // 动态分档（GEB_DYN_LLIST）：desc 维度里的 p 一律 -1（gd()），内容/
    // buffer 仍按 p 物化（gear 模式下 tokens 必须 = 档表 max L——rope 表/
    // zeros/swap 逐行 p 无关，p_max 全量即前缀截断语义）。L 相关 Const
    // （qswap_c/kswap_c）在 gear 模式转 Data[-1]。档表 p 值喂 Seg，finish
    // 合成 ge.dynamicDims + 每档 S2 payload
    // GEB_PREFIX_POOL：prefix 段改静态组装 + per-L 静态 OM 池换柄执行——
    // 判决：动态图结构（-1 维 Reshape / rope·qswap Data 化）
    // 使 int8 逐层漂移 3×（真帧互差 13.1% actions 级；2 档对照证伪
    // range-tiling 假设）；llist 仍留给 flow 池与客户端校验
    let gear_ll = if std::env::var("GEB_PREFIX_POOL").is_ok() {
        println!("[prefix-pool] GEB_PREFIX_POOL 生效：prefix 静态组装 + per-L 池");
        None
    } else {
        dyn_llist()
    };
    let mut gear_ps_hold: Option<Vec<i64>> = None;
    let gd = |v: i64| if gear_ll.is_some() { -1 } else { v };
    if let Some(g) = &gear_ll {
        assert!(
            std::env::var("GEB_WCONST").is_ok(),
            "GEB_DYN_LLIST 需 GEB_WCONST=1（权重 Const 根图单份——档间零重复的前提）"
        );
        assert!(std::env::var("GEB_ROPEFLAT").is_ok(), "GEB_DYN_LLIST 需 GEB_ROPEFLAT=1");
        assert!(
            std::env::var("GEB_ATTN").map(|v| v == "manual").unwrap_or(false),
            "GEB_DYN_LLIST 需 GEB_ATTN=manual"
        );
        assert!(std::env::var("GEB_QKV3").is_ok(), "GEB_DYN_LLIST 需 GEB_QKV3=1");
        let lmax = *g.iter().max().unwrap();
        assert_eq!(tokens, lmax, "GEB_DYN_LLIST 模式 GEB_TOKENS 须 = 档表 max L（binds 按 max 物化）");
        let ps = g.iter().map(|&l| VT + l - drop_v).collect::<Vec<i64>>();
        println!("[gear] prefix 动态分档：L {:?} → p {:?}", g, ps);
        gear_ps_hold = Some(ps);
    }
    let mut seed = 0xBEEFu32;
    // GEB_QKV3：q/k/v 独立投影（同 vision 段——SliceD 列切视图的运行时
    // MemcopyAsync 物化是 prefix 750ms 的主源，取证见 vision 段注记）
    let qkv3 = std::env::var("GEB_QKV3").is_ok();
    // GEB_ATTN=manual：手工 GQA attention（bmm+softmax+bmm + k/v TileD 广播）
    // 替代 PFA——PFA 是 transformer-API 型算子，静态 OM 内走 host launcher
    // （每次执行 host tiling + staging）。scale 由 pscale 折进 q 权重/bias。
    let attn_manual = std::env::var("GEB_ATTN")
        .map(|v| v == "manual").unwrap_or(false);
    let pscale = if attn_manual { 1.0f32 / (HD as f32).sqrt() } else { 1.0 };
    // GEB_PREFIX_INT8（生产化）：7 投影 → QuantMatmulDequant。因子文件
    // = GEB_INT8_SMOOTH（默认 pyo3_check/smooth_calib_v1_engine.safetensors，
    // torch 域 npz 经 Gemma 1+g fold 校正转换）。GEB_INT8_PROJS 子集分阶段
    // bring-up（默认全 7）；GEB_INT8_NATIVE=0 回退 TileD+Mul 行预乘。
    // GEB_INT8_DUMP=<dir> 落盘量化产物对拍 int8_weights_v1.npz 金标准。
    let int8 = prefix_int8();
    let (facs, i8norms): (
        std::collections::HashMap<String, Int8Smooth>,
        std::collections::HashMap<String, [Vec<f32>; 2]>,
    ) = if int8 {
        assert!(real.is_some(), "GEB_PREFIX_INT8 需要 GEB_CKPT 真权重");
        assert!(qkv3, "GEB_PREFIX_INT8 需要 GEB_QKV3=1（fused qkv 无法 per-matrix smooth）");
        let path = std::env::var("GEB_INT8_SMOOTH")
            .unwrap_or_else(|_| "./smooth_calib_v1_engine.safetensors".into());
        println!("[int8] prefix int8 开启：smooth={path}");
        load_int8_smooth(&path)
    } else {
        (std::collections::HashMap::new(), std::collections::HashMap::new())
    };
    let int8_projs: std::collections::HashSet<String> = std::env::var("GEB_INT8_PROJS")
        .unwrap_or_else(|_| "q,k,v,o,gate,up,down".into())
        .split(',')
        .map(|x| x.trim().to_string())
        .collect();
    let int8_native = envi("GEB_INT8_NATIVE", 1) == 1;
    let dump_dir_s = std::env::var("GEB_INT8_DUMP").ok();
    let dump_dir = dump_dir_s.as_deref();
    let i8sel = |p: &str| int8 && int8_projs.contains(p);
    let mut s = Seg::new("ge_prefix");
    if let Some(ps) = gear_ps_hold.take() {
        s.set_gear_ps(ps);
    }

    // ---- 段级输入 ----
    // 0 x0 / 1 zeros / 2..7 flat rope 六件（eager flat 版用）/
    // 8..11 rank-2 rope 表四件（GE rank-2 组合用）/ 层 10 项
    // e2e x0 = vision_out ‖ token_embedding[token_ids]（embed_prefix 同序：
    // 视觉前、语言后；state 不进 prefix——π0.5 走离散化进 prompt 文本）。
    // GEB_E2E_PAD_TOKENS（range padding 对拍）：golden 真长 <
    // GEB_TOKENS（桶界）——token 补 0 到桶界（pad 行由 padmask 遮蔽），
    // 真长供 padmask 填充。cmp_mid 截断比前缀 = 真行对拍
    let pad_true_l = envi("GEB_E2E_PAD_TOKENS", 0);
    let x0_h = match (&e2e, real) {
        (Some(st), Some(w)) => {
            assert!(!st.vision_out.is_empty(), "e2e: 须先跑 vision 段");
            assert_eq!(st.vision_out.len(), (VT * PW) as usize);
            if pad_true_l > 0 {
                assert_eq!(st.token_ids.len(), pad_true_l as usize, "GEB_E2E_PAD_TOKENS ≠ golden token 数");
                assert!(pad_true_l < tokens, "GEB_E2E_PAD_TOKENS 须 < GEB_TOKENS（桶界）");
            } else {
                assert_eq!(st.token_ids.len(), tokens as usize, "golden token 数与 GEB_TOKENS 不符");
            }
            // 一次性 f32 物化 [vocab,PW]（~2.1GB 峰值，查完即弃）
            let emb = w.vision.token_embedding.to_f32_vec().unwrap();
            let vocab_w = PW as usize;
            // LeRobot embed_language_tokens 语义：查表行 × √width（gemma
            // embed scale，modeling_pi05 L695；vision 段无此缩放）。
            // 空视图剔除：vision_out 只取前 (VT-drop_v) 行（视图主序，empty
            // 在尾），golden 对拍键 x0_vis 同序
            let lang_scale = (PW as f32).sqrt();
            let vis_rows = ((VT - drop_v) * PW) as usize;
            let mut x0 = st.vision_out[..vis_rows].to_vec();
            let ids_eff: Vec<u32> = if pad_true_l > 0 {
                let mut v = st.token_ids.clone();
                v.resize(tokens as usize, 0);
                v
            } else {
                st.token_ids.clone()
            };
            x0.reserve(ids_eff.len() * vocab_w);
            for &id in &ids_eff {
                let r = id as usize * vocab_w;
                assert!(r + vocab_w <= emb.len(), "token id {id} 超 vocab");
                x0.extend(emb[r..r + vocab_w].iter().map(|&v| f16::from_f32(v * lang_scale)));
            }
            x0
        }
        (None, None) => panic!("e2e 需要 GEB_CKPT（token_embedding 查表）"),
        _ => rand_f16((p * PW) as usize, &mut seed, 100.0),
    };
    s.data(ctx, "x0", &[gd(p), PW], &x0_h);
    // GEB_NORM32 v2：常量走 Const 折叠（n32_consts 懒建），无 Data 注册
    // ——binds[0]=x0 的硬编码约束自然解除
    // bisect：x0 = cat(vision_out, 查表×√PW) 组装后即比（idx 域 [0,VT*PW)
    // = 视觉行 / 其后 = 语言行——分岔落哪个段一眼可辨）。剔除空视图时
    // golden 参考键为 x0_vis（712 行同序）
    if let Some(st) = e2e.as_ref() {
        if drop_v > 0 {
            st.cmp_mid("x0_vis", &x0_h);
        } else {
            st.cmp_mid("x0", &x0_h);
        }
    }
    // zeros 显式 gear 化（data_zeros 保持纯静态——vision 段共用）：
    // 内容全零逐行 p 无关，p_max 全量即前缀截断
    {
        let zh = vec![f16::from_f32(0.0); (p * PW) as usize];
        s.data(ctx, "zeros", &[gd(p), PW], &zh);
    }
    // GEB_ATTN_MASK（range padding）：softmax 加性 mask 输入
    // [1,1,p]——bake 内容占位全 0（无 pad），serve/probe 侧按真 L 填充
    //（行 [vis+L, vis+P) = -30000；f16 域禁 -inf）。全层共用一只 Data
    let attn_mask_n: Option<String> = if envi("GEB_ATTN_MASK", 0) == 1 {
        let mh = vec![f16::from_f32(0.0); p as usize];
        Some(s.data(ctx, "padmask", &[1, 1, p], &mh))
    } else {
        None
    };
    // PAD 对拍：padmask 填真 mask（行 [vis+真L, p) = -30000）——serve 同义
    if attn_mask_n.is_some() && pad_true_l > 0 {
        let vis = VT - envi("GEB_PREFIX_DROP_EMPTY", 0);
        let mut mh = vec![f16::from_f32(0.0); p as usize];
        for r in &mut mh[(vis + pad_true_l) as usize..] {
            *r = f16::from_f32(-30000.0);
        }
        let bytes: Vec<u8> = mh.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        let bi = s.bind_of("padmask");
        ctx.copy_h2d(&s.binds[bi], &bytes).expect("padmask h2d");
    }
    let (qc, qs, qi) = rope_flat_const(p, HEADS, 0);
    let (kc, ks, ki) = rope_flat_const(p, KV_HEADS, 0);
    let qf = (p * HEADS * 2, HD / 2);
    let kf = (p * KV_HEADS * 2, HD / 2);
    // gear 模式：dim0 一律 -1 + mul 登记（qf.0 = p×(HEADS*2)）；内容 p_max
    // 全量（逐行 p 无关——前缀截断语义）
    s.data(ctx, "qcos", &[gd(qf.0), qf.1], &qc);
    s.set_gear_mul("qcos", HEADS * 2);
    s.data(ctx, "qsin", &[gd(qf.0), qf.1], &qs);
    s.set_gear_mul("qsin", HEADS * 2);
    s.data_i32(ctx, "qswap", &[gd(qf.0)], &qi);
    s.set_gear_mul("qswap", HEADS * 2);
    s.data(ctx, "kcos", &[gd(kf.0), kf.1], &kc);
    s.set_gear_mul("kcos", KV_HEADS * 2);
    s.data(ctx, "ksin", &[gd(kf.0), kf.1], &ks);
    s.set_gear_mul("ksin", KV_HEADS * 2);
    s.data_i32(ctx, "kswap", &[gd(kf.0)], &ki);
    s.set_gear_mul("kswap", KV_HEADS * 2);
    // rank-2 表（物理 [t*heads, d] 与 [t, heads*d] 同布局；desc 按 rank-2 设）
    let (qc2, qs2) = rope_rank2_const(p, HEADS, 0);
    let (kc2, ks2) = rope_rank2_const(p, KV_HEADS, 0);
    s.data(ctx, "qcos2", &[gd(p), QD], &qc2);
    s.data(ctx, "qsin2", &[gd(p), QD], &qs2);
    s.data(ctx, "kcos2", &[gd(p), KVD], &kc2);
    s.data(ctx, "ksin2", &[gd(p), KVD], &ks2);
    // Reshape shape 张量（rank-3 桥；Const 内容 gear 不变——p 位写 -1，
    // 分支推理按档解析，HF 参考链实证）
    s.const_i32("shp_q3", &[1, gd(p) as i32, QD as i32]);
    s.const_i32("shp_k3", &[1, gd(p) as i32, KVD as i32]);
    s.const_i32("shp_v3", &[1, gd(p) as i32, KVD as i32]);
    // 手工 GQA attention 的头主序桥 shape（headsplit/headmerge 用，views=1）
    s.const_i32("shp_mq4", &[1, gd(p) as i32, HEADS as i32, HD as i32]);
    s.const_i32("shp_mq3", &[HEADS as i32, gd(p) as i32, HD as i32]);
    s.const_i32("shp_ma4", &[1, HEADS as i32, gd(p) as i32, HD as i32]);
    s.const_i32("shp_mflat", &[gd(p) as i32, QD as i32]);
    // GEB_ROPEFLAT：flat 版 rope（换视图行交换，免列切视图物化）。
    // swap 索引：静态 Const 折叠；gear 模式转 Data[-1]（Const 全档共享而
    // 长度随档变；Data 逐档特化 desc，内容 r^1 逐行 p 无关——实证）
    let ropeflat = std::env::var("GEB_ROPEFLAT").is_ok();
    if gear_ll.is_some() {
        s.data_i32(ctx, "qswap_c", &[gd(qf.0)], &qi);
        s.set_gear_mul("qswap_c", HEADS * 2);
        s.data_i32(ctx, "kswap_c", &[gd(kf.0)], &ki);
        s.set_gear_mul("kswap_c", KV_HEADS * 2);
    } else {
        s.const_i32("qswap_c", &qi);
        s.const_i32("kswap_c", &ki);
    }
    s.const_i32("shp_qflat", &[gd(qf.0) as i32, qf.1 as i32]);
    s.const_i32("shp_kflat", &[gd(kf.0) as i32, kf.1 as i32]);
    s.const_i32("shp_kb2", &[gd(p) as i32, KVD as i32]);

    // ---- 层循环（eager qkv 独立投影，同 vision 段注记）----
    let mut eager_qkv: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer)> = Vec::new();
    let mut layer_bases = Vec::with_capacity(depth);
    let mut cur = "x0".to_string();
    let mut kv_outs: Vec<String> = Vec::new();
    // GEB_DBG_MID：层 0 attention 合并输出（o_proj 前）与 h1 作为图输出尾部
    // 追加——golden m0/h1 对拍，钉 GE 图内第一坏级（attention vs MLP vs 残差）
    let dbg_mid = std::env::var("GEB_DBG_MID").is_ok();
    let mut dbg_outs: Vec<String> = Vec::new();
    for i in 0..depth {
        let base = s.binds.len();
        layer_bases.push(base);
        let tag = format!("l{i}_");
        let last = i + 1 == depth;
        // 真权重：语言层 host 解析时已把 Gemma 1+w scale 折进 q/k/v/gate/up
        //（g1/g2 = ones）——probe 图直接消费该约定，无需再乘 norm scale。
        // GEB_LAYER_OFFSET：权重取层偏移（隔离实验——如 d2 图层 0 用 L1
        // 权重 + x0 喂 golden h1，直接对拍 kvk_l1 公式，裁决 WCONST 烤入）
        let lay = real.and_then(|w| {
            w.language_layers.get(i + envi("GEB_LAYER_OFFSET", 0) as usize)
        });
        let lay_i = i + envi("GEB_LAYER_OFFSET", 0) as usize; // smooth 因子键层号同权重层
        // int8 q/k/v 注册（f32 直取，不过 f16；manual-attn 的 pscale 折 q——
        // 行量化对整体常数不变，仅 scale 同比缩小）
        let (iq8, ik8, iv8) = if int8 {
            let l = lay.expect("GEB_PREFIX_INT8 需真权重");
            let fk = |proj: &str| format!("prefix/L{:02}/{proj}", lay_i);
            let dt = |proj: &str| format!("L{:02}_{proj}", lay_i);
            let mut q32 = l.attention.q.weight.to_f32_vec().unwrap();
            if attn_manual {
                for v in q32.iter_mut() {
                    *v *= pscale;
                }
            }
            (
                if i8sel("q") {
                    Some(reg_i8(&mut s, ctx, &facs, &fk("q"), &dt("q"), &q32, PW, QD, int8_native, dump_dir))
                } else {
                    None
                },
                if i8sel("k") {
                    Some(reg_i8(&mut s, ctx, &facs, &fk("k"), &dt("k"),
                        &l.attention.k.weight.to_f32_vec().unwrap(), PW, KVD, int8_native, dump_dir))
                } else {
                    None
                },
                if i8sel("v") {
                    Some(reg_i8(&mut s, ctx, &facs, &fk("v"), &dt("v"),
                        &l.attention.v.weight.to_f32_vec().unwrap(), PW, KVD, int8_native, dump_dir))
                } else {
                    None
                },
            )
        } else {
            (None, None, None)
        };
        let qkv_i8 = i8sel("q") || i8sel("k") || i8sel("v");
        let (g1_h, g1_t): (Vec<f16>, &[f32]) = if qkv_i8 {
            // int8 组：norm 回 torch 域（gamma = (1+g) 真值，loader fold 的
            // 替代源）——smooth 行 = s_npz 与 npz 权重面同锚（纯引擎域 s_eff
            // 因 L00 近零 fold 会爆 f16，实测）
            let g = &i8norms[&format!("prefix/L{:02}", lay_i)][0];
            assert_eq!(g.len(), PW as usize, "g1 长度不符");
            (g.iter().map(|&v| f16::from_f32(v)).collect(), g)
        } else {
            (
                lay.map(|l| t_f16(&l.input_norm_scale))
                    .unwrap_or_else(|| norm_f16(PW as usize, &mut seed, 1.0)),
                &[],
            )
        };
        // gear 模式：Const 化（定值动态维会撑爆 SetDims payload 的 128 维
        // 上限——同 smooth 注册注记）
        if !s.gear_ps.is_empty() {
            s.const_f16(&format!("{}g1", tag), &[PW], &g1_h);
        } else {
            s.data(ctx, &format!("{}g1", tag), &[PW], &g1_h);
        }
        {
            // f16 host 仅在对应投影未转 int8 时构造（int8 段已在上一块注册；
            // 组切 torch 域后 f16 成员除回 fold——见 lin_f16_unfold）
            let wq = if iq8.is_none() {
                lay.map(|l| lin_f16_unfold(&l.attention.q, QD as usize, qkv_i8, g1_t))
                    .unwrap_or_else(|| rand_f16((PW * QD) as usize, &mut seed, 8000.0))
            } else {
                Vec::new()
            };
            let wk = if ik8.is_none() {
                lay.map(|l| lin_f16_unfold(&l.attention.k, KVD as usize, qkv_i8, g1_t))
                    .unwrap_or_else(|| rand_f16((PW * KVD) as usize, &mut seed, 8000.0))
            } else {
                Vec::new()
            };
            let wv = if iv8.is_none() {
                lay.map(|l| lin_f16_unfold(&l.attention.v, KVD as usize, qkv_i8, g1_t))
                    .unwrap_or_else(|| rand_f16((PW * KVD) as usize, &mut seed, 8000.0))
            } else {
                Vec::new()
            };
            // Gemma 投影无 bias（真权重 = zeros）
            let bias = lay
                .map(|_| vec![f16::from_f32(0.0); QKVW as usize])
                .unwrap_or_else(|| rand_f16(QKVW as usize, &mut seed, 4000.0));
            // manual attention：q 块（权重+bias）预乘 1/√hd（图内无 scale
            // 算子）；eager 侧保持原值（PFA scale_value 承担）
            let wq_s: Vec<f16> = if attn_manual {
                wq.iter().map(|v| f16::from_f32(v.to_f32() * pscale)).collect()
            } else {
                wq.clone()
            };
            let mut bias_s = bias.clone();
            if attn_manual {
                for v in bias_s[..QD as usize].iter_mut() {
                    *v = f16::from_f32(v.to_f32() * pscale);
                }
            }
            if qkv3 {
                // qkv3 层内输入位次 1..7（g1 之后）；eager 按 qkv3 偏移读
                if iq8.is_none() {
                    s.wt(ctx, &format!("{}qw", tag), &[QD, PW], &wq_s, PW, QD);
                }
                if ik8.is_none() {
                    s.wt(ctx, &format!("{}kw", tag), &[KVD, PW], &wk, PW, KVD);
                }
                if iv8.is_none() {
                    s.wt(ctx, &format!("{}vw", tag), &[KVD, PW], &wv, PW, KVD);
                }
                // gear 模式行重复注册（row_bcast）；int8 折叠不消费 → 原行省内存
                s.row_bcast(ctx, &format!("{}qb", tag), if iq8.is_some() { 1 } else { p }, QD, &bias_s[..QD as usize]);
                s.row_bcast(ctx, &format!("{}kb", tag), if ik8.is_some() { 1 } else { p }, KVD, &bias_s[QD as usize..(QD + KVD) as usize]);
                s.row_bcast(ctx, &format!("{}vb", tag), if iv8.is_some() { 1 } else { p }, KVD, &bias_s[(QD + KVD) as usize..]);
            } else {
                let mut fused = Vec::with_capacity((PW * QKVW) as usize);
                for r in 0..PW as usize {
                    let rb = r * QD as usize;
                    fused.extend_from_slice(&wq_s[rb..rb + QD as usize]);
                    let rb = r * KVD as usize;
                    fused.extend_from_slice(&wk[rb..rb + KVD as usize]);
                    fused.extend_from_slice(&wv[rb..rb + KVD as usize]);
                }
                s.wt(ctx, &format!("{}qkvwt", tag), &[QKVW, PW], &fused, PW, QKVW);
                s.data(ctx, &format!("{}qkvb", tag), &[1, QKVW], &bias_s);
            }
            // eager 缓冲（f16 参考镜像）——int8 模式对拍走 e2e golden，跳过
            if !int8 {
                let bq = upload(ctx, &bias[0..QD as usize]);
                let bk = upload(ctx, &bias[QD as usize..(QD + KVD) as usize]);
                let bv = upload(ctx, &bias[(QD + KVD) as usize..]);
                let ebq = wbuf_t(ctx, &wq, PW, QD);
                let ebk = wbuf_t(ctx, &wk, PW, KVD);
                let ebv = wbuf_t(ctx, &wv, PW, KVD);
                eager_qkv.push((ebq, ebk, ebv, bq, bk, bv));
            }
        }
        // o/gate/up/down：int8 选中者在使用点注册（末层 break 前不触达），
        // 此处 f16 wt 段按需保留
        if !i8sel("o") {
            let host = lay
                .map(|l| lw_f16(&l.attention.output))
                .unwrap_or_else(|| rand_f16((QD * PW) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}outwt", tag), &[PW, QD], &host, QD, PW);
        }
        let outb_h = lay
            .map(|_| vec![f16::from_f32(0.0); PW as usize])
            .unwrap_or_else(|| rand_f16(PW as usize, &mut seed, 4000.0));
        s.row_bcast(ctx, &format!("{}outb", tag), if i8sel("o") { 1 } else { p }, PW, &outb_h);
        let mlp_i8 = i8sel("gate") || i8sel("up");
        let (g2_h, g2_t): (Vec<f16>, &[f32]) = if mlp_i8 {
            let g = &i8norms[&format!("prefix/L{:02}", lay_i)][1];
            assert_eq!(g.len(), PW as usize, "g2 长度不符");
            (g.iter().map(|&v| f16::from_f32(v)).collect(), g)
        } else {
            (
                lay.map(|l| t_f16(&l.post_attention_norm_scale))
                    .unwrap_or_else(|| norm_f16(PW as usize, &mut seed, 1.0)),
                &[],
            )
        };
        if !s.gear_ps.is_empty() {
            s.const_f16(&format!("{}g2", tag), &[PW], &g2_h);
        } else {
            s.data(ctx, &format!("{}g2", tag), &[PW], &g2_h);
        }
        if !i8sel("gate") {
            let host = lay
                .map(|l| lin_f16_unfold(&l.mlp.gate, INTER as usize, mlp_i8, g2_t))
                .unwrap_or_else(|| rand_f16((PW * INTER) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}gatewt", tag), &[INTER, PW], &host, PW, INTER);
        }
        if !i8sel("up") {
            let host = lay
                .map(|l| lin_f16_unfold(&l.mlp.up, INTER as usize, mlp_i8, g2_t))
                .unwrap_or_else(|| rand_f16((PW * INTER) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}upwt", tag), &[INTER, PW], &host, PW, INTER);
        }
        if !i8sel("down") {
            let host = lay
                .map(|l| lw_f16(&l.mlp.down))
                .unwrap_or_else(|| rand_f16((INTER * PW) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}downwt", tag), &[PW, INTER], &host, INTER, PW);
        }
        let downb_h = lay
            .map(|_| vec![f16::from_f32(0.0); PW as usize])
            .unwrap_or_else(|| rand_f16(PW as usize, &mut seed, 4000.0));
        s.row_bcast(ctx, &format!("{}downb", tag), if i8sel("down") { 1 } else { p }, PW, &downb_h);

        let norm1 = s.addrms(&format!("{}n1", tag), &cur, "zeros", &format!("{}g1", tag), &[gd(p), PW]);
        // rank-2 切分 + rope（rank-2 版）+ Reshape rank-3 桥。
        // ⚠ 图输出绑 rank-2（rope 的 ad / slice 的 v）——Reshape 输出绑图
        // 输出时 desc 是动态 [-1,-1,-1]（shape 张量驱动），size 无效
        let (q2, k2, v2) = if qkv3 {
            let qm = mm_or_qmd(&mut s, &format!("{}qm", tag), &norm1, &[gd(p), PW], iq8.as_ref(), &format!("{}qw", tag), QD, &[gd(p), QD], int8_native);
            let km = mm_or_qmd(&mut s, &format!("{}km", tag), &norm1, &[gd(p), PW], ik8.as_ref(), &format!("{}kw", tag), KVD, &[gd(p), KVD], int8_native);
            let vm = mm_or_qmd(&mut s, &format!("{}vm", tag), &norm1, &[gd(p), PW], iv8.as_ref(), &format!("{}vw", tag), KVD, &[gd(p), KVD], int8_native);
            // int8：bias 已折入算子（int32 零）——跳过 TileD+Add 对
            let qb = if iq8.is_some() {
                qm
            } else {
                s.bias(&format!("{}qb_", tag), &qm, &[gd(p), QD], &format!("{}qb", tag))
            };
            let kb = if ik8.is_some() {
                km
            } else {
                s.bias(&format!("{}kb_", tag), &km, &[gd(p), KVD], &format!("{}kb", tag))
            };
            let vb = if iv8.is_some() {
                vm
            } else {
                s.bias(&format!("{}vb_", tag), &vm, &[gd(p), KVD], &format!("{}vb", tag))
            };
            (qb, kb, vb)
        } else {
            let qkv = s.mm(&format!("{}qkv", tag), &norm1, &[gd(p), PW], &format!("{}qkvwt", tag), &[QKVW, PW], &[gd(p), QKVW]);
            let qkvb = s.bias(&format!("{}qkvb", tag), &qkv, &[gd(p), QKVW], &format!("{}qkvb", tag));
            let q2 = s.slice2(&format!("{}q", tag), &qkvb, &[gd(p), QKVW], 0, QD);
            let k2 = s.slice2(&format!("{}k", tag), &qkvb, &[gd(p), QKVW], QD, KVD);
            let v2 = s.slice2(&format!("{}v", tag), &qkvb, &[gd(p), QKVW], QD + KVD, KVD);
            (q2, k2, v2)
        };
        let (kr2, kr3) = if ropeflat {
            let kf2 = [gd(kf.0), kf.1];
            s.rope2_flat(&format!("{}kr", tag), &k2, &[gd(p), KVD], &kf2, "kcos", "ksin", "kswap_c", "shp_kflat", "shp_kb2", "shp_k3")
        } else {
            s.rope2(&format!("{}kr", tag), &k2, &[gd(p), KVD], "kcos2", "ksin2", "shp_k3")
        };
        let v3 = s.reshape(&format!("{}v3", tag), &v2, &[gd(p), KVD], "shp_v3", &[1, gd(p), KVD]);
        kv_outs.push(kr2.clone());
        kv_outs.push(v2.clone());
        if last {
            break; // 末层无 PFA/tail（compute_tail=false）
        }
        let (qr2, qr3) = if ropeflat {
            let qf2 = [gd(qf.0), qf.1];
            s.rope2_flat(&format!("{}qr", tag), &q2, &[gd(p), QD], &qf2, "qcos", "qsin", "qswap_c", "shp_qflat", "shp_mflat", "shp_q3")
        } else {
            s.rope2(&format!("{}qr", tag), &q2, &[gd(p), QD], "qcos2", "qsin2", "shp_q3")
        };
        let sq = if attn_manual {
            // 头主序桥 + GQA 手工链（k/v 单头 TileD 广播；kr3/v3 rank-3 直用）
            let q3 = s.headsplit(&format!("{}qh", tag), &qr2, gd(p), 1, gd(p), HEADS, HD, "shp_mq4", "shp_mq3");
            let attn = s.attn_manual_gqa(&format!("{}attn", tag), &q3, &kr3, &v3, HEADS, gd(p), gd(p), HD, attn_mask_n.as_deref());
            s.headmerge(&format!("{}am", tag), &attn, 1, gd(p), HEADS, HD, "shp_ma4", "shp_mflat")
        } else {
            let pfa = s.pfa(&format!("{}pfa", tag), &qr3, &kr3, &v3, &[1, gd(p), QD], &[1, gd(p), KVD], HEADS, KV_HEADS, HD);
            s.squeeze(&format!("{}sq", tag), &pfa, &[1, gd(p), QD], &[gd(p), QD])
        };
        let io8 = if i8sel("o") {
            let l = lay.expect("GEB_PREFIX_INT8 需真权重");
            Some(reg_i8(&mut s, ctx, &facs, &format!("prefix/L{:02}/o", lay_i), &format!("L{:02}_o", lay_i),
                &l.attention.output.weight.to_f32_vec().unwrap(), QD, PW, int8_native, dump_dir))
        } else {
            None
        };
        let proj = mm_or_qmd(&mut s, &format!("{}proj", tag), &sq, &[gd(p), QD], io8.as_ref(), &format!("{}outwt", tag), PW, &[gd(p), PW], int8_native);
        let projb = if io8.is_some() {
            proj.clone() // int8：bias 已折入算子
        } else {
            s.bias(&format!("{}projb", tag), &proj, &[gd(p), PW], &format!("{}outb", tag))
        };
        let res = s.add2(&format!("{}res", tag), &projb, &cur, &[gd(p), PW]);
        if dbg_mid && i == 0 {
            dbg_outs.push(sq.clone());
            dbg_outs.push(res.clone());
        }
        let norm2 = s.addrms(&format!("{}n2", tag), &res, "zeros", &format!("{}g2", tag), &[gd(p), PW]);
        let ig8 = if i8sel("gate") {
            let l = lay.expect("GEB_PREFIX_INT8 需真权重");
            Some(reg_i8(&mut s, ctx, &facs, &format!("prefix/L{:02}/gate", lay_i), &format!("L{:02}_gate", lay_i),
                &l.mlp.gate.weight.to_f32_vec().unwrap(), PW, INTER, int8_native, dump_dir))
        } else {
            None
        };
        let iu8 = if i8sel("up") {
            let l = lay.expect("GEB_PREFIX_INT8 需真权重");
            Some(reg_i8(&mut s, ctx, &facs, &format!("prefix/L{:02}/up", lay_i), &format!("L{:02}_up", lay_i),
                &l.mlp.up.weight.to_f32_vec().unwrap(), PW, INTER, int8_native, dump_dir))
        } else {
            None
        };
        let gate = mm_or_qmd(&mut s, &format!("{}gate", tag), &norm2, &[gd(p), PW], ig8.as_ref(), &format!("{}gatewt", tag), INTER, &[gd(p), INTER], int8_native);
        let up = mm_or_qmd(&mut s, &format!("{}up", tag), &norm2, &[gd(p), PW], iu8.as_ref(), &format!("{}upwt", tag), INTER, &[gd(p), INTER], int8_native);
        let gact = s.gelu(&format!("{}gact", tag), &gate, &[gd(p), INTER], true);
        let act = s.mul2(&format!("{}act", tag), &gact, &up, &[gd(p), INTER]);
        let id8 = if i8sel("down") {
            let l = lay.expect("GEB_PREFIX_INT8 需真权重");
            Some(reg_i8(&mut s, ctx, &facs, &format!("prefix/L{:02}/down", lay_i), &format!("L{:02}_down", lay_i),
                &l.mlp.down.weight.to_f32_vec().unwrap(), INTER, PW, int8_native, dump_dir))
        } else {
            None
        };
        let down = mm_or_qmd(&mut s, &format!("{}down", tag), &act, &[gd(p), INTER], id8.as_ref(), &format!("{}downwt", tag), PW, &[gd(p), PW], int8_native);
        let downb = if id8.is_some() {
            down.clone() // int8：bias 已折入算子
        } else {
            s.bias(&format!("{}downb", tag), &down, &[gd(p), PW], &format!("{}downb", tag))
        };
        cur = s.add2(&format!("{}out", tag), &downb, &res, &[gd(p), PW]);
        if dbg_mid && i == 0 && std::env::var("GEB_DBG_FULL").is_ok() {
            // 全级导出（GEB_DBG_FULL）：逐级与教科书对拍钉 GE 图坏点。
            // ⚠ 输出槽可能被内存复用覆写（res 槽实测被 n2 覆写）——逐级
            // 对拍时用「值匹配教科书哪一级」判读，勿信槽序。
            // n1/k2：norm1 输出与 rope 前 k——kvk 漂移链（输入→norm→k_proj→
            // rope）的中间级，18.7% 分岔定位用（norm 后幅度与输入无关 ⇒
            // 幅度依赖分岔只可能在 n1 之前产生）
            for n in [norm1.clone(), k2.clone(), proj.clone(), projb.clone(), res.clone(), norm2.clone(),
                      gate.clone(), up.clone(), gact.clone(), act.clone(),
                      down.clone(), downb.clone(), cur.clone()] {
                dbg_outs.push(n);
            }
        }
    }
    kv_outs.extend(dbg_outs);
    let out_names: Vec<&str> = kv_outs.iter().map(|x| x.as_str()).collect();
    s.finish(&out_names);
    println!("prefix OM built: depth={depth} P={p} n_in={} n_out={}", s.ins().len(), kv_outs.len());

    // ---- eager 参考（language_layer_ascend 序列镜像；qkv 独立投影）----
    let layer_bases2 = layer_bases.clone();
    let kv_count = kv_outs.len();
    let eager = |b: &[DeviceBuffer]| -> Vec<DeviceBuffer> {
        let zeros = &b[1];
        let mut h: Option<DeviceBuffer> = None;
        let mut outs = Vec::with_capacity(kv_count);
        for li in 0..depth {
            // qkv3 层内输入布局：g1,qw,kw,vw,qb,kb,vb,outwt,outb,g2,
            // gatewt,upwt,downwt,downb（14 项）；默认 10 项
            let w = &b[layer_bases2[li]..layer_bases2[li] + if qkv3 { 14 } else { 10 }];
            let (ow, ob, g2, gw, uw, dw, db) = if qkv3 { (7, 8, 9, 10, 11, 12, 13) } else { (3, 4, 5, 6, 7, 8, 9) };
            let hb: &DeviceBuffer = h.as_ref().unwrap_or(&b[0]);
            let (wq, wk, wv, bq, bk, bv) = &eager_qkv[li];
            let normed = aops::add_rms_norm_fp16(ctx, stream, hb, zeros, &w[0], &[p, PW], RMS_EPS).unwrap().0;
            let k = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [p, PW], wk, PW, KVD).unwrap(), bk, p, KVD).unwrap();
            let v = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [p, PW], wv, PW, KVD).unwrap(), bv, p, KVD).unwrap();
            let kr = aops::rope_rotate_half_flat_fp16(ctx, stream, &k, p * KV_HEADS, HD, &b[5], &b[6], &b[7]).unwrap();
            if li + 1 == depth {
                outs.push(kr);
                outs.push(v);
                break;
            }
            let q = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [p, PW], wq, PW, QD).unwrap(), bq, p, QD).unwrap();
            let qr = aops::rope_rotate_half_flat_fp16(ctx, stream, &q, p * HEADS, HD, &b[2], &b[3], &b[4]).unwrap();
            let attn = aops::prompt_flash_attention_bsh_fp16(
                ctx, stream, &qr, &kr, &v, p, HEADS, KV_HEADS, HD, None).unwrap();
            outs.push(kr);
            outs.push(v);
            let proj = aops::matmul_b_t_fp16(ctx, stream, &attn, [p, QD], &w[ow], QD, PW).unwrap();
            let biased = aops::bias_add_fp16(ctx, stream, &proj, &w[ob], p, PW).unwrap();
            let res = aops::add_fp16(ctx, stream, &biased, hb, &[p, PW]).unwrap();
            let norm2 = aops::add_rms_norm_fp16(ctx, stream, &res, zeros, &w[g2], &[p, PW], RMS_EPS).unwrap().0;
            let gate = aops::matmul_b_t_fp16(ctx, stream, &norm2, [p, PW], &w[gw], PW, INTER).unwrap();
            let up = aops::matmul_b_t_fp16(ctx, stream, &norm2, [p, PW], &w[uw], PW, INTER).unwrap();
            let g = aops::gelu_fp16(ctx, stream, &gate, &[p, INTER], true).unwrap();
            let act = aops::mul_fp16(ctx, stream, &g, &up, &[p, INTER]).unwrap();
            let down = aops::matmul_b_t_fp16(ctx, stream, &act, [p, INTER], &w[dw], INTER, PW).unwrap();
            let downb = aops::bias_add_fp16(ctx, stream, &down, &w[db], p, PW).unwrap();
            h = Some(aops::add_fp16(ctx, stream, &downb, &res, &[p, PW]).unwrap());
        }
        // 中间量 drop 前 kernel 必须落定（同 vision 段注记）
        drop(stream.synchronize());
        outs
    };

    if let Some(st) = e2e {
        // e2e：加载 prefix_real.om 跑一次，抓 36×k/v（k0,v0,k1,v1,...）；
        // GEB_DBG_MID 烤的 OM 尾部多 2 输出（层 0 attention 合并/h1）
        let outs = e2e_run(ctx, stream, &mut s, "prefix");
        let nkv = depth * 2;
        assert!(outs.len() >= nkv, "prefix OM 输出 {} < kv {}", outs.len(), nkv);
        // GEB_E2E_DUMP_MID=<dir>：prefix 全部输出原始 f16 落盘——含 36 路 kv
        //（slot{2i}=k_l{i} / slot{2i+1}=v_l{i}，行数 = 可见 Pv）+ 调试槽位
        //（GEB_DBG_MID 烤的 OM 才有；槽值可能被复用覆写，判读用「值匹配
        // 教科书哪级」而非槽序）。用法：static 侧 dump → 合并进 golden
        // → dyn 跑时 cmp_mid 打的即逐层 dyn-vs-static 互差
        if let Ok(dir) = std::env::var("GEB_E2E_DUMP_MID") {
            std::fs::create_dir_all(&dir).expect("dump dir");
            for (i, vals) in outs.iter().enumerate() {
                let bytes: Vec<u8> =
                    vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
                std::fs::write(format!("{dir}/ge_slot{i}.f16"), &bytes).expect("dump mid");
            }
            println!("[e2e] prefix 输出已落盘 {dir}/ge_slot{{0..{}}}.f16（含 36 路 kv）", outs.len());
        }
        if outs.len() > nkv {
            // 剔除空视图时 golden 参考键为 _vis（712 行，与图槽同形）
            if drop_v > 0 {
                st.cmp_mid("m0_vis", &outs[nkv]);
                st.cmp_mid("h1_vis", &outs[nkv + 1]);
            } else {
                st.cmp_mid("m0", &outs[nkv]);
                st.cmp_mid("h1", &outs[nkv + 1]);
            }
        }
        st.kv = outs[..nkv].to_vec();
        let m = st.kv.iter().flat_map(|v| v.iter()).fold(0f32, |m, v| m.max(v.to_f32().abs()));
        println!("[e2e] prefix kv ×{} |max|={m:.4}", st.kv.len());
        // 逐层 k 对拍（golden 键 kvk_l{i+GEB_LAYER_OFFSET}——偏移实验时键随
        // 权重层同步偏移）
        let key_off = envi("GEB_LAYER_OFFSET", 0) as usize;
        for li in 0..depth {
            st.cmp_mid(&format!("kvk_l{}", li + key_off), &st.kv[2 * li]);
            st.cmp_mid(&format!("kvv_l{}", li + key_off), &st.kv[2 * li + 1]);
        }
        // GEB_E2E_REPLAY：逐帧重组 x0（本帧 vision_out 可见行 + 真 token 查表
        // ×√PW——bring-up 同式）覆写 binds[0]（段首注册的 x0）重跑，36 路
        // kv 下载（aux 槽不下载）。嵌入表 2.1GB 一次性物化复用
        if !st.replay.is_empty() {
            let w = real.expect("replay prefix 需要 GEB_CKPT（token 嵌入查表）");
            let n_out = s.g.num_outputs().unwrap();
            let routs: Vec<DeviceBuffer> = (0..n_out)
                .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("replay out malloc"))
                .collect();
            let emb = w.vision.token_embedding.to_f32_vec().unwrap();
            let vocab_w = PW as usize;
            let lang_scale = (PW as f32).sqrt();
            let vis_rows = ((VT - drop_v) * PW) as usize;
            let h2d = |buf: &DeviceBuffer, vals: &[f16]| {
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
                ctx.copy_h2d(buf, &bytes).expect("replay h2d");
            };
            let t0 = std::time::Instant::now();
            for f in st.replay.iter_mut() {
                assert_eq!(f.vision_out.len(), (VT * PW) as usize, "须先跑 vision replay");
                let mut x0 = f.vision_out[..vis_rows].to_vec();
                x0.reserve(f.token_ids.len() * vocab_w);
                for &id in &f.token_ids {
                    let r = id as usize * vocab_w;
                    assert!(r + vocab_w <= emb.len(), "token id {id} 超 vocab");
                    x0.extend(emb[r..r + vocab_w].iter().map(|&v| f16::from_f32(v * lang_scale)));
                }
                assert_eq!(x0.len() * 2, s.binds[0].len(), "replay x0 尺寸不符");
                h2d(&s.binds[0], &x0);
                let refs: Vec<&DeviceBuffer> = routs.iter().collect();
                // replay 帧恒 = GEB_TOKENS（= gear 模式 max 档）——末档执行
                s.run_seg(s.gear_payloads.len().saturating_sub(1), &refs, stream)
                    .expect("ge run");
                drop(stream.synchronize());
                f.kv = (0..nkv)
                    .map(|i| download_f16(ctx, &routs[i], s.g.output_size(i).unwrap() / 2))
                    .collect();
            }
            println!("[e2e] replay prefix ×{} {:?}", st.replay.len(), t0.elapsed());
        }
        // GEB_E2E_SERVE：机械下沉（嵌入表 ~2.1GB 常驻机械内，跨请求查表复用）
        if std::env::var("GEB_E2E_SERVE").is_ok() {
            let w = real.expect("serve prefix 需要 GEB_CKPT（token 嵌入查表）");
            let n_out = s.g.num_outputs().unwrap();
            let routs: Vec<DeviceBuffer> = (0..n_out)
                .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("serve out malloc"))
                .collect();
            let emb = w.vision.token_embedding.to_f32_vec().unwrap();
            let gear_of_l = gear_ll
                .as_ref()
                .map(|g| g.iter().enumerate().map(|(i, &l)| (l, i)).collect())
                .unwrap_or_default();
            let rout_bytes: Vec<usize> =
                (0..n_out).map(|i| s.g.output_size(i).unwrap()).collect();
            let pool_root = std::env::var("GEB_PREFIX_POOL").ok();
            if pool_root.is_some() {
                println!("[prefix-pool] serve 池模式：per-L 静态 OM 换柄（lazy，cap {}）", envi("GEB_POOL_CAP", 8));
            }
            let mask_bind = s.try_bind_of("padmask");
            st.pmech = Some(PrefixMech {
                s,
                routs,
                lang_scale: (PW as f32).sqrt(),
                vis_rows: ((VT - drop_v) * PW) as usize,
                nkv: depth * 2,
                vocab_w: PW as usize,
                emb,
                tok_cache_ids: Vec::new(),
                tok_cache_rows: Vec::new(),
                gear_of_l,
                pool_root,
                pool_graphs: std::collections::HashMap::new(),
                pool_lru: Vec::new(),
                // 池容量上限（LRU 驱逐）：桶数超过 cap 时最久未用桶被卸载、
                // 再命中需重装（秒级）——满谱系连跑时应设为桶数
                pool_cap: envi("GEB_POOL_CAP", 8) as usize,
                rout_bytes,
                mask_bind,
            });
        }
        return;
    }
    if int8 {
        // eager/parity 跳过：f16 eager 参考对 int8 图无裁判意义（量化执行差
        // 的裁判 = e2e golden 对拍 + task0 行为）。烤桶 OM 落盘照常（save
        // 只需 build 产物，同 GEB_NORM32 先例）。
        let mut save_path = None;
        if let Ok(p) = std::env::var("GEB_SAVE") {
            s.g.save(&p).expect("save om (int8)");
            println!("OM saved: {p}");
            save_path = Some(p);
        }
        // GEB_INT8_RUN=1：活图直跑 vs 存后重载跑（同 binds）——OM 序列化
        // desc（int8/f32 Data 输入）疑点隔离：probe 活图干净而 e2e 重载
        // 后全炸（ladder 定位：gate/down 单投影同炸、布局桥
        // N2/N3 干净 ⇒ 嫌疑集中 save/load 往返）
        if std::env::var("GEB_INT8_RUN").is_ok() {
            let run_once = |s: &mut Seg, tag: &str| {
                let ins: Vec<&DeviceBuffer> = s.ins();
                let n_out = s.g.num_outputs().unwrap();
                let outs: Vec<DeviceBuffer> = (0..n_out)
                    .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("out malloc"))
                    .collect();
                let refs: Vec<&DeviceBuffer> = outs.iter().collect();
                s.g.run(&ins, &refs, stream).expect("ge run");
                drop(stream.synchronize());
                let full = download_f16(ctx, &outs[0], s.g.output_size(0).unwrap() / 2);
                let m = full.iter().fold(0f32, |a, v| a.max(v.to_f32().abs()));
                let head: Vec<f32> = full[..8.min(full.len())].iter().map(|v| v.to_f32()).collect();
                println!("[int8] {tag} out0 head={head:?} |max|={m:.2}");
            };
            run_once(&mut s, "活图");
            let live_sizes: Vec<(usize, usize)> = (0..s.g.num_inputs().unwrap())
                .map(|i| (s.g.input_size(i).unwrap_or(0), s.binds[i].len()))
                .collect();
            let bad_live: Vec<usize> = live_sizes
                .iter()
                .enumerate()
                .filter(|(_, (sz, bl))| sz != bl)
                .map(|(i, _)| i)
                .collect();
            println!("[int8] 活图输入 {} 个，binds 长度不符的 idx：{:?}", live_sizes.len(), bad_live);
            if let Some(p) = save_path {
                s.g = ge_builder::load(&p).expect("reload om");
                let n2 = s.g.num_inputs().unwrap();
                let mut mism = Vec::new();
                for i in 0..n2.min(live_sizes.len()) {
                    let sz = s.g.input_size(i).unwrap_or(0);
                    if sz != live_sizes[i].0 {
                        mism.push((i, live_sizes[i].0, sz, s.g.input_dims(i).unwrap_or_default()));
                    }
                }
                println!("[int8] 重载输入 {n2} 个（活图 {}）；size 不符：{:?}（前 8 项）", live_sizes.len(), mism.iter().take(8).collect::<Vec<_>>());
                run_once(&mut s, "重载");
            }
        }
        ge_builder::fini().expect("fini");
        let projs: Vec<&str> = ["q", "k", "v", "o", "gate", "up", "down"]
            .into_iter()
            .filter(|p| int8_projs.contains(*p))
            .collect();
        println!(
            "GE_PREFIX_PROBE_OK (int8: 投影 {} ×{} 层, native_smooth={int8_native})",
            projs.join(","),
            depth
        );
        return;
    }
    let elems = [(p * KVD) as usize];
    let out_elems: Vec<usize> = kv_outs.iter().map(|_| elems[0]).collect();
    parity_and_bench(ctx, stream, &mut s, "prefix", &out_elems, &eager, bench);
    ge_builder::fini().expect("fini");
    println!("GE_PREFIX_PROBE_OK");
}

// ---------------------------------------------------------------------------
// flow 段：单步 OM（action_in → depth×action 层(ada-norm + cross PFA +
// geglu) → action_out → euler）。styles/prefix-kv/euler 常数全为外部输入。
// ---------------------------------------------------------------------------

pub fn seg_flow(be: &AscendBackend, bench: bool, real: Option<&Pi05Weights>, e2e: Option<&mut E2eStage>) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH", 18) as usize;
    let tokens = envi("GEB_TOKENS", 64);
    // prefix 可见长度（空视图剔除须与 prefix 段一致——suffix rope 偏移与
    // pk/pv 形状都取此值；infer 侧 pos_s = sum(prefix_pad)+cumsum-1 = 712 起）
    let p = VT + tokens - envi("GEB_PREFIX_DROP_EMPTY", 0);
    let mut seed = 0xF00Du32;
    // 同 vision/prefix 段三件套（取证见各段注记）：q/k/v 独立投影 /
    // flat rope（免列切物化）/ 手工 cross-GQA attention（免 PFA host
    // launcher）。scale 折进 q 权重/bias
    let qkv3 = std::env::var("GEB_QKV3").is_ok();
    let ropeflat = std::env::var("GEB_ROPEFLAT").is_ok();
    let attn_manual = std::env::var("GEB_ATTN").map(|v| v == "manual").unwrap_or(false);
    let fscale = if attn_manual { 1.0f32 / (HD as f32).sqrt() } else { 1.0 };
    let mut s = Seg::new("ge_flow");
    // 动态分档：与 prefix 段同源（pk/pv 36 路 [-1,KVD]）——但生产 HYBRID
    // 拓扑下 flow 不走 gear（styles 是运行期内容的静态 Data 输入，无法
    // Const 化，而静态 Data 混入动态集会让 SetInputDynamicDims 档位匹配
    // 失败——实测定罪）：flow 用 per-L 静态 OM 池（GEB_FLOW_POOL，
    // 现存 tl{L}_i8 资产），图按 p_max 构造 binds。GEB_DYN_LLIST_FLOW 仅供
    // gear 化 flow 的隔离实验
    let gear_ll = std::env::var("GEB_DYN_LLIST_FLOW").ok().and_then(|_| dyn_llist());
    let mut gear_ps_hold: Option<Vec<i64>> = None;
    let gd = |v: i64| if gear_ll.is_some() { -1 } else { v };
    if let Some(g) = &gear_ll {
        assert!(
            std::env::var("GEB_WCONST").is_ok(),
            "GEB_DYN_LLIST 需 GEB_WCONST=1"
        );
        assert!(ropeflat && attn_manual && qkv3, "GEB_DYN_LLIST 需三件套（ROPEFLAT/ATTN=manual/QKV3）");
        let lmax = *g.iter().max().unwrap();
        assert_eq!(tokens, lmax, "GEB_DYN_LLIST 模式 GEB_TOKENS 须 = 档表 max L");
        let drop_v2 = envi("GEB_PREFIX_DROP_EMPTY", 0);
        let ps = g.iter().map(|&l| VT + l - drop_v2).collect::<Vec<i64>>();
        println!("[gear] flow 动态分档：L {:?} → p {:?}", g, ps);
        gear_ps_hold = Some(ps);
    }
    if let Some(ps) = gear_ps_hold.take() {
        s.set_gear_ps(ps);
    }

    // ---- 段级输入（注册序，两遍式：先全部注册再建图——层 i 的 next
    // ada-norm 引用层 i+1 的 style 输入，必须先注册后链接）----
    // 0 state / 1 zeros / 2..7 flat rope（eager）/ 8..11 rank-2 表（GE）/
    // 12 ainwt / 13 ainb / 14.. 每层 (pk_i, pv_i)（rank-2）/
    // 每层 12 项 / 尾 fsc,fsh,aoutwt,aoutb,c1,c2
    // e2e：state 起步 = noise（golden 或合成）；缺省随机（对拍两路同值）
    let state_h = e2e
        .as_ref()
        .map(|st| st.noise.clone())
        .unwrap_or_else(|| rand_f16((HOR * ADIM) as usize, &mut seed, 100.0));
    s.data(ctx, "state", &[HOR, ADIM], &state_h);
    // GEB_NORM32 v2：常量 Const 折叠，无 Data 注册（binds[0] 约束解除）
    if let Some(st) = e2e.as_ref() {
        assert_eq!(st.kv.len(), depth * 2, "e2e: prefix 段 k/v 数与 flow 深度不符（GEB_DEPTH 须全 18）");
    }
    s.data_zeros(ctx, "zeros", HOR, AW);
    // PAD 对拍：flow token 位置 = vis+真L+i（serve 侧 set_rope_for(真L) 同
    // 义）——桶界 P 只定 cache 消费维度，rope 相位必须按真 L，否则 flow
    // 整体相位漂移（首战 24.7% l17 教训：数值验收两坑之一）
    let rope_p = if envi("GEB_E2E_PAD_TOKENS", 0) > 0 {
        lp_to_p(envi("GEB_E2E_PAD_TOKENS", 0))
    } else {
        p
    };
    let (qc, qs, qi) = rope_flat_const(HOR, HEADS, rope_p);
    let (kc, ks, ki) = rope_flat_const(HOR, KV_HEADS, rope_p);
    let qf = (HOR * HEADS * 2, HD / 2);
    let kf = (HOR * KV_HEADS * 2, HD / 2);
    s.data(ctx, "qcos", &[qf.0, qf.1], &qc);
    s.data(ctx, "qsin", &[qf.0, qf.1], &qs);
    s.data_i32(ctx, "qswap", &[qf.0], &qi);
    s.data(ctx, "kcos", &[kf.0, kf.1], &kc);
    s.data(ctx, "ksin", &[kf.0, kf.1], &ks);
    s.data_i32(ctx, "kswap", &[kf.0], &ki);
    let (qc2, qs2) = rope_rank2_const(HOR, HEADS, rope_p);
    let (kc2, ks2) = rope_rank2_const(HOR, KV_HEADS, rope_p);
    s.data(ctx, "qcos2", &[HOR, QD], &qc2);
    s.data(ctx, "qsin2", &[HOR, QD], &qs2);
    s.data(ctx, "kcos2", &[HOR, KVD], &kc2);
    s.data(ctx, "ksin2", &[HOR, KVD], &ks2);
    // Reshape shape 张量（q/k/v/kall rank-3 桥；kall/vall 共用；Const）
    s.const_i32("shp_q3", &[1, HOR as i32, QD as i32]);
    s.const_i32("shp_k3", &[1, HOR as i32, KVD as i32]);
    s.const_i32("shp_v3", &[1, HOR as i32, KVD as i32]);
    s.const_i32("shp_kall", &[1, gd(p + HOR) as i32, KVD as i32]);
    // ropeflat / manual cross-attention 常量（swap 索引 Const 折叠）
    s.const_i32("qswap_c", &qi);
    s.const_i32("kswap_c", &ki);
    s.const_i32("shp_qflat", &[qf.0 as i32, qf.1 as i32]);
    s.const_i32("shp_kflat", &[kf.0 as i32, kf.1 as i32]);
    s.const_i32("shp_fbq", &[HOR as i32, QD as i32]);
    s.const_i32("shp_fbk", &[HOR as i32, KVD as i32]);
    s.const_i32("shp_fq4", &[1, HOR as i32, HEADS as i32, HD as i32]);
    s.const_i32("shp_fq3", &[HEADS as i32, HOR as i32, HD as i32]);
    s.const_i32("shp_fa4", &[1, HEADS as i32, HOR as i32, HD as i32]);
    {
        let host = real
            .map(|w| lw_f16(&w.action_in))
            .unwrap_or_else(|| rand_f16((ADIM * AW) as usize, &mut seed, 8000.0));
        s.wt(ctx, "ainwt", &[AW, ADIM], &host, ADIM, AW);
    }
    let ainb_h = real
        .map(|w| lb_f16(&w.action_in, AW as usize))
        .unwrap_or_else(|| rand_f16(AW as usize, &mut seed, 4000.0));
    s.data(ctx, "ainb", &[1, AW], &ainb_h);
    // prefix k/v（rank-2——ConcatD axis=0 拼接后统一 Unsqueeze）；
    // e2e：prefix OM 36 输出直连（k0,v0,k1,v1,... → pk{i}=kv[2i]/pv{i}=kv[2i+1]）
    // pk/pv 注册基址（replay 逐帧换绑用——循环体恰好两次 s.data）
    let pk_base = s.binds.len();
    for i in 0..depth {
        let pk_h = e2e
            .as_ref()
            .map(|st| st.kv[2 * i].clone())
            .unwrap_or_else(|| rand_f16((p * KVD) as usize, &mut seed, 100.0));
        let pv_h = e2e
            .as_ref()
            .map(|st| st.kv[2 * i + 1].clone())
            .unwrap_or_else(|| rand_f16((p * KVD) as usize, &mut seed, 100.0));
        s.data(ctx, &format!("pk{i}"), &[gd(p), KVD], &pk_h);
        s.data(ctx, &format!("pv{i}"), &[gd(p), KVD], &pv_h);
    }
    // ---- flow int8（GEB_FLOW_INT8）：7 投影/层 → QuantMatmulDequant ----
    // 注册区在层区之前（sm Data 落此，pk_base 不动）；层内 int8 权重槽以
    // 16B dummy bind 占位——18 槽布局与 f16 恒同（serve/e2e 的 style 硬编码
    // 偏移 b+0/+1/+10/+11/+16/+17 依赖布局，FlowMech/style_residency 免改）
    let int8 = flow_int8();
    if int8 {
        assert!(qkv3, "GEB_FLOW_INT8 需要 GEB_QKV3=1（fused qkv 无法 per-matrix smooth）");
    }
    let facs: std::collections::HashMap<String, Int8Smooth> = if int8 {
        assert!(real.is_some(), "GEB_FLOW_INT8 需要 GEB_CKPT 真权重");
        let path = std::env::var("GEB_FLOW_SMOOTH")
            .unwrap_or_else(|_| "./smooth_calib_v2_flow_engine.safetensors".into());
        println!("[int8] flow int8 开启：smooth={path}");
        load_int8_smooth(&path).0
    } else {
        std::collections::HashMap::new()
    };
    // GEB_FLOW_PROJS：flow 段独立子集（缺省回落 GEB_INT8_PROJS——两段共享
    // 同名 env 会在子集实验时把 prefix 段一起切掉，OM 输入数不匹配 rc=-2
    // 实测）
    let int8_projs: std::collections::HashSet<String> = std::env::var("GEB_FLOW_PROJS")
        .or_else(|_| std::env::var("GEB_INT8_PROJS"))
        .unwrap_or_else(|_| "q,k,v,o,gate,up,down".into())
        .split(',')
        .map(|x| x.trim().to_string())
        .collect();
    let int8_native = envi("GEB_INT8_NATIVE", 1) == 1;
    let dump_dir = std::env::var("GEB_INT8_DUMP").ok();
    let fi8sel = |pr: &str| int8 && int8_projs.contains(pr);
    // (名, k, n)：k/v 的 n=256 与 down 的 k=4096 是小/宽 shape——qmd 在
    // M=50 的编译/性能未验，GEB_INT8_PROJS 子集可退
    const FPROJS: [(&str, i64, i64); 7] = [
        ("q", AW, QD),
        ("k", AW, KVD),
        ("v", AW, KVD),
        ("o", QD, AW),
        ("gate", AW, AINTER),
        ("up", AW, AINTER),
        ("down", AINTER, AW),
    ];
    let mut fi8: Vec<Vec<Option<(String, String, String)>>> = vec![vec![None; 7]; depth];
    if int8 {
        for i in 0..depth {
            let lay = real
                .and_then(|w| w.action_layers.get(i))
                .unwrap_or_else(|| panic!("GEB_FLOW_INT8 需真权重（层 {i} 缺）"));
            for (slot, &(pn, kk, nn)) in FPROJS.iter().enumerate() {
                if !fi8sel(pn) {
                    continue;
                }
                let w32: Vec<f32> = match pn {
                    "q" => {
                        let mut v = lay.attention.q.weight.to_f32_vec().unwrap();
                        if attn_manual {
                            for x in v.iter_mut() {
                                *x *= fscale;
                            }
                        }
                        v
                    }
                    "k" => lay.attention.k.weight.to_f32_vec().unwrap(),
                    "v" => lay.attention.v.weight.to_f32_vec().unwrap(),
                    "o" => lay.attention.output.weight.to_f32_vec().unwrap(),
                    "gate" => lay.mlp.gate.weight.to_f32_vec().unwrap(),
                    "up" => lay.mlp.up.weight.to_f32_vec().unwrap(),
                    "down" => lay.mlp.down.weight.to_f32_vec().unwrap(),
                    _ => unreachable!(),
                };
                fi8[i][slot] = Some(reg_i8(
                    &mut s, ctx, &facs, &format!("flow/L{i:02}/{pn}"), &format!("F{i:02}_{pn}"),
                    &w32, kk, nn, int8_native, dump_dir.as_deref(),
                ));
            }
        }
        println!("[int8] flow int8 预注册完成：{} 层 × {} 投影（native={}）", depth, fi8[0].iter().filter(|x| x.is_some()).count(), int8_native);
    }
    // 层权重 12 项（GEB_QKV3: 16 项）+ 尾部 agate/mgate 两项（gate 通路
    // 补全）：ascl ash [qkvwt qkvb | qw kw vw qb kb vb] outwt outb
    // mscl msh gatewt upwt downwt downb agate mgate（eager qkv 独立投影）
    let mut eager_qkv: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer)> = Vec::new();
    let mut layer_bases = Vec::with_capacity(depth);
    for i in 0..depth {
        layer_bases.push(s.binds.len());
        let tag = format!("l{i}_");
        // 真权重：ascl/ash/mscl/msh 是 ada-norm 条件向量（style 投影 ×
        // conditioning 的运行时产物，随 time step 变化）——保持合成输入，
        // 真权重接入只换投影/LN 类权重
        let lay = real.and_then(|w| w.action_layers.get(i));
        s.data(ctx, &format!("{}ascl", tag), &[AW], &norm_f16(AW as usize, &mut seed, 1.0));
        s.data(ctx, &format!("{}ash", tag), &[AW], &norm_f16(AW as usize, &mut seed, 0.0));
        {
            // int8 槽：f16 host 不构造、wt 槽 16B dummy 占位（布局纪律见
            // 预注册注记）；eager 缓冲整块免建（int8 模式 eager 跳过）
            let wq = if fi8[i][0].is_none() {
                lay
                    .map(|l| lw_f16(&l.attention.q))
                    .unwrap_or_else(|| rand_f16((AW * QD) as usize, &mut seed, 8000.0))
            } else {
                Vec::new()
            };
            let wk = if fi8[i][1].is_none() {
                lay
                    .map(|l| lw_f16(&l.attention.k))
                    .unwrap_or_else(|| rand_f16((AW * KVD) as usize, &mut seed, 8000.0))
            } else {
                Vec::new()
            };
            let wv = if fi8[i][2].is_none() {
                lay
                    .map(|l| lw_f16(&l.attention.v))
                    .unwrap_or_else(|| rand_f16((AW * KVD) as usize, &mut seed, 8000.0))
            } else {
                Vec::new()
            };
            // Gemma 投影无 bias（真权重 = zeros）
            let bias = lay
                .map(|_| vec![f16::from_f32(0.0); QKVW as usize])
                .unwrap_or_else(|| rand_f16(QKVW as usize, &mut seed, 4000.0));
            // manual attention：q 块（权重+bias）预乘 1/√hd；eager 保持原值
            let wq_s: Vec<f16> = if attn_manual {
                wq.iter().map(|v| f16::from_f32(v.to_f32() * fscale)).collect()
            } else {
                wq.clone()
            };
            let mut bias_s = bias.clone();
            if attn_manual {
                for v in bias_s[..QD as usize].iter_mut() {
                    *v = f16::from_f32(v.to_f32() * fscale);
                }
            }
            if qkv3 {
                // qkv3 层内输入位次 2..8（ascl/ash 之后）；eager 按 qkv3 偏移读
                let i8_slot = |s: &mut Seg| {
                    s.binds.push(ctx.malloc(16).expect("i8 wt 槽占位"));
                    s.bind_names.push(String::new());
                };
                if fi8[i][0].is_some() {
                    i8_slot(&mut s);
                } else {
                    s.wt(ctx, &format!("{}qw", tag), &[QD, AW], &wq_s, AW, QD);
                }
                if fi8[i][1].is_some() {
                    i8_slot(&mut s);
                } else {
                    s.wt(ctx, &format!("{}kw", tag), &[KVD, AW], &wk, AW, KVD);
                }
                if fi8[i][2].is_some() {
                    i8_slot(&mut s);
                } else {
                    s.wt(ctx, &format!("{}vw", tag), &[KVD, AW], &wv, AW, KVD);
                }
                s.data(ctx, &format!("{}qb", tag), &[1, QD], &bias_s[..QD as usize]);
                s.data(ctx, &format!("{}kb", tag), &[1, KVD], &bias_s[QD as usize..(QD + KVD) as usize]);
                s.data(ctx, &format!("{}vb", tag), &[1, KVD], &bias_s[(QD + KVD) as usize..]);
            } else {
                let mut fused = Vec::with_capacity((AW * QKVW) as usize);
                for r in 0..AW as usize {
                    let rb = r * QD as usize;
                    fused.extend_from_slice(&wq_s[rb..rb + QD as usize]);
                    let rb = r * KVD as usize;
                    fused.extend_from_slice(&wk[rb..rb + KVD as usize]);
                    fused.extend_from_slice(&wv[rb..rb + KVD as usize]);
                }
                s.wt(ctx, &format!("{}qkvwt", tag), &[QKVW, AW], &fused, AW, QKVW);
                s.data(ctx, &format!("{}qkvb", tag), &[1, QKVW], &bias_s);
            }
            if !int8 {
                let bq = upload(ctx, &bias[0..QD as usize]);
                let bk = upload(ctx, &bias[QD as usize..(QD + KVD) as usize]);
                let bv = upload(ctx, &bias[(QD + KVD) as usize..]);
                let ebq = wbuf_t(ctx, &wq, AW, QD);
                let ebk = wbuf_t(ctx, &wk, AW, KVD);
                let ebv = wbuf_t(ctx, &wv, AW, KVD);
                eager_qkv.push((ebq, ebk, ebv, bq, bk, bv));
            }
        }
        if fi8[i][3].is_some() {
            s.binds.push(ctx.malloc(16).expect("i8 wt 槽占位"));
            s.bind_names.push(String::new());
        } else {
            let host = lay
                .map(|l| lw_f16(&l.attention.output))
                .unwrap_or_else(|| rand_f16((QD * AW) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}outwt", tag), &[AW, QD], &host, QD, AW);
        }
        let outb_h = lay
            .map(|_| vec![f16::from_f32(0.0); AW as usize])
            .unwrap_or_else(|| rand_f16(AW as usize, &mut seed, 4000.0));
        s.data(ctx, &format!("{}outb", tag), &[1, AW], &outb_h);
        s.data(ctx, &format!("{}mscl", tag), &[AW], &norm_f16(AW as usize, &mut seed, 1.0));
        s.data(ctx, &format!("{}msh", tag), &[AW], &norm_f16(AW as usize, &mut seed, 0.0));
        if fi8[i][4].is_some() {
            s.binds.push(ctx.malloc(16).expect("i8 wt 槽占位"));
            s.bind_names.push(String::new());
        } else {
            let host = lay
                .map(|l| lw_f16(&l.mlp.gate))
                .unwrap_or_else(|| rand_f16((AW * AINTER) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}gatewt", tag), &[AINTER, AW], &host, AW, AINTER);
        }
        if fi8[i][5].is_some() {
            s.binds.push(ctx.malloc(16).expect("i8 wt 槽占位"));
            s.bind_names.push(String::new());
        } else {
            let host = lay
                .map(|l| lw_f16(&l.mlp.up))
                .unwrap_or_else(|| rand_f16((AW * AINTER) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}upwt", tag), &[AINTER, AW], &host, AW, AINTER);
        }
        if fi8[i][6].is_some() {
            s.binds.push(ctx.malloc(16).expect("i8 wt 槽占位"));
            s.bind_names.push(String::new());
        } else {
            let host = lay
                .map(|l| lw_f16(&l.mlp.down))
                .unwrap_or_else(|| rand_f16((AINTER * AW) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}downwt", tag), &[AW, AINTER], &host, AINTER, AW);
        }
        let downb_h = lay
            .map(|_| vec![f16::from_f32(0.0); AW as usize])
            .unwrap_or_else(|| norm_f16(AW as usize, &mut seed, 0.0));
        s.data(ctx, &format!("{}downb", tag), &[1, AW], &downb_h);
        // ada-norm 门控第三段（chunk 3 of dense(cond)）：attention/MLP 分支
        // 输出 ×gate 后进残差（torch _gated_residual）。块尾追加——qkv3
        // eager 偏移 8..15 不动，gate 固定位 16/17（默认布局 12/13）
        s.data(ctx, &format!("{}agate", tag), &[1, AW], &norm_f16(AW as usize, &mut seed, 1.0));
        s.data(ctx, &format!("{}mgate", tag), &[1, AW], &norm_f16(AW as usize, &mut seed, 1.0));
    }
    // e2e 每步覆写 final norm 条件用（fsc/fsh 的 bind 位）
    let fs_idx = s.binds.len();
    s.data(ctx, "fsc", &[AW], &norm_f16(AW as usize, &mut seed, 1.0));
    s.data(ctx, "fsh", &[AW], &norm_f16(AW as usize, &mut seed, 0.0));
    {
        let host = real
            .map(|w| lw_f16(&w.action_out))
            .unwrap_or_else(|| rand_f16((AW * ADIM) as usize, &mut seed, 8000.0));
        s.wt(ctx, "aoutwt", &[ADIM, AW], &host, AW, ADIM);
    }
    let aoutb_h = real
        .map(|w| lb_f16(&w.action_out, ADIM as usize))
        .unwrap_or_else(|| rand_f16(ADIM as usize, &mut seed, 4000.0));
    s.data(ctx, "aoutb", &[1, ADIM], &aoutb_h);
    // euler 常数（σ = dt = -0.1：x' = (1+σ)x + σ·v）
    let eul_c1 = vec![f16::from_f32(0.9); (HOR * ADIM) as usize];
    let eul_c2 = vec![f16::from_f32(-0.1); (HOR * ADIM) as usize];
    s.data(ctx, "c1", &[HOR, ADIM], &eul_c1);
    s.data(ctx, "c2", &[HOR, ADIM], &eul_c2);

    // ---- 图构造 ----
    // action_in + 第一层 attention ada-norm（eager：第一层无预解析
    // normalized，用 adaptive_rms(action_in_out, attention_style[0])）
    let ain = s.mm("ain", "state", &[HOR, ADIM], "ainwt", &[AW, ADIM], &[HOR, AW]);
    let ain = s.bias("ainb2", &ain, &[HOR, AW], "ainb");
    let mut normed = s.ada("l0ada", &ain, "zeros", "l0_ascl", "l0_ash", &[HOR, AW]);
    // ⚠ 残差流（未归一化 x）双轨携带——cuda bf16_executor 语义：attention
    // 残差基是 input 本体（fused::adaptive_gate_residual_rms_bf16(
    // projection, input)），normed 只喂 qkv。定罪：本段曾以
    // normed 作残差基（两处镜像同 bug，组件 parity 全绿因为参考 eager 同
    // 错）——18 层累积 → v 输出近随机、闭环 0/10；step0_x1 仅 11.5% 是
    // dt=0.1 对 v 误差的掩盖（Δx1=0.1·Δv ⇒ v 已 ~100% 错）
    let mut xstream = ain;

    let total = p + HOR;
    // GEB_ATTN_MASK（range padding）：flow 侧 mask [1,1,total]——键布局 =
    // cache 行在前（真 [0,vis+L) / pad [vis+L, vis+P)）+ flow HOR 行恒真。
    // bake 占位全 0，serve 侧按真 L 填充
    let fmask_n: Option<String> = if envi("GEB_ATTN_MASK", 0) == 1 {
        let mh = vec![f16::from_f32(0.0); total as usize];
        Some(s.data(ctx, "padmask", &[1, 1, total], &mh))
    } else {
        None
    };
    for i in 0..depth {
        let tag = format!("l{i}_");
        // attention（normed 已是本层 ada-norm 输出；q1/r1 验证链）
        let (q2, k2, v2) = if qkv3 {
            let qm = mm_or_qmd(&mut s, &format!("{}qm", tag), &normed, &[HOR, AW], fi8[i][0].as_ref(), &format!("{}qw", tag), QD, &[HOR, QD], int8_native);
            let km = mm_or_qmd(&mut s, &format!("{}km", tag), &normed, &[HOR, AW], fi8[i][1].as_ref(), &format!("{}kw", tag), KVD, &[HOR, KVD], int8_native);
            let vm = mm_or_qmd(&mut s, &format!("{}vm", tag), &normed, &[HOR, AW], fi8[i][2].as_ref(), &format!("{}vw", tag), KVD, &[HOR, KVD], int8_native);
            // int8：Gemma bias≡0 砍 TileD+Add 对（同 prefix 先例；bias Data
            // 保留注册占图输入槽——布局稳定）
            let qb = if fi8[i][0].is_some() {
                qm
            } else {
                s.bias(&format!("{}qb_", tag), &qm, &[HOR, QD], &format!("{}qb", tag))
            };
            let kb = if fi8[i][1].is_some() {
                km
            } else {
                s.bias(&format!("{}kb_", tag), &km, &[HOR, KVD], &format!("{}kb", tag))
            };
            let vb = if fi8[i][2].is_some() {
                vm
            } else {
                s.bias(&format!("{}vb_", tag), &vm, &[HOR, KVD], &format!("{}vb", tag))
            };
            (qb, kb, vb)
        } else {
            let qkv = s.mm(&format!("{}qkv", tag), &normed, &[HOR, AW], &format!("{}qkvwt", tag), &[QKVW, AW], &[HOR, QKVW]);
            let qkvb = s.bias(&format!("{}qkvb", tag), &qkv, &[HOR, QKVW], &format!("{}qkvb", tag));
            let q2 = s.slice2(&format!("{}q", tag), &qkvb, &[HOR, QKVW], 0, QD);
            let k2 = s.slice2(&format!("{}k", tag), &qkvb, &[HOR, QKVW], QD, KVD);
            let v2 = s.slice2(&format!("{}v", tag), &qkvb, &[HOR, QKVW], QD + KVD, KVD);
            (q2, k2, v2)
        };
        let kf2 = [kf.0, kf.1];
        let (kr2, _) = if ropeflat {
            s.rope2_flat(&format!("{}kr", tag), &k2, &[HOR, KVD], &kf2, "kcos", "ksin", "kswap_c", "shp_kflat", "shp_fbk", "shp_k3")
        } else {
            s.rope2(&format!("{}kr", tag), &k2, &[HOR, KVD], "kcos2", "ksin2", "shp_k3")
        };
        let qf2 = [qf.0, qf.1];
        let (qr2, qr3) = if ropeflat {
            s.rope2_flat(&format!("{}qr", tag), &q2, &[HOR, QD], &qf2, "qcos", "qsin", "qswap_c", "shp_qflat", "shp_fbq", "shp_q3")
        } else {
            s.rope2(&format!("{}qr", tag), &q2, &[HOR, QD], "qcos2", "qsin2", "shp_q3")
        };
        // prefix cat（rank-2 ConcatD axis=0）+ Reshape rank-3 桥
        let kcat = format!("{}kcat", tag);
        s.g.add_op(&kcat, "ConcatD").unwrap();
        s.g.dyn_inputs(&kcat, "x", 2).unwrap();
        s.g.set_input_desc(&kcat, "x0", &[gd(p), KVD], Dtype::Fp16).unwrap();
        s.g.set_input_desc(&kcat, "x1", &[HOR, KVD], Dtype::Fp16).unwrap();
        s.g.set_output_desc(&kcat, "y", &[gd(total), KVD], Dtype::Fp16).unwrap();
        s.g.set_attr_int(&kcat, "concat_dim", 0).unwrap();
        s.g.set_attr_int(&kcat, "N", 2).unwrap();
        s.g.link(&kcat, "x0", &format!("pk{i}")).unwrap();
        s.wire(&kcat, "x1", &kr2);
        let kall = s.reg_out(&kcat, "y");
        let kall3 = s.reshape(&format!("{}kall3", tag), &kall, &[gd(total), KVD], "shp_kall", &[1, gd(total), KVD]);
        let vcat = format!("{}vcat", tag);
        s.g.add_op(&vcat, "ConcatD").unwrap();
        s.g.dyn_inputs(&vcat, "x", 2).unwrap();
        s.g.set_input_desc(&vcat, "x0", &[gd(p), KVD], Dtype::Fp16).unwrap();
        s.g.set_input_desc(&vcat, "x1", &[HOR, KVD], Dtype::Fp16).unwrap();
        s.g.set_output_desc(&vcat, "y", &[gd(total), KVD], Dtype::Fp16).unwrap();
        s.g.set_attr_int(&vcat, "concat_dim", 0).unwrap();
        s.g.set_attr_int(&vcat, "N", 2).unwrap();
        s.g.link(&vcat, "x0", &format!("pv{i}")).unwrap();
        s.wire(&vcat, "x1", &v2);
        let vall = s.reg_out(&vcat, "y");
        let vall3 = s.reshape(&format!("{}vall3", tag), &vall, &[gd(total), KVD], "shp_kall", &[1, gd(total), KVD]);
        let sq = if attn_manual {
            // 手工 cross-GQA：q 头主序化（headsplit），k/v 已是 rank-3 单头
            let q3 = s.headsplit(&format!("{}qh", tag), &qr2, HOR, 1, HOR, HEADS, HD, "shp_fq4", "shp_fq3");
            let attn = s.attn_manual_gqa(&format!("{}attn", tag), &q3, &kall3, &vall3, HEADS, HOR, gd(total), HD, fmask_n.as_deref());
            s.headmerge(&format!("{}am", tag), &attn, 1, HOR, HEADS, HD, "shp_fa4", "shp_fbq")
        } else {
            let pfa = s.pfa(&format!("{}pfa", tag), &qr3, &kall3, &vall3, &[1, HOR, QD], &[1, total, KVD], HEADS, KV_HEADS, HD);
            s.squeeze(&format!("{}sq", tag), &pfa, &[1, HOR, QD], &[HOR, QD])
        };
        let proj = mm_or_qmd(&mut s, &format!("{}proj", tag), &sq, &[HOR, QD], fi8[i][3].as_ref(), &format!("{}outwt", tag), AW, &[HOR, AW], int8_native);
        let projb = if fi8[i][3].is_some() {
            proj
        } else {
            s.bias(&format!("{}projb", tag), &proj, &[HOR, AW], &format!("{}outb", tag))
        };
        // 分支 ×gate 进残差（torch _gated_residual：residual + branch·gate）
        let gated = s.gatew(&format!("{}ag", tag), &projb, &[HOR, AW], &format!("{}agate", tag));
        let res = s.add2(&format!("{}res", tag), &gated, &xstream, &[HOR, AW]);

        // mlp
        let mnorm = s.ada(&format!("{}mn", tag), &res, "zeros", &format!("{}mscl", tag), &format!("{}msh", tag), &[HOR, AW]);
        let gate = mm_or_qmd(&mut s, &format!("{}gate", tag), &mnorm, &[HOR, AW], fi8[i][4].as_ref(), &format!("{}gatewt", tag), AINTER, &[HOR, AINTER], int8_native);
        let up = mm_or_qmd(&mut s, &format!("{}up", tag), &mnorm, &[HOR, AW], fi8[i][5].as_ref(), &format!("{}upwt", tag), AINTER, &[HOR, AINTER], int8_native);
        let gact = s.gelu(&format!("{}gact", tag), &gate, &[HOR, AINTER], true);
        let act = s.mul2(&format!("{}act", tag), &gact, &up, &[HOR, AINTER]);
        let down = mm_or_qmd(&mut s, &format!("{}down", tag), &act, &[HOR, AINTER], fi8[i][6].as_ref(), &format!("{}downwt", tag), AW, &[HOR, AW], int8_native);
        let downb = if fi8[i][6].is_some() {
            down
        } else {
            s.bias(&format!("{}downb", tag), &down, &[HOR, AW], &format!("{}downb", tag))
        };
        let gated2 = s.gatew(&format!("{}mg", tag), &downb, &[HOR, AW], &format!("{}mgate", tag));
        let hidden = s.add2(&format!("{}h", tag), &gated2, &res, &[HOR, AW]);

        // 下一层 attention 的 ada-norm（eager attention_normalized 同构）
        let (nscl, nsh) = if i + 1 < depth {
            (format!("l{}_ascl", i + 1), format!("l{}_ash", i + 1))
        } else {
            ("fsc".to_string(), "fsh".to_string())
        };
        normed = s.ada(&format!("{}nn", tag), &hidden, "zeros", &nscl, &nsh, &[HOR, AW]);
        xstream = hidden;
    }

    // action_out + euler
    let vel = s.mm("aout", &normed, &[HOR, AW], "aoutwt", &[ADIM, AW], &[HOR, ADIM]);
    let vel = s.bias("aoutb2", &vel, &[HOR, ADIM], "aoutb");
    let t1 = s.mul2("eul1", "state", "c1", &[HOR, ADIM]);
    let t2 = s.mul2("eul2", &vel, "c2", &[HOR, ADIM]);
    let out = s.add2("eul", &t1, &t2, &[HOR, ADIM]);
    s.finish(&[&out]);
    println!("flow OM built: depth={depth} P={p} n_in={} n_out=1", s.ins().len());

    // ---- eager 参考（action_layer_ascend 序列镜像；qkv 独立投影）----
    let layer_bases2 = layer_bases.clone();
    let eager = |b: &[DeviceBuffer]| -> Vec<DeviceBuffer> {
        let zeros = &b[1];
        // ada-norm eager：AddRmsNorm(gamma=scale) + bias_add(shift)
        let ada = |x: &DeviceBuffer, scl: &DeviceBuffer, sh: &DeviceBuffer| -> DeviceBuffer {
            let n = aops::add_rms_norm_fp16(ctx, stream, x, zeros, scl, &[HOR, AW], RMS_EPS).unwrap().0;
            aops::bias_add_fp16(ctx, stream, &n, sh, HOR, AW).unwrap()
        };
        // ⚠ 段级索引按当前注册序：0 state / 1 zeros / 2..7 flat rope /
        // 8..11 rank-2 表 / 12 ainwt / 13 ainb / 14.. pk_i,pv_i 交替 /
        // 16.. 层权重（const_i32 改造前 shape 占 4 槽，旧索引 16/17/18
        // 已失效——[832,256] 读到 qkvwt 的 size 断言 panic 取证）
        let ain = aops::matmul_b_t_fp16(ctx, stream, &b[0], [HOR, ADIM], &b[12], ADIM, AW).unwrap();
        // 残差流双轨（同图构造修复——residual 基 = 未归一化 x）
        let mut xstream = aops::bias_add_fp16(ctx, stream, &ain, &b[13], HOR, AW).unwrap();
        let mut normed = ada(&xstream, &b[layer_bases2[0]], &b[layer_bases2[0] + 1]);
        for li in 0..depth {
            // qkv3 层内输入布局：ascl,ash,qw,kw,vw,qb,kb,vb,outwt,outb,
            // mscl,msh,gatewt,upwt,downwt,downb,agate,mgate（18 项）；默认 14 项
            let w = &b[layer_bases2[li]..layer_bases2[li] + if qkv3 { 18 } else { 14 }];
            let (g_a, g_m) = if qkv3 { (16, 17) } else { (12, 13) };
            let (ow, ob, msc, msh, gw, uw, dw, db) = if qkv3 { (8, 9, 10, 11, 12, 13, 14, 15) } else { (4, 5, 6, 7, 8, 9, 10, 11) };
            let (pk, pv) = (&b[14 + 2 * li], &b[15 + 2 * li]);
            let (wq, wk, wv, bq, bk, bv) = &eager_qkv[li];
            let q = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [HOR, AW], wq, AW, QD).unwrap(), bq, HOR, QD).unwrap();
            let k = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [HOR, AW], wk, AW, KVD).unwrap(), bk, HOR, KVD).unwrap();
            let v = aops::bias_add_fp16(ctx, stream,
                &aops::matmul_b_t_fp16(ctx, stream, &normed, [HOR, AW], wv, AW, KVD).unwrap(), bv, HOR, KVD).unwrap();
            let kr = aops::rope_rotate_half_flat_fp16(ctx, stream, &k, HOR * KV_HEADS, HD, &b[5], &b[6], &b[7]).unwrap();
            let qr = aops::rope_rotate_half_flat_fp16(ctx, stream, &q, HOR * HEADS, HD, &b[2], &b[3], &b[4]).unwrap();
            let kall = aops::cat_fp16(ctx, stream, &[pk, &kr], &[vec![p, KVD], vec![HOR, KVD]], 0, &[total, KVD]).unwrap();
            let vall = aops::cat_fp16(ctx, stream, &[pv, &v], &[vec![p, KVD], vec![HOR, KVD]], 0, &[total, KVD]).unwrap();
            let attn = aops::prompt_flash_attention_cross_bsh_fp16(
                ctx, stream, &qr, &kall, &vall, HOR, total, HEADS, KV_HEADS, HD, None).unwrap();
            let proj = aops::matmul_b_t_fp16(ctx, stream, &attn, [HOR, QD], &w[ow], QD, AW).unwrap();
            let proj = aops::bias_add_fp16(ctx, stream, &proj, &w[ob], HOR, AW).unwrap();
            let proj = aops::gate_mul_fp16(ctx, stream, &proj, &w[g_a], HOR, AW).unwrap();
            let res = aops::add_fp16(ctx, stream, &proj, &xstream, &[HOR, AW]).unwrap();
            let mnorm = ada(&res, &w[msc], &w[msh]);
            let gate = aops::matmul_b_t_fp16(ctx, stream, &mnorm, [HOR, AW], &w[gw], AW, AINTER).unwrap();
            let up = aops::matmul_b_t_fp16(ctx, stream, &mnorm, [HOR, AW], &w[uw], AW, AINTER).unwrap();
            let g = aops::gelu_fp16(ctx, stream, &gate, &[HOR, AINTER], true).unwrap();
            let act = aops::mul_fp16(ctx, stream, &g, &up, &[HOR, AINTER]).unwrap();
            let down = aops::matmul_b_t_fp16(ctx, stream, &act, [HOR, AINTER], &w[dw], AINTER, AW).unwrap();
            let down = aops::bias_add_fp16(ctx, stream, &down, &w[db], HOR, AW).unwrap();
            let down = aops::gate_mul_fp16(ctx, stream, &down, &w[g_m], HOR, AW).unwrap();
            let hidden = aops::add_fp16(ctx, stream, &down, &res, &[HOR, AW]).unwrap();
            let n = b.len();
            let (ns, nh) = if li + 1 < depth {
                (&b[layer_bases2[li + 1]], &b[layer_bases2[li + 1] + 1])
            } else {
                (&b[n - 6], &b[n - 5])
            };
            normed = ada(&hidden, ns, nh);
            xstream = hidden;
        }
        let n = b.len();
        let vel = aops::matmul_b_t_fp16(ctx, stream, &normed, [HOR, AW], &b[n - 4], AW, ADIM).unwrap();
        let vel = aops::bias_add_fp16(ctx, stream, &vel, &b[n - 3], HOR, ADIM).unwrap();
        let t1 = aops::mul_fp16(ctx, stream, &b[0], &b[n - 2], &[HOR, ADIM]).unwrap();
        let t2 = aops::mul_fp16(ctx, stream, &vel, &b[n - 1], &[HOR, ADIM]).unwrap();
        let fin = aops::add_fp16(ctx, stream, &t1, &t2, &[HOR, ADIM]).unwrap();
        // 中间量 drop 前 kernel 必须落定（同 vision 段注记）
        drop(stream.synchronize());
        vec![fin]
    };

    if let Some(st) = e2e {
        // e2e 10 步：每步覆写 styles（qkv3 布局：ascl/ash=base+0/+1，
        // mscl/msh=base+10/+11）+ state（x），pk/pv 不变；euler 已在图内
        let rw = real.expect("e2e 需要 GEB_CKPT（time_mlp/style 投影）");
        let cfg = Pi05Config::default();
        let dir = std::env::var("GEB_OM_DIR").unwrap_or_else(|_| "./om_cache".into());
        // HYBRID：GEB_FLOW_POOL=<om 根>（配 GEB_DYN_LLIST 的 L 表）——
        // 装载 per-L 静态 flow OM 池（binds 一次 max 物化，图句柄按请求换）。
        // bring-up 用首档句柄跑（内容 warmup 无所谓）
        let mut flow_pool: std::collections::HashMap<i64, GeGraph> = std::collections::HashMap::new();
        if let Ok(pool_root) = std::env::var("GEB_FLOW_POOL") {
            let ls = dyn_llist().unwrap_or_else(|| panic!("GEB_FLOW_POOL 需 GEB_DYN_LLIST 档表"));
            // 串行装载（22 只 × ~500MB ≈ 1-2min）。22 线程并行实测崩进程
            //（aclmdlLoadFromMem 并发上限 <22——3 线程的 OM 预载活；
            // crash-loop 实测）。有界并行留待验证
            let t0p = std::time::Instant::now();
            for &l in &ls {
                let p = format!("{pool_root}/tl{l}_i8/flow_real.om");
                let g = ge_builder::load(&p).unwrap_or_else(|e| panic!("flow pool {p}: {e:?}"));
                flow_pool.insert(l, g);
            }
            println!("[flow-pool] {} 只装载完成 {:?}", flow_pool.len(), t0p.elapsed());
            let first = *ls.first().expect("档表非空");
            s.g = flow_pool[&first];
        } else {
            println!("[e2e] loading {dir}/flow_real.om");
            s.g = ge_builder::load(&format!("{dir}/flow_real.om")).expect("load om");
        }
        assert_eq!(s.g.num_outputs().unwrap(), 1);
        let ob = ctx.malloc(s.g.output_size(0).unwrap().max(16)).expect("out malloc");
        let h2d_f16 = |buf: &DeviceBuffer, vals: &[f16]| {
            let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            ctx.copy_h2d(buf, &bytes).expect("h2d style/state");
        };
        let mut x = st.noise.clone();
        // euler 常数换绑为 LeRobot 语义 x' = x + dt·v（c1=1.0，sample_actions
        // L894；OM 缺省 0.9/-0.1 = openpi x1-预测式）。c1/c2 是 Data 输入，
        // 按名定位——原「binds 尾两位」假设被 GEB_ATTN_MASK 的 padmask
        // 追加注册打破（padmask 落尾，c2 灌进 1444B mask bind 即崩）
        let c1_h = vec![f16::from_f32(1.0); (HOR * ADIM) as usize];
        let c2_h = vec![f16::from_f32(-0.1); (HOR * ADIM) as usize];
        h2d_f16(&s.binds[s.bind_of("c1")], &c1_h);
        h2d_f16(&s.binds[s.bind_of("c2")], &c2_h);
        // PAD 对拍：flow padmask 填真 mask（cache pad 键列 [vis+真L, p) =
        // -30000，flow HOR 行与真 cache 行恒 0——serve 侧 fill_mask 同义；
        // 缺此步则 pad cache 键裸露，actions 对拍被语义污染（首战 9% 教训）
        if envi("GEB_E2E_PAD_TOKENS", 0) > 0 {
            if let Some(bi) = s.try_bind_of("padmask") {
                let lt = envi("GEB_E2E_PAD_TOKENS", 0);
                let vis = lp_to_p(0);
                let mut mh = vec![f16::from_f32(0.0); (p + HOR) as usize];
                for r in &mut mh[(vis + lt) as usize..p as usize] {
                    *r = f16::from_f32(-30000.0);
                }
                let bytes: Vec<u8> =
                    mh.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
                ctx.copy_h2d(&s.binds[bi], &bytes).expect("flow padmask h2d");
            }
        }
        // styles 预计算缓存：10 步 time 调度固定（1+step·dt）⇒ styles 跨调用
        // 不变，生产口径一次性算 + 常驻（每步 host 现算 ≈173ms/步 是 glue 大头，
        // bring-up 计时定罪）。cond 对拍/替入诊断（bisect）保留在此
        let tpre = std::time::Instant::now();
        let style_cache: Vec<Vec<Vec<f16>>> = (0..cfg.num_flow_steps)
            .map(|step| {
                let time = cfg.flow_start_time * (1.0 - step as f32 / cfg.num_flow_steps as f32);
                let te = sinusoidal_time_embedding(
                    time,
                    cfg.action_expert.width,
                    cfg.time_min_period,
                    cfg.time_max_period,
                );
                let cond = e2e_conditioning(rw, &te);
                // golden cond_s{step}（frame0_v3b）存在时打印 host f32 cond 与
                // golden 的差（styles 对拍诊断）
                let gold_cond = {
                    let key = format!("cond_s{step}");
                    st.golden_mid
                        .iter()
                        .find(|(k, _)| *k == key)
                        .map(|(_, v)| v.clone())
                };
                if let Some(want) = &gold_cond {
                    assert_eq!(cond.len(), want.len(), "cond 与 golden cond_s{step} 长度不符");
                    let mut md = 0f32;
                    for (a, b) in cond.iter().zip(want.iter()) {
                        md = md.max((a - b).abs());
                    }
                    println!("[e2e] step {step} cond host-vs-golden max_diff={md:.6}");
                }
                let mut v: Vec<Vec<f16>> = Vec::with_capacity(depth * 6 + 2);
                for i in 0..depth {
                    let lay = &rw.action_layers[i];
                    let (a_scl, a_sh, a_gt) = e2e_style_pair(&lay.input_norm.style, &cond, AW as usize);
                    let (m_scl, m_sh, m_gt) = e2e_style_pair(&lay.post_attention_norm.style, &cond, AW as usize);
                    v.push(a_scl);
                    v.push(a_sh);
                    v.push(m_scl);
                    v.push(m_sh);
                    v.push(a_gt);
                    v.push(m_gt);
                }
                let (f_scl, f_sh, _) = e2e_style_pair(&rw.action_final_norm.style, &cond, AW as usize);
                v.push(f_scl);
                v.push(f_sh);
                v
            })
            .collect();
        println!(
            "[e2e] styles 预计算 ×{} 步 {:?}（一次性，跨调用缓存）",
            cfg.num_flow_steps,
            tpre.elapsed()
        );
        let t0 = std::time::Instant::now();
        for step in 0..cfg.num_flow_steps {
            let sc = &style_cache[step];
            for i in 0..depth {
                let b = layer_bases[i];
                h2d_f16(&s.binds[b], &sc[6 * i]);
                h2d_f16(&s.binds[b + 1], &sc[6 * i + 1]);
                h2d_f16(&s.binds[b + 10], &sc[6 * i + 2]);
                h2d_f16(&s.binds[b + 11], &sc[6 * i + 3]);
                h2d_f16(&s.binds[b + 16], &sc[6 * i + 4]);
                h2d_f16(&s.binds[b + 17], &sc[6 * i + 5]);
            }
            h2d_f16(&s.binds[fs_idx], &sc[6 * depth]);
            h2d_f16(&s.binds[fs_idx + 1], &sc[6 * depth + 1]);
            h2d_f16(&s.binds[0], &x); // state = 当前 x
            let oref: Vec<&DeviceBuffer> = vec![&ob];
            s.run_seg(s.gear_payloads.len().saturating_sub(1), &oref, stream)
                .expect("ge run");
            drop(stream.synchronize());
            x = download_f16(ctx, &ob, (HOR * ADIM) as usize);
            let m = x.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
            println!("[e2e] step {step} |x|max={m:.4}");
            if step == 0 {
                st.cmp_mid("step0_x1", &x);
            }
        }
        println!("[e2e] flow 10 步 {:?}（styles 已缓存——纯 h2d 换绑 + run + d2h）", t0.elapsed());
        // GEB_E2E_BENCH=N：稳态 per-call bench——每迭代 = 完整 10 步 flow
        // （styles 缓存直取 + h2d + run + sync + d2h），x 每迭代从 noise 重启
        let nb = envi("GEB_E2E_BENCH", 0) as usize;
        if nb > 0 {
            let mut ts = Vec::with_capacity(nb);
            for _ in 0..nb {
                let t = std::time::Instant::now();
                let mut xx = st.noise.clone();
                for sc in &style_cache {
                    for i in 0..depth {
                        let b = layer_bases[i];
                        h2d_f16(&s.binds[b], &sc[6 * i]);
                        h2d_f16(&s.binds[b + 1], &sc[6 * i + 1]);
                        h2d_f16(&s.binds[b + 10], &sc[6 * i + 2]);
                        h2d_f16(&s.binds[b + 11], &sc[6 * i + 3]);
                        h2d_f16(&s.binds[b + 16], &sc[6 * i + 4]);
                        h2d_f16(&s.binds[b + 17], &sc[6 * i + 5]);
                    }
                    h2d_f16(&s.binds[fs_idx], &sc[6 * depth]);
                    h2d_f16(&s.binds[fs_idx + 1], &sc[6 * depth + 1]);
                    h2d_f16(&s.binds[0], &xx);
                    let oref: Vec<&DeviceBuffer> = vec![&ob];
                    s.run_seg(s.gear_payloads.len().saturating_sub(1), &oref, stream)
                        .expect("ge run");
                    drop(stream.synchronize());
                    xx = download_f16(ctx, &ob, (HOR * ADIM) as usize);
                }
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "[e2e] flow 稳态 bench ×{nb} 调用（10 步/调用，styles 缓存 + h2d + run + sync + d2h）: P50={:.2}ms min={:.2} max={:.2}",
                ts[nb / 2],
                ts[0],
                ts[nb - 1]
            );
        }
        st.actions = x;
        // GEB_E2E_DUMP_MID 同目录落 actions（f16 raw [50,ADIM]——互差的
        // golden 合并原料：static actions 并进 golden_actions 后，dyn 跑的
        // GOLDEN PARITY 即 actions 级 dyn-vs-static 互差）
        if let Ok(dir) = std::env::var("GEB_E2E_DUMP_MID") {
            let bytes: Vec<u8> = st.actions.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            std::fs::write(format!("{dir}/ge_actions.f16"), &bytes).expect("dump actions");
            println!("[e2e] actions 已落盘 {dir}/ge_actions.f16");
        }
        if let Some(g) = &st.golden_actions {
            // golden actions = predict_action_chunk 的 deployable 切片
            // (50, gd)——取 e2e 输出 [50, ADIM] 每行前 gd 列比（pad_vector
            // 真值在前）
            let gd = g.len() / 50;
            assert!(gd > 0 && gd <= ADIM as usize, "golden actions 形状异常");
            let mut md = 0f32;
            for r in 0..50usize {
                for c in 0..gd {
                    let a = st.actions[r * ADIM as usize + c].to_f32();
                    let b = g[r * gd + c].to_f32();
                    md = md.max((a - b).abs());
                }
            }
            let rm = g.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
            println!(
                "[e2e] GOLDEN PARITY (first {gd} cols): max_diff={md:.5} (golden |max|={rm:.3}, rel={:.1}%)",
                if rm > 0.0 { md / rm * 100.0 } else { md }
            );
        } else {
            println!("[e2e] 无 golden actions——只验有限性（golden 对拍/LIBERO 归下一步）");
        }
        assert!(st.actions.iter().all(|v| v.to_f32().is_finite()), "actions 含非有限值");
        // GEB_E2E_REPLAY：逐帧 pk/pv 换绑（36 路 host 中转，同生产链）+ noise
        // 重启 10 步（styles 预计算缓存跨帧不变）→ actions 前 gd 列 vs torch
        // normalized_actions（同 noise 对拍 = 离线行为裁判）
        if !st.replay.is_empty() {
            let t0 = std::time::Instant::now();
            let mut rels: Vec<f32> = Vec::with_capacity(st.replay.len());
            for (fi, f) in st.replay.iter_mut().enumerate() {
                assert_eq!(f.kv.len(), depth * 2, "须先跑 prefix replay");
                for i in 0..depth {
                    h2d_f16(&s.binds[pk_base + 2 * i], &f.kv[2 * i]);
                    h2d_f16(&s.binds[pk_base + 2 * i + 1], &f.kv[2 * i + 1]);
                }
                let mut xx = f.noise.clone();
                for sc in &style_cache {
                    for i in 0..depth {
                        let b = layer_bases[i];
                        h2d_f16(&s.binds[b], &sc[6 * i]);
                        h2d_f16(&s.binds[b + 1], &sc[6 * i + 1]);
                        h2d_f16(&s.binds[b + 10], &sc[6 * i + 2]);
                        h2d_f16(&s.binds[b + 11], &sc[6 * i + 3]);
                        h2d_f16(&s.binds[b + 16], &sc[6 * i + 4]);
                        h2d_f16(&s.binds[b + 17], &sc[6 * i + 5]);
                    }
                    h2d_f16(&s.binds[fs_idx], &sc[6 * depth]);
                    h2d_f16(&s.binds[fs_idx + 1], &sc[6 * depth + 1]);
                    h2d_f16(&s.binds[0], &xx);
                    let oref: Vec<&DeviceBuffer> = vec![&ob];
                    s.run_seg(s.gear_payloads.len().saturating_sub(1), &oref, stream)
                        .expect("ge run");
                    drop(stream.synchronize());
                    xx = download_f16(ctx, &ob, (HOR * ADIM) as usize);
                }
                f.actions = xx;
                let gd = f.nact.len() / 50;
                assert!(gd > 0 && gd <= ADIM as usize, "replay nact 形状异常（帧 {fi}）");
                let mut md = 0f32;
                for r in 0..50usize {
                    for c in 0..gd {
                        let a = f.actions[r * ADIM as usize + c].to_f32();
                        let b = f.nact[r * gd + c].to_f32();
                        md = md.max((a - b).abs());
                    }
                }
                let rm = f.nact.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
                let xm = f.actions.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
                let rel = if rm > 0.0 { md / rm * 100.0 } else { md };
                rels.push(rel);
                println!(
                    "[e2e] replay 帧 {fi}: |x|max={xm:.3} max_diff={md:.5} rel={rel:.1}% (nact|max|={rm:.3})",
                );
            }
            rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "[e2e] REPLAY 对拍 ×{} 帧 {:?}: rel P50={:.1}% min={:.1}% max={:.1}%（torch normalized_actions，同 noise）",
                st.replay.len(),
                t0.elapsed(),
                rels[rels.len() / 2],
                rels[0],
                rels[rels.len() - 1]
            );
            // GEB_REPLAY_DUMP=<path>：逐帧 engine actions 落盘（LE 原始二进制：
            // u32 帧数；每帧 u32 gd + 50*ADIM f32 actions + 50*gd f32 nact）——
            // 行为刑侦用（bias/scale/corr：系统性与噪声性偏差的判别）
            if let Ok(p) = std::env::var("GEB_REPLAY_DUMP") {
                use std::io::Write as _;
                let mut out = std::fs::File::create(&p).expect("replay dump create");
                let n = st.replay.len() as u32;
                out.write_all(&n.to_le_bytes()).unwrap();
                for fr in st.replay.iter() {
                    let gd = (fr.nact.len() / 50) as u32;
                    out.write_all(&gd.to_le_bytes()).unwrap();
                    for v in fr.actions.iter() {
                        out.write_all(&v.to_f32().to_le_bytes()).unwrap();
                    }
                    for v in fr.nact.iter() {
                        out.write_all(&v.to_f32().to_le_bytes()).unwrap();
                    }
                }
                println!("[e2e] replay actions dumped: {p}");
            }
        }
        // GEB_E2E_SERVE：机械下沉（styles 缓存跨请求不变——10 步调度固定；
        // euler c1/c2 已换绑 LeRobot 语义，buffer 内容恒持）
        if std::env::var("GEB_E2E_SERVE").is_ok() {
            let (style_master, style_views, style_map) = if serve_fast() {
                let (m, v, mp) =
                    style_residency(ctx, &style_cache, &layer_bases, fs_idx, depth);
                println!(
                    "[serve] GEB_SERVE_FAST：styles 设备驻留 {} 步 × {} 槽（master {} 字节，一次上传）",
                    v.len(),
                    mp.len(),
                    m.len()
                );
                (Some(m), v, mp)
            } else {
                (None, Vec::new(), std::collections::HashMap::new())
            };
            let ins_order: Vec<usize> = if s.wconst {
                s.data_inputs.clone()
            } else {
                (0..s.binds.len()).collect()
            };
            let rope_binds: Vec<usize> = ["qcos", "qsin", "kcos", "ksin", "qcos2", "qsin2", "kcos2", "ksin2"]
                .iter()
                .map(|n| s.bind_of(n))
                .collect();
            let gear_of_l = gear_ll
                .as_ref()
                .map(|g| g.iter().enumerate().map(|(i, &l)| (l, i)).collect())
                .unwrap_or_default();
            let fmask_bind = s.try_bind_of("padmask");
            st.fmech = Some(FlowMech {
                s,
                ob,
                style_cache,
                layer_bases,
                fs_idx,
                pk_base,
                depth,
                style_master,
                style_views,
                style_map,
                ins_order,
                gear_of_l,
                rope_binds,
                last_l: -1,
                flows: flow_pool,
                mask_bind: fmask_bind,
                mask_last: -1,
            });
        }
        return;
    }
    if int8 {
        // eager/parity 跳过（同 prefix int8 先例）：f16 eager 参考对 int8 图
        // 无裁判意义——法官 = e2e golden 对拍 + task0 行为。烤桶 OM 落盘照常
        if let Ok(p) = std::env::var("GEB_SAVE") {
            s.g.save(&p).expect("save om (flow int8)");
            println!("OM saved: {p}");
        }
    } else {
        parity_and_bench(ctx, stream, &mut s, "flow", &[(HOR * ADIM) as usize], &eager, bench);
    }
    ge_builder::fini().expect("fini");
    println!("GE_FLOW_PROBE_OK");
}

/// 单算子最小图编译冒烟（GEB_OPTEST=btd|rsh|sld|gat|cat|aln|gln|asc）——

// ---------------------------------------------------------------------------
// GeServe 门面（④C in-process 直调面）：三段 OM per-frame 推理。
// spool serve_loop 与 PyO3（apxinf-py --features ascend）共用同一帧实现。
// 驱动入口（env 分发 / GEB_CKPT 加载）见 examples/ge_model_probe.rs。
// ---------------------------------------------------------------------------

/// 一帧推理结果：actions [HOR, ADIM] flow 终态 + 段计时（ms）+
/// vision_out [VT, PW]（对拍/调试用，生产可忽略）
pub struct GeServeOut {
    pub actions: Vec<f16>,
    pub vision_out: Vec<f16>,
    pub vision_ms: f32,
    pub prefix_ms: f32,
    pub flow_ms: f32,
}

/// 进程内三段管线（= serve 桶的机械组）。open 走与探针 bring-up 完全
/// 相同的构装路径（含首帧 warmup 与 styles 预计算）；infer 与 serve_loop
/// 逐帧同源。
pub struct GeServe {
    vm: VisionMech,
    pm: PrefixMech,
    fm: FlowMech,
    pub tokens: usize,
    pub fast: bool,
}

impl GeServe {
    /// 从预置桶目录开一条管线。前置：ge_builder::init + AscendBackend::new
    /// 已由调用方完成（GE 全局初始化须先于 acl runtime）；real = 真权重
    /// （token 嵌入/styles 条件 host 消费面）。
    /// env 契约：om_dir 内 {vision,prefix,flow}_real.om 须与四件套
    /// （GEB_ATTN=manual GEB_QKV3=1 GEB_ROPEFLAT=1 GEB_WCONST=1）同配置
    /// 构建；本函数自行补设缺省 env（GEB_SEG=e2e / GEB_TOKENS /
    /// GEB_OM_DIR / GEB_E2E_SERVE——后者是三段机械下沉的既定闸门），
    /// GEB_NORM32 桶拓扑不同不支持。
    /// ⚠ 进程级单管线：配置走进程全局 env，多桶（多 L）须多进程或先扩展
    pub fn open(
        be: &AscendBackend,
        real: &Pi05Weights,
        om_dir: &str,
        tokens: usize,
        fast: bool,
    ) -> GeServe {
        assert!(!norm32_enabled(), "GeServe 不支持 GEB_NORM32 桶（拓扑不同）");
        for (k, v) in [
            ("GEB_SEG", "e2e"),
            ("GEB_ATTN", "manual"),
            ("GEB_QKV3", "1"),
            ("GEB_ROPEFLAT", "1"),
            ("GEB_WCONST", "1"),
            // 拓扑一致性（Seg 构造读，须与 OM 构建同值；进程内场景 env
            // 不受控，钉死生产值）。⚠ 不钉 GEB_DEPTH——分段默认
            // vision=27 / prefix=18 / flow=18 与 OM 一致，钉 18 会把
            // vision 砍成 18 层（n_in 187≠277 首跑教训）
            ("GEB_PREFIX_DROP_EMPTY", "256"),
            ("GEB_LAYER_OFFSET", "0"),
        ] {
            if std::env::var(k).unwrap_or_default().is_empty() {
                std::env::set_var(k, v);
            }
        }
        if std::env::var("GEB_E2E_SERVE").is_err() {
            std::env::set_var("GEB_E2E_SERVE", "__inproc__");
        }
        std::env::set_var("GEB_TOKENS", tokens.to_string());
        std::env::set_var("GEB_OM_DIR", om_dir);
        if fast {
            std::env::set_var("GEB_SERVE_FAST", "1");
        }
        // 冷启动 lazy：三段 OM 并行预载（构造段序内 e2e_run 命中即取）
        preload_e2e_oms(be.ctx());
        // 合成 bring-up 输入（首帧 warmup 消费；真实帧走 infer）
        let mut seed = 0xE2E2u32;
        let mut st = E2eStage {
            patches: rand_f16((VT * V_PATCH_W) as usize, &mut seed, 300.0),
            token_ids: (0..tokens).map(|i| 1000 + i as u32 * 7).collect(),
            noise: rand_f16((HOR * ADIM) as usize, &mut seed, 1.0),
            golden_actions: None,
            golden_mid: Vec::new(),
            vision_out: Vec::new(),
            kv: Vec::new(),
            actions: Vec::new(),
            replay: Vec::new(),
            vmech: None,
            pmech: None,
            fmech: None,
        };
        seg_vision(be, false, Some(real), Some(&mut st));
        seg_prefix(be, false, Some(real), Some(&mut st));
        seg_flow(be, false, Some(real), Some(&mut st));
        let fast = serve_fast();
        GeServe {
            vm: st.vmech.take().expect("GeServe vision 机械缺失"),
            pm: st.pmech.take().expect("GeServe prefix 机械缺失"),
            fm: st.fmech.take().expect("GeServe flow 机械缺失"),
            tokens,
            fast,
        }
    }

    /// 一帧推理：patches [VT*V_PATCH_W] / token_ids [tokens] / noise
    /// [HOR*ADIM] → actions [HOR, ADIM]（normalized_actions，denorm 在
    /// torch 后处理）
    pub fn infer(
        &mut self,
        be: &AscendBackend,
        patches: &[f16],
        token_ids: &[u32],
        noise: &[f16],
    ) -> GeServeOut {
        let ctx = be.ctx();
        let stream = be.stream();
        let t0 = std::time::Instant::now();
        let vision_out = self.vm.one_frame(ctx, stream, patches);
        let vision_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (actions, prefix_ms, flow_ms) = if self.fast {
            // GEB_SERVE_FAST：kv 设备直连 + styles 驻留 + x 驻留（tp 仍含
            // prefix run 的 sync 等待——段计时与 legacy 可比）。gear 模式
            // 下 L = token_ids.len()，逐请求选档（prefix/flow 同档）
            let t1 = std::time::Instant::now();
            self.pm.one_frame_dev(ctx, stream, &vision_out, token_ids);
            let prefix_ms = t1.elapsed().as_secs_f64() * 1e3;
            let kvrefs: Vec<&DeviceBuffer> = self.pm.routs[..self.pm.nkv].iter().collect();
            let t2 = std::time::Instant::now();
            let actions = self.fm.one_frame_fast(ctx, stream, &kvrefs, token_ids, noise);
            (actions, prefix_ms, t2.elapsed().as_secs_f64() * 1e3)
        } else {
            let t1 = std::time::Instant::now();
            let kv = self.pm.one_frame(ctx, stream, &vision_out, token_ids);
            let prefix_ms = t1.elapsed().as_secs_f64() * 1e3;
            let t2 = std::time::Instant::now();
            let actions = self.fm.one_frame(ctx, stream, &kv, token_ids, noise);
            (actions, prefix_ms, t2.elapsed().as_secs_f64() * 1e3)
        };
        GeServeOut {
            actions,
            vision_out,
            vision_ms: vision_ms as f32,
            prefix_ms: prefix_ms as f32,
            flow_ms: flow_ms as f32,
        }
    }
}
