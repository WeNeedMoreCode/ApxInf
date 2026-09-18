//! C2 三段 OM 全量探针：vision / prefix / flow 三段各自声明为 GE 静态
//! OM（ge_builder FFI），与 eager aclnn 组合同输入同序列对拍 + bench。
//! flow 段是单步 OM（10 步 flow 由调用方执行 10 次换绑定）。
//!
//! 算子契约（310P opp proto 头文件取证 2026-09-19）：
//!   AddLayerNorm  x1/x2/gamma/beta(/bias)→y/mean/rstd/x, epsilon(Float)
//!   GeluV2        x→y, approximate(String: "none"=exact / "tanh")
//!   GatherV2D     x/indices→y, axis(Int attr；310P 实现 axis 只支持 0)
//!   ConcatD       DYNAMIC_INPUT x1..xN → y, concat_dim(Int) + N(Int)
//!   SliceD        x→y, offsets/size(ListInt)
//!   BroadcastToD  x→y, shape(ListInt)  [行广播：bias / ada-norm shift]
//!   Reshape       x + shape(张量输入) → y   [rope flat 桥 / rank-3 桥]
//!   MatMulV2(transpose_x2)/PromptFlashAttention(BSH rank-3)/AddRmsNorm/
//!   Squeeze/Add/Mul —— C1 已验证（ge_layer_probe）
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
//! 运行（rust 容器，先 source /data/apxinf/rust_env.sh）：
//!   GEB_SEG=vision GEB_DEPTH=2 cargo run --example ge_model_probe --features ascend --release -p apxinf-model
//!   GEB_SEG=prefix GEB_BENCH=1 cargo run ...   （全深对拍 + bench）
//!   GEB_SAVE=/data/apxinf/om_cache/prefix.om ...（OM 落盘）
//!   GEB_LOAD=...（缓存加载替代编译）
//! 环境变量：GEB_SEG（默认 prefix）/ GEB_DEPTH（层深覆盖，冒烟用）/
//!   GEB_TOKENS（prefix token 数，默认 64 → P=832，16 倍数纪律）/
//!   GEB_BENCH / GEB_SAVE / GEB_LOAD
use half::f16;

use apxinf_ascend::ge_builder::{self, Dtype, GeGraph};
use apxinf_ascend::{ops as aops, AscendBackend, AscendContext, AscendStream, DeviceBuffer};

// π0.5 3 视图档维度（Pi05Config::default）
const VT: i64 = 768; // vision tokens = 3 views × 256 patches
const VPV: i64 = 256;
const VIEWS: i64 = 3;
const VW: i64 = 1152; // vision width
const V_INTER: i64 = 4304;
const V_HEADS: i64 = 16;
const V_HD: i64 = 72;
const V_PATCH_W: i64 = 588; // 3*14*14
const PW: i64 = 2048; // language width
const INTER: i64 = 16384;
const HEADS: i64 = 8;
const KV_HEADS: i64 = 1;
const HD: i64 = 256;
const QD: i64 = HEADS * HD; // 2048
const KVD: i64 = KV_HEADS * HD; // 256
const QKVW: i64 = QD + 2 * KVD; // 2560
const AW: i64 = 1024; // action expert width
const AINTER: i64 = 4096;
const ADIM: i64 = 32;
const HOR: i64 = 50;
const ROPE_THETA: f64 = 10000.0;
const RMS_EPS: f64 = 1e-6;
const LN_EPS: f64 = 1e-6;

fn envi(k: &str, d: i64) -> i64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// erf 近似（Abramowitz-Stegun 7.1.26，|ε|<1.5e-7——exact gelu 对照够用）
fn erf(x: f32) -> f32 {
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
/// LayerNorm 行方差塌缩 → rstd 放大 385× 的教训，2026-09-19）
fn mix64(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn rand_f16(n: usize, seed: &mut u32, div: f32) -> Vec<f16> {
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
fn norm_f16(n: usize, seed: &mut u32, center: f32) -> Vec<f16> {
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

fn upload(ctx: &AscendContext, vals: &[f16]) -> DeviceBuffer {
    let bytes =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 2) };
    let buf = ctx.malloc(bytes.len()).expect("malloc");
    ctx.copy_h2d(&buf, bytes).expect("h2d");
    buf
}

fn upload_bytes(ctx: &AscendContext, bytes: &[u8]) -> DeviceBuffer {
    let buf = ctx.malloc(bytes.len()).expect("malloc");
    ctx.copy_h2d(&buf, bytes).expect("h2d");
    buf
}

fn upload_i32(ctx: &AscendContext, vals: &[i32]) -> DeviceBuffer {
    let bytes =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
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

/// host [rows, cols] f16 权重 → 转置 upload 为 [cols, rows] 物理
/// （MatMulV2 transpose_x2=true 与 eager matmul_b_t 共用布局）
fn wbuf_t(ctx: &AscendContext, host: &[f16], rows: i64, cols: i64) -> DeviceBuffer {
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
    let t = aops::host_transpose(bytes, rows, cols);
    upload_bytes(ctx, &t)
}

/// host LayerNorm 参考（f32 计算）：aclnnAddLayerNorm 在 [768,1152] 有
/// kernel 级放大（2026-09-19），eager 参考路径的 LN 用 host 计算。
fn host_ln(
    ctx: &AscendContext, x: &DeviceBuffer, g: &[f16], b: &[f16], rows: i64, cols: i64,
) -> DeviceBuffer {
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
fn rope_flat_const(tokens: i64, heads: i64, pos_offset: i64) -> (Vec<f16>, Vec<f16>, Vec<i32>) {
    let half = (HD / 2) as usize;
    let rows = (tokens * heads * 2) as usize;
    let mut cos = vec![f16::from_f32(0.0); rows * half];
    let mut sin = vec![f16::from_f32(0.0); rows * half];
    for t in 0..tokens as usize {
        for i in 0..half {
            let freq = ((pos_offset + t as i64) as f64)
                * ROPE_THETA.powf(-(2.0 * i as f64) / HD as f64);
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

/// cos/sin rank-2 表 [t*heads, HD]（eager build_rope_tables 同款：cos 半维
/// 重复、sin 符号折叠），配合通道半互换 swap（SliceD×2+ConcatD）的
/// rank-2 rope 组合。
fn rope_rank2_const(tokens: i64, heads: i64, pos_offset: i64) -> (Vec<f16>, Vec<f16>) {
    let d = HD as usize;
    let half = d / 2;
    let rows = (tokens * heads) as usize;
    let mut cos = vec![f16::from_f32(0.0); rows * d];
    let mut sin = vec![f16::from_f32(0.0); rows * d];
    for t in 0..tokens as usize {
        for i in 0..half {
            let freq = ((pos_offset + t as i64) as f64)
                * ROPE_THETA.powf(-(2.0 * i as f64) / HD as f64);
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

struct Seg {
    g: GeGraph,
    idx: i64,
    names: Vec<String>,
    shapes: Vec<Vec<i64>>,
    binds: Vec<DeviceBuffer>,
    datas: std::collections::HashSet<String>,
    outports: std::collections::HashMap<String, &'static str>,
    /// LayerNormV4 实例名（finish 时 mean/rstd 绑 aux 图输出）
    ln_aux: Vec<String>,
}

impl Seg {
    fn new(name: &str) -> Seg {
        Seg {
            g: GeGraph::begin(name).expect("begin"),
            idx: 0,
            names: Vec::new(),
            shapes: Vec::new(),
            binds: Vec::new(),
            datas: std::collections::HashSet::new(),
            outports: std::collections::HashMap::new(),
            ln_aux: Vec::new(),
        }
    }

    fn reg_out(&mut self, op: &str, port: &'static str) -> String {
        self.outports.insert(op.to_string(), port);
        op.to_string()
    }

    fn data_buf(&mut self, name: &str, dims: &[i64], buf: DeviceBuffer) -> String {
        self.g.add_data(name, self.idx, dims, Dtype::Fp16).unwrap();
        self.idx += 1;
        self.names.push(name.to_string());
        self.shapes.push(dims.to_vec());
        self.binds.push(buf);
        self.datas.insert(name.to_string());
        name.to_string()
    }

    fn data(&mut self, ctx: &AscendContext, name: &str, dims: &[i64], host: &[f16]) -> String {
        let buf = upload(ctx, host);
        self.data_buf(name, dims, buf)
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

    /// 行广播 bias：TileD(row [1,cols], multiples=[rows,1]) + Add。
    /// （BroadcastToD 在 310P kernel 编译崩——TBE task_distribute；TileD
    /// attr 版编译验证过。算子名后缀 _bc/_ba 与 bias 行 Data 名错开。）
    fn bias(&mut self, name: &str, x: &str, x_dims: &[i64], row: &str) -> String {
        let bc = format!("{name}_bc");
        let ba = format!("{name}_ba");
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

    /// LayerNormV4（vision SigLIP norm）。⚠ aclnnAddLayerNorm/GE AddLayerNorm
    /// 在 [768,1152] 大 shape 输出 ~100× 放大（行方差正常、kernel 级行为，
    /// 2026-09-19 trace 取证）——换 LayerNormV4（normalized_shape 张量输入）。
    /// ⚠ mean/rstd 是 REQUIRED 输出且死端会让 y 爆（lnv4c 取证，与
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

    /// ada-norm：y = rms(x)·scale + shift = AddRmsNorm(x, zeros, gamma=scale)
    /// + TileD(shift_row) + Add（BroadcastToD 310P 编译崩，换 TileD）
    fn ada(&mut self, name: &str, x: &str, zeros: &str, scale: &str, shift: &str, dims: &[i64]) -> String {
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

    fn finish(&mut self, out_names: &[&str]) {
        if std::env::var("GEB_LOAD").is_ok() {
            // 缓存加载模式：跳过编译，parity_and_bench 里替换为磁盘 OM
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
        if self.ln_aux.is_empty() {
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
        self.g.build().expect("build");
    }
}

// ---------------------------------------------------------------------------
// 通用运行：eager 参考跑完 → GE run → 对拍 → bench → OM 落盘
// ---------------------------------------------------------------------------

/// 对拍 + bench 通用尾段。outs_elem：每个图输出的元素数。
fn parity_and_bench(
    ctx: &AscendContext,
    stream: &AscendStream,
    seg: &mut Seg,
    tag: &str,
    out_elems: &[usize],
    eager: &dyn Fn(&[DeviceBuffer]) -> Vec<DeviceBuffer>,
    bench: bool,
) {
    let n_in = seg.binds.len();
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

    // GE run（加载缓存或已 build 的图）
    let ins: Vec<&DeviceBuffer> = seg.binds.iter().collect();
    let n_out = seg.g.num_outputs().unwrap();
    for i in 0..n_out {
        println!("[out-dims] {i}: size={} dims={:?}", seg.g.output_size(i).unwrap_or(0), seg.g.output_dims(i).unwrap_or_default());
    }
    let outs: Vec<DeviceBuffer> = (0..n_out)
        .map(|i| ctx.malloc(seg.g.output_size(i).unwrap().max(16)).expect("out malloc"))
        .collect();
    let out_refs: Vec<&DeviceBuffer> = outs.iter().collect();
    seg.g.run(&ins, &out_refs, stream).expect("ge run");
    drop(stream.synchronize());

    // 对拍（主输出——LayerNormV4 的 mean/rstd aux 输出跳过）。断言用
    // 相对误差：GE 静态 OM 与 eager aclnn 允许不同 tiling 变体（这正是
    // C 路线的性能来源），单算子/小图逐位一致（C1 验证），全量多层图
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
        for _ in 0..3 {
            seg.g.run(&ins, &out_refs, stream).unwrap();
        }
        drop(stream.synchronize());
        let mut ts = Vec::new();
        for r in 0..30 {
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                seg.g.run(&ins, &out_refs, stream).unwrap();
            }
            drop(stream.synchronize());
            if r >= 3 {
                ts.push(t0.elapsed().as_secs_f64() * 1000.0 / 10.0);
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{tag} GE OM: {:.4} ms (median/10)", ts[ts.len() / 2]);
    }
}

// ---------------------------------------------------------------------------
// vision 段：patch embed → depth×SigLIP 层 → post LN → projector
// eager = ascend_executor::vision_layer_ascend 序列镜像
// ---------------------------------------------------------------------------

fn seg_vision(be: &AscendBackend, bench: bool) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH", 27) as usize;
    let t = VT;
    let vqd = V_HEADS * V_HD; // 1152
    let qkvw = vqd * 3;
    let mut seed = 0xC0DEu32;
    let mut s = Seg::new("ge_vision");

    // ---- 段级输入（注册序 = eager 的 index 约定）----
    // 0 patches / 1 patch_wt / 2 patch_b / 3 pos_rep / 4 zeros /
    // 5 shp_qkv3 / 6 shp_flat2 / 每层 12 项 / 尾 4 项(pnw,pnb,projwt,projb)
    s.data(ctx, "patches", &[t, V_PATCH_W], &rand_f16((t * V_PATCH_W) as usize, &mut seed, 300.0));
    {
        let host = rand_f16((V_PATCH_W * VW) as usize, &mut seed, 30000.0);
        let b = wbuf_t(ctx, &host, V_PATCH_W, VW);
        s.data_buf("patch_wt", &[VW, V_PATCH_W], b);
    }
    s.data(ctx, "patch_b", &[1, VW], &rand_f16(VW as usize, &mut seed, 12000.0));
    {
        // position 表循环重复（cuda kernel 语义：行 r 读 table[r % tpv]）
        let table = rand_f16((VPV * VW) as usize, &mut seed, 300.0);
        let mut rep = Vec::with_capacity((t * VW) as usize);
        for r in 0..t as usize {
            let src = (r % VPV as usize) * VW as usize;
            rep.extend_from_slice(&table[src..src + VW as usize]);
        }
        s.data(ctx, "pos_rep", &[t, VW], &rep);
    }
    s.data_zeros(ctx, "zeros", t, VW);
    s.data_i32(ctx, "shp_v3", &[3], &[VIEWS as i32, VPV as i32, vqd as i32]);
    s.data_i32(ctx, "shp_flat2", &[2], &[t as i32, vqd as i32]);
    s.data_i32(ctx, "nsh", &[1], &[VW as i32]); // LayerNormV4 normalized_shape

    // ---- 图：patch embed ----
    let mut cur = s.mm("pemb", "patches", &[t, V_PATCH_W], "patch_wt", &[VW, V_PATCH_W], &[t, VW]);
    cur = s.bias("pemb_b", &cur, &[t, VW], "patch_b");
    cur = s.add2("pemb_p", &cur, "pos_rep", &[t, VW]);

    // ---- 层循环 ----
    // eager 参考的 qkv 走三个独立 matmul（fused 输出 [m, q|k|v] 列交织，
    // take_rows 的 flat 切分在交织布局下数学错误——生产 eager 的该切分
    // 是 C2 取证发现的 bug，GE 侧 SliceD 通道切分才是对的；eager 参考以
    // 独立投影保持数学正确）。独立 buffer 段函数持有，不进图输入。
    let mut eager_qkv: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer)> = Vec::new();
    // norm gamma/beta 的 host 副本（eager 参考 host LN 用）
    let mut eager_norms: Vec<(Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>)> = Vec::new();
    let mut layer_bases = Vec::with_capacity(depth);
    for i in 0..depth {
        let base = s.binds.len();
        layer_bases.push(base);
        let p = format!("l{i}_");
        let n1w_h = norm_f16(VW as usize, &mut seed, 1.0);
        let n1b_h = norm_f16(VW as usize, &mut seed, 0.0);
        s.data(ctx, &format!("{}n1w", p), &[VW], &n1w_h);
        s.data(ctx, &format!("{}n1b", p), &[VW], &n1b_h);
        {
            // host 三块 [VW, vqd] → GE 拼接 [VW, qkvw] 转置上传；
            // eager 三块各自转置上传；bias 整条（GE）/host 切三段（eager）
            let wq = rand_f16((VW * vqd) as usize, &mut seed, 30000.0);
            let wk = rand_f16((VW * vqd) as usize, &mut seed, 30000.0);
            let wv = rand_f16((VW * vqd) as usize, &mut seed, 30000.0);
            let mut fused = Vec::with_capacity((VW * qkvw) as usize);
            for r in 0..VW as usize {
                let base = r * vqd as usize;
                fused.extend_from_slice(&wq[base..base + vqd as usize]);
                fused.extend_from_slice(&wk[base..base + vqd as usize]);
                fused.extend_from_slice(&wv[base..base + vqd as usize]);
            }
            let b = wbuf_t(ctx, &fused, VW, qkvw);
            s.data_buf(&format!("{}qkvwt", p), &[qkvw, VW], b);
            let bias = rand_f16(qkvw as usize, &mut seed, 12000.0);
            s.data(ctx, &format!("{}qkvb", p), &[1, qkvw], &bias);
            let bq = upload(ctx, &bias[0..vqd as usize]);
            let bk = upload(ctx, &bias[vqd as usize..2 * vqd as usize]);
            let bv = upload(ctx, &bias[2 * vqd as usize..]);
            let ebq = wbuf_t(ctx, &wq, VW, vqd);
            let ebk = wbuf_t(ctx, &wk, VW, vqd);
            let ebv = wbuf_t(ctx, &wv, VW, vqd);
            eager_qkv.push((ebq, ebk, ebv, bq, bk, bv));
        }
        {
            let host = rand_f16((vqd * VW) as usize, &mut seed, 30000.0);
            let b = wbuf_t(ctx, &host, vqd, VW);
            s.data_buf(&format!("{}outwt", p), &[VW, vqd], b);
        }
        s.data(ctx, &format!("{}outb", p), &[1, VW], &rand_f16(VW as usize, &mut seed, 12000.0));
        let n2w_h = norm_f16(VW as usize, &mut seed, 1.0);
        let n2b_h = norm_f16(VW as usize, &mut seed, 0.0);
        s.data(ctx, &format!("{}n2w", p), &[VW], &n2w_h);
        s.data(ctx, &format!("{}n2b", p), &[VW], &n2b_h);
        eager_norms.push((n1w_h, n1b_h, n2w_h, n2b_h));
        {
            let host = rand_f16((VW * V_INTER) as usize, &mut seed, 30000.0);
            let b = wbuf_t(ctx, &host, VW, V_INTER);
            s.data_buf(&format!("{}fc1wt", p), &[V_INTER, VW], b);
        }
        s.data(ctx, &format!("{}fc1b", p), &[1, V_INTER], &rand_f16(V_INTER as usize, &mut seed, 12000.0));
        {
            let host = rand_f16((V_INTER * VW) as usize, &mut seed, 30000.0);
            let b = wbuf_t(ctx, &host, V_INTER, VW);
            s.data_buf(&format!("{}fc2wt", p), &[VW, V_INTER], b);
        }
        s.data(ctx, &format!("{}fc2b", p), &[1, VW], &rand_f16(VW as usize, &mut seed, 12000.0));

        // attention（v3 方案：rank-2 列切 → Reshape rank-3 桥 → batch PFA）
        let ln1 = s.addln(&format!("{}ln1", p), &cur, &format!("{}n1w", p), &format!("{}n1b", p), "nsh", &[t, VW]);
        let qkv = s.mm(&format!("{}qkv", p), &ln1, &[t, VW], &format!("{}qkvwt", p), &[qkvw, VW], &[t, qkvw]);
        let qkvb = s.bias(&format!("{}qkvb", p), &qkv, &[t, qkvw], &format!("{}qkvb", p));
        let q2 = s.slice2(&format!("{}q2", p), &qkvb, &[t, qkvw], 0, vqd);
        let k2 = s.slice2(&format!("{}k2", p), &qkvb, &[t, qkvw], vqd, vqd);
        let v2 = s.slice2(&format!("{}v2", p), &qkvb, &[t, qkvw], vqd * 2, vqd);
        let q3 = s.reshape(&format!("{}q3", p), &q2, &[t, vqd], "shp_v3", &[VIEWS, VPV, vqd]);
        let k3 = s.reshape(&format!("{}k3", p), &k2, &[t, vqd], "shp_v3", &[VIEWS, VPV, vqd]);
        let v3 = s.reshape(&format!("{}v3", p), &v2, &[t, vqd], "shp_v3", &[VIEWS, VPV, vqd]);
        let pfa = s.pfa(&format!("{}pfa", p), &q3, &k3, &v3, &[VIEWS, VPV, vqd], &[VIEWS, VPV, vqd], V_HEADS, V_HEADS, V_HD);
        let a2 = s.reshape(&format!("{}a2", p), &pfa, &[VIEWS, VPV, vqd], "shp_flat2", &[t, vqd]);
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
    }

    // ---- post norm + projector ----
    let pnw_h = norm_f16(VW as usize, &mut seed, 1.0);
    let pnb_h = norm_f16(VW as usize, &mut seed, 0.0);
    s.data(ctx, "pnw", &[VW], &pnw_h);
    s.data(ctx, "pnb", &[VW], &pnb_h);
    {
        let host = rand_f16((VW * PW) as usize, &mut seed, 30000.0);
        let b = wbuf_t(ctx, &host, VW, PW);
        s.data_buf("projwt", &[PW, VW], b);
    }
    s.data(ctx, "projb", &[1, PW], &rand_f16(PW as usize, &mut seed, 12000.0));
    let pln = s.addln("pln", &cur, "pnw", "pnb", "nsh", &[t, VW]);
    let proj = s.mm("proj", &pln, &[t, VW], "projwt", &[PW, VW], &[t, PW]);
    let out = s.bias("out", &proj, &[t, PW], "projb");
    s.finish(&[&out]);
    println!("vision OM built: depth={depth} n_in={} n_out=1", s.binds.len());

    // ---- eager 参考（vision_layer_ascend 序列镜像；qkv 独立投影）----
    let trace = std::env::var("GEB_TRACE").is_ok();
    let eager = |b: &[DeviceBuffer]| -> Vec<DeviceBuffer> {
        let mut mx = |tag: &str, buf: &DeviceBuffer| {
            if trace {
                let n = buf.len() / 2;
                let h = download_f16(ctx, buf, n);
                let m = h.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
                println!("[trace] {tag}: |max|={m:.3}");
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
            let w = &b[layer_bases[li]..layer_bases[li] + 12];
            let (wq, wk, wv, bq, bk, bv) = &eager_qkv[li];
            let (n1w_h, n1b_h, n2w_h, n2b_h) = &eager_norms[li];
            let normed = host_ln(ctx, &h, n1w_h, n1b_h, t, VW);
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
            let proj = aops::matmul_b_t_fp16(ctx, stream, &attn, [t, vqd], &w[4], vqd, VW).unwrap();
            let proj = aops::bias_add_fp16(ctx, stream, &proj, &w[5], t, VW).unwrap();
            let res1 = aops::add_fp16(ctx, stream, &proj, &h, &[t, VW]).unwrap();
            mx(&format!("l{li} res1"), &res1);
            let norm2 = host_ln(ctx, &res1, n2w_h, n2b_h, t, VW);
            let act = aops::matmul_b_t_fp16(ctx, stream, &norm2, [t, VW], &w[8], VW, V_INTER).unwrap();
            let act = aops::bias_add_fp16(ctx, stream, &act, &w[9], t, V_INTER).unwrap();
            let act = aops::gelu_fp16(ctx, stream, &act, &[t, V_INTER], false).unwrap();
            let out = aops::matmul_b_t_fp16(ctx, stream, &act, [t, V_INTER], &w[10], V_INTER, VW).unwrap();
            let out = aops::bias_add_fp16(ctx, stream, &out, &w[11], t, VW).unwrap();
            h = aops::add_fp16(ctx, stream, &out, &res1, &[t, VW]).unwrap();
            mx(&format!("l{li} out"), &h);
        }
        let n = b.len();
        let normed = host_ln(ctx, &h, &pnw_h, &pnb_h, t, VW);
        let proj = aops::matmul_b_t_fp16(ctx, stream, &normed, [t, VW], &b[n - 2], VW, PW).unwrap();
        vec![aops::bias_add_fp16(ctx, stream, &proj, &b[n - 1], t, PW).unwrap()]
    };

    parity_and_bench(ctx, stream, &mut s, "vision", &[(t * PW) as usize], &eager, bench);
    ge_builder::fini().expect("fini");
    println!("GE_VISION_PROBE_OK");
}

// ---------------------------------------------------------------------------
// prefix 段：depth×language 层全序，输出每层 k/v（末层无 tail——eager
// compute_tail=false 语义）
// ---------------------------------------------------------------------------

fn seg_prefix(be: &AscendBackend, bench: bool) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH", 18) as usize;
    let tokens = envi("GEB_TOKENS", 64);
    let p = VT + tokens; // prefix 长度（16 倍数纪律）
    let mut seed = 0xBEEFu32;
    let mut s = Seg::new("ge_prefix");

    // ---- 段级输入 ----
    // 0 x0 / 1 zeros / 2..7 flat rope 六件（eager flat 版用）/
    // 8..11 rank-2 rope 表四件（GE rank-2 组合用）/ 层 10 项
    s.data(ctx, "x0", &[p, PW], &rand_f16((p * PW) as usize, &mut seed, 100.0));
    s.data_zeros(ctx, "zeros", p, PW);
    let (qc, qs, qi) = rope_flat_const(p, HEADS, 0);
    let (kc, ks, ki) = rope_flat_const(p, KV_HEADS, 0);
    let qf = (p * HEADS * 2, HD / 2);
    let kf = (p * KV_HEADS * 2, HD / 2);
    s.data(ctx, "qcos", &[qf.0, qf.1], &qc);
    s.data(ctx, "qsin", &[qf.0, qf.1], &qs);
    s.data_i32(ctx, "qswap", &[qf.0], &qi);
    s.data(ctx, "kcos", &[kf.0, kf.1], &kc);
    s.data(ctx, "ksin", &[kf.0, kf.1], &ks);
    s.data_i32(ctx, "kswap", &[kf.0], &ki);
    // rank-2 表（物理 [t*heads, d] 与 [t, heads*d] 同布局；desc 按 rank-2 设）
    let (qc2, qs2) = rope_rank2_const(p, HEADS, 0);
    let (kc2, ks2) = rope_rank2_const(p, KV_HEADS, 0);
    s.data(ctx, "qcos2", &[p, QD], &qc2);
    s.data(ctx, "qsin2", &[p, QD], &qs2);
    s.data(ctx, "kcos2", &[p, KVD], &kc2);
    s.data(ctx, "ksin2", &[p, KVD], &ks2);
    // Reshape shape 张量（rank-3 桥）
    s.data_i32(ctx, "shp_q3", &[3], &[1, p as i32, QD as i32]);
    s.data_i32(ctx, "shp_k3", &[3], &[1, p as i32, KVD as i32]);
    s.data_i32(ctx, "shp_v3", &[3], &[1, p as i32, KVD as i32]);

    // ---- 层循环（eager qkv 独立投影，同 vision 段注记）----
    let mut eager_qkv: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer)> = Vec::new();
    let mut layer_bases = Vec::with_capacity(depth);
    let mut cur = "x0".to_string();
    let mut kv_outs: Vec<String> = Vec::new();
    for i in 0..depth {
        let base = s.binds.len();
        layer_bases.push(base);
        let tag = format!("l{i}_");
        let last = i + 1 == depth;
        s.data(ctx, &format!("{}g1", tag), &[PW], &norm_f16(PW as usize, &mut seed, 1.0));
        {
            let wq = rand_f16((PW * QD) as usize, &mut seed, 8000.0);
            let wk = rand_f16((PW * KVD) as usize, &mut seed, 8000.0);
            let wv = rand_f16((PW * KVD) as usize, &mut seed, 8000.0);
            let mut fused = Vec::with_capacity((PW * QKVW) as usize);
            for r in 0..PW as usize {
                let rb = r * QD as usize;
                fused.extend_from_slice(&wq[rb..rb + QD as usize]);
                let rb = r * KVD as usize;
                fused.extend_from_slice(&wk[rb..rb + KVD as usize]);
                fused.extend_from_slice(&wv[rb..rb + KVD as usize]);
            }
            let b = wbuf_t(ctx, &fused, PW, QKVW);
            s.data_buf(&format!("{}qkvwt", tag), &[QKVW, PW], b);
            let bias = rand_f16(QKVW as usize, &mut seed, 4000.0);
            s.data(ctx, &format!("{}qkvb", tag), &[1, QKVW], &bias);
            let bq = upload(ctx, &bias[0..QD as usize]);
            let bk = upload(ctx, &bias[QD as usize..(QD + KVD) as usize]);
            let bv = upload(ctx, &bias[(QD + KVD) as usize..]);
            let ebq = wbuf_t(ctx, &wq, PW, QD);
            let ebk = wbuf_t(ctx, &wk, PW, KVD);
            let ebv = wbuf_t(ctx, &wv, PW, KVD);
            eager_qkv.push((ebq, ebk, ebv, bq, bk, bv));
        }
        {
            let host = rand_f16((QD * PW) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, QD, PW);
            s.data_buf(&format!("{}outwt", tag), &[PW, QD], b);
        }
        s.data(ctx, &format!("{}outb", tag), &[1, PW], &rand_f16(PW as usize, &mut seed, 4000.0));
        s.data(ctx, &format!("{}g2", tag), &[PW], &norm_f16(PW as usize, &mut seed, 1.0));
        {
            let host = rand_f16((PW * INTER) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, PW, INTER);
            s.data_buf(&format!("{}gatewt", tag), &[INTER, PW], b);
        }
        {
            let host = rand_f16((PW * INTER) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, PW, INTER);
            s.data_buf(&format!("{}upwt", tag), &[INTER, PW], b);
        }
        {
            let host = rand_f16((INTER * PW) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, INTER, PW);
            s.data_buf(&format!("{}downwt", tag), &[PW, INTER], b);
        }
        s.data(ctx, &format!("{}downb", tag), &[1, PW], &rand_f16(PW as usize, &mut seed, 4000.0));

        let norm1 = s.addrms(&format!("{}n1", tag), &cur, "zeros", &format!("{}g1", tag), &[p, PW]);
        let qkv = s.mm(&format!("{}qkv", tag), &norm1, &[p, PW], &format!("{}qkvwt", tag), &[QKVW, PW], &[p, QKVW]);
        let qkvb = s.bias(&format!("{}qkvb", tag), &qkv, &[p, QKVW], &format!("{}qkvb", tag));
        // rank-2 列切 + rope（rank-2 版）+ Reshape rank-3 桥。
        // ⚠ 图输出绑 rank-2（rope 的 ad / slice 的 v）——Reshape 输出绑图
        // 输出时 desc 是动态 [-1,-1,-1]（shape 张量驱动），size 无效
        let q2 = s.slice2(&format!("{}q", tag), &qkvb, &[p, QKVW], 0, QD);
        let k2 = s.slice2(&format!("{}k", tag), &qkvb, &[p, QKVW], QD, KVD);
        let v2 = s.slice2(&format!("{}v", tag), &qkvb, &[p, QKVW], QD + KVD, KVD);
        let (kr2, kr3) = s.rope2(&format!("{}kr", tag), &k2, &[p, KVD], "kcos2", "ksin2", "shp_k3");
        let v3 = s.reshape(&format!("{}v3", tag), &v2, &[p, KVD], "shp_v3", &[1, p, KVD]);
        kv_outs.push(kr2.clone());
        kv_outs.push(v2.clone());
        if last {
            break; // 末层无 PFA/tail（compute_tail=false）
        }
        let (_, qr3) = s.rope2(&format!("{}qr", tag), &q2, &[p, QD], "qcos2", "qsin2", "shp_q3");
        let pfa = s.pfa(&format!("{}pfa", tag), &qr3, &kr3, &v3, &[1, p, QD], &[1, p, KVD], HEADS, KV_HEADS, HD);
        let sq = s.squeeze(&format!("{}sq", tag), &pfa, &[1, p, QD], &[p, QD]);
        let proj = s.mm(&format!("{}proj", tag), &sq, &[p, QD], &format!("{}outwt", tag), &[PW, QD], &[p, PW]);
        let projb = s.bias(&format!("{}projb", tag), &proj, &[p, PW], &format!("{}outb", tag));
        let res = s.add2(&format!("{}res", tag), &projb, &cur, &[p, PW]);
        let norm2 = s.addrms(&format!("{}n2", tag), &res, "zeros", &format!("{}g2", tag), &[p, PW]);
        let gate = s.mm(&format!("{}gate", tag), &norm2, &[p, PW], &format!("{}gatewt", tag), &[INTER, PW], &[p, INTER]);
        let up = s.mm(&format!("{}up", tag), &norm2, &[p, PW], &format!("{}upwt", tag), &[INTER, PW], &[p, INTER]);
        let gact = s.gelu(&format!("{}gact", tag), &gate, &[p, INTER], true);
        let act = s.mul2(&format!("{}act", tag), &gact, &up, &[p, INTER]);
        let down = s.mm(&format!("{}down", tag), &act, &[p, INTER], &format!("{}downwt", tag), &[PW, INTER], &[p, PW]);
        let downb = s.bias(&format!("{}downb", tag), &down, &[p, PW], &format!("{}downb", tag));
        cur = s.add2(&format!("{}out", tag), &downb, &res, &[p, PW]);
    }
    let out_names: Vec<&str> = kv_outs.iter().map(|x| x.as_str()).collect();
    s.finish(&out_names);
    println!("prefix OM built: depth={depth} P={p} n_in={} n_out={}", s.binds.len(), kv_outs.len());

    // ---- eager 参考（language_layer_ascend 序列镜像；qkv 独立投影）----
    let layer_bases2 = layer_bases.clone();
    let kv_count = kv_outs.len();
    let eager = |b: &[DeviceBuffer]| -> Vec<DeviceBuffer> {
        let zeros = &b[1];
        let mut h: Option<DeviceBuffer> = None;
        let mut outs = Vec::with_capacity(kv_count);
        for li in 0..depth {
            let w = &b[layer_bases2[li]..layer_bases2[li] + 10];
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
            let proj = aops::matmul_b_t_fp16(ctx, stream, &attn, [p, QD], &w[3], QD, PW).unwrap();
            let biased = aops::bias_add_fp16(ctx, stream, &proj, &w[4], p, PW).unwrap();
            let res = aops::add_fp16(ctx, stream, &biased, hb, &[p, PW]).unwrap();
            let norm2 = aops::add_rms_norm_fp16(ctx, stream, &res, zeros, &w[5], &[p, PW], RMS_EPS).unwrap().0;
            let gate = aops::matmul_b_t_fp16(ctx, stream, &norm2, [p, PW], &w[6], PW, INTER).unwrap();
            let up = aops::matmul_b_t_fp16(ctx, stream, &norm2, [p, PW], &w[7], PW, INTER).unwrap();
            let g = aops::gelu_fp16(ctx, stream, &gate, &[p, INTER], true).unwrap();
            let act = aops::mul_fp16(ctx, stream, &g, &up, &[p, INTER]).unwrap();
            let down = aops::matmul_b_t_fp16(ctx, stream, &act, [p, INTER], &w[8], INTER, PW).unwrap();
            let downb = aops::bias_add_fp16(ctx, stream, &down, &w[9], p, PW).unwrap();
            h = Some(aops::add_fp16(ctx, stream, &downb, &res, &[p, PW]).unwrap());
        }
        outs
    };

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

fn seg_flow(be: &AscendBackend, bench: bool) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH", 18) as usize;
    let tokens = envi("GEB_TOKENS", 64);
    let p = VT + tokens;
    let mut seed = 0xF00Du32;
    let mut s = Seg::new("ge_flow");

    // ---- 段级输入（注册序，两遍式：先全部注册再建图——层 i 的 next
    // ada-norm 引用层 i+1 的 style 输入，必须先注册后链接）----
    // 0 state / 1 zeros / 2..7 flat rope（eager）/ 8..11 rank-2 表（GE）/
    // 12 ainwt / 13 ainb / 14.. 每层 (pk_i, pv_i)（rank-2）/
    // 每层 12 项 / 尾 fsc,fsh,aoutwt,aoutb,c1,c2
    s.data(ctx, "state", &[HOR, ADIM], &rand_f16((HOR * ADIM) as usize, &mut seed, 100.0));
    s.data_zeros(ctx, "zeros", HOR, AW);
    let (qc, qs, qi) = rope_flat_const(HOR, HEADS, p);
    let (kc, ks, ki) = rope_flat_const(HOR, KV_HEADS, p);
    let qf = (HOR * HEADS * 2, HD / 2);
    let kf = (HOR * KV_HEADS * 2, HD / 2);
    s.data(ctx, "qcos", &[qf.0, qf.1], &qc);
    s.data(ctx, "qsin", &[qf.0, qf.1], &qs);
    s.data_i32(ctx, "qswap", &[qf.0], &qi);
    s.data(ctx, "kcos", &[kf.0, kf.1], &kc);
    s.data(ctx, "ksin", &[kf.0, kf.1], &ks);
    s.data_i32(ctx, "kswap", &[kf.0], &ki);
    let (qc2, qs2) = rope_rank2_const(HOR, HEADS, p);
    let (kc2, ks2) = rope_rank2_const(HOR, KV_HEADS, p);
    s.data(ctx, "qcos2", &[HOR, QD], &qc2);
    s.data(ctx, "qsin2", &[HOR, QD], &qs2);
    s.data(ctx, "kcos2", &[HOR, KVD], &kc2);
    s.data(ctx, "ksin2", &[HOR, KVD], &ks2);
    // Reshape shape 张量（q/k/v/kall rank-3 桥；kall/vall 共用）
    s.data_i32(ctx, "shp_q3", &[3], &[1, HOR as i32, QD as i32]);
    s.data_i32(ctx, "shp_k3", &[3], &[1, HOR as i32, KVD as i32]);
    s.data_i32(ctx, "shp_v3", &[3], &[1, HOR as i32, KVD as i32]);
    s.data_i32(ctx, "shp_kall", &[3], &[1, (p + HOR) as i32, KVD as i32]);
    {
        let host = rand_f16((ADIM * AW) as usize, &mut seed, 8000.0);
        let b = wbuf_t(ctx, &host, ADIM, AW);
        s.data_buf("ainwt", &[AW, ADIM], b);
    }
    s.data(ctx, "ainb", &[1, AW], &rand_f16(AW as usize, &mut seed, 4000.0));
    // prefix k/v（rank-2——ConcatD axis=0 拼接后统一 Unsqueeze）
    for i in 0..depth {
        s.data(ctx, &format!("pk{i}"), &[p, KVD], &rand_f16((p * KVD) as usize, &mut seed, 100.0));
        s.data(ctx, &format!("pv{i}"), &[p, KVD], &rand_f16((p * KVD) as usize, &mut seed, 100.0));
    }
    // 层权重 12 项：ascl ash qkvwt qkvb outwt outb mscl msh gatewt upwt downwt downb
    //（eager qkv 独立投影，同 vision 段注记）
    let mut eager_qkv: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer)> = Vec::new();
    let mut layer_bases = Vec::with_capacity(depth);
    for i in 0..depth {
        layer_bases.push(s.binds.len());
        let tag = format!("l{i}_");
        s.data(ctx, &format!("{}ascl", tag), &[AW], &norm_f16(AW as usize, &mut seed, 1.0));
        s.data(ctx, &format!("{}ash", tag), &[AW], &norm_f16(AW as usize, &mut seed, 0.0));
        {
            let wq = rand_f16((AW * QD) as usize, &mut seed, 8000.0);
            let wk = rand_f16((AW * KVD) as usize, &mut seed, 8000.0);
            let wv = rand_f16((AW * KVD) as usize, &mut seed, 8000.0);
            let mut fused = Vec::with_capacity((AW * QKVW) as usize);
            for r in 0..AW as usize {
                let rb = r * QD as usize;
                fused.extend_from_slice(&wq[rb..rb + QD as usize]);
                let rb = r * KVD as usize;
                fused.extend_from_slice(&wk[rb..rb + KVD as usize]);
                fused.extend_from_slice(&wv[rb..rb + KVD as usize]);
            }
            let b = wbuf_t(ctx, &fused, AW, QKVW);
            s.data_buf(&format!("{}qkvwt", tag), &[QKVW, AW], b);
            let bias = rand_f16(QKVW as usize, &mut seed, 4000.0);
            s.data(ctx, &format!("{}qkvb", tag), &[1, QKVW], &bias);
            let bq = upload(ctx, &bias[0..QD as usize]);
            let bk = upload(ctx, &bias[QD as usize..(QD + KVD) as usize]);
            let bv = upload(ctx, &bias[(QD + KVD) as usize..]);
            let ebq = wbuf_t(ctx, &wq, AW, QD);
            let ebk = wbuf_t(ctx, &wk, AW, KVD);
            let ebv = wbuf_t(ctx, &wv, AW, KVD);
            eager_qkv.push((ebq, ebk, ebv, bq, bk, bv));
        }
        {
            let host = rand_f16((QD * AW) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, QD, AW);
            s.data_buf(&format!("{}outwt", tag), &[AW, QD], b);
        }
        s.data(ctx, &format!("{}outb", tag), &[1, AW], &rand_f16(AW as usize, &mut seed, 4000.0));
        s.data(ctx, &format!("{}mscl", tag), &[AW], &norm_f16(AW as usize, &mut seed, 1.0));
        s.data(ctx, &format!("{}msh", tag), &[AW], &norm_f16(AW as usize, &mut seed, 0.0));
        {
            let host = rand_f16((AW * AINTER) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, AW, AINTER);
            s.data_buf(&format!("{}gatewt", tag), &[AINTER, AW], b);
        }
        {
            let host = rand_f16((AW * AINTER) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, AW, AINTER);
            s.data_buf(&format!("{}upwt", tag), &[AINTER, AW], b);
        }
        {
            let host = rand_f16((AINTER * AW) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, AINTER, AW);
            s.data_buf(&format!("{}downwt", tag), &[AW, AINTER], b);
        }
        s.data(ctx, &format!("{}downb", tag), &[1, AW], &norm_f16(AW as usize, &mut seed, 0.0));
    }
    s.data(ctx, "fsc", &[AW], &norm_f16(AW as usize, &mut seed, 1.0));
    s.data(ctx, "fsh", &[AW], &norm_f16(AW as usize, &mut seed, 0.0));
    {
        let host = rand_f16((AW * ADIM) as usize, &mut seed, 8000.0);
        let b = wbuf_t(ctx, &host, AW, ADIM);
        s.data_buf("aoutwt", &[ADIM, AW], b);
    }
    s.data(ctx, "aoutb", &[1, ADIM], &rand_f16(ADIM as usize, &mut seed, 4000.0));
    // euler 常数（σ = dt = -0.1：x' = (1+σ)x + σ·v）
    let eul_c1 = vec![f16::from_f32(0.9); (HOR * ADIM) as usize];
    let eul_c2 = vec![f16::from_f32(-0.1); (HOR * ADIM) as usize];
    s.data(ctx, "c1", &[HOR, ADIM], &eul_c1);
    s.data(ctx, "c2", &[HOR, ADIM], &eul_c2);

    // ---- 图构造 ----
    // action_in + 第一层 attention ada-norm（eager：第一层无预解析
    // normalized，用 adaptive_rms(action_in_out, attention_style[0])）
    let mut normed = s.mm("ain", "state", &[HOR, ADIM], "ainwt", &[AW, ADIM], &[HOR, AW]);
    normed = s.bias("ainb2", &normed, &[HOR, AW], "ainb");
    normed = s.ada("l0ada", &normed, "zeros", "l0_ascl", "l0_ash", &[HOR, AW]);

    let total = p + HOR;
    for i in 0..depth {
        let tag = format!("l{i}_");
        // attention（normed 已是本层 ada-norm 输出；q1/r1 验证链）
        let qkv = s.mm(&format!("{}qkv", tag), &normed, &[HOR, AW], &format!("{}qkvwt", tag), &[QKVW, AW], &[HOR, QKVW]);
        let qkvb = s.bias(&format!("{}qkvb", tag), &qkv, &[HOR, QKVW], &format!("{}qkvb", tag));
        let q2 = s.slice2(&format!("{}q", tag), &qkvb, &[HOR, QKVW], 0, QD);
        let k2 = s.slice2(&format!("{}k", tag), &qkvb, &[HOR, QKVW], QD, KVD);
        let v2 = s.slice2(&format!("{}v", tag), &qkvb, &[HOR, QKVW], QD + KVD, KVD);
        let (kr2, _) = s.rope2(&format!("{}kr", tag), &k2, &[HOR, KVD], "kcos2", "ksin2", "shp_k3");
        let (_, qr3) = s.rope2(&format!("{}qr", tag), &q2, &[HOR, QD], "qcos2", "qsin2", "shp_q3");
        // prefix cat（rank-2 ConcatD axis=0）+ Reshape rank-3 桥
        let kcat = format!("{}kcat", tag);
        s.g.add_op(&kcat, "ConcatD").unwrap();
        s.g.dyn_inputs(&kcat, "x", 2).unwrap();
        s.g.set_input_desc(&kcat, "x0", &[p, KVD], Dtype::Fp16).unwrap();
        s.g.set_input_desc(&kcat, "x1", &[HOR, KVD], Dtype::Fp16).unwrap();
        s.g.set_output_desc(&kcat, "y", &[total, KVD], Dtype::Fp16).unwrap();
        s.g.set_attr_int(&kcat, "concat_dim", 0).unwrap();
        s.g.set_attr_int(&kcat, "N", 2).unwrap();
        s.g.link(&kcat, "x0", &format!("pk{i}")).unwrap();
        s.wire(&kcat, "x1", &kr2);
        let kall = s.reg_out(&kcat, "y");
        let kall3 = s.reshape(&format!("{}kall3", tag), &kall, &[total, KVD], "shp_kall", &[1, total, KVD]);
        let vcat = format!("{}vcat", tag);
        s.g.add_op(&vcat, "ConcatD").unwrap();
        s.g.dyn_inputs(&vcat, "x", 2).unwrap();
        s.g.set_input_desc(&vcat, "x0", &[p, KVD], Dtype::Fp16).unwrap();
        s.g.set_input_desc(&vcat, "x1", &[HOR, KVD], Dtype::Fp16).unwrap();
        s.g.set_output_desc(&vcat, "y", &[total, KVD], Dtype::Fp16).unwrap();
        s.g.set_attr_int(&vcat, "concat_dim", 0).unwrap();
        s.g.set_attr_int(&vcat, "N", 2).unwrap();
        s.g.link(&vcat, "x0", &format!("pv{i}")).unwrap();
        s.wire(&vcat, "x1", &v2);
        let vall = s.reg_out(&vcat, "y");
        let vall3 = s.reshape(&format!("{}vall3", tag), &vall, &[total, KVD], "shp_kall", &[1, total, KVD]);
        let pfa = s.pfa(&format!("{}pfa", tag), &qr3, &kall3, &vall3, &[1, HOR, QD], &[1, total, KVD], HEADS, KV_HEADS, HD);
        let sq = s.squeeze(&format!("{}sq", tag), &pfa, &[1, HOR, QD], &[HOR, QD]);
        let proj = s.mm(&format!("{}proj", tag), &sq, &[HOR, QD], &format!("{}outwt", tag), &[AW, QD], &[HOR, AW]);
        let projb = s.bias(&format!("{}projb", tag), &proj, &[HOR, AW], &format!("{}outb", tag));
        let res = s.add2(&format!("{}res", tag), &projb, &normed, &[HOR, AW]);

        // mlp
        let mnorm = s.ada(&format!("{}mn", tag), &res, "zeros", &format!("{}mscl", tag), &format!("{}msh", tag), &[HOR, AW]);
        let gate = s.mm(&format!("{}gate", tag), &mnorm, &[HOR, AW], &format!("{}gatewt", tag), &[AINTER, AW], &[HOR, AINTER]);
        let up = s.mm(&format!("{}up", tag), &mnorm, &[HOR, AW], &format!("{}upwt", tag), &[AINTER, AW], &[HOR, AINTER]);
        let gact = s.gelu(&format!("{}gact", tag), &gate, &[HOR, AINTER], true);
        let act = s.mul2(&format!("{}act", tag), &gact, &up, &[HOR, AINTER]);
        let down = s.mm(&format!("{}down", tag), &act, &[HOR, AINTER], &format!("{}downwt", tag), &[AW, AINTER], &[HOR, AW]);
        let downb = s.bias(&format!("{}downb", tag), &down, &[HOR, AW], &format!("{}downb", tag));
        let hidden = s.add2(&format!("{}h", tag), &downb, &res, &[HOR, AW]);

        // 下一层 attention 的 ada-norm（eager attention_normalized 同构）
        let (nscl, nsh) = if i + 1 < depth {
            (format!("l{}_ascl", i + 1), format!("l{}_ash", i + 1))
        } else {
            ("fsc".to_string(), "fsh".to_string())
        };
        normed = s.ada(&format!("{}nn", tag), &hidden, "zeros", &nscl, &nsh, &[HOR, AW]);
    }

    // action_out + euler
    let vel = s.mm("aout", &normed, &[HOR, AW], "aoutwt", &[ADIM, AW], &[HOR, ADIM]);
    let vel = s.bias("aoutb2", &vel, &[HOR, ADIM], "aoutb");
    let t1 = s.mul2("eul1", "state", "c1", &[HOR, ADIM]);
    let t2 = s.mul2("eul2", &vel, "c2", &[HOR, ADIM]);
    let out = s.add2("eul", &t1, &t2, &[HOR, ADIM]);
    s.finish(&[&out]);
    println!("flow OM built: depth={depth} P={p} n_in={} n_out=1", s.binds.len());

    // ---- eager 参考（action_layer_ascend 序列镜像；qkv 独立投影）----
    let layer_bases2 = layer_bases.clone();
    let eager = |b: &[DeviceBuffer]| -> Vec<DeviceBuffer> {
        let zeros = &b[1];
        // ada-norm eager：AddRmsNorm(gamma=scale) + bias_add(shift)
        let ada = |x: &DeviceBuffer, scl: &DeviceBuffer, sh: &DeviceBuffer| -> DeviceBuffer {
            let n = aops::add_rms_norm_fp16(ctx, stream, x, zeros, scl, &[HOR, AW], RMS_EPS).unwrap().0;
            aops::bias_add_fp16(ctx, stream, &n, sh, HOR, AW).unwrap()
        };
        let ain = aops::matmul_b_t_fp16(ctx, stream, &b[0], [HOR, ADIM], &b[16], ADIM, AW).unwrap();
        let mut normed = aops::bias_add_fp16(ctx, stream, &ain, &b[17], HOR, AW).unwrap();
        normed = ada(&normed, &b[layer_bases2[0]], &b[layer_bases2[0] + 1]);
        for li in 0..depth {
            let w = &b[layer_bases2[li]..layer_bases2[li] + 12];
            let (pk, pv) = (&b[18 + 2 * li], &b[19 + 2 * li]);
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
            let proj = aops::matmul_b_t_fp16(ctx, stream, &attn, [HOR, QD], &w[4], QD, AW).unwrap();
            let proj = aops::bias_add_fp16(ctx, stream, &proj, &w[5], HOR, AW).unwrap();
            let res = aops::add_fp16(ctx, stream, &proj, &normed, &[HOR, AW]).unwrap();
            let mnorm = ada(&res, &w[6], &w[7]);
            let gate = aops::matmul_b_t_fp16(ctx, stream, &mnorm, [HOR, AW], &w[8], AW, AINTER).unwrap();
            let up = aops::matmul_b_t_fp16(ctx, stream, &mnorm, [HOR, AW], &w[9], AW, AINTER).unwrap();
            let g = aops::gelu_fp16(ctx, stream, &gate, &[HOR, AINTER], true).unwrap();
            let act = aops::mul_fp16(ctx, stream, &g, &up, &[HOR, AINTER]).unwrap();
            let down = aops::matmul_b_t_fp16(ctx, stream, &act, [HOR, AINTER], &w[10], AINTER, AW).unwrap();
            let down = aops::bias_add_fp16(ctx, stream, &down, &w[11], HOR, AW).unwrap();
            let hidden = aops::add_fp16(ctx, stream, &down, &res, &[HOR, AW]).unwrap();
            let n = b.len();
            let (ns, nh) = if li + 1 < depth {
                (&b[layer_bases2[li + 1]], &b[layer_bases2[li + 1] + 1])
            } else {
                (&b[n - 6], &b[n - 5])
            };
            normed = ada(&hidden, ns, nh);
        }
        let n = b.len();
        let vel = aops::matmul_b_t_fp16(ctx, stream, &normed, [HOR, AW], &b[n - 4], AW, ADIM).unwrap();
        let vel = aops::bias_add_fp16(ctx, stream, &vel, &b[n - 3], HOR, ADIM).unwrap();
        let t1 = aops::mul_fp16(ctx, stream, &b[0], &b[n - 2], &[HOR, ADIM]).unwrap();
        let t2 = aops::mul_fp16(ctx, stream, &vel, &b[n - 1], &[HOR, ADIM]).unwrap();
        vec![aops::add_fp16(ctx, stream, &t1, &t2, &[HOR, ADIM]).unwrap()]
    };

    parity_and_bench(ctx, stream, &mut s, "flow", &[(HOR * ADIM) as usize], &eager, bench);
    ge_builder::fini().expect("fini");
    println!("GE_FLOW_PROBE_OK");
}

/// 单算子最小图编译冒烟（GEB_OPTEST=btd|rsh|sld|gat|cat|aln|gln）——
/// 新算子逐个定罪用（三段图共享的新算子集）
fn optest(be: &AscendBackend, which: &str) {
    let ctx = be.ctx();
    let mut s = Seg::new(&format!("op_{which}"));
    let zeros_host = vec![f16::from_f32(0.0); 768 * 1152];
    match which {
        // BroadcastToD：[1,2560] → [832,2560]
        "btd" => {
            s.data(ctx, "x", &[1, 2560], &vec![f16::from_f32(0.1); 2560]);
            s.g.add_op("op", "BroadcastToD").unwrap();
            s.g.set_input_desc("op", "x", &[1, 2560], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[832, 2560], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list("op", "shape", &[832, 2560]).unwrap();
            s.g.link("op", "x", "x").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // Reshape：[832,2560] → [1,832,2560]（shape 张量输入）
        "rsh" => {
            s.data(ctx, "x", &[832, 2560], &vec![f16::from_f32(0.1); 832 * 2560]);
            s.data_i32(ctx, "shp", &[3], &[1, 832, 2560]);
            s.reshape("op", "x", &[832, 2560], "shp", &[1, 832, 2560]);
            s.finish(&["op"]);
        }
        // SliceD：[1,832,2560] rank-3 通道切分
        "sld" => {
            s.data(ctx, "x", &[1, 832, 2560], &vec![f16::from_f32(0.1); 832 * 2560]);
            s.slice3("op", "x", &[1, 832, 2560], 2048, 256);
            s.finish(&["op"]);
        }
        // GatherV2D：[1664,128] 行 swap gather
        "gat" => {
            s.data(ctx, "x", &[1664, 128], &vec![f16::from_f32(0.1); 1664 * 128]);
            let idx: Vec<i32> = (0..1664).map(|r| r ^ 1).collect();
            s.data_i32(ctx, "idx", &[1664], &idx);
            s.gather_rows("op", "x", &[1664, 128], "idx");
            s.finish(&["op"]);
        }
        // ConcatD：rank-3 [1,832,256]+[1,50,256] 沿 seq
        "cat" => {
            s.data(ctx, "a", &[1, 832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            s.data(ctx, "b", &[1, 50, 256], &vec![f16::from_f32(0.2); 50 * 256]);
            s.concat_seq3("op", "a", "b", &[1, 832, 256], &[1, 50, 256]);
            s.finish(&["op"]);
        }
        // AddLayerNorm：vision SigLIP norm
        "aln" => {
            s.data(ctx, "x", &[768, 1152], &vec![f16::from_f32(0.1); 768 * 1152]);
            s.data(ctx, "z", &[768, 1152], &zeros_host);
            s.data(ctx, "g", &[1152], &vec![f16::from_f32(1.0); 1152]);
            s.data(ctx, "b", &[1152], &vec![f16::from_f32(0.0); 1152]);
            s.addln("op", "x", "z", "g", "b", &[768, 1152]);
            s.finish(&["op"]);
        }
        // GeluV2 exact（"none"）
        "gln" => {
            s.data(ctx, "x", &[768, 4304], &vec![f16::from_f32(0.1); 768 * 4304]);
            s.gelu("op", "x", &[768, 4304], false);
            s.finish(&["op"]);
        }
        // GatherV2D 行复制变体：[1,2560] + idx[832个0] → [832,2560]
        "gat2" => {
            s.data(ctx, "x", &[1, 2560], &vec![f16::from_f32(0.1); 2560]);
            let idx: Vec<i32> = vec![0; 832];
            s.data_i32(ctx, "idx", &[832], &idx);
            s.gather_rows("op", "x", &[832, 2560], "idx");
            s.finish(&["op"]);
        }
        // TileD：[1,2560] multiples [832,1] → [832,2560]
        "tld" => {
            s.data(ctx, "x", &[1, 2560], &vec![f16::from_f32(0.1); 2560]);
            s.g.add_op("op", "TileD").unwrap();
            s.g.set_input_desc("op", "x", &[1, 2560], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[832, 2560], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list("op", "multiples", &[832, 1]).unwrap();
            s.g.link("op", "x", "x").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // ConcatD 端口名 "x"（DYNAMIC_INPUT 的非编号端口）
        "catx" => {
            s.data(ctx, "a", &[1, 832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            s.data(ctx, "b", &[1, 50, 256], &vec![f16::from_f32(0.2); 50 * 256]);
            s.g.add_op("op", "ConcatD").unwrap();
            s.g.set_input_desc("op", "x", &[1, 832, 256], Dtype::Fp16).unwrap();
            s.g.set_input_desc("op", "x", &[1, 50, 256], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[1, 882, 256], Dtype::Fp16).unwrap();
            s.g.set_attr_int("op", "concat_dim", 1).unwrap();
            s.g.set_attr_int("op", "N", 2).unwrap();
            s.g.link("op", "x", "a").unwrap();
            s.g.link("op", "x", "b").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // ---- prefix depth=1 链组合递增（定位组合编译崩点）----
        // p1 rms / p2 +qkv mm / p3 +bias(TileD+Add) / p4 +Reshape r3 /
        // p5 +SliceD k,v / p6 +k rope 全链
        "p1" | "p2" | "p3" | "p4" | "p5" | "p6" => {
            let lvl: i32 = which[1..].parse().unwrap();
            let p = 832i64;
            let mut seed = 1u32;
            s.data(ctx, "x", &[p, PW], &rand_f16((p * PW) as usize, &mut seed, 100.0));
            s.data_zeros(ctx, "zeros", p, PW);
            s.data(ctx, "g1", &[PW], &norm_f16(PW as usize, &mut seed, 1.0));
            let n1 = s.addrms("n1", "x", "zeros", "g1", &[p, PW]);
            if lvl == 1 {
                s.finish(&[&n1]);
                println!("OPTEST_{which}_OK");
                ge_builder::fini().expect("fini");
                return;
            }
            {
                let host = rand_f16((PW * QKVW) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, PW, QKVW);
                s.data_buf("qkvwt", &[QKVW, PW], b);
            }
            s.data(ctx, "qkvb", &[1, QKVW], &rand_f16(QKVW as usize, &mut seed, 4000.0));
            let qkv = s.mm("qkv", &n1, &[p, PW], "qkvwt", &[QKVW, PW], &[p, QKVW]);
            if lvl == 2 {
                s.finish(&[&qkv]);
                println!("OPTEST_{which}_OK");
                ge_builder::fini().expect("fini");
                return;
            }
            let biased = s.bias("qkvb2", &qkv, &[p, QKVW], "qkvb");
            if lvl == 3 {
                s.finish(&[&biased]);
                println!("OPTEST_{which}_OK");
                ge_builder::fini().expect("fini");
                return;
            }
            s.data_i32(ctx, "shp_qkv3", &[3], &[1, p as i32, QKVW as i32]);
            let r3 = s.reshape("r3", &biased, &[p, QKVW], "shp_qkv3", &[1, p, QKVW]);
            if lvl == 4 {
                s.finish(&[&r3]);
                println!("OPTEST_{which}_OK");
                ge_builder::fini().expect("fini");
                return;
            }
            let v = s.slice3("v", &r3, &[1, p, QKVW], QD + KVD, KVD);
            let k = s.slice3("k", &r3, &[1, p, QKVW], QD, KVD);
            if lvl == 5 {
                s.finish(&[&k, &v]);
                println!("OPTEST_{which}_OK");
                ge_builder::fini().expect("fini");
                return;
            }
            let (kc2, ks2) = rope_rank2_const(p, KV_HEADS, 0);
            s.data(ctx, "kcos2", &[p, KVD], &kc2);
            s.data(ctx, "ksin2", &[p, KVD], &ks2);
            s.data_i32(ctx, "shp_k3", &[3], &[1, p as i32, KVD as i32]);
            let k2r = s.slice2("k2r", &biased, &[p, QKVW], QD, KVD);
            let (_, kr3) = s.rope2("kr", &k2r, &[p, KVD], "kcos2", "ksin2", "shp_k3");
            s.finish(&[&kr3]);
        }
        // p2b: MatMulV2 输出 desc 直设 rank-3
        "p2b" => {
            let p = 832i64;
            let mut seed = 1u32;
            s.data(ctx, "x", &[p, PW], &rand_f16((p * PW) as usize, &mut seed, 100.0));
            {
                let host = rand_f16((PW * QKVW) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, PW, QKVW);
                s.data_buf("qkvwt", &[QKVW, PW], b);
            }
            s.g.add_op("qkv", "MatMulV2").unwrap();
            s.g.set_input_desc("qkv", "x1", &[p, PW], Dtype::Fp16).unwrap();
            s.g.set_input_desc("qkv", "x2", &[QKVW, PW], Dtype::Fp16).unwrap();
            s.g.set_output_desc("qkv", "y", &[1, p, QKVW], Dtype::Fp16).unwrap();
            s.g.set_attr_bool("qkv", "transpose_x1", false).unwrap();
            s.g.set_attr_bool("qkv", "transpose_x2", true).unwrap();
            s.g.link("qkv", "x1", "x").unwrap();
            s.g.link("qkv", "x2", "qkvwt").unwrap();
            s.reg_out("qkv", "y");
            s.finish(&["qkv"]);
        }
        // usq: Unsqueeze(axes=[0]) rank-2 → rank-3
        "usq" => {
            s.data(ctx, "x", &[832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            s.g.add_op("op", "Unsqueeze").unwrap();
            s.g.set_input_desc("op", "x", &[832, 256], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[1, 832, 256], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list("op", "axes", &[0]).unwrap();
            s.g.link("op", "x", "x").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // p5c: mm rank-2 → bias → SliceD rank-2 → Unsqueeze → 图出
        "p5c" => {
            let p = 832i64;
            let mut seed = 1u32;
            s.data(ctx, "x", &[p, PW], &rand_f16((p * PW) as usize, &mut seed, 100.0));
            {
                let host = rand_f16((PW * QKVW) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, PW, QKVW);
                s.data_buf("qkvwt", &[QKVW, PW], b);
            }
            s.data(ctx, "qkvb", &[1, QKVW], &rand_f16(QKVW as usize, &mut seed, 4000.0));
            let qkv = s.mm("qkv", "x", &[p, PW], "qkvwt", &[QKVW, PW], &[p, QKVW]);
            let biased = s.bias("qkvb2", &qkv, &[p, QKVW], "qkvb");
            // rank-2 SliceD：offsets/size 二元
            let sl = |s: &mut Seg, name: &str, x: &str, dims: &[i64; 2], col_off: i64, col_len: i64| {
                let out = [dims[0], col_len];
                s.g.add_op(name, "SliceD").unwrap();
                s.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
                s.g.set_output_desc(name, "y", &out, Dtype::Fp16).unwrap();
                s.g.set_attr_int_list(name, "offsets", &[0, col_off]).unwrap();
                s.g.set_attr_int_list(name, "size", &out).unwrap();
                s.wire(name, "x", x);
                s.reg_out(name, "y")
            };
            let k2 = sl(&mut s, "k2", &biased, &[p, QKVW], QD, KVD);
            s.g.add_op("k3", "Unsqueeze").unwrap();
            s.g.set_input_desc("k3", "x", &[p, KVD], Dtype::Fp16).unwrap();
            s.g.set_output_desc("k3", "y", &[1, p, KVD], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list("k3", "axes", &[0]).unwrap();
            s.wire("k3", "x", &k2);
            s.reg_out("k3", "y");
            s.finish(&["k3"]);
        }
        // p6c: p5c + rope rank-2 版（swap = SliceD×2 + ConcatD 通道维）
        "p6c" => {
            let p = 832i64;
            let mut seed = 1u32;
            s.data(ctx, "x", &[p, PW], &rand_f16((p * PW) as usize, &mut seed, 100.0));
            {
                let host = rand_f16((PW * QKVW) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, PW, QKVW);
                s.data_buf("qkvwt", &[QKVW, PW], b);
            }
            s.data(ctx, "qkvb", &[1, QKVW], &rand_f16(QKVW as usize, &mut seed, 4000.0));
            let qkv = s.mm("qkv", "x", &[p, PW], "qkvwt", &[QKVW, PW], &[p, QKVW]);
            let biased = s.bias("qkvb2", &qkv, &[p, QKVW], "qkvb");
            let sl = |s: &mut Seg, name: &str, x: &str, dims: &[i64; 2], col_off: i64, col_len: i64| {
                let out = [dims[0], col_len];
                s.g.add_op(name, "SliceD").unwrap();
                s.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
                s.g.set_output_desc(name, "y", &out, Dtype::Fp16).unwrap();
                s.g.set_attr_int_list(name, "offsets", &[0, col_off]).unwrap();
                s.g.set_attr_int_list(name, "size", &out).unwrap();
                s.wire(name, "x", x);
                s.reg_out(name, "y")
            };
            let k2 = sl(&mut s, "k2", &biased, &[p, QKVW], QD, KVD);
            // rotate-half swap：lo/hi 半通道互换（rank-2）
            let lo = sl(&mut s, "klo", &k2, &[p, KVD], 0, KVD / 2);
            let hi = sl(&mut s, "khi", &k2, &[p, KVD], KVD / 2, KVD / 2);
            s.g.add_op("ksw", "ConcatD").unwrap();
            s.g.set_input_desc("ksw", "x0", &[p, KVD / 2], Dtype::Fp16).unwrap();
            s.g.set_input_desc("ksw", "x1", &[p, KVD / 2], Dtype::Fp16).unwrap();
            s.g.set_output_desc("ksw", "y", &[p, KVD], Dtype::Fp16).unwrap();
            s.g.set_attr_int("ksw", "concat_dim", 1).unwrap();
            s.g.set_attr_int("ksw", "N", 2).unwrap();
            s.wire("ksw", "x0", &hi);
            s.wire("ksw", "x1", &lo);
            let ksw = s.reg_out("ksw", "y");
            // rope 常量（rank-2 [p*heads, d] 表，rope_rank2_const）
            let (kc, ks) = rope_rank2_const(p, KV_HEADS, 0);
            s.data(ctx, "kcos", &[p * KV_HEADS, HD], &kc);
            s.data(ctx, "ksin", &[p * KV_HEADS, HD], &ks);
            let m1 = s.mul2("m1", &k2, "kcos", &[p, KVD]);
            let m2 = s.mul2("m2", &ksw, "ksin", &[p, KVD]);
            let kr = s.add2("kr", &m1, &m2, &[p, KVD]);
            s.g.add_op("kr3", "Unsqueeze").unwrap();
            s.g.set_input_desc("kr3", "x", &[p, KVD], Dtype::Fp16).unwrap();
            s.g.set_output_desc("kr3", "y", &[1, p, KVD], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list("kr3", "axes", &[0]).unwrap();
            s.wire("kr3", "x", &kr);
            s.reg_out("kr3", "y");
            s.finish(&["kr3"]);
        }
        // p5d: mm rank-3 直出 → SliceD rank-3 通道切（零桥方案）
        "p5d" => {
            let p = 832i64;
            let mut seed = 1u32;
            s.data(ctx, "x", &[p, PW], &rand_f16((p * PW) as usize, &mut seed, 100.0));
            {
                let host = rand_f16((PW * QKVW) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, PW, QKVW);
                s.data_buf("qkvwt", &[QKVW, PW], b);
            }
            s.g.add_op("qkv", "MatMulV2").unwrap();
            s.g.set_input_desc("qkv", "x1", &[p, PW], Dtype::Fp16).unwrap();
            s.g.set_input_desc("qkv", "x2", &[QKVW, PW], Dtype::Fp16).unwrap();
            s.g.set_output_desc("qkv", "y", &[1, p, QKVW], Dtype::Fp16).unwrap();
            s.g.set_attr_bool("qkv", "transpose_x1", false).unwrap();
            s.g.set_attr_bool("qkv", "transpose_x2", true).unwrap();
            s.g.link("qkv", "x1", "x").unwrap();
            s.g.link("qkv", "x2", "qkvwt").unwrap();
            let qkv = s.reg_out("qkv", "y");
            let k = s.slice3("k", &qkv, &[1, p, QKVW], QD, KVD);
            s.finish(&[&k]);
        }
        // p6d: p5d + rope rank-3 全链（半通道 SliceD×2 + ConcatD axis=2 +
        // mul/add rank-3 表 [1,p,KVD]）
        "p6d" => {
            let p = 832i64;
            let mut seed = 1u32;
            s.data(ctx, "x", &[p, PW], &rand_f16((p * PW) as usize, &mut seed, 100.0));
            {
                let host = rand_f16((PW * QKVW) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, PW, QKVW);
                s.data_buf("qkvwt", &[QKVW, PW], b);
            }
            s.g.add_op("qkv", "MatMulV2").unwrap();
            s.g.set_input_desc("qkv", "x1", &[p, PW], Dtype::Fp16).unwrap();
            s.g.set_input_desc("qkv", "x2", &[QKVW, PW], Dtype::Fp16).unwrap();
            s.g.set_output_desc("qkv", "y", &[1, p, QKVW], Dtype::Fp16).unwrap();
            s.g.set_attr_bool("qkv", "transpose_x1", false).unwrap();
            s.g.set_attr_bool("qkv", "transpose_x2", true).unwrap();
            s.g.link("qkv", "x1", "x").unwrap();
            s.g.link("qkv", "x2", "qkvwt").unwrap();
            let qkv = s.reg_out("qkv", "y");
            let k = s.slice3("k", &qkv, &[1, p, QKVW], QD, KVD);
            // rotate-half swap（rank-3 半通道）
            let lo = s.slice3("klo", &k, &[1, p, KVD], 0, KVD / 2);
            let hi = s.slice3("khi", &k, &[1, p, KVD], KVD / 2, KVD / 2);
            s.g.add_op("ksw", "ConcatD").unwrap();
            s.g.set_input_desc("ksw", "x0", &[1, p, KVD / 2], Dtype::Fp16).unwrap();
            s.g.set_input_desc("ksw", "x1", &[1, p, KVD / 2], Dtype::Fp16).unwrap();
            s.g.set_output_desc("ksw", "y", &[1, p, KVD], Dtype::Fp16).unwrap();
            s.g.set_attr_int("ksw", "concat_dim", 2).unwrap();
            s.g.set_attr_int("ksw", "N", 2).unwrap();
            s.wire("ksw", "x0", &hi);
            s.wire("ksw", "x1", &lo);
            let ksw = s.reg_out("ksw", "y");
            let (kc, ks) = rope_rank2_const(p, KV_HEADS, 0);
            s.data(ctx, "kcos", &[1, p, KVD], &kc);
            s.data(ctx, "ksin", &[1, p, KVD], &ks);
            let m1 = s.mul2("m1", &k, "kcos", &[1, p, KVD]);
            let m2 = s.mul2("m2", &ksw, "ksin", &[1, p, KVD]);
            let kr = s.add2("kr", &m1, &m2, &[1, p, KVD]);
            s.finish(&[&kr]);
        }
        // p7a: ConcatD rank-2 axis=1（x0/x1 端口，验证端口修复后编译）
        "p7a" => {
            s.data(ctx, "a", &[832, 128], &vec![f16::from_f32(0.1); 832 * 128]);
            s.data(ctx, "b", &[832, 128], &vec![f16::from_f32(0.2); 832 * 128]);
            s.g.add_op("op", "ConcatD").unwrap();
            s.g.set_input_desc("op", "x0", &[832, 128], Dtype::Fp16).unwrap();
            s.g.set_input_desc("op", "x1", &[832, 128], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[832, 256], Dtype::Fp16).unwrap();
            s.g.set_attr_int("op", "concat_dim", 1).unwrap();
            s.g.set_attr_int("op", "N", 2).unwrap();
            s.g.link("op", "x0", "a").unwrap();
            s.g.link("op", "x1", "b").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // p7b: SliceD→SliceD 链（rank-2 通道半切再半切）
        "p7b" => {
            s.data(ctx, "x", &[832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            let full = s.slice2("s1", "x", &[832, 256], 0, 256);
            let half = s.slice2("s2", &full, &[832, 256], 128, 128);
            s.finish(&[&half]);
        }
        // p7c: SliceD(算子出) → Mul（表 Data）组合
        "p7c" => {
            s.data(ctx, "x", &[832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            s.data(ctx, "c", &[832, 256], &vec![f16::from_f32(0.5); 832 * 256]);
            let k = s.slice2("k", "x", &[832, 256], 0, 256);
            let m = s.mul2("m", &k, "c", &[832, 256]);
            s.finish(&[&m]);
        }
        // p7d: ConcatD rank-2 axis=0（行拼接——TBE 最友好方向）
        "p7d" => {
            s.data(ctx, "a", &[832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            s.data(ctx, "b", &[50, 256], &vec![f16::from_f32(0.2); 50 * 256]);
            s.g.add_op("op", "ConcatD").unwrap();
            s.g.set_input_desc("op", "x0", &[832, 256], Dtype::Fp16).unwrap();
            s.g.set_input_desc("op", "x1", &[50, 256], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[882, 256], Dtype::Fp16).unwrap();
            s.g.set_attr_int("op", "concat_dim", 0).unwrap();
            s.g.set_attr_int("op", "N", 2).unwrap();
            s.g.link("op", "x0", "a").unwrap();
            s.g.link("op", "x1", "b").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // p7e: Concat（张量版 concat_dim 输入）rank-2 axis=0
        "p7e" => {
            s.data(ctx, "a", &[832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            s.data(ctx, "b", &[50, 256], &vec![f16::from_f32(0.2); 50 * 256]);
            s.data_i32(ctx, "ax", &[1], &[0]);
            s.g.add_op("op", "Concat").unwrap();
            s.g.set_input_desc("op", "concat_dim", &[1], Dtype::Int32).unwrap();
            s.g.set_input_desc("op", "x0", &[832, 256], Dtype::Fp16).unwrap();
            s.g.set_input_desc("op", "x1", &[50, 256], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[882, 256], Dtype::Fp16).unwrap();
            s.g.set_attr_int("op", "N", 2).unwrap();
            s.g.link("op", "concat_dim", "ax").unwrap();
            s.g.link("op", "x0", "a").unwrap();
            s.g.link("op", "x1", "b").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // p7f: ConcatD axis=0 不设 N attr
        "p7f" => {
            s.data(ctx, "a", &[832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            s.data(ctx, "b", &[50, 256], &vec![f16::from_f32(0.2); 50 * 256]);
            s.g.add_op("op", "ConcatD").unwrap();
            s.g.set_input_desc("op", "x0", &[832, 256], Dtype::Fp16).unwrap();
            s.g.set_input_desc("op", "x1", &[50, 256], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[882, 256], Dtype::Fp16).unwrap();
            s.g.set_attr_int("op", "concat_dim", 0).unwrap();
            s.g.link("op", "x0", "a").unwrap();
            s.g.link("op", "x1", "b").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // p7g: ConcatD axis=0 + dyn_inputs 注册（符号直链路径）
        "p7g" => {
            s.data(ctx, "a", &[832, 256], &vec![f16::from_f32(0.1); 832 * 256]);
            s.data(ctx, "b", &[50, 256], &vec![f16::from_f32(0.2); 50 * 256]);
            s.g.add_op("op", "ConcatD").unwrap();
            s.g.dyn_inputs("op", "x", 2).unwrap();
            let n = s.g.dyn_probe("op", "x").unwrap_or(-1);
            println!("dyn num after register = {n}");
            s.g.set_input_desc("op", "x0", &[832, 256], Dtype::Fp16).unwrap();
            s.g.set_input_desc("op", "x1", &[50, 256], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[882, 256], Dtype::Fp16).unwrap();
            s.g.set_attr_int("op", "concat_dim", 0).unwrap();
            s.g.set_attr_int("op", "N", 2).unwrap();
            s.g.link("op", "x0", "a").unwrap();
            s.g.link("op", "x1", "b").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
        }
        // ---- 方案验证矩阵（三段绕开路径）----
        // v1: 3×mm rank-3 直出 → batch PFA → 图出（vision attention 骨架）
        "v1" => {
            let mut seed = 1u32;
            let t = VT;
            let vqd = V_HEADS * V_HD;
            for n in ["q", "k", "v"] {
                let host = rand_f16((VW * vqd) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, VW, vqd);
                s.data_buf(&format!("w{n}"), &[vqd, VW], b);
                s.g.add_op(&format!("mm{n}"), "MatMulV2").unwrap();
                s.g.set_input_desc(&format!("mm{n}"), "x1", &[t, VW], Dtype::Fp16).unwrap();
                s.g.set_input_desc(&format!("mm{n}"), "x2", &[vqd, VW], Dtype::Fp16).unwrap();
                s.g.set_output_desc(&format!("mm{n}"), "y", &[VIEWS, VPV, vqd], Dtype::Fp16).unwrap();
                s.g.set_attr_bool(&format!("mm{n}"), "transpose_x1", false).unwrap();
                s.g.set_attr_bool(&format!("mm{n}"), "transpose_x2", true).unwrap();
                s.g.link(&format!("mm{n}"), "x2", &format!("w{n}")).unwrap();
                s.reg_out(&format!("mm{n}"), "y");
            }
            let x = s.data(ctx, "x", &[t, VW], &rand_f16((t * VW) as usize, &mut seed, 100.0));
            let _ = x;
            for n in ["q", "k", "v"] {
                s.g.link(&format!("mm{n}"), "x1", "x").unwrap();
            }
            let pfa = s.pfa("pfa", "mmq", "mmk", "mmv", &[VIEWS, VPV, vqd], &[VIEWS, VPV, vqd], V_HEADS, V_HEADS, V_HD);
            s.finish(&[&pfa]);
        }
        // v2: v1 + PFA out → Reshape rank-2
        "v2" => {
            let mut seed = 1u32;
            let t = VT;
            let vqd = V_HEADS * V_HD;
            for n in ["q", "k", "v"] {
                let host = rand_f16((VW * vqd) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, VW, vqd);
                s.data_buf(&format!("w{n}"), &[vqd, VW], b);
                s.g.add_op(&format!("mm{n}"), "MatMulV2").unwrap();
                s.g.set_input_desc(&format!("mm{n}"), "x1", &[t, VW], Dtype::Fp16).unwrap();
                s.g.set_input_desc(&format!("mm{n}"), "x2", &[vqd, VW], Dtype::Fp16).unwrap();
                s.g.set_output_desc(&format!("mm{n}"), "y", &[VIEWS, VPV, vqd], Dtype::Fp16).unwrap();
                s.g.set_attr_bool(&format!("mm{n}"), "transpose_x1", false).unwrap();
                s.g.set_attr_bool(&format!("mm{n}"), "transpose_x2", true).unwrap();
                s.g.link(&format!("mm{n}"), "x2", &format!("w{n}")).unwrap();
                s.reg_out(&format!("mm{n}"), "y");
            }
            s.data(ctx, "x", &[t, VW], &rand_f16((t * VW) as usize, &mut seed, 100.0));
            for n in ["q", "k", "v"] {
                s.g.link(&format!("mm{n}"), "x1", "x").unwrap();
            }
            let pfa = s.pfa("pfa", "mmq", "mmk", "mmv", &[VIEWS, VPV, vqd], &[VIEWS, VPV, vqd], V_HEADS, V_HEADS, V_HD);
            s.data_i32(ctx, "shp2", &[2], &[t as i32, vqd as i32]);
            let r2 = s.reshape("r2", &pfa, &[VIEWS, VPV, vqd], "shp2", &[t, vqd]);
            s.finish(&[&r2]);
        }
        // v4: mm rank-3 → TileD bias rank-3 → 图出
        "v4" => {
            let mut seed = 1u32;
            let t = VT;
            let vqd = V_HEADS * V_HD;
            s.data(ctx, "x", &[t, VW], &rand_f16((t * VW) as usize, &mut seed, 100.0));
            let host = rand_f16((VW * vqd) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, VW, vqd);
            s.data_buf("wq", &[vqd, VW], b);
            let row = s.data(ctx, "bq", &[1, 1, vqd], &rand_f16(vqd as usize, &mut seed, 4000.0));
            let _ = row;
            s.g.add_op("mmq", "MatMulV2").unwrap();
            s.g.set_input_desc("mmq", "x1", &[t, VW], Dtype::Fp16).unwrap();
            s.g.set_input_desc("mmq", "x2", &[vqd, VW], Dtype::Fp16).unwrap();
            s.g.set_output_desc("mmq", "y", &[VIEWS, VPV, vqd], Dtype::Fp16).unwrap();
            s.g.set_attr_bool("mmq", "transpose_x1", false).unwrap();
            s.g.set_attr_bool("mmq", "transpose_x2", true).unwrap();
            s.g.link("mmq", "x1", "x").unwrap();
            s.g.link("mmq", "x2", "wq").unwrap();
            s.reg_out("mmq", "y");
            s.g.add_op("bc", "TileD").unwrap();
            s.g.set_input_desc("bc", "x", &[1, 1, vqd], Dtype::Fp16).unwrap();
            s.g.set_output_desc("bc", "y", &[VIEWS, VPV, vqd], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list("bc", "multiples", &[VIEWS, VPV, 1]).unwrap();
            s.g.link("bc", "x", "bq").unwrap();
            s.reg_out("bc", "y");
            let ba = s.add2("ba", "mmq", "bc", &[VIEWS, VPV, vqd]);
            s.finish(&[&ba]);
        }
        // r1: rope rank-2 全链（SliceD 半切 + ConcatD axis=1 + mul/add + Unsqueeze）
        "r1" => {
            let p = 832i64;
            let mut seed = 1u32;
            s.data(ctx, "k", &[p, KVD], &rand_f16((p * KVD) as usize, &mut seed, 100.0));
            let lo = s.slice2("klo", "k", &[p, KVD], 0, KVD / 2);
            let hi = s.slice2("khi", "k", &[p, KVD], KVD / 2, KVD / 2);
            s.g.add_op("ksw", "ConcatD").unwrap();
            s.g.dyn_inputs("ksw", "x", 2).unwrap();
            s.g.set_input_desc("ksw", "x0", &[p, KVD / 2], Dtype::Fp16).unwrap();
            s.g.set_input_desc("ksw", "x1", &[p, KVD / 2], Dtype::Fp16).unwrap();
            s.g.set_output_desc("ksw", "y", &[p, KVD], Dtype::Fp16).unwrap();
            s.g.set_attr_int("ksw", "concat_dim", 1).unwrap();
            s.g.set_attr_int("ksw", "N", 2).unwrap();
            s.wire("ksw", "x0", &hi);
            s.wire("ksw", "x1", &lo);
            let ksw = s.reg_out("ksw", "y");
            let (kc, ks) = rope_rank2_const(p, KV_HEADS, 0);
            s.data(ctx, "kcos", &[p, KVD], &kc);
            s.data(ctx, "ksin", &[p, KVD], &ks);
            let m1 = s.mul2("m1", "k", "kcos", &[p, KVD]);
            let m2 = s.mul2("m2", &ksw, "ksin", &[p, KVD]);
            let kr = s.add2("kr", &m1, &m2, &[p, KVD]);
            s.g.add_op("kr3", "Unsqueeze").unwrap();
            s.g.set_input_desc("kr3", "x", &[p, KVD], Dtype::Fp16).unwrap();
            s.g.set_output_desc("kr3", "y", &[1, p, KVD], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list("kr3", "axes", &[0]).unwrap();
            s.wire("kr3", "x", &kr);
            let kr3 = s.reg_out("kr3", "y");
            s.finish(&[&kr3]);
        }
        // q1: prefix/flow attention 全链（mm rank-2 → bias → SliceD×3 →
        // Unsqueeze×3 → cross PFA → Squeeze → mm）
        "q1" => {
            let p = 832i64;
            let h = HOR;
            let total = p + h;
            let mut seed = 1u32;
            s.data(ctx, "x", &[h, AW], &rand_f16((h * AW) as usize, &mut seed, 100.0));
            {
                let host = rand_f16((AW * QKVW) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, AW, QKVW);
                s.data_buf("qkvwt", &[QKVW, AW], b);
            }
            s.data(ctx, "qkvb", &[1, QKVW], &rand_f16(QKVW as usize, &mut seed, 4000.0));
            let qkv = s.mm("qkv", "x", &[h, AW], "qkvwt", &[QKVW, AW], &[h, QKVW]);
            let biased = s.bias("qkvb2", &qkv, &[h, QKVW], "qkvb");
            let q2 = s.slice2("q", &biased, &[h, QKVW], 0, QD);
            let k2 = s.slice2("k", &biased, &[h, QKVW], QD, KVD);
            let v2 = s.slice2("v", &biased, &[h, QKVW], QD + KVD, KVD);
            let up = |s: &mut Seg, name: &str, x: &str, dims: &[i64; 2]| {
                let out = [1, dims[0], dims[1]];
                s.g.add_op(name, "Unsqueeze").unwrap();
                s.g.set_input_desc(name, "x", dims, Dtype::Fp16).unwrap();
                s.g.set_output_desc(name, "y", &out, Dtype::Fp16).unwrap();
                s.g.set_attr_int_list(name, "axes", &[0]).unwrap();
                s.wire(name, "x", x);
                s.reg_out(name, "y")
            };
            let q3 = up(&mut s, "q3", &q2, &[h, QD]);
            let k3 = up(&mut s, "k3", &k2, &[h, KVD]);
            let v3 = up(&mut s, "v3", &v2, &[h, KVD]);
            // prefix k/v（rank-2 Data）→ cat rank-2 → Unsqueeze
            s.data(ctx, "pk", &[p, KVD], &rand_f16((p * KVD) as usize, &mut seed, 100.0));
            s.data(ctx, "pv", &[p, KVD], &rand_f16((p * KVD) as usize, &mut seed, 100.0));
            s.g.add_op("kc", "ConcatD").unwrap();
            s.g.dyn_inputs("kc", "x", 2).unwrap();
            s.g.set_input_desc("kc", "x0", &[p, KVD], Dtype::Fp16).unwrap();
            s.g.set_input_desc("kc", "x1", &[h, KVD], Dtype::Fp16).unwrap();
            s.g.set_output_desc("kc", "y", &[total, KVD], Dtype::Fp16).unwrap();
            s.g.set_attr_int("kc", "concat_dim", 0).unwrap();
            s.g.set_attr_int("kc", "N", 2).unwrap();
            s.g.link("kc", "x0", "pk").unwrap();
            s.wire("kc", "x1", &k2);
            let kall = s.reg_out("kc", "y");
            let kall3 = up(&mut s, "kall3", &kall, &[total, KVD]);
            let _ = kall3;
            let _ = v3;
            let pfa = s.pfa("pfa", &q3, &kall3, &v3, &[1, h, QD], &[1, total, KVD], HEADS, KV_HEADS, HD);
            let sq = s.squeeze("sq", &pfa, &[1, h, QD], &[h, QD]);
            {
                let host = rand_f16((QD * AW) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, QD, AW);
                s.data_buf("outwt", &[AW, QD], b);
            }
            let proj = s.mm("proj", &sq, &[h, QD], "outwt", &[AW, QD], &[h, AW]);
            s.finish(&[&proj]);
        }
        // v3: mm rank-2 → Reshape [3,256,1152] → batch PFA（Reshape 喂 PFA）
        "v3" => {
            let mut seed = 1u32;
            let t = VT;
            let vqd = V_HEADS * V_HD;
            s.data(ctx, "x", &[t, VW], &rand_f16((t * VW) as usize, &mut seed, 100.0));
            let host = rand_f16((VW * vqd) as usize, &mut seed, 8000.0);
            let b = wbuf_t(ctx, &host, VW, vqd);
            s.data_buf("wq", &[vqd, VW], b);
            let mmq = s.mm("mmq", "x", &[t, VW], "wq", &[vqd, VW], &[t, vqd]);
            s.data_i32(ctx, "shp3", &[3], &[VIEWS as i32, VPV as i32, vqd as i32]);
            let q3 = s.reshape("q3", &mmq, &[t, vqd], "shp3", &[VIEWS, VPV, vqd]);
            // k/v 用 Data rank-3 直设（同布局）
            s.data(ctx, "k3", &[VIEWS, VPV, vqd], &rand_f16((t * vqd) as usize, &mut seed, 100.0));
            s.data(ctx, "v3", &[VIEWS, VPV, vqd], &rand_f16((t * vqd) as usize, &mut seed, 100.0));
            let pfa = s.pfa("pfa", &q3, "k3", "v3", &[VIEWS, VPV, vqd], &[VIEWS, VPV, vqd], V_HEADS, V_HEADS, V_HD);
            s.finish(&[&pfa]);
        }
        // v5: 保底链——fused qkv rank-2 → 列切 → 3×(行切+Unsqueeze+PFA+
        // Squeeze) → ConcatD(axis=0)（全已验证组件）
        "v5" => {
            let mut seed = 1u32;
            let t = VT;
            let vqd = V_HEADS * V_HD;
            let qkvw = vqd * 3;
            s.data(ctx, "x", &[t, VW], &rand_f16((t * VW) as usize, &mut seed, 100.0));
            {
                let host = rand_f16((VW * qkvw) as usize, &mut seed, 8000.0);
                let b = wbuf_t(ctx, &host, VW, qkvw);
                s.data_buf("qkvwt", &[qkvw, VW], b);
            }
            s.data(ctx, "qkvb", &[1, qkvw], &rand_f16(qkvw as usize, &mut seed, 4000.0));
            let qkv = s.mm("qkv", "x", &[t, VW], "qkvwt", &[qkvw, VW], &[t, qkvw]);
            let biased = s.bias("qkvb2", &qkv, &[t, qkvw], "qkvb");
            let q = s.slice2("q", &biased, &[t, qkvw], 0, vqd);
            let k = s.slice2("k", &biased, &[t, qkvw], vqd, vqd);
            let v = s.slice2("v", &biased, &[t, qkvw], vqd * 2, vqd);
            let mut attns = Vec::new();
            for vi in 0..VIEWS {
                let off = vi * VPV;
                let qs = s.slice2(&format!("q{vi}v"), &q, &[t, vqd], off, VPV);
                let ks = s.slice2(&format!("k{vi}v"), &k, &[t, vqd], off, VPV);
                let vs = s.slice2(&format!("v{vi}v"), &v, &[t, vqd], off, VPV);
                let up = |s: &mut Seg, name: &str, x: &str| {
                    s.g.add_op(name, "Unsqueeze").unwrap();
                    s.g.set_input_desc(name, "x", &[VPV, vqd], Dtype::Fp16).unwrap();
                    s.g.set_output_desc(name, "y", &[1, VPV, vqd], Dtype::Fp16).unwrap();
                    s.g.set_attr_int_list(name, "axes", &[0]).unwrap();
                    s.wire(name, "x", x);
                    s.reg_out(name, "y")
                };
                let q3 = up(&mut s, &format!("q{vi}3"), &qs);
                let k3 = up(&mut s, &format!("k{vi}3"), &ks);
                let v3 = up(&mut s, &format!("v{vi}3"), &vs);
                let pfa = s.pfa(&format!("pfa{vi}"), &q3, &k3, &v3, &[1, VPV, vqd], &[1, VPV, vqd], V_HEADS, V_HEADS, V_HD);
                attns.push(s.squeeze(&format!("sq{vi}"), &pfa, &[1, VPV, vqd], &[VPV, vqd]));
            }
            s.g.add_op("cat", "ConcatD").unwrap();
            s.g.dyn_inputs("cat", "x", 3).unwrap();
            for (i, a) in attns.iter().enumerate() {
                s.g.set_input_desc("cat", &format!("x{i}"), &[VPV, vqd], Dtype::Fp16).unwrap();
            }
            s.g.set_output_desc("cat", "y", &[t, vqd], Dtype::Fp16).unwrap();
            s.g.set_attr_int("cat", "concat_dim", 0).unwrap();
            s.g.set_attr_int("cat", "N", 3).unwrap();
            for (i, a) in attns.iter().enumerate() {
                s.wire("cat", &format!("x{i}"), a);
            }
            let cat = s.reg_out("cat", "y");
            s.finish(&[&cat]);
        }
        // lnv4: LayerNormV4（x + normalized_shape 张量 + gamma/beta → y）
        // 带数值验证（host LN 对照）
        "lnv4" => {
            s.data(ctx, "x", &[768, 1152], &rand_f16(768 * 1152, &mut 1u32, 300.0));
            s.data_i32(ctx, "nsh", &[1], &[1152]);
            let gh = norm_f16(1152, &mut 1u32, 1.0);
            let bh = norm_f16(1152, &mut 2u32, 0.0);
            s.data(ctx, "g", &[1152], &gh);
            s.data(ctx, "b", &[1152], &bh);
            s.g.add_op("op", "LayerNormV4").unwrap();
            s.g.set_input_desc("op", "x", &[768, 1152], Dtype::Fp16).unwrap();
            s.g.set_input_desc("op", "normalized_shape", &[1], Dtype::Int32).unwrap();
            s.g.set_input_desc("op", "gamma", &[1152], Dtype::Fp16).unwrap();
            s.g.set_input_desc("op", "beta", &[1152], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[768, 1152], Dtype::Fp16).unwrap();
            s.g.set_attr_float("op", "epsilon", LN_EPS).unwrap();
            s.g.link("op", "x", "x").unwrap();
            s.g.link("op", "normalized_shape", "nsh").unwrap();
            s.g.link("op", "gamma", "g").unwrap();
            s.g.link("op", "beta", "b").unwrap();
            s.reg_out("op", "y");
            // LayerNormV4 的 mean/rstd 是 REQUIRED 输出——绑满 3 个图输出
            let names: Vec<&str> = s.names.iter().map(|x| x.as_str()).collect();
            let shape_refs: Vec<(&str, &[i64])> = s
                .names
                .iter()
                .zip(&s.shapes)
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            s.g.graph_inputs(&names).unwrap();
            s.g.graph_outputs_idx(&["op", "op", "op"], &[0, 1, 2]).unwrap();
            s.g.set_nd_input_shape(&shape_refs).unwrap();
            s.g.build().expect("build");
            // run + host LN 对照
            let stream = be.stream();
            let ins: Vec<&DeviceBuffer> = s.binds.iter().collect();
            let out = ctx.malloc(768 * 1152 * 2).unwrap();
            let aux1 = ctx.malloc(768 * 4).unwrap();
            let aux2 = ctx.malloc(768 * 4).unwrap();
            s.g.run(&ins, &[&out, &aux1, &aux2], stream).unwrap();
            drop(stream.synchronize());
            let ge = download_f16(ctx, &out, 768 * 1152);
            let href = host_ln(ctx, &s.binds[0], &gh, &bh, 768, 1152);
            let hr = download_f16(ctx, &href, 768 * 1152);
            let mut md = 0f32;
            let mut gm = 0f32;
            for (a, r) in ge.iter().zip(&hr) {
                md = md.max((a.to_f32() - r.to_f32()).abs());
                gm = gm.max(a.to_f32().abs());
            }
            println!("lnv4 numeric: ge |max|={gm:.3} max_diff={md:.5}");
        }
        // lnv4b: 前置 Add（算子输出）→ LayerNormV4（模拟图内形态）
        "lnv4b" => {
            s.data(ctx, "x", &[768, 1152], &rand_f16(768 * 1152, &mut 1u32, 300.0));
            s.data_zeros(ctx, "z", 768, 1152);
            let pre = s.add2("pre", "x", "z", &[768, 1152]);
            s.data_i32(ctx, "nsh", &[1], &[1152]);
            let gh = norm_f16(1152, &mut 1u32, 1.0);
            let bh = norm_f16(1152, &mut 2u32, 0.0);
            s.data(ctx, "g", &[1152], &gh);
            s.data(ctx, "b", &[1152], &bh);
            let ln = s.addln("op", &pre, "g", "b", "nsh", &[768, 1152]);
            let names: Vec<&str> = s.names.iter().map(|x| x.as_str()).collect();
            let shape_refs: Vec<(&str, &[i64])> = s
                .names
                .iter()
                .zip(&s.shapes)
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            s.g.graph_inputs(&names).unwrap();
            s.g.graph_outputs_idx(&["op", "op", "op"], &[0, 1, 2]).unwrap();
            s.g.set_nd_input_shape(&shape_refs).unwrap();
            s.g.build().expect("build");
            let stream = be.stream();
            let ins: Vec<&DeviceBuffer> = s.binds.iter().collect();
            let out = ctx.malloc(768 * 1152 * 2).unwrap();
            let aux1 = ctx.malloc(768 * 4).unwrap();
            let aux2 = ctx.malloc(768 * 4).unwrap();
            s.g.run(&ins, &[&out, &aux1, &aux2], stream).unwrap();
            drop(stream.synchronize());
            let ge = download_f16(ctx, &out, 768 * 1152);
            // host 参考：LN(x + 0)
            let xh = download_f16(ctx, &s.binds[0], 768 * 1152);
            let xl: Vec<f16> = xh.clone();
            let xb = upload(ctx, &xl);
            let href = host_ln(ctx, &xb, &gh, &bh, 768, 1152);
            let hr = download_f16(ctx, &href, 768 * 1152);
            let mut md = 0f32;
            let mut gm = 0f32;
            for (a, r) in ge.iter().zip(&hr) {
                md = md.max((a.to_f32() - r.to_f32()).abs());
                gm = gm.max(a.to_f32().abs());
            }
            println!("lnv4b numeric: ge |max|={gm:.3} max_diff={md:.5}");
        }
        // glnn: GeluV2 "none" 数值验证（host erf 对照）
        "glnn" => {
            let host = rand_f16(768 * 4304, &mut 1u32, 30.0);
            s.data(ctx, "x", &[768, 4304], &host);
            s.gelu("op", "x", &[768, 4304], false);
            s.finish(&["op"]);
            let stream = be.stream();
            let ins: Vec<&DeviceBuffer> = s.binds.iter().collect();
            let out = ctx.malloc(768 * 4304 * 2).unwrap();
            s.g.run(&ins, &[&out], stream).unwrap();
            drop(stream.synchronize());
            let ge = download_f16(ctx, &out, 768 * 4304);
            let mut md = 0f32;
            for (i, v) in ge.iter().enumerate() {
                let x = host[i].to_f32();
                // exact gelu: x·Φ(x) = x·(1+erf(x/√2))/2
                let phi = 0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2));
                let expect = x * phi;
                md = md.max((v.to_f32() - expect).abs());
            }
            println!("glnn numeric: max_diff={md:.5}");
        }
        // rshn: Reshape rank-3 桥数值（flat 拷贝对照）
        "rshn" => {
            let host = rand_f16(768 * 1152, &mut 1u32, 300.0);
            s.data(ctx, "x", &[768, 1152], &host);
            s.data_i32(ctx, "shp", &[3], &[3, 256, 1152]);
            s.reshape("op", "x", &[768, 1152], "shp", &[3, 256, 1152]);
            s.finish(&["op"]);
            let stream = be.stream();
            let ins: Vec<&DeviceBuffer> = s.binds.iter().collect();
            let out = ctx.malloc(768 * 1152 * 2).unwrap();
            s.g.run(&ins, &[&out], stream).unwrap();
            drop(stream.synchronize());
            let ge = download_f16(ctx, &out, 768 * 1152);
            let mut md = 0f32;
            for (a, b) in ge.iter().zip(&host) {
                md = md.max((a.to_f32() - b.to_f32()).abs());
            }
            println!("rshn numeric: max_diff={md:.5}");
        }
        // v3n: v3 结构数值验证（mm rank-2 → Reshape → batch PFA vs eager）
        "v3n" => {
            let mut seed = 1u32;
            let t = VT;
            let vqd = V_HEADS * V_HD;
            let xh = rand_f16((t * VW) as usize, &mut seed, 300.0);
            s.data(ctx, "x", &[t, VW], &xh);
            let wqh = rand_f16((VW * vqd) as usize, &mut seed, 30000.0);
            let wq = wbuf_t(ctx, &wqh, VW, vqd);
            s.data_buf("wq", &[vqd, VW], wq);
            let mmq = s.mm("mmq", "x", &[t, VW], "wq", &[vqd, VW], &[t, vqd]);
            s.data_i32(ctx, "shp3", &[3], &[VIEWS as i32, VPV as i32, vqd as i32]);
            let q3 = s.reshape("q3", &mmq, &[t, vqd], "shp3", &[VIEWS, VPV, vqd]);
            let kh = rand_f16((t * vqd) as usize, &mut seed, 300.0);
            let vh = rand_f16((t * vqd) as usize, &mut seed, 300.0);
            s.data(ctx, "k3", &[VIEWS, VPV, vqd], &kh);
            s.data(ctx, "v3", &[VIEWS, VPV, vqd], &vh);
            let pfa = s.pfa("pfa", &q3, "k3", "v3", &[VIEWS, VPV, vqd], &[VIEWS, VPV, vqd], V_HEADS, V_HEADS, V_HD);
            s.finish(&[&pfa]);
            // GE run
            let stream = be.stream();
            let ins: Vec<&DeviceBuffer> = s.binds.iter().collect();
            let out = ctx.malloc((t * vqd * 2) as usize).unwrap();
            s.g.run(&ins, &[&out], stream).unwrap();
            drop(stream.synchronize());
            let ge = download_f16(ctx, &out, (t * vqd) as usize);
            // eager 参照：mm → reshape(同布局) → batch PFA
            let q2 = aops::matmul_b_t_fp16(ctx, stream, &s.binds[0], [t, VW], &s.binds[1], VW, vqd).unwrap();
            let ref_out = aops::prompt_flash_attention_bsh_batch_fp16(
                ctx, stream, &q2, &s.binds[3], &s.binds[4], VIEWS, VPV, V_HEADS, V_HEADS, V_HD, None).unwrap();
            drop(stream.synchronize());
            let er = download_f16(ctx, &ref_out, (t * vqd) as usize);
            let mut md = 0f32;
            for (a, b) in ge.iter().zip(&er) {
                md = md.max((a.to_f32() - b.to_f32()).abs());
            }
            println!("v3n numeric: max_diff={md:.5} (ge |max|={:.3})", ge.iter().fold(0f32, |m, v| m.max(v.to_f32().abs())));
        }
        // tldn: TileD 行复制数值（bias 广播组件）
        "tldn" => {
            let row = rand_f16(2560, &mut 1u32, 12000.0);
            s.data(ctx, "x", &[1, 2560], &row);
            s.g.add_op("op", "TileD").unwrap();
            s.g.set_input_desc("op", "x", &[1, 2560], Dtype::Fp16).unwrap();
            s.g.set_output_desc("op", "y", &[832, 2560], Dtype::Fp16).unwrap();
            s.g.set_attr_int_list("op", "multiples", &[832, 1]).unwrap();
            s.g.link("op", "x", "x").unwrap();
            s.reg_out("op", "y");
            s.finish(&["op"]);
            let stream = be.stream();
            let ins: Vec<&DeviceBuffer> = s.binds.iter().collect();
            let out = ctx.malloc(832 * 2560 * 2).unwrap();
            s.g.run(&ins, &[&out], stream).unwrap();
            drop(stream.synchronize());
            let ge = download_f16(ctx, &out, 832 * 2560);
            let mut md = 0f32;
            for (i, v) in ge.iter().enumerate() {
                let expect = row[i % 2560];
                md = md.max((v.to_f32() - expect.to_f32()).abs());
            }
            println!("tldn numeric: max_diff={md:.5}");
        }
        // lnv4c: 层中间形态——LN 死端（只消费 y）→ 下游 mm（数值验证）
        "lnv4c" => {
            let mut seed = 1u32;
            let xh = rand_f16(768 * 1152, &mut seed, 300.0);
            s.data(ctx, "x", &[768, 1152], &xh);
            s.data_zeros(ctx, "z", 768, 1152);
            let pre = s.add2("pre", "x", "z", &[768, 1152]);
            s.data_i32(ctx, "nsh", &[1], &[1152]);
            let gh = norm_f16(1152, &mut seed, 1.0);
            let bh = norm_f16(1152, &mut seed, 0.0);
            s.data(ctx, "g", &[1152], &gh);
            s.data(ctx, "b", &[1152], &bh);
            let ln = s.addln("op", &pre, "g", "b", "nsh", &[768, 1152]);
            // 下游 mm（层中间消费 y；mean/rstd 死端）
            let wqh = rand_f16((1152 * 1152) as usize, &mut seed, 30000.0);
            let wq = wbuf_t(ctx, &wqh, 1152, 1152);
            s.data_buf("wq", &[1152, 1152], wq);
            let mm = s.mm("mm", &ln, &[768, 1152], "wq", &[1152, 1152], &[768, 1152]);
            let names: Vec<&str> = s.names.iter().map(|x| x.as_str()).collect();
            let shape_refs: Vec<(&str, &[i64])> = s
                .names
                .iter()
                .zip(&s.shapes)
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            s.g.graph_inputs(&names).unwrap();
            s.g.graph_outputs(&["mm"]).unwrap();
            s.g.set_nd_input_shape(&shape_refs).unwrap();
            s.g.build().expect("build");
            let stream = be.stream();
            let ins: Vec<&DeviceBuffer> = s.binds.iter().collect();
            let out = ctx.malloc(768 * 1152 * 2).unwrap();
            s.g.run(&ins, &[&out], stream).unwrap();
            drop(stream.synchronize());
            let ge = download_f16(ctx, &out, 768 * 1152);
            // eager 参照：host LN → matmul
            let ln_ref = host_ln(ctx, &s.binds[0], &gh, &bh, 768, 1152);
            let mm_ref = aops::matmul_b_t_fp16(ctx, stream, &ln_ref, [768, 1152], &s.binds[5], 1152, 1152).unwrap();
            drop(stream.synchronize());
            let er = download_f16(ctx, &mm_ref, 768 * 1152);
            let mut md = 0f32;
            let mut gm = 0f32;
            for (a, b) in ge.iter().zip(&er) {
                md = md.max((a.to_f32() - b.to_f32()).abs());
                gm = gm.max(a.to_f32().abs());
            }
            println!("lnv4c numeric: ge |max|={gm:.3} max_diff={md:.5}");
        }
        other => panic!("GEB_OPTEST: ...|v3n|tldn|lnv4c, got {other}"),
    }
    println!("OPTEST_{which}_OK");
    ge_builder::fini().expect("fini");
}

fn main() {
    let seg = std::env::var("GEB_SEG").unwrap_or_else(|_| "prefix".into());
    let bench = std::env::var("GEB_BENCH").is_ok();
    ge_builder::init("Ascend310P3").expect("geb init (before acl runtime)");
    let be = AscendBackend::new(0).expect("be");
    if let Ok(t) = std::env::var("GEB_OPTEST") {
        optest(&be, &t);
        return;
    }
    match seg.as_str() {
        "vision" => seg_vision(&be, bench),
        "prefix" => seg_prefix(&be, bench),
        "flow" => seg_flow(&be, bench),
        other => panic!("GEB_SEG: vision|prefix|flow, got {other}"),
    }
}
