//! PI-0.5 three-segment Ascend GE executor (vision / prefix / flow OMs):
//! segment graph building, per-frame mechanics (VisionMech / PrefixMech /
//! FlowMech), spool serve loop, and the GeServe in-process facade shared by
//! PyO3 and serve. Probe entry points (env dispatch, GEB_CKPT loading) live
//! in examples/ge_model_probe.rs. The flow OM is single-step — a 10-step
//! flow is 10 executions with rebound inputs.
//!
//! Operator contract (verified against 310P opp protos):
//!   AddLayerNorm x1/x2/gamma/beta(/bias) -> y/mean/rstd/x, epsilon(Float)
//!   GeluV2       x -> y, approximate("none"|"tanh")
//!   GatherV2D    x/indices -> y, axis(Int; 310P supports axis 0 only)
//!   ConcatD      DYNAMIC_INPUT x1..xN -> y, concat_dim(Int) + N(Int)
//!   SliceD       x -> y, offsets/size(ListInt)
//!   BroadcastToD x -> y, shape(ListInt)          [row broadcast]
//!   Reshape      x + shape(tensor input) -> y    [flat / rank-3 bridge]
//!   MatMulV2(transpose_x2), PromptFlashAttention (BSH rank-3), AddRmsNorm,
//!   Squeeze, Add, Mul — verified at single-layer depth.
//!
//! Graph conventions:
//!   - Weights are host-transposed to [n,k] and MatMulV2 runs with
//!     transpose_x2=true; weights/biases/constants enter as Data inputs.
//!   - qkv column split: full-width bias broadcast -> rank-3 Reshape ->
//!     SliceD x3 (PFA requires rank-3 descs).
//!   - rope is composed in-graph (flat adjacent-row swap via GatherV2D);
//!     tables, swap indices and shape tensors are segment-level constant
//!     inputs shared across layers.
//!   - ada-norm: AddRmsNorm(gamma=scale_row) + BroadcastToD(shift_row)+Add.
//!   - Data sources use `link`; operator sources need `link_out` with an
//!     explicit port — default resolution silently fails on multi-output
//!     operators.
//!
//! Environment knobs: GEB_SEG (probe segment, default prefix), GEB_DEPTH
//! (smoke-depth override), GEB_TOKENS (default 64 -> P=832, multiple of 16),
//! GEB_BENCH / GEB_SAVE / GEB_LOAD, GEB_ATTN=manual, GEB_QKV3=1 (independent
//! q/k/v projections; avoids SliceD view materialization), GEB_CKPT=<dir>
//! (real checkpoint baked into the OM, per checkpoint; without it synthetic
//! weights), GEB_OM_DIR (default ./om_cache).
use half::f16;

use apxinf_ascend::ge_builder::{self, Dtype, GeGraph};
use apxinf_ascend::{ops as aops, AscendBackend, AscendContext, AscendStream, DeviceBuffer};
use crate::pi05::{sinusoidal_time_embedding, LinearWeights, Pi05Config, Pi05Weights};

// PI-0.5 3-view profile dims (Pi05Config::default).
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
// Dynamic gears (ge.dynamicDims). GEB_DYN_LLIST="137,...,200" lists the L
// gears. Rules learned the hard way:
//   - every p-dependent desc dim and every Reshape shape Const carries -1
//     (Consts are shared across gears, so their content must not depend on
//     the gear; -1 is resolved per gear at run time)
//   - L-length tensors (rope tables, swap indices) become Data[-1] inputs
//     sized to p_max — row content is gear-independent and reading the
//     first p rows is prefix truncation
//   - per-request gear selection = one aclmdlSetInputDynamicDims call that
//     carries the dims of ALL dynamic inputs at once.
// ---------------------------------------------------------------------------

/// Parses GEB_DYN_LLIST (comma-separated positive ints); None = static
/// single-L mode.
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

/// Process-wide gear switch (env read once; shared by build and run sides).
/// Forced false under GEB_PREFIX_POOL: pool-mode prefix assembly must match
/// the static-bucket OM interface (gear folding drops 90 rows==1 bias Data
/// inputs — n_in 170 != 260, verified). Flow descs are identical in both
/// modes (rope stays Data), but set_rope_for's early-exit gate checks the
/// flows pool instead — see there.
pub fn gear_mode() -> bool {
    *GEAR_MODE.get_or_init(|| dyn_llist().is_some() && std::env::var("GEB_PREFIX_POOL").is_err())
}

/// Dynamic-dim slot: -1 in gear mode (MultiBatchClone specializes per
/// gear), the literal value in static mode. Route every p-dependent dim
/// through here so helpers serve both modes without branching.
pub fn gdyn(v: i64) -> i64 {
    if gear_mode() {
        -1
    } else {
        v
    }
}

/// L -> p (visible prefix length; drop term shared with segment code).
pub fn lp_to_p(l: i64) -> i64 {
    VT + l - envi("GEB_PREFIX_DROP_EMPTY", 0)
}

/// Range-padding grid map (GEB_POOL_GRID = stride): true L -> bucket
/// boundary P (round up to a stride multiple, clamp [MIN, MAX]); no env =
/// identity. Rows between L and P are hidden by the additive softmax mask
/// of GEB_ATTN_MASK, so every L <= MAX has a bucket.
pub fn pool_grid_p(l: i64) -> i64 {
    let stride = envi("GEB_POOL_GRID", 0);
    if stride <= 1 {
        return l;
    }
    let hi = envi("GEB_POOL_GRID_MAX", 200);
    assert!(l <= hi, "range-padding: L={l} exceeds bucket table max {hi}");
    let p = ((l + stride - 1) / stride) * stride;
    p.clamp(envi("GEB_POOL_GRID_MIN", 136), hi)
}

/// splitmix64 step. Do not swap in a plain LCG: high-bits mod-200 LCGs
/// have short-period structure that collapses LayerNorm row variance
/// (rstd inflated x385 here once).
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

/// norm gamma/beta: center +/- 0.05 (realistic magnitude).
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

/// Checkpoint tensor (bf16/f32) -> host f16 (310P has no bf16; fp16 is the
/// precision policy).
pub fn t_f16(t: &apxinf_core::Tensor) -> Vec<f16> {
    t.to_f32_vec().unwrap().into_iter().map(f16::from_f32).collect()
}

/// LinearWeights.weight [in,out] -> host f16 (same host layout as wt()).
pub fn lw_f16(l: &LinearWeights) -> Vec<f16> {
    t_f16(&l.weight)
}

/// LinearWeights.bias -> f16; None (Gemma projections have no bias) -> zeros.
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

/// fp16 device buffer -> host Vec<f16> (LE u16 reassembly of the d2h stream).
pub fn download_f16(ctx: &AscendContext, buf: &DeviceBuffer, n: usize) -> Vec<f16> {
    let mut back = vec![0u8; n * 2];
    ctx.copy_d2h(buf, &mut back).expect("d2h");
    back.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

// ---------------------------------------------------------------------------
// Production int8: prefix/flow projection mms -> QuantMatmulDequant.
// ---------------------------------------------------------------------------

/// GEB_PREFIX_INT8: prefix 7 projections use QuantMatmulDequant (int8 ND
/// [n,k] + per-row scale + smooth; operator contract verified by single-op
/// parity). Unset = f16 fallback. Requires real weights + GEB_QKV3 +
/// GEB_INT8_SMOOTH factor file.
pub fn prefix_int8() -> bool {
    std::env::var("GEB_PREFIX_INT8").map(|v| !v.is_empty()).unwrap_or(false)
}

/// GEB_FLOW_INT8: flow 7 projections use QuantMatmulDequant (factor file
/// GEB_FLOW_SMOOTH, keys flow/L{nn}/{proj}). Unlike prefix, the action
/// loader does not fold Gemma (1+g) — weights.rs takes values verbatim —
/// so in-graph activations are already in the torch domain: s_eff is
/// s_npz directly and no g table applies. Motivation: M=50 flow steps
/// cannot amortize weight reads (~622MB f16 streamed per step), so
/// halving weight bytes is a direct win.
pub fn flow_int8() -> bool {
    std::env::var("GEB_FLOW_INT8").map(|v| !v.is_empty()).unwrap_or(false)
}

/// Engine-domain smooth factors (GEB_INT8_SMOOTH safetensors, keys
/// prefix/L{nn}/{proj}), converted from torch-domain calibration by
/// convert_smooth_engine.py v2: int8 layers keep the true (1+g) gamma in
/// AddRmsNorm and push activations back to the torch domain. A pure
/// engine-domain s_eff would blow up to +/-e7 where the (1+g) fold is
/// near zero (not f16-representable — measured).
pub struct Int8Smooth {
    /// Weight-side absorb factor f32[k] (= s_npz/(1+g); for f32 weight quant).
    pub s_eff: Vec<f32>,
    /// Native smooth_scale input f32[k] (= s_npz, torch domain, f16-safe).
    pub smooth: Vec<f32>,
    /// 1/s_npz (row broadcast on the GEB_INT8_NATIVE=0 fallback path).
    pub inv: Vec<f32>,
}

/// Loads the smooth factor file: matrix factors (s_eff/smooth/inv per
/// base) plus group-norm truths (prefix/L{nn}/g1, g2 = 1+norm_scale, used
/// to return int8 layers to the torch domain).
pub fn load_int8_smooth(
    path: &str,
) -> (
    std::collections::HashMap<String, Int8Smooth>,
    std::collections::HashMap<String, [Vec<f32>; 2]>,
) {
    let (t, _) = apxinf_loader::safetensors::load_native_path(std::path::Path::new(path))
        .unwrap_or_else(|e| panic!("GEB_INT8_SMOOTH load failed {path}: {e}"));
    let mut out = std::collections::HashMap::new();
    let bases: Vec<String> = t
        .keys()
        .filter_map(|k| k.strip_suffix("/s_eff").map(|b| b.to_string()))
        .collect();
    assert!(!bases.is_empty(), "{path} has no */s_eff keys (needs a converted factor file)");
    for base in bases {
        let get = |suf: &str| -> Vec<f32> {
            t.get(&format!("{base}{suf}"))
                .unwrap_or_else(|| panic!("int8 smooth missing {base}{suf}"))
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

/// W8 row quantization (after smooth absorb): host w32 [k,n] in-major ->
/// int8 [n,k] physical + per-row scale [n]. Math must stay identical to
/// int8_bake_weights.py (f32-domain multiply, max().max(1e-12)/127,
/// ties-even round, clamp +/-127) — GEB_INT8_DUMP output is compared
/// bit-exactly against the npz golden. Tiled transpose: a naive
/// element-wise pass is all cache misses at [2048,16384] (~130s/18 layers).
pub fn quant_w8(w32: &[f32], k: usize, s_eff: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n = w32.len() / k;
    assert_eq!(w32.len(), n * k, "int8 weights: not whole k rows");
    assert_eq!(s_eff.len(), k, "smooth factor length != k");
    let mut ws = vec![0f32; n * k]; // [n,k] physical (transpose + absorb in one pass)
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

/// Flat rope constants: cos/sin tables plus swap indices for the flat
/// layout x [t, heads*d] -> [t*heads*2, d/2] (even rows = first half
/// channels; position = pos_offset + t). sin rows fold the rotate-half
/// sign (hf=0 -> -s, hf=1 -> +s); swap = adjacent-row exchange r^1.
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

/// Rope angle, golden-matched: inv_freq = theta^(-2i/HD) is quantized to
/// f16 FIRST — torch's .to(f16) converts the rotary inv_freq buffer too,
/// and HF rotary computes angles from the f16 frequencies. Full-precision
/// frequencies drift O(0.1 rad) at high positions (~968): prefix kv is
/// 12.6% off at l0, amplifying to 85% by l17. Replicating the f16 step
/// measured 0.08%.
pub fn rope_angle(pos: i64, i: usize) -> f64 {
    let inv = f16::from_f32((ROPE_THETA as f32).powf(-(2.0 * i as f32) / HD as f32));
    pos as f64 * inv.to_f32() as f64
}

/// Rank-2 rope tables [t*heads, HD]: cos half-dim duplicated, sin sign
/// folded — pairs with the channel-half-swap (SliceD x2 + ConcatD) rope
/// composition.
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
// Segment builder: data (weights/constants) and graph in one place;
// input registration order == runtime binding order. wire() distinguishes
// Data sources (link) from operator sources (link_out with an explicit
// port — default port resolution fails silently on multi-output ops).
// ---------------------------------------------------------------------------

pub struct Seg {
    pub g: GeGraph,
    idx: i64,
    names: Vec<String>,
    shapes: Vec<Vec<i64>>,
    pub binds: Vec<DeviceBuffer>,
    /// Name table parallel to binds. names/shapes only cover Data inputs —
    /// WCONST weights (wt Const branch) and i8 placeholder slots push into
    /// binds but not names, so those two tables do NOT align; bind_names
    /// does, and name-based lookup no longer depends on registration order.
    pub bind_names: Vec<String>,
    /// Indices of graph inputs inside binds (under GEB_WCONST weights are
    /// Consts and not graph inputs; binds still holds every upload so the
    /// indices are identical in both modes).
    data_inputs: Vec<usize>,
    /// GEB_NORM32 constant pool, lazily built per width (sc/kc/ones Const
    /// names).
    n32_c: std::collections::HashMap<i64, (String, String, String)>,
    /// GEB_WCONST: mm weights baked as Consts (ND->NZ folded at compile
    /// time, zero runtime tax; measured 0.3945 -> 0.2945 ms/mm, +8.4MB in
    /// the OM). Cost: the OM becomes per-weight-set (checkpoint version
    /// joins the cache key).
    wconst: bool,
    datas: std::collections::HashSet<String>,
    outports: std::collections::HashMap<String, &'static str>,
    /// LayerNormV4 instance names (finish() binds mean/rstd as aux graph
    /// outputs).
    ln_aux: Vec<String>,
    /// Gear mode (GEB_DYN_LLIST): p value per gear (VT+L-drop_v, fed by
    /// the segment). finish() derives the ge.dynamicDims gear table and
    /// one S2 payload per gear (all dynamic-input dims concatenated in
    /// registration order) for run_gear.
    pub gear_ps: Vec<i64>,
    gear_mul: std::collections::HashMap<String, i64>,
    /// S2 payload per gear (non-empty in gear mode; empty = plain run).
    pub gear_payloads: Vec<Vec<i64>>,
    /// Pure -1-slot values per gear, for the ge.dynamicDims option string
    /// only — atc wants just the -1 positions there; mixing in static
    /// trailing dims from the payload fails compilation with rc=-7.
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
            gear_payloads: Vec::new(),
            gear_optvals: Vec::new(),
        }
    }

    /// Graph-input assembly (under GEB_WCONST, Const-baked weights are
    /// excluded; otherwise all binds).
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

    /// Gear setup: p value per gear, fed by the segment (dynamic-input
    /// dim0 = mul*p, default mul=1).
    pub fn set_gear_ps(&mut self, ps: Vec<i64>) {
        self.gear_ps = ps;
    }

    pub fn set_gear_mul(&mut self, name: &str, mul: i64) {
        self.gear_mul.insert(name.to_string(), mul);
    }

    /// Name -> binds index (bind_names is strictly parallel to binds,
    /// including WCONST weights and i8 placeholders; names/shapes cover
    /// only Data inputs and do not align). try_ variant returns None for
    /// absent names (used for the optional GEB_ATTN_MASK padmask input).
    pub fn try_bind_of(&self, name: &str) -> Option<usize> {
        self.bind_names.iter().position(|n| n == name)
    }

    pub fn bind_of(&self, name: &str) -> usize {
        self.bind_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("bind_of: graph input {name} not found"))
    }

    /// S2 payload per gear: the FULL dims of every input carrying a -1
    /// slot (dynamic slots = mul*p, static trailing dims verbatim; one
    /// SetInputDynamicDims call carries all of them). The
    /// ge.dynamicDims option string stays pure -1-slot values (atc
    /// semantics).
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
                    "gear input {n} has {ndyn} -1 dims (only dim0 supported)"
                );
                if ndyn == 1 {
                    assert_eq!(shp[0], -1, "gear input {n}: -1 must be dim0");
                    let v = mul * pg;
                    pl.push(v);
                    pl.extend_from_slice(&shp[1..]);
                    ov.push(v);
                }
            }
            self.gear_payloads.push(pl);
            self.gear_optvals.push(ov);
        }
        println!(
            "[gear] {} gears; payload dims {} (first 8 of gear 0: {:?})",
            self.gear_ps.len(),
            self.gear_payloads[0].len(),
            &self.gear_payloads[0][..8.min(self.gear_payloads[0].len())]
        );
    }

    /// Gear-run convenience: payload from self (static mode falls through
    /// to plain run).
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

    /// mm weight registration (host [rows,cols] -> transposed [cols,rows]):
    /// under GEB_WCONST baked as a Const (binds still holds the uploaded
    /// copy; no graph-input slot or idx consumed); otherwise a Data input
    /// (pays a device-side ND->NZ TransData at ~63GB/s every execution).
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

    /// fp32 Data input (GEB_NORM32 semantics; binds are dtype-agnostic
    /// bytes).
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

    /// int8 Data input (QuantMatmulDequant weights/scales).
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

    /// Const shape tensor (not a graph input; folded at compile time).
    /// Shape inputs MUST be Consts: as Data inputs they turn consumer
    /// outputs (Reshape/LayerNormV4) into unknown descs, and the
    /// DynamicShapePartitioner then splits the graph and schedules the
    /// unknown parts on the host (~20ms stall per boundary; 3220 unknown
    /// marks measured in a vision OM).
    fn const_i32(&mut self, name: &str, vals: &[i32]) -> String {
        self.g.add_const_i32(name, vals).unwrap();
        self.datas.insert(name.to_string());
        name.to_string()
    }

    /// Const fp16 tensor (weights baked into the graph).
    fn const_f16(&mut self, name: &str, dims: &[i64], host: &[f16]) -> String {
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        self.g.add_const_raw(name, dims, Dtype::Fp16, &bytes).unwrap();
        self.datas.insert(name.to_string());
        name.to_string()
    }

    /// dst.port <- src (Data: default resolution; operators: explicit
    /// output port).
    fn wire(&self, dst: &str, port: &str, src: &str) {
        if self.datas.contains(src) {
            self.g.link(dst, port, src).unwrap();
        } else {
            let out = self.outports.get(src).copied().unwrap_or("y");
            self.g.link_out(dst, port, src, out).unwrap();
        }
    }

    // ---- Operator helpers (return the op name) ----

    /// MatMulV2: a [m,k] x w' (physical [n,k], transposed) -> [m,n].
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

    /// QuantMatmulDequant: a [m,k] x int8 wq [n,k] (transpose_weight=true,
    /// linear style) -> y [m,n]; weight_scale f32 [n]; optional
    /// smooth_scale f16 [k] and bias int32 [n] (None = input absent).
    /// Contract verified by single-op parity on 310P legacy. Attributes
    /// must be set explicitly — bare op defaults do not apply (rc=-7).
    /// Folding bias in removes a per-layer TileD+Add pair (Gemma bias=0 ->
    /// int32 zero identity; non-zero bias would need pre-quantization by
    /// weight_scale — unsupported, v1 is zero-only).
    fn qmd(
        &mut self, name: &str, a: &str, a_dims: &[i64], wq: &str, w_dims: &[i64],
        ws: &str, sm: Option<&str>, zb: Option<&str>, n: i64, o: &[i64],
    ) -> String {
        self.g.add_op(name, "QuantMatmulDequant").unwrap();
        self.g.set_input_desc(name, "x", a_dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "quantized_weight", w_dims, Dtype::Int8).unwrap();
        self.g.set_input_desc(name, "weight_scale", &[n], Dtype::Fp32).unwrap();
        if sm.is_some() {
            self.g.set_input_desc(name, "smooth_scale", &[a_dims[1]], Dtype::Fp16).unwrap();
        }
        if zb.is_some() {
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

    /// Row-broadcast bias registration: static mode registers the [1,cols]
    /// row (consumer TileD broadcasts); gear mode registers the row
    /// repeated rows times as [gdyn,cols] — TileD multiples cannot carry a
    /// gear-dependent p (attrs are shared across gears), so the content
    /// itself becomes the broadcast (rows identical = prefix truncation).
    /// The int8 folding path does not consume the bias: in gear mode it is
    /// simply not registered (an unconsumed static Data input still takes a
    /// graph-input slot and mixing static content into the dynamic set
    /// trips SetDims 145012); static mode passes rows=1 to register the
    /// bare row and save memory.
    fn row_bcast(
        &mut self,
        ctx: &AscendContext,
        name: &str,
        rows: i64,
        cols: i64,
        row: &[f16],
    ) -> String {
        // Gear mode: row repeated into [-1,cols]; rows==1 and unconsumed -> not registered.
        if gear_mode() {
            if rows > 1 {
                let mut rep = Vec::with_capacity((rows * cols) as usize);
                for _ in 0..rows {
                    rep.extend_from_slice(row);
                }
                self.data(ctx, name, &[gdyn(rows), cols], &rep)
            } else {
                // Folded and unconsumed: not registered (placeholder name,
                // never wired).
                format!("{name}_folded")
            }
        } else {
            self.data(ctx, name, &[1, cols], row)
        }
    }

    fn bias(&mut self, name: &str, x: &str, x_dims: &[i64], row: &str) -> String {
        let ba = format!("{name}_ba");
        if gear_mode() {
            // Row already repeated into [-1,cols] — plain Add, no TileD.
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

    /// Row-broadcast gated multiply: TileD(gate [1,cols]) x Mul — the
    /// gate path of torch _gated_residual (residual + branch*gate).
    /// Broadcasts along dim0 whole-row like bias().
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
        // GEB_NORM32: hand-composed fp32 variance (no zeros input needed —
        // rms(x+0)=rms(x)).
        if norm32_enabled() {
            // rms32 v2 emits f16 directly (no Cast tail).
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

    /// ada-norm: y = rms(x)*scale + shift = AddRmsNorm(x, zeros,
    /// gamma=scale) + TileD(shift_row) + Add (TileD instead of
    /// BroadcastToD, which fails to compile on 310P).
    fn ada(&mut self, name: &str, x: &str, zeros: &str, scale: &str, shift: &str, dims: &[i64]) -> String {
        // GEB_NORM32: torch ada semantics = rms32(x, scale) + shift
        // (all-f16 domain; 1+s0 is folded into the host scale).
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

    /// NORM32 constant pool, lazily built per width (sc = s prescale,
    /// kc = s*sqrt(w), ones [1,w]) — Consts fold at compile time and cost
    /// no model input; norms of the same width share one node.
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

    /// NORM32 core v2 (f16 domain, fp32 accumulation via MatMul cube):
    /// 310P tbe ReduceSumD/ReduceSum with f32 descs are fake support — the
    /// FE falls back to f16 accumulation and row sums saturate above
    /// 65504 (92% deviation measured); Cast to f32 is real semantics.
    /// Variance precision therefore rides the mm cube's fp32 accumulation
    /// (0.05% measured = f16 quantization level):
    ///   xs = x*s (guards outlier square overflow) -> sq = xs^2 ->
    ///   ssum = mm(sq, ones^T) -> inv = rsqrt(ssum)*k (k = s*sqrt(w)) ->
    ///   1-D tile broadcast -> x*invb*gamma.
    /// All kernels f16; GemmaRMSNorm alignment hinges on a non-saturating,
    /// non-drifting variance (fp32 accumulate); f16 mul rounding ~0.05%/op
    /// is the residual.
    fn rms32(&mut self, name: &str, x: &str, gamma: &str, dims: &[i64]) -> String {
        let (rows, w) = (dims[0], dims[1]);
        let (sc, kc, ones) = self.n32_consts(w);
        // xs = x*s
        let xs = format!("{name}_xs");
        self.g.add_op(&xs, "Mul").unwrap();
        self.g.set_input_desc(&xs, "x1", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(&xs, "x2", &[1], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&xs, "y", dims, Dtype::Fp16).unwrap();
        self.wire(&xs, "x1", x);
        self.g.link(&xs, "x2", &sc).unwrap();
        let xs = self.reg_out(&xs, "y");
        // sq = xs^2
        let sq = self.mul2_f16(&format!("{name}_sq"), &xs, &xs, dims);
        // ssum = mm(sq, ones^T) [rows,1] — fp32 accumulation inside the cube.
        let ssum = self.mm(&format!("{name}_mm"), &sq, dims, &ones, &[1, w], &[rows, 1]);
        // inv = rsqrt(ssum) * k (k = s*sqrt(w) folds the prescale back in).
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
        // Broadcast, row-wise shape: [rows,1] -> Reshape [1,rows] ->
        // TileD multiples=[w,1] -> [w,rows] -> TransposeD(1,0) -> [rows,w].
        // The contiguous-tile variants (rank-1 [rows], or [1,rows] tiled
        // by [1,w]) corrupt scattered f16 elements by ~8% on 310P; this
        // row-wise shape measures a clean 0.08%.
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
        // nx = x * invb
        let nx = self.mul2_f16(&format!("{name}_nx"), x, &invt, dims);
        // gamma [w] -> Reshape [1,w] (rank bridge) -> Tile [rows,w]
        let gshp = format!("{name}_gshp");
        self.const_i32(&gshp, &[1, w as i32]);
        let gfr = self.reshape(&format!("{name}_gfr"), gamma, &[w], &gshp, &[1, w]);
        let gt = format!("{name}_gt");
        self.g.add_op(&gt, "TileD").unwrap();
        self.g.set_input_desc(&gt, "x", &[1, w], Dtype::Fp16).unwrap();
        self.g.set_output_desc(&gt, "y", dims, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(&gt, "multiples", &[rows, 1]).unwrap();
        self.wire(&gt, "x", &gfr); // gfr is an op output — plain link takes Data names only
        let gft = self.reg_out(&gt, "y");
        self.mul2_f16(name, &nx, &gft, dims)
    }

    /// Same-shape f16 Mul (NORM32 v2, all-f16 domain).
    fn mul2_f16(&mut self, name: &str, a: &str, b: &str, dims: &[i64]) -> String {
        self.g.add_op(name, "Mul").unwrap();
        self.g.set_input_desc(name, "x1", dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "x2", dims, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", dims, Dtype::Fp16).unwrap();
        self.wire(name, "x1", a);
        self.wire(name, "x2", b);
        self.reg_out(name, "y")
    }

    /// LayerNormV4 (vision SigLIP norm). aclnnAddLayerNorm/GE AddLayerNorm
    /// amplify output ~100x at [768,1152] (row variance normal — kernel
    /// behavior), hence LayerNormV4 with its normalized_shape tensor
    /// input. mean/rstd are REQUIRED outputs and leaving them dead blows
    /// up y (unlike AddRmsNorm, whose dead ends are harmless) — every LN
    /// instance is recorded in ln_aux and bound as aux graph outputs at
    /// finish().
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

    /// PromptFlashAttention (BSH rank-3; q/kv seq may differ = cross).
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

    /// Squeeze(axis=[0]): [1,m,*] -> [m,*].
    fn squeeze(&mut self, name: &str, x: &str, in3: &[i64; 3], out2: &[i64; 2]) -> String {
        self.g.add_op(name, "Squeeze").unwrap();
        self.g.set_input_desc(name, "x", in3, Dtype::Fp16).unwrap();
        self.g.set_output_desc(name, "y", out2, Dtype::Fp16).unwrap();
        self.g.set_attr_int_list(name, "axis", &[0]).unwrap();
        self.wire(name, "x", x);
        self.reg_out(name, "y")
    }

    /// Reshape(x, shape as a constant tensor input).
    fn reshape(&mut self, name: &str, x: &str, in_dims: &[i64], shape_in: &str, out: &[i64]) -> String {
        self.g.add_op(name, "Reshape").unwrap();
        self.g.set_input_desc(name, "x", in_dims, Dtype::Fp16).unwrap();
        self.g.set_input_desc(name, "shape", &[out.len() as i64], Dtype::Int32).unwrap();
        self.g.set_output_desc(name, "y", out, Dtype::Fp16).unwrap();
        self.wire(name, "x", x);
        self.g.link(name, "shape", shape_in).unwrap();
        self.reg_out(name, "y")
    }

    /// SliceD rank-2 channel split.
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

    /// GatherV2D row gather (axis=0 attr; used for swaps and row copies).
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

    /// Rank-2 rope composition: x [t, heads*d] -> half-channel SliceD x2
    /// -> ConcatD(axis=1) swap -> mul+mul+add (rank-2 tables) -> Reshape
    /// [1,t,heads*d]. Feeding PFA through Reshape is bit-identical;
    /// Unsqueeze showed cumulative drift, so it is not used. Returns
    /// (rank-2, rank-3).
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

    /// Flat rope (no column split): [t, wd] -> Reshape [t*2, half] ->
    /// GatherV2D adjacent-row swap (Const idx) -> mul+mul+add -> Reshape
    /// back. Replaces rope2's SliceD x2 + ConcatD: column-split views
    /// trigger a runtime MemcopyAsync materialization, and GatherV2D on
    /// 310P is axis=0 only, hence the flat view for a row swap.
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

    /// Manual attention (MHA): q/k/v3 [bh,s,d] -> bmm(q,k^T) ->
    /// SoftmaxV2(axes=-1) -> bmm(.,v). Scale is folded into the q-side
    /// weights+bias by the caller. This replaces PFA, whose InnerPFA host
    /// callback stalls ~20ms per execution (AttentionScore has no 310P
    /// kernel).
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

    /// GQA variant: k3/v3 [1,skv,d] tiled to bh heads; q3 [bh,sq,d];
    /// returns [bh,sq,d]. With mask (GEB_ATTN_MASK, range padding) the
    /// additive mask is broadcast-added before softmax (x2 [1,1,skv]:
    /// large negative on pad key columns, 0 on real ones) — exact for
    /// real rows (x+0 identity; exp(-30000+max) underflows to 0).
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

    /// Token-major [views*s, h*d] -> head-major [views*h, s, d] (bmm
    /// layout). A direct Reshape is a mispermute (98.2% parity error
    /// measured); the correct chain is Reshape [views,s,h,d] ->
    /// TransposeD(0,2,1,3) -> Reshape [views*h,s,d].
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

    /// Inverse of headsplit: [views*h, s, d] -> [views*s, h*d].
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
        // Gear payload composition precedes every early return (serve/e2e
        // skip compilation but mechs still need payloads to pick a gear per
        // request; names/shapes are final at this point).
        self.compose_gear_payloads();
        if std::env::var("GEB_LOAD").is_ok() {
            // Cached-load mode: skip compilation, use the on-disk OM.
            return;
        }
        if std::env::var("GEB_SEG").ok().as_deref() == Some("e2e") {
            // e2e mode: skip compilation too ({seg}_real.om loaded in
            // e2e_run).
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
        // GEB_NO_AUX=1: skip the LN aux-output binding (dead mean/rstd
        // blow up y — numerics break but bench timing stays valid).
        if self.ln_aux.is_empty() || std::env::var("GEB_NO_AUX").is_ok() {
            self.g.graph_outputs(out_names).unwrap();
        } else {
            // Main outputs idx0 + mean(idx1)/rstd(idx2) per LayerNormV4.
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
        // Gear mode: ge.dynamicDims gear table = pure -1-slot values in
        // registration order (mixing static trailing dims from the payload
        // into the option fails with rc=-7 — hence the two structures).
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
        // GEB_OPT_<key>=<value>: build-option passthrough (A/B
        // experiments, e.g.
        // GEB_OPT_ge.streamMaxParallelNum=AIcoreEngine:1,VectorEngine:1).
        for (k, v) in std::env::vars().filter(|(k, _)| k.starts_with("GEB_OPT_")) {
            let key = &k["GEB_OPT_".len()..];
            self.g.set_option(key, &v).unwrap();
            println!("[geb-opt] {key} = {v}");
        }
        self.g.build().expect("build");
    }
}

// ---------------------------------------------------------------------------
// e2e: the three segment OMs wired into the real inference chain
// (GEB_SEG=e2e, in-process chaining, host relay between segments).
// Assembly mirrors the CUDA executor:
//   x0 = vision_out || token_embedding[token_ids] (vision rows first)
//   state does not enter prefix — PI-0.5 discretizes state into the
//   prompt (pi05_prompt)
//   cond = silu(W_out*silu(W_in*te+b_in)+b_out); style split [0:w]/[w:2w]
//   (ascl = 1+s0, ash = s1)
// ---------------------------------------------------------------------------

pub struct E2eStage {
    /// frame inputs
    patches: Vec<f16>,          // [VT, V_PATCH_W]
    token_ids: Vec<u32>,        // lang tokens
    noise: Vec<f16>,            // [HOR, ADIM]
    /// stage outputs (host-relayed between segments)
    vision_out: Vec<f16>,       // [VT, PW] post-projector
    kv: Vec<Vec<f16>>,          // prefix 36 outputs (k0,v0,k1,v1,... each [p,KVD])
    actions: Vec<f16>,          // flow terminal state [HOR, ADIM]
    /// GEB_E2E_SERVE resident mechs (per-frame executors; ctx/stream passed
    /// in by the serve loop, mechs hold owned state only — one process per bucket)
    vmech: Option<VisionMech>,
    pmech: Option<PrefixMech>,
    fmech: Option<FlowMech>,
}

// ---------------------------------------------------------------------------
// Resident per-frame segment mechs (closed-loop eval). One process per
// bucket (GEB_TOKENS=L + GEB_OM_DIR=tl{L}); the Python side picks a
// bucket per frame and the supervisor spawns one serve process per
// bucket inside the apxinf_rust container (the engine binary binds the
// 9.0.1 GE). ctx/stream live outside; mechs hold only owned state: Seg
// (OM + binds), output buffers, embedding table, styles cache.
// ---------------------------------------------------------------------------

pub struct VisionMech {
    s: Seg,
    routs: Vec<DeviceBuffer>, // 108 LN aux slots, allocated once (only the main output is downloaded)
}

impl VisionMech {
    /// One vision frame: overwrite binds[0] (patches), rerun, download
    /// only main output idx0.
    fn one_frame(&mut self, ctx: &AscendContext, stream: &AscendStream, patches: &[f16]) -> Vec<f16> {
        assert_eq!(patches.len() * 2, self.s.binds[0].len(), "serve patches size mismatch");
        // Hot path: zero-copy byte views (on LE hosts to_le_bytes == raw
        // bytes) and all three stages on one stream (h2d -> run -> d2h in
        // stream FIFO order; the host waits once at the final sync). Bit
        // -identical to the per-element conversion path; removes the d2h
        // 3.4ms + dlconv 5.1ms once measured here.
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
        let _ = stream.synchronize();
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
    emb: Vec<f32>,     // [vocab, PW] host lookup table (~2.1GB resident)
    lang_scale: f32,   // sqrt(PW) (gemma embed scale)
    vis_rows: usize,   // (VT-drop_v)*PW — visible rows after empty-view drop
    nkv: usize,        // depth*2
    vocab_w: usize,
    /// Token lookup cache: the prompt is constant across replans (same
    /// token_ids within a task's L bucket), so the lookup+scale f16 rows
    /// are computed once. Bit-stable: same ids, same deterministic f32
    /// arithmetic.
    tok_cache_ids: Vec<u32>,
    tok_cache_rows: Vec<f16>,
    /// Gear mode: L -> gear index (empty in static mode = fixed L).
    gear_of_l: std::collections::HashMap<i64, usize>,
    /// GEB_PREFIX_POOL: per-L static prefix OM pool (handle swap on
    /// execute; routs shared across Ls — max-sized materialization, no
    /// cross-request state).
    pool_root: Option<String>,
    pool_graphs: std::collections::HashMap<i64, ge_builder::GeGraph>,
    pool_lru: Vec<i64>,
    pool_cap: usize,
    /// Per-slot routs byte size (= max-gear OM output size; pool OMs d2h
    /// the whole buffer then truncate).
    rout_bytes: Vec<usize>,
    /// GEB_ATTN_MASK (range padding): padmask input bind index (None for
    /// maskless OMs; must be Some under pool-mode grid mapping or pad
    /// rows go unmasked).
    mask_bind: Option<usize>,
}

impl PrefixMech {
    /// L -> gear index (always 0 in static mode — run_seg takes the plain
    /// run branch).
    fn gear_idx(&self, l: i64) -> usize {
        if self.s.gear_payloads.is_empty() {
            return 0;
        }
        *self
            .gear_of_l
            .get(&l)
            .unwrap_or_else(|| panic!("prefix gear table missing L={l} (not in GEB_DYN_LLIST)"))
    }
}

impl PrefixMech {
    /// Pool mode: swap the static OM handle for L (lazy load tl{l}_i8;
    /// LRU cap keeps every gear resident — one OM is 1.88GB). routs/binds
    /// stay put (max-sized, shared across Ls).
    fn swap_prefix(&mut self, l: i64) {
        if !self.pool_graphs.contains_key(&l) {
            let p = format!("{}/tl{l}_i8/prefix_real.om",
                            self.pool_root.as_deref().unwrap_or_default());
            println!("[prefix-pool] lazy load L={l} ({p})");
            let g = ge_builder::load(&p)
                .unwrap_or_else(|e| panic!("prefix pool missing L={l} ({e:?}) — bake {p} first"));
            let n_out = g.num_outputs().unwrap();
            assert!(n_out <= self.rout_bytes.len(),
                    "pool OM outputs {n_out} > max slots {}", self.rout_bytes.len());
            self.pool_graphs.insert(l, g);
            self.pool_lru.push(l);
            while self.pool_lru.len() > self.pool_cap {
                let victim = self.pool_lru.remove(0);
                if let Some(g) = self.pool_graphs.remove(&victim) {
                    let _ = g.unload();
                }
                println!("[prefix-pool] LRU evict L={victim} (cap={})", self.pool_cap);
            }
        } else {
            let i = self.pool_lru.iter().position(|&x| x == l).unwrap();
            self.pool_lru.remove(i);
            self.pool_lru.push(l);
        }
        self.s.g = *self.pool_graphs.get(&l).unwrap();
    }

    /// One prefix frame (device state): reassemble x0 (this frame's
    /// visible vision_out rows + real token lookups * lang_scale),
    /// overwrite binds[0], rerun + sync; the 36 kv outputs stay in
    /// self.routs on device — consumed by FlowMech's async d2d under
    /// GEB_SERVE_FAST, zero host relay.
    fn one_frame_dev(
        &mut self,
        ctx: &AscendContext,
        stream: &AscendStream,
        vision_out: &[f16],
        token_ids: &[u32],
    ) {
        assert_eq!(vision_out.len(), (VT * PW) as usize, "serve vision_out size mismatch");
        // Range padding (GEB_POOL_GRID, pool mode only): true L -> bucket
        // boundary P, tokens padded with 0 (embedding[0]) — pad rows are
        // masked by padmask; non-pool mode is the identity and unaffected.
        let l_true = token_ids.len() as i64;
        let p_bkt = if self.pool_root.is_some() {
            pool_grid_p(l_true)
        } else {
            l_true
        };
        assert!(
            p_bkt == l_true || self.mask_bind.is_some(),
            "range-padding: padded to P={p_bkt} but the OM has no padmask (bake with GEB_ATTN_MASK=1)"
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
        // Timed: asm (x0 assembly + lookup) / h2d_enq / run / wait.
        let tim = serve_timing();
        let t0 = std::time::Instant::now();
        let gi = self.gear_idx(p_bkt);
        if self.pool_root.is_some() {
            self.swap_prefix(p_bkt);
        }
        // Token lookup cache: recompute only when ids change (the prompt
        // is constant across replans). Same f32 arithmetic on the same
        // ids yields identical f16 bytes.
        if self.tok_cache_ids != token_ids {
            let mut rows: Vec<f16> = Vec::with_capacity(token_ids.len() * self.vocab_w);
            for &id in token_ids {
                let r = id as usize * self.vocab_w;
                assert!(r + self.vocab_w <= self.emb.len(), "token id {id} beyond vocab");
                rows.extend(
                    self.emb[r..r + self.vocab_w]
                        .iter()
                        .map(|&v| f16::from_f32(v * self.lang_scale)),
                );
            }
            self.tok_cache_ids = token_ids.to_vec();
            self.tok_cache_rows = rows;
            // padmask refresh rides the token change (no-op when L is
            // unchanged): rows [vis+L, vis+P) = -30000, real/vision rows
            // 0 — rebuilt whole, no cross-bucket contamination.
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
        // Gear mode: binds[0] is materialized to p_max (x0's actual p
        // rows <= max — sized h2d).
        assert!(x0.len() * 2 <= self.s.binds[0].len(), "serve x0 exceeds max gear size");
        // Zero-copy byte view (x0 outlives the function; async h2d borrow
        // is safe).
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(x0.as_ptr() as *const u8, x0.len() * 2) };
        if self.s.gear_payloads.is_empty() && self.pool_root.is_none() {
            ctx.copy_h2d_async(&self.s.binds[0], bytes, stream).expect("serve h2d x0");
        } else {
            // gear / prefix-pool: binds[0] is max-sized; this frame's p
            // rows go through the sized copy.
            ctx.copy_h2d_async_sized(&self.s.binds[0], bytes, stream)
                .expect("serve h2d x0 (sized)");
        }
        let t2 = std::time::Instant::now();
        let refs: Vec<&DeviceBuffer> = self.routs.iter().collect();
        self.s.run_seg(gi, &refs, stream).expect("ge run");
        let t3 = std::time::Instant::now();
        let _ = stream.synchronize();
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

    /// One prefix frame (legacy host relay): one_frame_dev then download
    /// the 36 kv outputs (aux slots not downloaded).
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

    /// kv download sized to the currently executing OM's real row count
    /// (routs/binds are max-sized; the whole-buffer d2h contract means
    /// download-then-truncate).
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
    style_cache: Vec<Vec<Vec<f16>>>, // 10 steps x (6*depth+2) segments (incl. gate; request-invariant)
    layer_bases: Vec<usize>,
    fs_idx: usize,
    pk_base: usize,
    depth: usize,
    /// Backing allocation for style_views — the views are raw slices, so
    /// this field exists purely to keep the master buffer alive for the
    /// mech's lifetime (never read).
    #[allow(dead_code)]
    style_master: Option<DeviceBuffer>,
    /// GEB_SERVE_FAST device-resident styles: each of the 10 steps gets
    /// its own view (frame-invariant, so one upload serves every frame —
    /// unlike the legacy 110 slots reused across steps, re-uploaded per
    /// step).
    style_views: Vec<Vec<DeviceBuffer>>,
    /// style bind index -> segment order within a step.
    style_map: std::collections::HashMap<usize, usize>,
    /// Graph-input assembly order (= data_inputs under wconst, same
    /// semantics as Seg::ins).
    ins_order: Vec<usize>,
    /// Gear mode: L -> gear index, plus rope table bind indices (tables
    /// are HOR-shaped; their content shifts with pos_offset=p, so they are
    /// re-uploaded on L change — no-op while L holds).
    gear_of_l: std::collections::HashMap<i64, usize>,
    rope_binds: Vec<usize>,
    last_l: i64,
    /// HYBRID: per-L static flow OM pool (non-empty = swap handles per
    /// request; flow does not use gears — its styles are static Data, see
    /// seg_flow).
    flows: std::collections::HashMap<i64, GeGraph>,
    /// GEB_ATTN_MASK (range padding): padmask bind index + last filled L
    /// (no-op on match; rebuilt whole, no cross-bucket contamination).
    mask_bind: Option<usize>,
    mask_last: i64,
}

/// Builds device-resident styles: the 10 steps' (6*depth+2) segments are
/// packed into one master (single h2d), one view per step-segment;
/// style_map matches the legacy per-slot h2d loop's bind mapping.
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
    /// L -> gear index (always 0 in static mode).
    fn gear_idx(&self, l: i64) -> usize {
        if self.s.gear_payloads.is_empty() {
            return 0;
        }
        *self
            .gear_of_l
            .get(&l)
            .unwrap_or_else(|| panic!("flow gear table missing L={l} (not in GEB_DYN_LLIST)"))
    }

    /// HYBRID: swap the static flow OM handle for L (empty pool = keep
    /// the single OM).
    fn swap_flow(&mut self, l: i64) {
        if !self.flows.is_empty() {
            self.s.g = *self
                .flows
                .get(&l)
                .unwrap_or_else(|| panic!("flow OM pool missing L={l}"));
        }
    }

    /// Re-upload the rope tables on L change (flat quartet + rank-2
    /// quartet; content shifts with pos_offset=p — the tables are
    /// HOR-shaped regardless of p). No-op when L holds. The early-exit
    /// gate checks the flows pool: a non-empty pool means HYBRID swaps a
    /// static flow OM per L, so rope must follow L even when payloads are
    /// empty (the GEB_PREFIX_POOL case, where gear_mode is globally
    /// false).
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

    /// Range padding: fill the flow padmask — key layout = cache rows
    /// first (real [0,vis+L), pad [vis+L, vis+P)) plus this bucket's HOR
    /// flow rows always real. No-op when L holds. Pool mode only (flows
    /// non-empty). The whole length is filled (copy_h2d asserts equal
    /// length): the bind is max-bucket sized (762 rows) and unconsumed
    /// rows beyond this bucket are harmless at -30000.
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

    /// One flow frame: rebind the 36 pk/pv, restart from noise, 10 steps
    /// (styles from cache; euler c1/c2 carry LeRobot semantics with the
    /// buffer contents held).
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
            // sized: under HYBRID the pk/pv content is p real rows < the
            // max-sized bind.
            ctx.copy_h2d_async_sized(buf, &bytes, stream).expect("serve h2d");
        };
        assert_eq!(kv.len(), self.depth * 2, "serve kv count mismatch");
        // Range padding: swap the flow OM handle by bucket boundary P,
        // upload rope content by true L (flow token positions = vis+L+i,
        // independent of P; tables are HOR-shaped); refresh the mask.
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
            h2d(&self.s.binds[0], &x); // state = current x
            let oref: Vec<&DeviceBuffer> = vec![&self.ob];
            self.s.run_seg(gi, &oref, stream).expect("ge run");
            let _ = stream.synchronize();
            x = download_f16(ctx, &self.ob, (HOR * ADIM) as usize);
        }
        x
    }

    /// One flow frame (GEB_SERVE_FAST): the 36 pk/pv go over async d2d
    /// (straight from prefix routs) -> noise h2d -> 10 steps chained on
    /// one stream (zero-copy style views, x resident via d2d, no
    /// per-step sync) -> one sync at the end + actions d2h. Bit-identical
    /// to one_frame: same byte streams (d2d and views are exact bytes)
    /// executed in the same stream order — moving syncs only moves wait
    /// points, not the computation order.
    fn one_frame_fast(
        &mut self,
        ctx: &AscendContext,
        stream: &AscendStream,
        kv_dev: &[&DeviceBuffer],
        token_ids: &[u32],
        noise: &[f16],
    ) -> Vec<f16> {
        assert_eq!(kv_dev.len(), self.depth * 2, "serve kv count mismatch");
        assert!(!self.style_views.is_empty(), "fast path requires resident styles");
        // Range padding: swap the flow OM handle by bucket boundary P,
        // upload rope content by true L; refresh the mask.
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
        // Timed: kv (36x d2d) / h2d_enq / steps (per-step enqueue) +
        // sync / dl.
        let tim = serve_timing();
        let t0 = std::time::Instant::now();
        for i in 0..self.depth * 2 {
            // Gear mode: both sides are max-gear buffers (d2d copies in
            // full; the branch desc trims trailing invalid rows). The
            // equal-length assert stays for static mode.
            if self.s.gear_payloads.is_empty() {
                assert_eq!(
                    kv_dev[i].len(),
                    self.s.binds[self.pk_base + i].len(),
                    "fast kv size mismatch (lane {i})"
                );
            }
            ctx.copy_d2d_async(&self.s.binds[self.pk_base + i], kv_dev[i], stream)
                .expect("serve d2d kv");
        }
        let t1 = std::time::Instant::now();
        // Zero-copy byte view (the noise borrow outlives the function).
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
            // Gear mode: ins assembled here (run_seg's ins() knows
            // nothing of style-view substitution) — run_gear directly
            // with payload gi.
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
                // x stays resident: this step's output feeds the next
                // step's state (same-stream ordering; no d2h+h2d+sync).
                ctx.copy_d2d_async(&self.s.binds[0], &self.ob, stream).expect("serve d2d x");
            }
        }
        let t4 = std::time::Instant::now();
        let _ = stream.synchronize();
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
}

pub fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GEB_NORM32: torch GemmaRMSNorm alignment — variance in fp32. Dropping
/// the fp32 upcast flipped behavior in the norm16 control; AddRmsNorm's
/// fp32 desc is fake support (GE casts back to an f16 kernel), so the
/// composition is hand-built.
pub fn norm32_enabled() -> bool {
    std::env::var("GEB_NORM32").is_ok()
}

/// GEB_SERVE_FAST: device-direct segment handoff in serve — prefix's 36
/// kv outputs go over async d2d (zero host relay), flow x stays resident
/// (per-step d2h->h2d becomes a same-stream d2d), per-step syncs are
/// dropped, styles become device-resident (one upload, per-step views).
/// Bit-identical numerics: same byte streams in the same stream order.
/// An empty value does not count as on (avoids accidental trips in
/// supervisor env passthrough).
pub fn serve_fast() -> bool {
    std::env::var("GEB_SERVE_FAST").map(|v| !v.is_empty()).unwrap_or(false)
}

/// GEB_SERVE_TIMING: per-stage timing inside the serve hot path
/// (conv/h2d/run/sync/dl) for glue-overhead attribution. Instant stamps
/// only; data flow untouched.
pub fn serve_timing() -> bool {
    std::env::var("GEB_SERVE_TIMING").is_ok()
}

/// Host linear layer (full f32; dims like 1x32x3072, negligible cost).
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

/// te [AW] -> conditioning [ADIM] (two time_mlp layers, each
/// matmul -> bias -> silu).
pub fn e2e_conditioning(real: &Pi05Weights, te: &[f32]) -> Vec<f32> {
    let h = host_lin_f32(&real.time_mlp_in, te, te.len());
    let h: Vec<f32> = h.iter().map(|&v| silu_f32(v)).collect();
    let c = host_lin_f32(&real.time_mlp_out, &h, h.len());
    c.iter().map(|&v| silu_f32(v)).collect()
}

/// Style projection [ADIM,3W] -> (scale=1+s0, shift=s1, gate=s2). The
/// third segment is openpi _gated_residual's branch gate (residual +
/// branch*gate, no activation, direct f16 cast) — the engine used to
/// drop it entirely, leaving flow output near-random (the main cause of
/// a 0/10 closed loop).
pub fn e2e_style_pair(w: &LinearWeights, cond: &[f32], width: usize) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
    let raw = host_lin_f32(w, cond, cond.len());
    let scl = raw[..width].iter().map(|&v| f16::from_f32(1.0 + v)).collect();
    let sh = raw[width..2 * width].iter().map(|&v| f16::from_f32(v)).collect();
    let gt = raw[2 * width..3 * width].iter().map(|&v| f16::from_f32(v)).collect();
    (scl, sh, gt)
}

/// Three-segment OM preload cache (cold-start lazy): loaded in parallel
/// before GeServe::open (locked on the ge_builder side), consumed by
/// e2e_run on hit — collapses the serial 52s OM load to ~max(one
/// segment).
static OM_PRELOAD: std::sync::Mutex<
    Option<std::collections::HashMap<String, ge_builder::GeGraph>>,
> = std::sync::Mutex::new(None);

/// Preloads {vision,prefix,flow}_real.om from GEB_OM_DIR in parallel
/// (idempotent; called once on the serve cold-start path). Loader
/// threads must bind the main context — aclmdlLoadFromMem follows the
/// calling thread's current context (a bare thread hits rc=-3). Failures
/// do not panic: e2e_run falls back to serial loading with the real
/// error.
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
                // Bind the context (&AscendContext is not Send — the
                // transmute borrows by reference; ctx outlives the join,
                // and this is a controlled one-shot cold-start path).
                if let Err(e) = ctx2.set_current() {
                    eprintln!("[preload] {seg} context bind failed: {e:?}");
                    return (seg, None);
                }
                let g = ge_builder::load(&path);
                if let Err(e) = &g {
                    eprintln!("[preload] {seg} OM load failed {path}: {e:?}");
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
    println!("[preload] {} segment OMs loaded in parallel {:?}", map.len(), t0.elapsed());
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
            println!("[e2e] {seg} OM preload hit");
            g
        }
        None => {
            println!("[e2e] loading {path}");
            ge_builder::load(&path).expect("load om")
        }
    };
    // Gear mode: bring-up content is max-gear materialized — run the
    // last gear (static = run_seg gear 0).
    let gi = s.gear_payloads.len().saturating_sub(1);
    if !s.gear_payloads.is_empty() {
        let n = s.g.gear_count().unwrap_or(0);
        println!("[gear] {seg} OM has {n} gears (bring-up runs the last)");
    }
    let n_out = s.g.num_outputs().unwrap();
    let outs: Vec<DeviceBuffer> = (0..n_out)
        .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("out malloc"))
        .collect();
    let refs: Vec<&DeviceBuffer> = outs.iter().collect();
    s.run_seg(gi, &refs, stream).expect("ge run");
    let _ = stream.synchronize();
    // Gear mode: buffers are max-gear sized (712 rows) while this gear
    // writes only the first 666 — d2h must span the whole buffer
    // (copy_d2h length assert), then the host truncates to
    // cur_output_dims rows.
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
    // GEB_E2E_BENCH=N: steady-state rerun bench (same input binds; the
    // full-output d2h is included — prefix's 36 kv downloads are the
    // production host-relay glue and are charged as such).
    let nb = envi("GEB_E2E_BENCH", 0) as usize;
    if nb > 0 {
        let mut ts = Vec::with_capacity(nb);
        for _ in 0..nb {
            let t = std::time::Instant::now();
            s.run_seg(gi, &refs, stream).expect("ge run");
            let _ = stream.synchronize();
            let _: Vec<_> = (0..n_out)
                .map(|i| download_f16(ctx, &outs[i], s.g.output_size(i).unwrap() / 2))
                .collect();
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "[e2e] {seg} steady bench x{nb}: P50={:.2}ms min={:.2} max={:.2} (run + sync + {n_out} d2h)",
            ts[nb / 2],
            ts[0],
            ts[nb - 1]
        );
    }
    dl
}

pub fn seg_e2e(be: &AscendBackend, bench: bool, real: Option<&Pi05Weights>) {
    let _ = bench;
    // e2e reads *_real.om; the Seg rebuild must use the same env (input
    // order must match).
    let need = |k: &str| std::env::var(k).is_ok();
    assert!(
        std::env::var("GEB_ATTN").is_ok()
            && need("GEB_QKV3") && need("GEB_ROPEFLAT") && need("GEB_WCONST"),
        "GEB_SEG=e2e requires GEB_ATTN(=manual|pfa) GEB_QKV3=1 GEB_ROPEFLAT=1 GEB_WCONST=1 \
         (must match how *_real.om under GEB_OM_DIR was built)"
    );
    let real = real.expect("GEB_SEG=e2e requires GEB_CKPT (token_embedding/time_mlp/style projections are consumed on the host)");
    let tokens = envi("GEB_TOKENS", 64) as usize;
    // Cold-start lazy: preload the three OMs in parallel (e2e_run picks
    // them up).
    preload_e2e_oms(be.ctx());
    // Note: GE static OMs accept non-multiple-of-16 M (GEB_TOKENS=200 ->
    // P=968 compiled and ran fine; the eager aclnn 16x-M crash does not
    // apply to the GE path).
    let mut st = {
        // synthetic bring-up frame (smoke): real frames arrive via serve spool
        let mut seed = 0xE2E2u32;
        let ids = (0..tokens).map(|i| 1000 + i as u32 * 7).collect();
        E2eStage {
            patches: rand_f16((VT * V_PATCH_W) as usize, &mut seed, 300.0),
            token_ids: ids,
            noise: rand_f16((HOR * ADIM) as usize, &mut seed, 1.0),
            vision_out: Vec::new(),
            kv: Vec::new(),
            actions: Vec::new(),
            vmech: None,
            pmech: None,
            fmech: None,
        }
    };
    let t0 = std::time::Instant::now();
    seg_vision(be, Some(real), Some(&mut st));
    let t1 = std::time::Instant::now();
    println!("[e2e] vision segment {:?} (incl. OM load)", t1.duration_since(t0));
    seg_prefix(be, Some(real), Some(&mut st));
    let t2 = std::time::Instant::now();
    println!("[e2e] prefix segment {:?} (incl. OM load + 2.1GB embedding lookup)", t2.duration_since(t1));
    if std::env::var("GEB_E2E_STOP").ok().as_deref() == Some("prefix") {
        // bisect hook: run only vision+prefix (e.g. a GEB_DEPTH=2 small
        // graph to verify output-slot reuse)
        ge_builder::fini().expect("fini");
        println!("GE_E2E_PROBE_OK");
        return;
    }
    seg_flow(be, Some(real), Some(&mut st));
    let t3 = std::time::Instant::now();
    println!("[e2e] flow segment {:?} (incl. OM load + 10 steps)", t3.duration_since(t2));
    let head: Vec<f32> = st.actions[..8.min(st.actions.len())].iter().map(|v| v.to_f32()).collect();
    println!("[e2e] actions head={head:?}");
    // GEB_E2E_SERVE=<dir>: resident closed-loop eval service (mechs
    // already sunk into E2eStage by each segment's e2e branch; multiple
    // buckets = one serve process per bucket on the Python side).
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
// GEB_E2E_SERVE resident service loop (closed-loop eval).
// ---------------------------------------------------------------------------

/// GEB_E2E_SERVE=<dir>: resident file-polling service. One process per
/// bucket (GEB_TOKENS=L, GEB_OM_DIR=tl{L}); the Python side picks a
/// bucket per frame with one serve process per bucket (supervisor spawns
/// inside the apxinf_rust container — the engine binary binds the 9.0.1
/// GE; the eval's torch side lives in apxinf_npu and they meet over the
/// /data shared directory).
/// Protocol (spool dir, tmp+rename atomic delivery, single serial
/// client):
///   ready              — three mechs ready marker (content "L=<tokens>")
///   req_{seq:06}.bin   — u32 magic 0x47514531 / u32 L / patches f32
///                        x VT*V_PATCH_W / token_ids u32 x L / noise f32
///                        x HOR*ADIM
///   resp_{seq:06}.bin  — u32 magic 0x47525331 / u32 status /
///                        f32 vision_ms/prefix_ms/flow_ms /
///                        actions f32 x HOR*7 (first 7 cols of the x_t
///                        terminal state)
///   shutdown           — exit signal
pub fn serve_loop(be: &AscendBackend, st: &mut E2eStage, dir: &str, tokens: usize) {
    // Frame implementation is shared with the in-process path
    // (GeServe::infer) — the spool is just the IO shell.
    let mut pipe = GeServe {
        vm: st.vmech.take().expect("serve requires the vision mech"),
        pm: st.pmech.take().expect("serve requires the prefix mech"),
        fm: st.fmech.take().expect("serve requires the flow mech"),
        tokens,
        fast: serve_fast(),
    };
    std::fs::create_dir_all(dir).expect("serve spool create");
    // Clear stale files (last run's req/resp/ready would confuse seq and
    // readiness). Only protocol files are deleted: pid is the
    // supervisor's liveness marker and stdout.log is this process's
    // redirect target — deleting those makes the supervisor judge the
    // process dead and respawn a second engine onto the same spool dir
    // (this has happened).
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
    println!("[serve] ready: {dir} L={tokens} (file polling; shutdown exits)");
    let mut served: u64 = 0;
    let exp_patches = (VT * V_PATCH_W) as usize;
    let exp_noise = (HOR * ADIM) as usize;
    loop {
        if std::path::Path::new(&format!("{dir}/shutdown")).exists() {
            println!("[serve] shutdown signal, exiting (served {served} requests)");
            return;
        }
        // Process the smallest live seq (no last_seq watermark: an eval
        // restart rewinds the python seq to 1 and a watermark would
        // discard the new requests as stale; with one serial client,
        // "smallest req in the dir" is the correct order — a dead
        // request from the previous process is at worst run for nothing).
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
        assert_eq!(magic, 0x4751_4531, "serve request magic mismatch");
        let l = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
        // Static mode = single-L bucket (L is pinned); gear mode checks
        // length per request and picks a gear (an off-table L panics in
        // gear_idx — the supervisor's gear table is the lineage's full
        // set); prefix pool mode (GEB_PREFIX_POOL) carries any L via a
        // handle swap.
        if pipe.pm.s.gear_payloads.is_empty() && pipe.pm.pool_root.is_none() {
            assert_eq!(
                l, tokens,
                "serve request L={l} != this process's bucket GEB_TOKENS={tokens} (python picked the wrong bucket)"
            );
        }
        assert_eq!(
            raw.len(),
            8 + exp_patches * 4 + l * 4 + exp_noise * 4,
            "serve request length mismatch (seq {seq} L={l})"
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
        // actions [HOR, ADIM] -> first 7 columns (LIBERO deployable dims;
        // normalized_actions is a slice of the x_t terminal state, denorm
        // happens in torch postprocess).
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
// vision segment: patch embed -> depth x SigLIP layers -> post LN ->
// projector.
// ---------------------------------------------------------------------------

pub fn seg_vision(be: &AscendBackend, real: Option<&Pi05Weights>, e2e: Option<&mut E2eStage>) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH_VISION", envi("GEB_DEPTH", 27)) as usize;
    let t = VT;
    let vqd = V_HEADS * V_HD; // 1152
    let qkvw = vqd * 3;
    // GEB_ATTN=manual: hand-built attention (bmm+softmax+bmm, all
    // static). Default is pfa (PromptFlashAttention — in a static OM its
    // InnerPFA host callback stalls ~20ms/layer).
    let attn_manual = std::env::var("GEB_ATTN").map(|v| v == "manual").unwrap_or(false);
    let vscale = if attn_manual { 1.0f32 / (V_HD as f32).sqrt() } else { 1.0 };
    // GEB_QKV3: independent q/k/v projections (3 mms + separate weight
    // inputs) instead of one fused qkv mm + SliceD x3. msprof showed the
    // slice views' downstream consumers (TransposeD/PFA) trigger 3
    // runtime MemcopyAsync materializations per layer (~1.77MB each, with
    // ~4.5-12ms host stalls) — the main source of ~450ms of vision gaps
    // (81 per execution).
    let qkv3 = std::env::var("GEB_QKV3").is_ok();
    let mut seed = 0xC0DEu32;
    let mut s = Seg::new("ge_vision");

    // ---- Segment inputs (registration order == runtime bind order) ----
    // 0 patches / 1 patch_wt / 2 patch_b / 3 pos_rep / 4 zeros / 5
    // shp_qkv3 / 6 shp_flat2 / 12 per layer (16 under GEB_QKV3) / 4 tail
    // items (pnw,pnb,projwt,projb). e2e supplies real patches; the
    // default is random (same values on both paths).
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
        // position table, cyclically repeated (cuda kernel semantics:
        // row r reads table[r % tpv]).
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
    // Shape inputs are always Consts (as Data they turn descs unknown and
    // stall host scheduling).
    s.const_i32("shp_v3", &[VIEWS as i32, VPV as i32, vqd as i32]);
    s.const_i32("shp_flat2", &[t as i32, vqd as i32]);
    s.const_i32("nsh", &[VW as i32]); // LayerNormV4 normalized_shape
    // head-major bridge shapes for manual attention
    // (headsplit/headmerge).
    s.const_i32("shp_at3", &[(VIEWS * V_HEADS) as i32, VPV as i32, V_HD as i32]);
    s.const_i32("shp_vs4", &[VIEWS as i32, VPV as i32, V_HEADS as i32, V_HD as i32]);
    s.const_i32("shp_vh4", &[VIEWS as i32, V_HEADS as i32, VPV as i32, V_HD as i32]);

    // ---- Graph: patch embed ----
    let mut cur = s.mm("pemb", "patches", &[t, V_PATCH_W], "patch_wt", &[VW, V_PATCH_W], &[t, VW]);
    cur = s.bias("pemb_b", &cur, &[t, VW], "patch_b");
    cur = s.add2("pemb_p", &cur, "pos_rep", &[t, VW]);

    // ---- Layer loop ----
    let mut layer_bases = Vec::with_capacity(depth);
    for i in 0..depth {
        let base = s.binds.len();
        layer_bases.push(base);
        let p = format!("l{i}_");
        // Real weights: SigLIP blocks are indexed by layer (GEB_DEPTH
        // smoke runs <= 27).
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
            // Three host blocks [VW, vqd]; the fused path concatenates to
            // [VW, qkvw] before the transposed upload. Under manual
            // attention the q block (weight+bias) is premultiplied by
            // 1/sqrt(hd) — the graph has no scale operator (PFA carries it
            // in scale_value instead).
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
            // Real-weight qkv bias = q.bias || k.bias || v.bias (each
            // SigLIP projection has its own bias).
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
                // Independent weight/bias inputs (one host split, no
                // runtime materialization); wq_s/bias_s already carry the
                // manual-attention vscale fold. The fused qkvwt/qkvb is
                // deliberately not registered in this mode — an input with
                // no graph consumer may be eliminated from the model's
                // input list, shifting every later input index.
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

        // attention: manual = [b*h,s,d] bridge + bmm/softmax/bmm (all
        // static); pfa = batch PromptFlashAttention.
        let ln1 = s.addln(&format!("{}ln1", p), &cur, &format!("{}n1w", p), &format!("{}n1b", p), "nsh", &[t, VW]);
        let (q2, k2, v2) = if qkv3 {
            let qm = s.mm(&format!("{}qm", p), &ln1, &[t, VW], &format!("{}qw", p), &[vqd, VW], &[t, vqd]);
            let km = s.mm(&format!("{}km", p), &ln1, &[t, VW], &format!("{}kw", p), &[vqd, VW], &[t, vqd]);
            let vm = s.mm(&format!("{}vm", p), &ln1, &[t, VW], &format!("{}vw", p), &[vqd, VW], &[t, vqd]);
            let qb = s.bias(&format!("{}qb_", p), &qm, &[t, vqd], &format!("{}qb", p));
            let kb = s.bias(&format!("{}kb_", p), &km, &[t, vqd], &format!("{}kb", p));
            let vb = s.bias(&format!("{}vb_", p), &vm, &[t, vqd], &format!("{}vb", p));
            (qb, kb, vb)
        } else {
            let qkv = s.mm(&format!("{}qkv", p), &ln1, &[t, VW], &format!("{}qkvwt", p), &[qkvw, VW], &[t, qkvw]);
            let qkvb = s.bias(&format!("{}qkvb", p), &qkv, &[t, qkvw], &format!("{}qkvb", p));
            let q2 = s.slice2(&format!("{}q2", p), &qkvb, &[t, qkvw], 0, vqd);
            let k2 = s.slice2(&format!("{}k2", p), &qkvb, &[t, qkvw], vqd, vqd);
            let v2 = s.slice2(&format!("{}v2", p), &qkvb, &[t, qkvw], vqd * 2, vqd);
            (q2, k2, v2)
        };
        let attn = if attn_manual {
            // head-major bridge (see headsplit) — a direct Reshape would
            // mispermute.
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

        // mlp (exact gelu)
        let ln2 = s.addln(&format!("{}ln2", p), &res1, &format!("{}n2w", p), &format!("{}n2b", p), "nsh", &[t, VW]);
        let fc1 = s.mm(&format!("{}fc1", p), &ln2, &[t, VW], &format!("{}fc1wt", p), &[V_INTER, VW], &[t, V_INTER]);
        let fc1b = s.bias(&format!("{}fc1b", p), &fc1, &[t, V_INTER], &format!("{}fc1b", p));
        let act = s.gelu(&format!("{}act", p), &fc1b, &[t, V_INTER], false);
        let fc2 = s.mm(&format!("{}fc2", p), &act, &[t, V_INTER], &format!("{}fc2wt", p), &[VW, V_INTER], &[t, VW]);
        let fc2b = s.bias(&format!("{}fc2b", p), &fc2, &[t, VW], &format!("{}fc2b", p));
        cur = s.add2(&format!("{}out", p), &fc2b, &res1, &[t, VW]);
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
    s.finish(&[&out]);
    println!("vision OM built: depth={depth} n_in={} n_out=1", s.ins().len());

    if let Some(st) = e2e {
        // e2e: load vision_real.om, run once, capture the post-projector
        // output [VT, PW].
        let outs = e2e_run(ctx, stream, &mut s, "vision");
        // The vision OM carries 108 LN aux outputs (dead mean/rstd would
        // blow up y) — the main output (post-projector) is idx0.
        assert!(outs.len() >= 1);
        st.vision_out = outs[0].clone();
        let m = st.vision_out.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
        println!("[e2e] vision_out |max|={m:.4}");
        // GEB_E2E_SERVE: sink the mech into E2eStage (per-frame execution
        // driven by seg_e2e's serve loop; routs allocated once, reused
        // across requests).
        if std::env::var("GEB_E2E_SERVE").is_ok() {
            let n_out = s.g.num_outputs().unwrap();
            let routs: Vec<DeviceBuffer> = (0..n_out)
                .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("serve out malloc"))
                .collect();
            st.vmech = Some(VisionMech { s, routs });
        }
        return;
    }
    ge_builder::fini().expect("fini");
    println!("GE_VISION_PROBE_OK");
}

// ---------------------------------------------------------------------------
// prefix segment: depth x language layers in order, emitting each layer's
// k/v (the last layer emits no tail).
// ---------------------------------------------------------------------------

/// int8 weight registration: s_eff-absorb quantization -> wq int8 [n,k]
/// (Data or Const) + scale f32 [n] + a smooth/inv f16 row Data (native
/// switch picks which). GEB_INT8_DUMP also writes L{nn}_{proj}.wq/.scale
/// for comparison against the int8_weights_v1.npz golden. Returns the
/// (wq, ws, smooth|inv) names.
#[allow(clippy::type_complexity)]
fn reg_i8(
    s: &mut Seg, ctx: &AscendContext, facs: &std::collections::HashMap<String, Int8Smooth>,
    key: &str, dump_tag: &str, w32: &[f32], k: i64, n: i64, native: bool, dump: Option<&str>,
) -> (String, String, String) {
    let f = facs.get(key).unwrap_or_else(|| panic!("int8 smooth missing {key}"));
    assert_eq!(f.s_eff.len(), k as usize, "{key} factor length {} != k={k}", f.s_eff.len());
    let (wq, scale) = quant_w8(w32, k as usize, &f.s_eff);
    if let Some(dir) = dump {
        let _ = std::fs::create_dir_all(dir);
        let wqb: Vec<u8> = wq.iter().map(|&v| v as u8).collect();
        std::fs::write(format!("{dir}/{dump_tag}.wq"), &wqb).expect("dump wq");
        let sb: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/{dump_tag}.scale"), &sb).expect("dump scale");
    }
    // GEB_INT8_WCONST=1 (default follows GEB_WCONST): bake wq/ws as
    // Consts — int8 Data weights pay a per-execution ND->NZ TransData
    // tax (prefix measured 129.5ms vs f16's 105 — the ~25ms delta is
    // 2GB/63GB/s). Consts fold the format at compile time, tax-free
    // (same as f16 wt()). The tiny [k] f16 smooth row stays Data.
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
        // Gear mode dilemma and its resolution: a purely static Data
        // poisons SetDims gear matching (measured); constant dynamic
        // dims would overflow the payload's ACL_MAX_DIM_CNT=128 (full
        // depth: 18 layers x 9 inputs + 24 base = 186). Default (=1) is
        // Const — the earlier "Const hits the legacy wall rc=-7" was a
        // misdiagnosis of a dynamicDims option-string bug (the old
        // option was polluted by payload static trailing dims; nothing
        // to do with Const).
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
    // The bias-fold-into-operator experiment is dead: an int32 Const
    // input fails aclgrphBuildModel with rc=-7 (the legacy operator's
    // bias branch is unusable, measured). Callers simply drop the bias
    // op instead (Gemma bias == 0 and int8 already asserts real weights
    // — zero semantic loss).
    (wq_n, ws_n, sm_n)
}

/// Unified int8/f16 projection: int8 -> QuantMatmulDequant (native
/// smooth feeds the operator directly; GEB_INT8_NATIVE=0 falls back to a
/// gatew row premultiply); f16 -> MatMulV2 (w16 = the original weight
/// segment name). Both paths' w dims are [n,k]. Bias is the caller's
/// business (the int8 path drops the op outright — Gemma bias == 0).
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

/// Host f16 member weights inside an int8 layer: once the group norm
/// returns to the torch domain (g1/g2 = 1+g), the engine's folded
/// weights (w*(1+g)) must divide the fold back out before f16 — else
/// the fold double-counts. unfold=false (group not on int8) keeps the
/// engine values (= lw_f16 semantics).
fn lin_f16_unfold(l: &LinearWeights, out_dim: usize, unfold: bool, g: &[f32]) -> Vec<f16> {
    let w32 = l.weight.to_f32_vec().unwrap();
    let w = if unfold {
        assert_eq!(g.len(), w32.len() / out_dim, "unfold factor length mismatch");
        w32.into_iter()
            .enumerate()
            .map(|(i, v)| v / g[i / out_dim])
            .collect::<Vec<f32>>()
    } else {
        w32
    };
    w.into_iter().map(f16::from_f32).collect()
}

pub fn seg_prefix(be: &AscendBackend, real: Option<&Pi05Weights>, e2e: Option<&mut E2eStage>) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH_PREFIX", envi("GEB_DEPTH", 18)) as usize;
    let tokens = envi("GEB_TOKENS", 64);
    // GEB_PREFIX_DROP_EMPTY: trailing empty-view rows (the 9/10 semantic
    // root-cause fix). LIBERO feeds only 2 real views; empty_camera takes
    // the missing path (-1 pad images enter SigLIP but pad_mask=0):
    // make_att_2d_masks' pad_2d hides those 256 keys from everyone and
    // position_ids=cumsum(pad)-1 collapses text positions to 512..711.
    // Pad columns contribute exactly 0 to softmax and visible tokens sit
    // exactly on arange, so this is mathematically the same as dropping
    // the view rows outright (968 -> 712; theory_check.py measured 0.128%
    // on the affected rows). The engine used to treat pad views as
    // first-class (visible keys squatting on 512..767) — the true root
    // cause of the 314% e2e drift.
    let drop_v = envi("GEB_PREFIX_DROP_EMPTY", 0);
    let p = VT + tokens - drop_v; // prefix length (multiple-of-16 discipline)
    // Dynamic gears (GEB_DYN_LLIST): every p in a desc becomes -1 (gd());
    // content/buffers still materialize at p (in gear mode tokens must
    // equal the gear table's max L — rope tables/zeros/swaps are
    // row-wise p-independent, and a full p_max upload is prefix
    // truncation). L-dependent Consts (qswap_c/kswap_c) become Data[-1]
    // in gear mode. The gear p values feed Seg; finish() composes
    // ge.dynamicDims + the per-gear S2 payloads.
    // GEB_PREFIX_POOL: prefix switches to static assembly + a per-L
    // static OM pool with handle swaps — the verdict was that dynamic
    // graph structure (-1-dim Reshape, rope/qswap as Data) drifts int8
    // layer-by-layer by 3x (13.1% actions-level cross-frame; a 2-gear
    // control falsified the range-tiling hypothesis). The llist stays
    // for the flow pool and client-side validation.
    let gear_ll = if std::env::var("GEB_PREFIX_POOL").is_ok() {
        println!("[prefix-pool] GEB_PREFIX_POOL active: static prefix assembly + per-L pool");
        None
    } else {
        dyn_llist()
    };
    let mut gear_ps_hold: Option<Vec<i64>> = None;
    let gd = |v: i64| if gear_ll.is_some() { -1 } else { v };
    if let Some(g) = &gear_ll {
        assert!(
            std::env::var("GEB_WCONST").is_ok(),
            "GEB_DYN_LLIST requires GEB_WCONST=1 (weights Const in one root graph — the precondition for zero duplication across gears)"
        );
        assert!(std::env::var("GEB_ROPEFLAT").is_ok(), "GEB_DYN_LLIST requires GEB_ROPEFLAT=1");
        assert!(
            std::env::var("GEB_ATTN").map(|v| v == "manual").unwrap_or(false),
            "GEB_DYN_LLIST requires GEB_ATTN=manual"
        );
        assert!(std::env::var("GEB_QKV3").is_ok(), "GEB_DYN_LLIST requires GEB_QKV3=1");
        let lmax = *g.iter().max().unwrap();
        assert_eq!(tokens, lmax, "GEB_DYN_LLIST requires GEB_TOKENS = the gear table's max L (binds materialize at max)");
        let ps = g.iter().map(|&l| VT + l - drop_v).collect::<Vec<i64>>();
        println!("[gear] prefix dynamic gears: L {:?} -> p {:?}", g, ps);
        gear_ps_hold = Some(ps);
    }
    let mut seed = 0xBEEFu32;
    // GEB_QKV3: independent q/k/v projections (same rationale as the
    // vision segment — the SliceD column views' runtime MemcopyAsync
    // materialization was the main source of prefix's 750ms).
    let qkv3 = std::env::var("GEB_QKV3").is_ok();
    // GEB_ATTN=manual: hand-built GQA attention (bmm+softmax+bmm + k/v
    // TileD broadcast) replacing PFA — PFA is a transformer-API-style
    // operator that runs a host launcher inside static OMs (host tiling
    // + staging per execution). pscale folds into the q weight/bias.
    let attn_manual = std::env::var("GEB_ATTN")
        .map(|v| v == "manual").unwrap_or(false);
    let pscale = if attn_manual { 1.0f32 / (HD as f32).sqrt() } else { 1.0 };
    // GEB_PREFIX_INT8 (production): 7 projections -> QuantMatmulDequant.
    // Factors come from GEB_INT8_SMOOTH (default
    // ./smooth_calib_v1_engine.safetensors, converted from the torch-domain
    // npz with the Gemma 1+g fold correction). GEB_INT8_PROJS selects a
    // subset for staged bring-up (default all 7); GEB_INT8_NATIVE=0 falls
    // back to a TileD+Mul row premultiply. GEB_INT8_DUMP=<dir> writes the
    // quantized artifacts for comparison against int8_weights_v1.npz.
    let int8 = prefix_int8();
    let (facs, i8norms): (
        std::collections::HashMap<String, Int8Smooth>,
        std::collections::HashMap<String, [Vec<f32>; 2]>,
    ) = if int8 {
        assert!(real.is_some(), "GEB_PREFIX_INT8 requires real weights via GEB_CKPT");
        assert!(qkv3, "GEB_PREFIX_INT8 requires GEB_QKV3=1 (fused qkv cannot take per-matrix smooth)");
        let path = std::env::var("GEB_INT8_SMOOTH")
            .unwrap_or_else(|_| "./smooth_calib_v1_engine.safetensors".into());
        println!("[int8] prefix int8 on: smooth={path}");
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

    // ---- Segment inputs ----
    // 0 x0 / 1 zeros / 2..7 flat rope sextet / 8..11 rank-2 rope tables
    // / 10 per layer. e2e x0 = vision_out || token_embedding[token_ids]
    // (embed_prefix order: vision rows first; state does not enter
    // prefix — PI-0.5 discretizes state into the prompt text).
    let x0_h = match (&e2e, real) {
        (Some(st), Some(w)) => {
            assert!(!st.vision_out.is_empty(), "e2e: run the vision segment first");
            assert_eq!(st.vision_out.len(), (VT * PW) as usize);
            assert_eq!(st.token_ids.len(), tokens as usize, "token ids != GEB_TOKENS");
            // One-shot f32 materialization of [vocab,PW] (~2.1GB peak,
            // discarded after the lookup).
            let emb = w.vision.token_embedding.to_f32_vec().unwrap();
            let vocab_w = PW as usize;
            // LeRobot embed_language_tokens semantics: lookup rows *
            // sqrt(width) (gemma embed scale; the vision segment has no
            // such scale). Empty-view drop: only the first (VT-drop_v)
            // vision_out rows are kept (view-major, empty views trail).
            let lang_scale = (PW as f32).sqrt();
            let vis_rows = ((VT - drop_v) * PW) as usize;
            let mut x0 = st.vision_out[..vis_rows].to_vec();
            x0.reserve(st.token_ids.len() * vocab_w);
            for &id in &st.token_ids {
                let r = id as usize * vocab_w;
                assert!(r + vocab_w <= emb.len(), "token id {id} beyond vocab");
                x0.extend(emb[r..r + vocab_w].iter().map(|&v| f16::from_f32(v * lang_scale)));
            }
            x0
        }
        (None, None) => panic!("e2e requires GEB_CKPT (token_embedding lookup)"),
        _ => rand_f16((p * PW) as usize, &mut seed, 100.0),
    };
    s.data(ctx, "x0", &[gd(p), PW], &x0_h);
    // zeros gets explicit gearing here (data_zeros stays purely static —
    // shared with the vision segment): all-zero content is row-wise
    // p-independent, and a full p_max upload is prefix truncation.
    {
        let zh = vec![f16::from_f32(0.0); (p * PW) as usize];
        s.data(ctx, "zeros", &[gd(p), PW], &zh);
    }
    // GEB_ATTN_MASK (range padding): additive softmax mask input [1,1,p]
    // — baked as all-zero placeholder (no pad); serve/probe fills it per
    // true L (rows [vis+L, vis+P) = -30000; -inf is illegal in f16). One
    // Data shared by all layers.
    let attn_mask_n: Option<String> = if envi("GEB_ATTN_MASK", 0) == 1 {
        let mh = vec![f16::from_f32(0.0); p as usize];
        Some(s.data(ctx, "padmask", &[1, 1, p], &mh))
    } else {
        None
    };
    let (qc, qs, qi) = rope_flat_const(p, HEADS, 0);
    let (kc, ks, ki) = rope_flat_const(p, KV_HEADS, 0);
    let qf = (p * HEADS * 2, HD / 2);
    let kf = (p * KV_HEADS * 2, HD / 2);
    // Gear mode: dim0 always -1 with a registered mul (qf.0 =
    // p*(HEADS*2)); content materializes at p_max (row-wise
    // p-independent — prefix truncation).
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
    // rank-2 tables (physical layout of [t*heads, d] matches
    // [t, heads*d]; descs are rank-2).
    let (qc2, qs2) = rope_rank2_const(p, HEADS, 0);
    let (kc2, ks2) = rope_rank2_const(p, KV_HEADS, 0);
    s.data(ctx, "qcos2", &[gd(p), QD], &qc2);
    s.data(ctx, "qsin2", &[gd(p), QD], &qs2);
    s.data(ctx, "kcos2", &[gd(p), KVD], &kc2);
    s.data(ctx, "ksin2", &[gd(p), KVD], &ks2);
    // Reshape shape tensors (rank-3 bridges; Const content is
    // gear-invariant — the p slot carries -1 and branch inference
    // resolves it per gear).
    s.const_i32("shp_q3", &[1, gd(p) as i32, QD as i32]);
    s.const_i32("shp_k3", &[1, gd(p) as i32, KVD as i32]);
    s.const_i32("shp_v3", &[1, gd(p) as i32, KVD as i32]);
    // head-major bridge shapes for manual GQA attention
    // (headsplit/headmerge, views=1).
    s.const_i32("shp_mq4", &[1, gd(p) as i32, HEADS as i32, HD as i32]);
    s.const_i32("shp_mq3", &[HEADS as i32, gd(p) as i32, HD as i32]);
    s.const_i32("shp_ma4", &[1, HEADS as i32, gd(p) as i32, HD as i32]);
    s.const_i32("shp_mflat", &[gd(p) as i32, QD as i32]);
    // GEB_ROPEFLAT: flat rope (row swap on a flat view — no column-split
    // materialization). Swap indices: a static Const when ungearred;
    // Data[-1] in gear mode (Consts are shared across gears but the
    // length varies; Data specializes the desc per gear, and the r^1
    // content is row-wise p-independent).
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

    // ---- Layer loop ----
    let mut layer_bases = Vec::with_capacity(depth);
    let mut cur = "x0".to_string();
    let mut kv_outs: Vec<String> = Vec::new();
    for i in 0..depth {
        let base = s.binds.len();
        layer_bases.push(base);
        let tag = format!("l{i}_");
        let last = i + 1 == depth;
        // Real weights: the host loader already folds the Gemma 1+w scale
        // into q/k/v/gate/up (g1/g2 = ones), so the graph consumes them
        // as-is.
        let lay = real.and_then(|w| w.language_layers.get(i));
        let lay_i = i; // smooth-factor key layer == weight layer
        // int8 q/k/v registration (f32 taken directly, no f16 pass; the
        // manual-attn pscale folds into q — row quantization is invariant
        // to a global constant, only the scale shrinks).
        let (iq8, ik8, iv8) = if int8 {
            let l = lay.expect("GEB_PREFIX_INT8 requires real weights");
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
            // int8 group: norm back to the torch domain (gamma = true
            // (1+g), replacing the loader fold); smooth rows anchor to
            // s_npz like the npz weight face (a pure engine-domain s_eff
            // blows up f16 where the L00 fold is near zero — measured).
            let g = &i8norms[&format!("prefix/L{:02}", lay_i)][0];
            assert_eq!(g.len(), PW as usize, "g1 length mismatch");
            (g.iter().map(|&v| f16::from_f32(v)).collect(), g)
        } else {
            (
                lay.map(|l| t_f16(&l.input_norm_scale))
                    .unwrap_or_else(|| norm_f16(PW as usize, &mut seed, 1.0)),
                &[],
            )
        };
        // Gear mode: bake as Const (constant dynamic dims would overflow
        // the SetDims payload's 128-dim cap — see the smooth
        // registration note).
        if !s.gear_ps.is_empty() {
            s.const_f16(&format!("{}g1", tag), &[PW], &g1_h);
        } else {
            s.data(ctx, &format!("{}g1", tag), &[PW], &g1_h);
        }
        {
            // f16 host blocks are built only when the projection is not
            // int8 (the int8 path registered above); after the group
            // switches to the torch domain the f16 members divide the
            // fold back out — see lin_f16_unfold.
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
            // Gemma projections carry no bias (real weights = zeros).
            let bias = lay
                .map(|_| vec![f16::from_f32(0.0); QKVW as usize])
                .unwrap_or_else(|| rand_f16(QKVW as usize, &mut seed, 4000.0));
            // Manual attention: q block (weight+bias) premultiplied by
            // 1/sqrt(hd) — no scale operator in the graph (PFA carries it
            // in scale_value).
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
                // qkv3 input slots within a layer are 1..7 (after g1).
                if iq8.is_none() {
                    s.wt(ctx, &format!("{}qw", tag), &[QD, PW], &wq_s, PW, QD);
                }
                if ik8.is_none() {
                    s.wt(ctx, &format!("{}kw", tag), &[KVD, PW], &wk, PW, KVD);
                }
                if iv8.is_none() {
                    s.wt(ctx, &format!("{}vw", tag), &[KVD, PW], &wv, PW, KVD);
                }
                // Gear mode: row-repeated registration (row_bcast); the
                // int8 fold does not consume it -> bare row saves memory.
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
        }
        // o/gate/up/down: int8-selected projections register at their use
        // site; the f16 wt blocks stay here as needed.
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
            assert_eq!(g.len(), PW as usize, "g2 length mismatch");
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
        // rank-2 split + rope + Reshape rank-3 bridge. Graph outputs bind
        // the rank-2 nodes (rope's ad / slice's v): binding a Reshape
        // output as a graph output yields a dynamic [-1,-1,-1] desc
        // (shape-tensor driven) and the declared size is ignored.
        let (q2, k2, v2) = if qkv3 {
            let qm = mm_or_qmd(&mut s, &format!("{}qm", tag), &norm1, &[gd(p), PW], iq8.as_ref(), &format!("{}qw", tag), QD, &[gd(p), QD], int8_native);
            let km = mm_or_qmd(&mut s, &format!("{}km", tag), &norm1, &[gd(p), PW], ik8.as_ref(), &format!("{}kw", tag), KVD, &[gd(p), KVD], int8_native);
            let vm = mm_or_qmd(&mut s, &format!("{}vm", tag), &norm1, &[gd(p), PW], iv8.as_ref(), &format!("{}vw", tag), KVD, &[gd(p), KVD], int8_native);
            // int8: bias is folded into the operator (int32 zero) — the
            // TileD+Add pair is skipped.
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
            break; // last layer: no attention/tail (compute_tail=false)
        }
        let (qr2, qr3) = if ropeflat {
            let qf2 = [gd(qf.0), qf.1];
            s.rope2_flat(&format!("{}qr", tag), &q2, &[gd(p), QD], &qf2, "qcos", "qsin", "qswap_c", "shp_qflat", "shp_mflat", "shp_q3")
        } else {
            s.rope2(&format!("{}qr", tag), &q2, &[gd(p), QD], "qcos2", "qsin2", "shp_q3")
        };
        let sq = if attn_manual {
            // head-major bridge + hand-built GQA chain (single-head k/v
            // TileD broadcast; kr3/v3 used as rank-3 directly).
            let q3 = s.headsplit(&format!("{}qh", tag), &qr2, gd(p), 1, gd(p), HEADS, HD, "shp_mq4", "shp_mq3");
            let attn = s.attn_manual_gqa(&format!("{}attn", tag), &q3, &kr3, &v3, HEADS, gd(p), gd(p), HD, attn_mask_n.as_deref());
            s.headmerge(&format!("{}am", tag), &attn, 1, gd(p), HEADS, HD, "shp_ma4", "shp_mflat")
        } else {
            let pfa = s.pfa(&format!("{}pfa", tag), &qr3, &kr3, &v3, &[1, gd(p), QD], &[1, gd(p), KVD], HEADS, KV_HEADS, HD);
            s.squeeze(&format!("{}sq", tag), &pfa, &[1, gd(p), QD], &[gd(p), QD])
        };
        let io8 = if i8sel("o") {
            let l = lay.expect("GEB_PREFIX_INT8 requires real weights");
            Some(reg_i8(&mut s, ctx, &facs, &format!("prefix/L{:02}/o", lay_i), &format!("L{:02}_o", lay_i),
                &l.attention.output.weight.to_f32_vec().unwrap(), QD, PW, int8_native, dump_dir))
        } else {
            None
        };
        let proj = mm_or_qmd(&mut s, &format!("{}proj", tag), &sq, &[gd(p), QD], io8.as_ref(), &format!("{}outwt", tag), PW, &[gd(p), PW], int8_native);
        let projb = if io8.is_some() {
            proj.clone() // int8: bias folded into the operator
        } else {
            s.bias(&format!("{}projb", tag), &proj, &[gd(p), PW], &format!("{}outb", tag))
        };
        let res = s.add2(&format!("{}res", tag), &projb, &cur, &[gd(p), PW]);
        let norm2 = s.addrms(&format!("{}n2", tag), &res, "zeros", &format!("{}g2", tag), &[gd(p), PW]);
        let ig8 = if i8sel("gate") {
            let l = lay.expect("GEB_PREFIX_INT8 requires real weights");
            Some(reg_i8(&mut s, ctx, &facs, &format!("prefix/L{:02}/gate", lay_i), &format!("L{:02}_gate", lay_i),
                &l.mlp.gate.weight.to_f32_vec().unwrap(), PW, INTER, int8_native, dump_dir))
        } else {
            None
        };
        let iu8 = if i8sel("up") {
            let l = lay.expect("GEB_PREFIX_INT8 requires real weights");
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
            let l = lay.expect("GEB_PREFIX_INT8 requires real weights");
            Some(reg_i8(&mut s, ctx, &facs, &format!("prefix/L{:02}/down", lay_i), &format!("L{:02}_down", lay_i),
                &l.mlp.down.weight.to_f32_vec().unwrap(), INTER, PW, int8_native, dump_dir))
        } else {
            None
        };
        let down = mm_or_qmd(&mut s, &format!("{}down", tag), &act, &[gd(p), INTER], id8.as_ref(), &format!("{}downwt", tag), PW, &[gd(p), PW], int8_native);
        let downb = if id8.is_some() {
            down.clone() // int8: bias folded into the operator
        } else {
            s.bias(&format!("{}downb", tag), &down, &[gd(p), PW], &format!("{}downb", tag))
        };
        cur = s.add2(&format!("{}out", tag), &downb, &res, &[gd(p), PW]);
    }
    let out_names: Vec<&str> = kv_outs.iter().map(|x| x.as_str()).collect();
    s.finish(&out_names);
    println!("prefix OM built: depth={depth} P={p} n_in={} n_out={}", s.ins().len(), kv_outs.len());

    if let Some(st) = e2e {
        // e2e: load prefix_real.om, run once, capture the 36 k/v outputs
        // (k0,v0,k1,v1,...).
        let outs = e2e_run(ctx, stream, &mut s, "prefix");
        let nkv = depth * 2;
        assert!(outs.len() >= nkv, "prefix OM outputs {} < kv {}", outs.len(), nkv);
        st.kv = outs[..nkv].to_vec();
        let m = st.kv.iter().flat_map(|v| v.iter()).fold(0f32, |m, v| m.max(v.to_f32().abs()));
        println!("[e2e] prefix kv x{} |max|={m:.4}", st.kv.len());
        // GEB_E2E_SERVE: sink the mech (embedding table ~2.1GB resident
        // in the mech, reused across requests).
        if std::env::var("GEB_E2E_SERVE").is_ok() {
            let w = real.expect("serve prefix requires GEB_CKPT (token embedding lookup)");
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
                println!("[prefix-pool] serve pool mode: per-L static OM handle swap (lazy, cap {})", envi("GEB_POOL_CAP", 8));
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
                // Pool capacity cap (LRU eviction): past the cap the least
                // recently used bucket unloads and a re-hit reloads
                // (seconds). Set to the bucket count for full-lineage
                // runs.
                pool_cap: envi("GEB_POOL_CAP", 8) as usize,
                rout_bytes,
                mask_bind,
            });
        }
        return;
    }
    if int8 {
        // int8 tail: save the built OM (GEB_SAVE needs only the build
        // artifact) and optionally run it (GEB_INT8_RUN).
        let mut save_path = None;
        if let Ok(p) = std::env::var("GEB_SAVE") {
            s.g.save(&p).expect("save om (int8)");
            println!("OM saved: {p}");
            save_path = Some(p);
        }
        // GEB_INT8_RUN=1: run the live graph, then the saved-and-reloaded
        // OM on the same binds — isolates OM serialization desc (int8/f32
        // Data input) suspicions: the probe's live graph was clean while
        // e2e blew up after reload (a ladder pinned gate/down single
        // projections blowing up with layout bridges N2/N3 clean ->
        // suspicion on the save/load roundtrip).
        if std::env::var("GEB_INT8_RUN").is_ok() {
            let run_once = |s: &mut Seg, tag: &str| {
                let ins: Vec<&DeviceBuffer> = s.ins();
                let n_out = s.g.num_outputs().unwrap();
                let outs: Vec<DeviceBuffer> = (0..n_out)
                    .map(|i| ctx.malloc(s.g.output_size(i).unwrap().max(16)).expect("out malloc"))
                    .collect();
                let refs: Vec<&DeviceBuffer> = outs.iter().collect();
                s.g.run(&ins, &refs, stream).expect("ge run");
                let _ = stream.synchronize();
                let full = download_f16(ctx, &outs[0], s.g.output_size(0).unwrap() / 2);
                let m = full.iter().fold(0f32, |a, v| a.max(v.to_f32().abs()));
                let head: Vec<f32> = full[..8.min(full.len())].iter().map(|v| v.to_f32()).collect();
                println!("[int8] {tag} out0 head={head:?} |max|={m:.2}");
            };
            run_once(&mut s, "live");
            let live_sizes: Vec<(usize, usize)> = (0..s.g.num_inputs().unwrap())
                .map(|i| (s.g.input_size(i).unwrap_or(0), s.binds[i].len()))
                .collect();
            let bad_live: Vec<usize> = live_sizes
                .iter()
                .enumerate()
                .filter(|(_, (sz, bl))| sz != bl)
                .map(|(i, _)| i)
                .collect();
            println!("[int8] live graph: {} inputs; binds length mismatches at {:?}", live_sizes.len(), bad_live);
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
                println!("[int8] reloaded: {n2} inputs (live {}); size mismatches (first 8): {:?}", live_sizes.len(), mism.iter().take(8).collect::<Vec<_>>());
                run_once(&mut s, "reloaded");
            }
        }
        ge_builder::fini().expect("fini");
        let projs: Vec<&str> = ["q", "k", "v", "o", "gate", "up", "down"]
            .into_iter()
            .filter(|p| int8_projs.contains(*p))
            .collect();
        println!(
            "GE_PREFIX_PROBE_OK (int8: projections {} x{} layers, native_smooth={int8_native})",
            projs.join(","),
            depth
        );
        return;
    }
    ge_builder::fini().expect("fini");
    println!("GE_PREFIX_PROBE_OK");
}

// ---------------------------------------------------------------------------
// flow segment: single-step OM (action_in -> depth x action layers
// (ada-norm + cross attention + geglu) -> action_out -> euler). Styles,
// prefix kv and euler constants are all external inputs.
// ---------------------------------------------------------------------------

pub fn seg_flow(be: &AscendBackend, real: Option<&Pi05Weights>, e2e: Option<&mut E2eStage>) {
    let ctx = be.ctx();
    let stream = be.stream();
    let depth = envi("GEB_DEPTH", 18) as usize;
    let tokens = envi("GEB_TOKENS", 64);
    // prefix visible length (the empty-view drop must match the prefix
    // segment — suffix rope offsets and pk/pv shapes all take this value;
    // the infer side starts pos_s = sum(prefix_pad)+cumsum-1 = 712).
    let p = VT + tokens - envi("GEB_PREFIX_DROP_EMPTY", 0);
    let mut seed = 0xF00Du32;
    // Same trio as vision/prefix (rationale in those segments):
    // independent q/k/v projections, flat rope (no column-split
    // materialization), hand-built cross-GQA attention (no PFA host
    // launcher). Scale folds into the q weight/bias.
    let qkv3 = std::env::var("GEB_QKV3").is_ok();
    let ropeflat = std::env::var("GEB_ROPEFLAT").is_ok();
    let attn_manual = std::env::var("GEB_ATTN").map(|v| v == "manual").unwrap_or(false);
    let fscale = if attn_manual { 1.0f32 / (HD as f32).sqrt() } else { 1.0 };
    let mut s = Seg::new("ge_flow");
    // Dynamic gears: same mechanism as prefix (36 pk/pv lanes as
    // [-1,KVD]) — but in the production HYBRID topology flow does NOT
    // gear (styles are static Data inputs with runtime content; they
    // cannot be Consts, and static Data mixed into the dynamic set breaks
    // SetInputDynamicDims gear matching — measured). Flow instead uses a
    // per-L static OM pool (GEB_FLOW_POOL over the tl{L}_i8 assets), with
    // binds built at p_max. GEB_DYN_LLIST_FLOW exists only for isolated
    // gear-mode flow experiments.
    let gear_ll = std::env::var("GEB_DYN_LLIST_FLOW").ok().and_then(|_| dyn_llist());
    let mut gear_ps_hold: Option<Vec<i64>> = None;
    let gd = |v: i64| if gear_ll.is_some() { -1 } else { v };
    if let Some(g) = &gear_ll {
        assert!(
            std::env::var("GEB_WCONST").is_ok(),
            "GEB_DYN_LLIST requires GEB_WCONST=1"
        );
        assert!(ropeflat && attn_manual && qkv3, "GEB_DYN_LLIST requires the trio (ROPEFLAT/ATTN=manual/QKV3)");
        let lmax = *g.iter().max().unwrap();
        assert_eq!(tokens, lmax, "GEB_DYN_LLIST requires GEB_TOKENS = the gear table's max L");
        let drop_v2 = envi("GEB_PREFIX_DROP_EMPTY", 0);
        let ps = g.iter().map(|&l| VT + l - drop_v2).collect::<Vec<i64>>();
        println!("[gear] flow dynamic gears: L {:?} -> p {:?}", g, ps);
        gear_ps_hold = Some(ps);
    }
    if let Some(ps) = gear_ps_hold.take() {
        s.set_gear_ps(ps);
    }

    // ---- Segment inputs (two-pass: register everything, then build —
    // layer i's next ada-norm references layer i+1's style input, so
    // registration must precede wiring) ----
    // 0 state / 1 zeros / 2..7 flat rope / 8..11 rank-2 tables /
    // 12 ainwt / 13 ainb / 14.. (pk_i, pv_i) per layer (rank-2) /
    // 12 per layer / tail fsc,fsh,aoutwt,aoutb,c1,c2.
    // e2e: state starts from noise; the default is random.
    let state_h = e2e
        .as_ref()
        .map(|st| st.noise.clone())
        .unwrap_or_else(|| rand_f16((HOR * ADIM) as usize, &mut seed, 100.0));
    s.data(ctx, "state", &[HOR, ADIM], &state_h);
    if let Some(st) = e2e.as_ref() {
        assert_eq!(st.kv.len(), depth * 2, "e2e: prefix kv count != flow depth (GEB_DEPTH must be 18)");
    }
    s.data_zeros(ctx, "zeros", HOR, AW);
    // The bucket boundary P fixes only the cache-consumed dim; serve's
    // set_rope_for(true L) fixes the phase by the true length.
    let rope_p = p;
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
    // Reshape shape tensors (q/k/v/kall rank-3 bridges; kall/vall share;
    // Consts).
    s.const_i32("shp_q3", &[1, HOR as i32, QD as i32]);
    s.const_i32("shp_k3", &[1, HOR as i32, KVD as i32]);
    s.const_i32("shp_v3", &[1, HOR as i32, KVD as i32]);
    s.const_i32("shp_kall", &[1, gd(p + HOR) as i32, KVD as i32]);
    // ropeflat / manual cross-attention constants (swap indices as
    // Consts).
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
    // prefix k/v (rank-2 — concatenated along axis=0 then Unsqueezed
    // together); e2e wires the prefix OM's 36 outputs straight in
    // (pk{i}=kv[2i], pv{i}=kv[2i+1]). pk/pv registration base (for
    // per-frame rebinding; the loop body does exactly two s.data).
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
    // ---- flow int8 (GEB_FLOW_INT8): 7 projections/layer ->
    // QuantMatmulDequant ----
    // The registration zone precedes the layer zone (sm Data lands here,
    // pk_base stays put); in-layer int8 weight slots take 16B dummy
    // binds — the 18-slot layout always matches f16 (serve/e2e's
    // hard-coded style offsets b+0/+1/+10/+11/+16/+17 depend on it, so
    // FlowMech/style_residency need no changes).
    let int8 = flow_int8();
    if int8 {
        assert!(qkv3, "GEB_FLOW_INT8 requires GEB_QKV3=1 (fused qkv cannot take per-matrix smooth)");
    }
    let facs: std::collections::HashMap<String, Int8Smooth> = if int8 {
        assert!(real.is_some(), "GEB_FLOW_INT8 requires real weights via GEB_CKPT");
        let path = std::env::var("GEB_FLOW_SMOOTH")
            .unwrap_or_else(|_| "./smooth_calib_v2_flow_engine.safetensors".into());
        println!("[int8] flow int8 on: smooth={path}");
        load_int8_smooth(&path).0
    } else {
        std::collections::HashMap::new()
    };
    // GEB_FLOW_PROJS: the flow segment's own subset (falling back to
    // GEB_INT8_PROJS; sharing one env between segments would switch
    // prefix off too during subset experiments — OM input-count
    // mismatch, rc=-2 measured).
    let int8_projs: std::collections::HashSet<String> = std::env::var("GEB_FLOW_PROJS")
        .or_else(|_| std::env::var("GEB_INT8_PROJS"))
        .unwrap_or_else(|_| "q,k,v,o,gate,up,down".into())
        .split(',')
        .map(|x| x.trim().to_string())
        .collect();
    let int8_native = envi("GEB_INT8_NATIVE", 1) == 1;
    let dump_dir = std::env::var("GEB_INT8_DUMP").ok();
    let fi8sel = |pr: &str| int8 && int8_projs.contains(pr);
    // (name, k, n): k/v's n=256 and down's k=4096 are small/wide shapes —
    // qmd compile/perf at M=50 was not pre-validated, so the
    // GEB_INT8_PROJS subset can bail on them.
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
                .unwrap_or_else(|| panic!("GEB_FLOW_INT8 requires real weights (layer {i} missing)"));
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
        println!("[int8] flow int8 preregistered: {} layers x {} projections (native={})", depth, fi8[0].iter().filter(|x| x.is_some()).count(), int8_native);
    }
    // Per-layer weight layout (GEB_QKV3 changes the bracketed part):
    // ascl ash [qkvwt qkvb | qw kw vw qb kb vb] outwt outb
    // mscl msh gatewt upwt downwt downb agate mgate
    let mut layer_bases = Vec::with_capacity(depth);
    for i in 0..depth {
        layer_bases.push(s.binds.len());
        let tag = format!("l{i}_");
        // Real weights: ascl/ash/mscl/msh are ada-norm conditioning
        // vectors (runtime products of the style projection and
        // conditioning, varying per time step) — they stay synthetic;
        // real-weight integration only swaps projection/LN-style weights.
        let lay = real.and_then(|w| w.action_layers.get(i));
        s.data(ctx, &format!("{}ascl", tag), &[AW], &norm_f16(AW as usize, &mut seed, 1.0));
        s.data(ctx, &format!("{}ash", tag), &[AW], &norm_f16(AW as usize, &mut seed, 0.0));
        {
            // int8 slots: no f16 host block is built and the wt slot gets
            // a 16B dummy placeholder (layout discipline — see the
            // preregistration note).
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
            // Gemma projections carry no bias (real weights = zeros).
            let bias = lay
                .map(|_| vec![f16::from_f32(0.0); QKVW as usize])
                .unwrap_or_else(|| rand_f16(QKVW as usize, &mut seed, 4000.0));
            // Manual attention: q block (weight+bias) premultiplied by
            // 1/sqrt(hd).
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
                // qkv3 input slots within a layer are 2..8 (after
                // ascl/ash).
                let i8_slot = |s: &mut Seg| {
                    s.binds.push(ctx.malloc(16).expect("i8 wt slot placeholder"));
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
        }
        if fi8[i][3].is_some() {
            s.binds.push(ctx.malloc(16).expect("i8 wt slot placeholder"));
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
            s.binds.push(ctx.malloc(16).expect("i8 wt slot placeholder"));
            s.bind_names.push(String::new());
        } else {
            let host = lay
                .map(|l| lw_f16(&l.mlp.gate))
                .unwrap_or_else(|| rand_f16((AW * AINTER) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}gatewt", tag), &[AINTER, AW], &host, AW, AINTER);
        }
        if fi8[i][5].is_some() {
            s.binds.push(ctx.malloc(16).expect("i8 wt slot placeholder"));
            s.bind_names.push(String::new());
        } else {
            let host = lay
                .map(|l| lw_f16(&l.mlp.up))
                .unwrap_or_else(|| rand_f16((AW * AINTER) as usize, &mut seed, 8000.0));
            s.wt(ctx, &format!("{}upwt", tag), &[AINTER, AW], &host, AW, AINTER);
        }
        if fi8[i][6].is_some() {
            s.binds.push(ctx.malloc(16).expect("i8 wt slot placeholder"));
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
        // ada-norm gated third segment (chunk 3 of dense(cond)): the
        // attention/MLP branch output scales by gate before the residual
        // (torch _gated_residual). Appended at the block tail — the qkv3
        // offsets 8..15 stay put and gate fixes at 16/17 (default layout
        // 12/13).
        s.data(ctx, &format!("{}agate", tag), &[1, AW], &norm_f16(AW as usize, &mut seed, 1.0));
        s.data(ctx, &format!("{}mgate", tag), &[1, AW], &norm_f16(AW as usize, &mut seed, 1.0));
    }
    // Bind slots of fsc/fsh — e2e overwrites them per step (final norm
    // conditions).
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
    // euler constants (sigma = dt = -0.1: x' = (1+sigma)x + sigma*v).
    let eul_c1 = vec![f16::from_f32(0.9); (HOR * ADIM) as usize];
    let eul_c2 = vec![f16::from_f32(-0.1); (HOR * ADIM) as usize];
    s.data(ctx, "c1", &[HOR, ADIM], &eul_c1);
    s.data(ctx, "c2", &[HOR, ADIM], &eul_c2);

    // ---- Graph build ----
    // action_in + the first layer's attention ada-norm (layer 0 has no
    // precomputed normalized input — adaptive_rms(action_in_out,
    // attention_style[0])).
    let ain = s.mm("ain", "state", &[HOR, ADIM], "ainwt", &[AW, ADIM], &[HOR, AW]);
    let ain = s.bias("ainb2", &ain, &[HOR, AW], "ainb");
    let mut normed = s.ada("l0ada", &ain, "zeros", "l0_ascl", "l0_ash", &[HOR, AW]);
    // The unnormalized residual stream is carried as a second track —
    // cuda bf16_executor semantics: the attention residual base is the
    // input itself (fused::adaptive_gate_residual_rms_bf16(projection,
    // input)); normed only feeds qkv. This segment once used normed as
    // the residual base (both mirrors shared the bug, and component
    // parity stayed green because the reference was equally wrong): 18
    // layers of accumulation left v near-random and the closed loop at
    // 0/10 — step0_x1's 11.5% was merely dt=0.1 masking the v error
    // (dx1 = 0.1*dv, so v was ~100% wrong).
    let mut xstream = ain;

    let total = p + HOR;
    // GEB_ATTN_MASK (range padding): the flow-side mask [1,1,total] —
    // key layout = cache rows first (real [0,vis+L), pad [vis+L, vis+P))
    // plus this bucket's HOR flow rows always real. Baked as all-zero
    // placeholder; serve fills it per true L.
    let fmask_n: Option<String> = if envi("GEB_ATTN_MASK", 0) == 1 {
        let mh = vec![f16::from_f32(0.0); total as usize];
        Some(s.data(ctx, "padmask", &[1, 1, total], &mh))
    } else {
        None
    };
    for i in 0..depth {
        let tag = format!("l{i}_");
        // attention (normed is this layer's ada-norm output).
        let (q2, k2, v2) = if qkv3 {
            let qm = mm_or_qmd(&mut s, &format!("{}qm", tag), &normed, &[HOR, AW], fi8[i][0].as_ref(), &format!("{}qw", tag), QD, &[HOR, QD], int8_native);
            let km = mm_or_qmd(&mut s, &format!("{}km", tag), &normed, &[HOR, AW], fi8[i][1].as_ref(), &format!("{}kw", tag), KVD, &[HOR, KVD], int8_native);
            let vm = mm_or_qmd(&mut s, &format!("{}vm", tag), &normed, &[HOR, AW], fi8[i][2].as_ref(), &format!("{}vw", tag), KVD, &[HOR, KVD], int8_native);
            // int8: Gemma bias == 0 drops the TileD+Add pair (prefix
            // precedent; the bias Data stays registered to hold the input
            // slot — layout stability).
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
        // prefix cat (rank-2 ConcatD axis=0) + Reshape rank-3 bridge
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
            // manual cross-GQA: q goes head-major (headsplit); k/v are
            // already single-head rank-3.
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
        // branch x gate into the residual (torch _gated_residual:
        // residual + branch*gate).
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

        // next layer's attention ada-norm (attention_normalized
        // isomorph).
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

    if let Some(st) = e2e {
        // e2e 10 steps: overwrite styles (qkv3 layout: ascl/ash =
        // base+0/+1, mscl/msh = base+10/+11) and state (x) per step;
        // pk/pv unchanged; euler is in-graph.
        let rw = real.expect("e2e requires GEB_CKPT (time_mlp/style projections)");
        let cfg = Pi05Config::default();
        let dir = std::env::var("GEB_OM_DIR").unwrap_or_else(|_| "./om_cache".into());
        // HYBRID: GEB_FLOW_POOL=<om root> (with GEB_DYN_LLIST's L table)
        // — loads the per-L static flow OM pool (binds materialize once
        // at max; graph handles swap per request). Bring-up runs the
        // first handle (warmup content doesn't matter).
        let mut flow_pool: std::collections::HashMap<i64, GeGraph> = std::collections::HashMap::new();
        if let Ok(pool_root) = std::env::var("GEB_FLOW_POOL") {
            let ls = dyn_llist().unwrap_or_else(|| panic!("GEB_FLOW_POOL requires a GEB_DYN_LLIST gear table"));
            // Serial loading (22 OMs x ~500MB = 1-2min). 22 parallel
            // threads crashed the process in testing (aclmdlLoadFromMem
            // concurrency cap < 22 — the 3-thread OM preload survives;
            // crash-loop measured). Bounded parallelism unverified.
            let t0p = std::time::Instant::now();
            for &l in &ls {
                let p = format!("{pool_root}/tl{l}_i8/flow_real.om");
                let g = ge_builder::load(&p).unwrap_or_else(|e| panic!("flow pool {p}: {e:?}"));
                flow_pool.insert(l, g);
            }
            println!("[flow-pool] {} OMs loaded {:?}", flow_pool.len(), t0p.elapsed());
            let first = *ls.first().expect("gear table non-empty");
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
        // Rebind the euler constants to LeRobot semantics x' = x + dt*v
        // (c1=1.0, sample_actions L894; the OM default 0.9/-0.1 is
        // openpi's x1-prediction form). c1/c2 are Data inputs located by
        // name — the old "last two binds" assumption broke when
        // GEB_ATTN_MASK's padmask got appended (it lands at the tail, and
        // pouring c2 into the 1444B mask bind crashes instantly).
        let c1_h = vec![f16::from_f32(1.0); (HOR * ADIM) as usize];
        let c2_h = vec![f16::from_f32(-0.1); (HOR * ADIM) as usize];
        h2d_f16(&s.binds[s.bind_of("c1")], &c1_h);
        h2d_f16(&s.binds[s.bind_of("c2")], &c2_h);
        // styles precompute cache: the 10-step time schedule is fixed
        // (1+step*dt), so styles never change across calls — computed
        // once and resident in production (per-step host computation at
        // ~173ms/step dominated the glue, per bring-up timing).
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
            "[e2e] styles precomputed x{} steps {:?} (once, cached across calls)",
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
            h2d_f16(&s.binds[0], &x); // state = current x
            let oref: Vec<&DeviceBuffer> = vec![&ob];
            s.run_seg(s.gear_payloads.len().saturating_sub(1), &oref, stream)
                .expect("ge run");
            let _ = stream.synchronize();
            x = download_f16(ctx, &ob, (HOR * ADIM) as usize);
            let m = x.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
            println!("[e2e] step {step} |x|max={m:.4}");
        }
        println!("[e2e] flow 10 steps {:?} (styles cached — pure h2d rebind + run + d2h)", t0.elapsed());
        // GEB_E2E_BENCH=N: steady per-call bench — each iteration is a
        // full 10-step flow (cached styles + h2d + run + sync + d2h); x
        // restarts from noise every iteration.
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
                    let _ = stream.synchronize();
                    xx = download_f16(ctx, &ob, (HOR * ADIM) as usize);
                }
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "[e2e] flow steady bench x{nb} calls (10 steps/call, cached styles + h2d + run + sync + d2h): P50={:.2}ms min={:.2} max={:.2}",
                ts[nb / 2],
                ts[0],
                ts[nb - 1]
            );
        }
        st.actions = x;
        // GEB_E2E_SERVE: sink the mech (styles cache is request-invariant
        // — the 10-step schedule is fixed; euler c1/c2 already rebound to
        // LeRobot semantics and the buffer contents persist).
        if std::env::var("GEB_E2E_SERVE").is_ok() {
            let (style_master, style_views, style_map) = if serve_fast() {
                let (m, v, mp) =
                    style_residency(ctx, &style_cache, &layer_bases, fs_idx, depth);
                println!(
                    "[serve] GEB_SERVE_FAST: styles device-resident, {} steps x {} slots (master {} bytes, one upload)",
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
        // int8 tail: save the built OM (GEB_SAVE needs only the build
        // artifact).
        if let Ok(p) = std::env::var("GEB_SAVE") {
            s.g.save(&p).expect("save om (flow int8)");
            println!("OM saved: {p}");
        }
    }
    ge_builder::fini().expect("fini");
    println!("GE_FLOW_PROBE_OK");
}

// ---------------------------------------------------------------------------
// GeServe facade (in-process direct call): per-frame inference over the
// three segment OMs. The spool serve_loop and PyO3 (apxinf-py --features
// ascend) share the same frame implementation. Driver entry points (env
// dispatch / GEB_CKPT loading) live in examples/ge_model_probe.rs.
// ---------------------------------------------------------------------------

/// One frame's inference result: actions [HOR, ADIM] (flow terminal
/// state), per-segment timings (ms), and vision_out [VT, PW] (debug;
/// production can ignore).
pub struct GeServeOut {
    pub actions: Vec<f16>,
    pub vision_out: Vec<f16>,
    pub vision_ms: f32,
    pub prefix_ms: f32,
    pub flow_ms: f32,
}

/// In-process three-segment pipeline (= a serve bucket's mech group).
/// open() walks the exact probe bring-up assembly path (including the
/// first-frame warmup and styles precompute); infer() is frame-for-frame
/// identical to serve_loop.
pub struct GeServe {
    vm: VisionMech,
    pm: PrefixMech,
    fm: FlowMech,
    pub tokens: usize,
    pub fast: bool,
}

impl GeServe {
    /// Opens a pipeline from a prepared bucket directory. Prerequisites:
    /// the caller has done ge_builder::init + AscendBackend::new (GE
    /// global init must precede acl runtime); real = real weights (the
    /// token embedding / styles conditions are consumed on the host).
    /// Env contract: om_dir's {vision,prefix,flow}_real.om must have been
    /// built with the quartet (GEB_ATTN=manual GEB_QKV3=1 GEB_ROPEFLAT=1
    /// GEB_WCONST=1); this function fills missing default envs itself
    /// (GEB_SEG=e2e / GEB_TOKENS / GEB_OM_DIR / GEB_E2E_SERVE — the
    /// latter being the gate that sinks the three mechs). GEB_NORM32
    /// buckets are a different topology and unsupported.
    /// One pipeline per process: configuration is process-global env;
    /// multiple buckets (multiple L) need multiple processes or an
    /// extension first.
    pub fn open(
        be: &AscendBackend,
        real: &Pi05Weights,
        om_dir: &str,
        tokens: usize,
        fast: bool,
    ) -> GeServe {
        assert!(!norm32_enabled(), "GeServe does not support GEB_NORM32 buckets (different topology)");
        for (k, v) in [
            ("GEB_SEG", "e2e"),
            ("GEB_ATTN", "manual"),
            ("GEB_QKV3", "1"),
            ("GEB_ROPEFLAT", "1"),
            ("GEB_WCONST", "1"),
            // Topology consistency (Seg construction reads these; they
            // must match the OM build; the in-process scenario cannot
            // control env, so pin production values). GEB_DEPTH is
            // deliberately NOT pinned — the per-segment defaults
            // vision=27 / prefix=18 / flow=18 match the OMs, and pinning
            // 18 would cut vision to 18 layers (n_in 187 != 277, learned
            // the hard way).
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
        // Cold-start lazy: preload the three OMs in parallel (e2e_run
        // inside the segment constructors picks them up).
        preload_e2e_oms(be.ctx());
        // Synthetic bring-up inputs (consumed by the first-frame warmup;
        // real frames arrive via infer).
        let mut seed = 0xE2E2u32;
        let mut st = E2eStage {
            patches: rand_f16((VT * V_PATCH_W) as usize, &mut seed, 300.0),
            token_ids: (0..tokens).map(|i| 1000 + i as u32 * 7).collect(),
            noise: rand_f16((HOR * ADIM) as usize, &mut seed, 1.0),
            vision_out: Vec::new(),
            kv: Vec::new(),
            actions: Vec::new(),
            vmech: None,
            pmech: None,
            fmech: None,
        };
        seg_vision(be, Some(real), Some(&mut st));
        seg_prefix(be, Some(real), Some(&mut st));
        seg_flow(be, Some(real), Some(&mut st));
        let fast = serve_fast();
        GeServe {
            vm: st.vmech.take().expect("GeServe vision mech missing"),
            pm: st.pmech.take().expect("GeServe prefix mech missing"),
            fm: st.fmech.take().expect("GeServe flow mech missing"),
            tokens,
            fast,
        }
    }

    /// One frame: patches [VT*V_PATCH_W] / token_ids [tokens] / noise
    /// [HOR*ADIM] -> actions [HOR, ADIM] (normalized_actions; denorm is
    /// torch postprocess).
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
            // GEB_SERVE_FAST: kv device-direct + styles resident + x
            // resident (tp still includes prefix run's sync wait — the
            // segment timings stay comparable with legacy). In gear mode
            // L = token_ids.len() picks the gear per request (prefix and
            // flow share it).
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
