//! Ascend π0.5 transformer-layer execution — mirror of bf16_executor.rs
//! over the aclnn op compositions in `apxinf-ascend::ops`.
//!
//! Where the CUDA path calls fused kernels, this path composes verified
//! aclnn ops (fused where available: AddRmsNorm / AddLayerNorm / PFA /
//! GeluV2; gathered/memcpyed where the fused rope entries reject pi0.5's
//! head_dims). Weights reuse the bf16_weights Tensor containers -- tensors
//! land as F16 through AscendBackend::to_device.

use apxinf_ascend::ops as aops;
use apxinf_ascend::{AclError, AscendBackend, DeviceBuffer};
use apxinf_core::{Backend as _, Error, Result, Tensor};
use std::sync::Arc;

use super::{
    Bf16DeviceActionLayer, Bf16DeviceLanguageLayer, Bf16DeviceVisionBlock, Bf16LinearWeights,
    GemmaVariantConfig,
};

/// Bridge aclnn errors into apxinf_core::Error.
pub(crate) fn acl<T>(r: std::result::Result<T, AclError>) -> Result<T> {
    r.map_err(|e| Error::Other(format!("aclnn: {e}")))
}

/// Combined executor caches threaded through layer calls.
pub struct AscendCaches {
    pub rope: AscendRopeCache,
    pub nz: NzCache,
    /// Pre-materialized ada-norm rows, keyed by style tensor pointer:
    /// (1+style[0:w], style[w:2w]) as device rows. Styles are computed
    /// once per flow step and reused across the layer loop -- caching
    /// removes the per-layer d2h+stream-sync that broke pipelining
    /// (~540 syncs per inference at full depth).
    style_rows: std::collections::HashMap<usize, (Arc<DeviceBuffer>, Arc<DeviceBuffer>)>,
    /// Pre-sliced qkv bias rows, keyed by (bias pointer, start, len).
    kv_biases: std::collections::HashMap<(usize, i64, i64), Arc<DeviceBuffer>>,
    /// Pinned rope position-index rows, keyed by (tokens, heads, offset).
    pos_idx: std::collections::HashMap<(i64, i64, i64), Arc<DeviceBuffer>>,
    /// Cached all-ones gamma rows for ada-norm rms (keyed by cols) -- the
    /// per-call h2d was a sync memcpy inside capture windows.
    ones_rows: std::collections::HashMap<usize, Arc<DeviceBuffer>>,
    /// Cached device token-index rows (keyed by the id sequence).
    pub(crate) token_idx: std::collections::HashMap<Vec<u32>, Arc<DeviceBuffer>>,
}

impl AscendCaches {
    pub fn new() -> Self {
        Self {
            rope: AscendRopeCache::new(),
            nz: NzCache::new(),
            style_rows: Default::default(),
            kv_biases: Default::default(),
            pos_idx: Default::default(),
            ones_rows: Default::default(),
            token_idx: Default::default(),
        }
    }
}

impl Default for AscendCaches {
    fn default() -> Self {
        Self::new()
    }
}

/// Lazy per-pointer transposed-weight cache. The plain [k, n] row-major
/// mat2 descriptor works in isolation but reads WRONG DATA (rel 0.79)
/// after an AddRmsNorm has run, while the transposed-b view ([k, n]
/// shape over physical [n, k], stride [1, k] -- torch's `a @ w.t()`
/// layout) is both immune and numerically correct (probe 2026-09-18).
/// So every linear weight is host-transposed once at first use and the
/// matmul runs through the transposed path.
pub struct NzCache {
    transposed: std::collections::HashMap<usize, Arc<DeviceBuffer>>,
    out_cols: std::collections::HashMap<usize, i64>,
}

impl NzCache {
    pub fn new() -> Self {
        Self { transposed: Default::default(), out_cols: Default::default() }
    }

    pub fn get(&mut self, be: &AscendBackend, w: &DeviceBuffer, cols_out: i64) -> Result<Arc<DeviceBuffer>> {
        let key = w.as_ptr() as usize;
        if let Some(v) = self.transposed.get(&key) {
            return Ok(v.clone());
        }
        let cols = *self.out_cols.entry(key).or_insert(cols_out);
        let total = (w.len() / 2) as i64;
        let rows = total / cols;
        if std::env::var("APXINF_ASCEND_TRACE").is_ok() {
            eprintln!("[trace] nz: transposing [{rows},{cols}] from {:p}", w.as_ptr());
        }
        let mut host = vec![0u8; w.len()];
        acl(be.ctx().copy_d2h(w, &mut host))?;
        let bytes = aops::host_transpose(&host, rows, cols);
        let buf = acl(be.ctx().malloc(bytes.len()))?;
        acl(be.ctx().copy_h2d(&buf, &bytes))?;
        acl(be.stream().synchronize())?;
        self.transposed.insert(key, Arc::new(buf));
        Ok(self.transposed.get(&key).unwrap().clone())
    }
}

impl Default for NzCache {
    fn default() -> Self {
        Self::new()
    }
}

/// acl()-bridged mirrors of the apxinf-ascend op functions so call sites
/// can use `?` directly against apxinf_core::Error.
pub(crate) mod aq {
    use super::*;

    pub fn matmul(be: &AscendBackend, nz: &mut NzCache, a: &DeviceBuffer, ash: [i64; 2], b: &DeviceBuffer, bsh: [i64; 2]) -> Result<DeviceBuffer> {
        let bt = nz.get(be, b, bsh[1])?;
        acl(aops::matmul_b_t_fp16(be.ctx(), be.stream(), a, ash, &bt, bsh[0], bsh[1]))
    }
    pub fn bias(be: &AscendBackend, x: &DeviceBuffer, bias: Option<&DeviceBuffer>, rows: i64, cols: i64) -> Result<DeviceBuffer> {
        match bias {
            Some(b) => acl(aops::bias_add_fp16(be.ctx(), be.stream(), x, b, rows, cols)),
            None => acl(aops::take_rows_fp16(be.ctx(), be.stream(), x, 0, (x.len() / 2) as i64, 1)),
        }
    }
    pub fn add(be: &AscendBackend, a: &DeviceBuffer, b: &DeviceBuffer, sh: &[i64]) -> Result<DeviceBuffer> {
        acl(aops::add_fp16(be.ctx(), be.stream(), a, b, sh))
    }
    pub fn gelu(be: &AscendBackend, x: &DeviceBuffer, sh: &[i64], tanh: bool) -> Result<DeviceBuffer> {
        acl(aops::gelu_fp16(be.ctx(), be.stream(), x, sh, tanh))
    }
    pub fn mul(be: &AscendBackend, a: &DeviceBuffer, b: &DeviceBuffer, sh: &[i64]) -> Result<DeviceBuffer> {
        acl(aops::mul_fp16(be.ctx(), be.stream(), a, b, sh))
    }
    pub fn take_rows(be: &AscendBackend, buf: &DeviceBuffer, off: i64, rows: i64, cols: i64) -> Result<DeviceBuffer> {
        acl(aops::take_rows_fp16(be.ctx(), be.stream(), buf, off, rows, cols))
    }
    pub fn rope(be: &AscendBackend, x: &DeviceBuffer, rows: i64, d: i64, cos: &DeviceBuffer, sin: &DeviceBuffer, rot: &DeviceBuffer) -> Result<DeviceBuffer> {
        acl(aops::rope_rotate_half_fp16(be.ctx(), be.stream(), x, rows, d, cos, sin, rot))
    }
    pub fn layer_norm(be: &AscendBackend, x: &DeviceBuffer, zeros: &DeviceBuffer, g: &DeviceBuffer, b: &DeviceBuffer, rows: i64, cols: i64, eps: f64) -> Result<DeviceBuffer> {
        acl(aops::layer_norm_fp16(be.ctx(), be.stream(), x, zeros, g, b, rows, cols, eps))
    }
    pub fn pfa_bsh(be: &AscendBackend, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, tokens: i64, heads: i64, kv: i64, hd: i64) -> Result<DeviceBuffer> {
        acl(aops::prompt_flash_attention_bsh_fp16(be.ctx(), be.stream(), q, k, v, tokens, heads, kv, hd, None))
    }
    pub fn silu(be: &AscendBackend, x: &DeviceBuffer, sh: &[i64]) -> Result<DeviceBuffer> {
        acl(aops::silu_fp16(be.ctx(), be.stream(), x, sh))
    }
    pub fn gather_rows(be: &AscendBackend, table: &DeviceBuffer, vocab: i64, dim: i64, indices: &DeviceBuffer, n: i64) -> Result<DeviceBuffer> {
        acl(aops::gather_rows_fp16(be.ctx(), be.stream(), table, vocab, dim, indices, n))
    }
    pub fn euler(be: &AscendBackend, x0: &DeviceBuffer, x1: &DeviceBuffer, sigma: f32, sh: &[i64]) -> Result<DeviceBuffer> {
        acl(aops::euler_update_fp16(be.ctx(), be.stream(), x0, x1, sigma, sh))
    }
    pub fn cat2(be: &AscendBackend, a: &DeviceBuffer, ash: [i64; 2], b: &DeviceBuffer, bsh: [i64; 2]) -> Result<DeviceBuffer> {
        let rows = ash[0] + bsh[0];
        let cols = ash[1];
        acl(aops::cat_fp16(be.ctx(), be.stream(), &[a, b], &[vec![ash[0], cols], vec![bsh[0], cols]], 0, &[rows, cols]))
    }
}

/// f32 slice -> fp16 little-endian bytes (no bytemuck dep in this crate).
fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    fn to_f16(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x007f_ffff;
        if ((bits >> 23) & 0xff) == 0 || exp <= 0 {
            return sign;
        }
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
        let m = (mant >> 13) as u16;
        let rem = mant & 0x1fff;
        let mut out = ((exp as u16) << 10) | m;
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            out += 1;
        }
        sign | out
    }
    v.iter().flat_map(|&x| to_f16(x).to_le_bytes()).collect()
}

pub struct LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct ActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

/// Borrow-free snapshot of the Arc'd rope table handles.
struct AscendRopeTableArcs {
    cos: Arc<DeviceBuffer>,
    sin_signed: Arc<DeviceBuffer>,
    rot_idx: Arc<DeviceBuffer>,
    max_pos: usize,
}

/// Per-(theta, head_dim) RoPE constants on device: cos table [max_pos, d],
/// sign-folded sin table [max_pos, d], channel gather index [d].
/// Built lazily by [`AscendRopeCache::tables`].
pub struct AscendRopeTables {
    pub cos: Arc<DeviceBuffer>,
    pub sin_signed: Arc<DeviceBuffer>,
    pub rot_idx: Arc<DeviceBuffer>,
    pub d: i64,
    pub theta: f32,
    pub max_pos: usize,
}

pub struct AscendRopeCache {
    tables: Vec<AscendRopeTables>,
}

impl AscendRopeCache {
    pub fn new() -> Self {
        Self { tables: Vec::new() }
    }

    /// Get (building on first use) an Arc snapshot of the (theta, d) tables.
    pub fn tables(
        &mut self,
        be: &AscendBackend,
        theta: f32,
        d: i64,
        max_pos: usize,
    ) -> Result<AscendRopeTableArcs> {
        let idx = match self.tables.iter().position(|t| t.theta == theta && t.d == d && t.max_pos >= max_pos) {
            Some(i) => i,
            None => {
                let t = build_rope_tables(be, theta, d, max_pos)?;
                self.tables.push(t);
                self.tables.len() - 1
            }
        };
        let t = &self.tables[idx];
        Ok(AscendRopeTableArcs {
            cos: t.cos.clone(),
            sin_signed: t.sin_signed.clone(),
            rot_idx: t.rot_idx.clone(),
            max_pos: t.max_pos,
        })
    }
}

impl Default for AscendRopeCache {
    fn default() -> Self {
        Self::new()
    }
}

fn build_rope_tables(be: &AscendBackend, theta: f32, d: i64, max_pos: usize) -> Result<AscendRopeTables> {
    let ctx = be.ctx();
    let stream = be.stream();
    let half = (d / 2) as usize;
    let mut cos_h = vec![0f32; max_pos * d as usize];
    let mut sin_h = vec![0f32; max_pos * d as usize];
    for pos in 0..max_pos {
        for i in 0..half {
            let freq = (pos as f32) / theta.powf(i as f32 * 2.0 / d as f32);
            let (c, s) = (freq.cos(), freq.sin());
            let base = pos * d as usize;
            cos_h[base + i] = c;
            cos_h[base + i + half] = c;
            sin_h[base + i] = -s; // rotate-half sign folded host-side
            sin_h[base + i + half] = s;
        }
    }
    let cos = acl(ctx.malloc(cos_h.len() * 2))?;
    let sin_signed = acl(ctx.malloc(sin_h.len() * 2))?;
    acl(ctx.copy_h2d(&cos, &f32_to_f16_bytes(&cos_h)))?;
    acl(ctx.copy_h2d(&sin_signed, &f32_to_f16_bytes(&sin_h)))?;
    let rot_idx: Vec<i32> = (0..d as usize).map(|i| if i < half { i + half } else { i - half } as i32).collect();
    let rot_idx_buf = acl(ctx.malloc(rot_idx.len() * 4))?;
    let idx_bytes =
        unsafe { std::slice::from_raw_parts(rot_idx.as_ptr() as *const u8, rot_idx.len() * 4) };
    acl(ctx.copy_h2d(&rot_idx_buf, idx_bytes))?;
    acl(stream.synchronize())?;
    Ok(AscendRopeTables { cos: Arc::new(cos), sin_signed: Arc::new(sin_signed), rot_idx: Arc::new(rot_idx_buf), d, theta, max_pos })
}

/// Apply rotate-half rope to a [tokens * heads, d] view of `x` using the
/// cached tables. Positions run 0..tokens for every head.
fn apply_rope_rows(
    be: &AscendBackend,
    cache: &mut AscendCaches,
    x: &DeviceBuffer,
    tokens: i64,
    heads: i64,
    theta: f32,
    pos_offset: usize,
) -> Result<DeviceBuffer> {
    let d = {
        let total = (x.len() / 2) as i64;
        debug_assert_eq!(total % (tokens * heads), 0);
        total / (tokens * heads)
    };
    let ctx = be.ctx();
    let stream = be.stream();
    // per-row cos/sin: gather table rows by position index, each position
    // repeated `heads` times. The index vector only depends on (tokens,
    // heads, offset) -- build once, keep the device copy pinned in the cache
    // (stable address matters for any future capture window).
    let pos_key = (tokens, heads, pos_offset as i64);
    let pos_buf = match cache.pos_idx.get(&pos_key) {
        Some(v) => v.clone(),
        None => {
            let mut pos_idx = Vec::with_capacity((tokens * heads) as usize);
            for t in 0..tokens {
                for _ in 0..heads {
                    pos_idx.push((pos_offset as i64 + t) as i32);
                }
            }
            let buf = acl(ctx.malloc(pos_idx.len() * 4))?;
            let pb = unsafe { std::slice::from_raw_parts(pos_idx.as_ptr() as *const u8, pos_idx.len() * 4) };
            acl(ctx.copy_h2d(&buf, pb))?;
            let buf = Arc::new(buf);
            cache.pos_idx.insert(pos_key, buf.clone());
            buf
        }
    };

    let tables: AscendRopeTableArcs = {
        let t = cache.rope.tables(be, theta, d, (pos_offset + tokens as usize).max(1))?;
        AscendRopeTableArcs { cos: t.cos.clone(), sin_signed: t.sin_signed.clone(), rot_idx: t.rot_idx.clone(), max_pos: t.max_pos }
    };
    let rows = tokens * heads;
    let max_pos = tables.max_pos as i64;
    let t_cos_tab = acl(apxinf_ascend::AclTensor::fp16_nd(&tables.cos, &[max_pos, d]))?;
    let t_sin_tab = acl(apxinf_ascend::AclTensor::fp16_nd(&tables.sin_signed, &[max_pos, d]))?;
    let t_pos = acl(apxinf_ascend::AclTensor::i32_nd(&pos_buf, &[rows]))?;
    let cos_pos = acl(ctx.scratch_buf((rows * d * 2) as usize))?;
    let sin_pos = acl(ctx.scratch_buf((rows * d * 2) as usize))?;
    let t_cp = acl(apxinf_ascend::AclTensor::fp16_nd(&cos_pos, &[rows, d]))?;
    let t_sp = acl(apxinf_ascend::AclTensor::fp16_nd(&sin_pos, &[rows, d]))?;
    for (tab, out) in [(t_cos_tab.handle(), t_cp.handle()), (t_sin_tab.handle(), t_sp.handle())] {
        let mut ws = 0u64;
        let mut ex: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let p = apxinf_ascend::ffi::aclnnGatherV2GetWorkspaceSize(tab, 0, t_pos.handle(), out, &mut ws, &mut ex);
            if p != 0 {
                return Err(Error::Other(format!("rope pos-gather plan {p}")));
            }
            let r = apxinf_ascend::ffi::aclnnGatherV2(std::ptr::null_mut(), ws, ex, stream.handle());
            if r != 0 {
                return Err(Error::Other(format!("rope pos-gather run {r}")));
            }
        }
    }
    // no sync here: gather outputs feed the next kernels on the same
    // stream -- ordering carries, and a sync in this spot would break a
    // capture window
    acl(aops::rope_rotate_half_fp16(ctx, stream, x, rows, d, &cos_pos, &sin_pos, &tables.rot_idx))
}

/// bias + reshape helper: y = x + bias (row broadcast), staying in buffers.
pub(crate) fn bias_add(be: &AscendBackend, x: &DeviceBuffer, bias: Option<&Tensor>, rows: i64, cols: i64) -> Result<DeviceBuffer> {
    let Some(bias) = bias else {
        // identity copy: one fp16 element per "row" covers the whole buffer
        return acl(aops::take_rows_fp16(be.ctx(), be.stream(), x, 0, (x.len() / 2) as i64, 1))
    };
    let b = tensor_buf(bias)?;
    acl(aops::bias_add_fp16(be.ctx(), be.stream(), x, b, rows, cols))
}

pub(crate) fn tensor_buf(t: &Tensor) -> Result<&DeviceBuffer> {
    match t.storage() {
        apxinf_core::Storage::Gpu { handle, .. } => handle
            ._prevent_leak
            .as_ref()
            .and_then(|any| any.downcast_ref::<DeviceBuffer>())
            .ok_or_else(|| Error::Other("tensor missing ascend storage".into())),
        _ => Err(Error::UnsupportedDevice(t.device())),
    }
}

fn buf_tensor(be: &AscendBackend, buf: DeviceBuffer, rows: usize, cols: usize) -> Tensor {
    be.wrap_fp16(buf, vec![rows, cols])
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_ascend(
    be: &AscendBackend,
    cache: &mut AscendCaches,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<LanguageLayerOutput> {
    let ctx = be.ctx();
    let stream = be.stream();
    let tokens = input.shape().dims()[0] as i64;
    let hidden = config.head_dim as i64 * (config.num_heads.max(config.num_kv_heads)) as i64;

    let normalized = be.rms_norm(input, &weights.input_norm_scale, rms_eps)?;
    let norm_b = tensor_buf(&normalized)?;
    let qkv = aq::matmul(be, &mut cache.nz,norm_b, [tokens, hidden],
        tensor_buf(&weights.qkv.weight)?, [hidden, config.num_heads as i64 * config.head_dim as i64 + 2 * config.num_kv_heads as i64 * config.head_dim as i64])?;

    let qd = config.num_heads as i64 * config.head_dim as i64;
    let kv_d = config.num_kv_heads as i64 * config.head_dim as i64;
    let q_raw = aq::take_rows(be,&qkv, 0, tokens, qd)?;
    let k_raw = aq::take_rows(be,&qkv, tokens, tokens, kv_d)?;
    let v_raw = aq::take_rows(be,&qkv, tokens * 2, tokens, kv_d)?;
    // qkv bias spans the fused width; slice per part.
    let q_bias = kv_bias(be, cache, &weights.qkv.bias, 0, qd)?;
    let k_bias = kv_bias(be, cache, &weights.qkv.bias, qd, kv_d)?;
    let v_bias = kv_bias(be, cache, &weights.qkv.bias, qd + kv_d, kv_d)?;
    let q = aq::bias(be,&q_raw, q_bias.as_deref(), tokens, qd)?;
    let k = aq::bias(be,&k_raw, k_bias.as_deref(), tokens, kv_d)?;
    let v = aq::bias(be,&v_raw, v_bias.as_deref(), tokens, kv_d)?;

    let q_rope = apply_rope_rows(be, cache, &q, tokens, config.num_heads as i64, rope_theta, position_offset)?;
    let k_rope = apply_rope_rows(be, cache, &k, tokens, config.num_kv_heads as i64, rope_theta, position_offset)?;

    // attention first (borrows k_rope/v); key/value wrap moves them after.
    let attn = pfa_full(be, &q_rope, &k_rope, &v, tokens as i64,
        config.num_heads as i64, config.num_kv_heads as i64, config.head_dim as i64)?;
    let key = buf_tensor(be, k_rope, tokens as usize, kv_d as usize);
    let value = buf_tensor(be, v, tokens as usize, kv_d as usize);
    if !compute_tail {
        let hidden_out = input.clone();
        return Ok(LanguageLayerOutput { hidden: hidden_out, key, value });
    }
    let projected = aq::matmul(be, &mut cache.nz,&attn, [tokens, config.num_heads as i64 * config.head_dim as i64],
        tensor_buf(&weights.output.weight)?, [config.num_heads as i64 * config.head_dim as i64, input.shape().dims()[1] as i64])?;
    let input_b = tensor_buf(input)?;
    let width = input.shape().dims()[1] as i64;
    let biased = bias_add(be, &projected, weights.output.bias.as_ref(), tokens, width)?;
    // residual + post norm
    let res = aq::add(be,&biased, input_b, &[tokens, width])?;
    let res_t = buf_tensor(be, aq::take_rows(be, &res, 0, tokens, width)?, tokens as usize, width as usize);
    let fused_norm = be.rms_norm(&res_t, &weights.post_attention_norm_scale, rms_eps)?;
    let fused_norm_b = tensor_buf(&fused_norm)?;

    // geglu mlp: gate_up weight holds [hidden, 2*inter]; split halves.
    let inter = (tensor_buf(&weights.gate_up.weight)?.len() / 2 / (width as usize * 2)) as i64;
    let gate_up = aq::matmul(be, &mut cache.nz,fused_norm_b, [tokens, width],
        tensor_buf(&weights.gate_up.weight)?, [width, inter * 2])?;
    // from_host_parts packs [gate, up] -- first half is the gelu input.
    let gate = aq::take_rows(be,&gate_up, 0, tokens, inter)?;
    let up = aq::take_rows(be,&gate_up, tokens, tokens, inter)?;
    let gate_g = aq::gelu(be,&gate, &[tokens, inter], true)?;
    let activated = aq::mul(be,&gate_g, &up, &[tokens, inter])?;
    let projected2 = aq::matmul(be, &mut cache.nz,&activated, [tokens, inter],
        tensor_buf(&weights.down.weight)?, [inter, width])?;
    let biased2 = bias_add(be, &projected2, weights.down.bias.as_ref(), tokens, width)?;
    let hidden_out = aq::add(be,&biased2, &res, &[tokens, width])?;
    Ok(LanguageLayerOutput {
        hidden: buf_tensor(be, hidden_out, tokens as usize, width as usize),
        key,
        value,
    })
}

fn tensor_opt(t: &Option<Tensor>) -> Result<Option<&DeviceBuffer>> {
    match t {
        Some(t) => Ok(Some(tensor_buf(t)?)),
        None => Ok(None),
    }
}

/// Slice a fused qkv bias into a kv-slice device row. Biases are static
/// weights: the d2h+slice+h2d round trip happens once per (pointer,
/// offset) and lands in `cache.kv_biases` -- steady-state calls (and any
/// future capture window) stay device-only.
fn kv_bias(be: &AscendBackend, cache: &mut AscendCaches, bias: &Option<Tensor>, start: i64, len: i64) -> Result<Option<Arc<DeviceBuffer>>> {
    let Some(bias) = bias else { return Ok(None) };
    let b = tensor_buf(bias)?;
    let key = (b.as_ptr() as usize, start, len);
    if let Some(v) = cache.kv_biases.get(&key) {
        return Ok(Some(v.clone()));
    }
    let bytes = (len * 2) as usize;
    let off = (start * 2) as usize;
    // full d2h then host-side slice (bias is tiny and static)
    let mut full = vec![0u8; b.len()];
    acl(be.ctx().copy_d2h(b, &mut full))?;
    let out = acl(be.ctx().malloc(bytes))?;
    acl(be.ctx().copy_h2d(&out, &full[off..off + bytes]))?;
    let out = Arc::new(out);
    cache.kv_biases.insert(key, out.clone());
    Ok(Some(out))
}

unsafe fn from_raw<'a>(b: &'a DeviceBuffer, off: usize, len: usize) -> &'a [u8] {
    std::slice::from_raw_parts((b.as_ptr() as *const u8).add(off), len)
}

/// Full (non-causal) attention over the given k/v, straight in the
/// [tokens, heads*d] linear layout (BSH inside PFA -- no transposes).
fn pfa_full(
    be: &AscendBackend,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    tokens: i64,
    heads: i64,
    kv_heads: i64,
    head_dim: i64,
) -> Result<DeviceBuffer> {
    acl(aops::prompt_flash_attention_bsh_fp16(
        be.ctx(), be.stream(), q, k, v, tokens, heads, kv_heads, head_dim, None,
    ))
}

/// Vision (SigLIP) layer: pre-norm LayerNorm + attention + residual,
/// pre-norm MLP with gelu — mirror of vision_layer_bf16.
#[allow(clippy::too_many_arguments)]
pub fn vision_layer_ascend(
    be: &AscendBackend,
    cache: &mut AscendCaches,
    weights: &Bf16DeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    let ctx = be.ctx();
    let stream = be.stream();
    let tokens = input.shape().dims()[0] as i64;
    let width = input.shape().dims()[1] as i64;
    let input_b = tensor_buf(input)?;
    let zeros = be.zeros(width as usize * tokens as usize * 2)?;

    let normalized = aq::layer_norm(be,input_b, &zeros,
        tensor_buf(&weights.norm1.weight)?, tensor_buf(&weights.norm1.bias)?,
        tokens, width, layer_norm_eps as f64)?;
    let qkv = aq::matmul(be, &mut cache.nz,&normalized, [tokens, width],
        tensor_buf(&weights.qkv.weight)?, [width, (heads + 2 * heads) as i64 * head_dim as i64])?;
    let hd = head_dim as i64;
    let qd = heads as i64 * hd;
    let q_raw = aq::take_rows(be,&qkv, 0, tokens, qd)?;
    let k_raw = aq::take_rows(be,&qkv, tokens, tokens, qd)?;
    let v_raw = aq::take_rows(be,&qkv, tokens * 2, tokens, qd)?;
    let q_bias = vision_kv_bias(be, cache, &weights.qkv.bias, 0, qd)?;
    let k_bias = vision_kv_bias(be, cache, &weights.qkv.bias, qd, qd)?;
    let v_bias = vision_kv_bias(be, cache, &weights.qkv.bias, qd * 2, qd)?;
    let q = aq::bias(be,&q_raw, q_bias.as_deref(), tokens, qd)?;
    let k = aq::bias(be,&k_raw, k_bias.as_deref(), tokens, qd)?;
    let v = aq::bias(be,&v_raw, v_bias.as_deref(), tokens, qd)?;

    // SigLIP attends WITHIN each view's window: [views, tpv] batched PFA
    // over the same contiguous [tokens, qd] buffers.
    let views = tokens / patches_per_view as i64;
    debug_assert_eq!(views * patches_per_view as i64, tokens);
    let attn = acl(aops::prompt_flash_attention_bsh_batch_fp16(
        be.ctx(), be.stream(), &q, &k, &v,
        views, patches_per_view as i64, heads as i64, heads as i64, hd, None,
    ))?;
    let proj = aq::matmul(be, &mut cache.nz,&attn, [tokens, qd],
        tensor_buf(&weights.output.weight)?, [qd, width])?;
    let proj = bias_add(be, &proj, weights.output.bias.as_ref(), tokens, width)?;
    let res1 = aq::add(be,&proj, input_b, &[tokens, width])?;
    let norm2 = aq::layer_norm(be,&res1, &zeros,
        tensor_buf(&weights.norm2.weight)?, tensor_buf(&weights.norm2.bias)?, tokens, width, layer_norm_eps as f64)?;

    let fc1_w = tensor_buf(&weights.fc1.weight)?;
    let inter = (fc1_w.len() / 2) as i64 / width;
    let act = aq::matmul(be, &mut cache.nz,&norm2, [tokens, width], fc1_w, [width, inter])?;
    let act = bias_add(be, &act, weights.fc1.bias.as_ref(), tokens, inter)?;
    let act = aq::gelu(be,&act, &[tokens, inter], false)?; // siglip uses exact gelu
    let out = aq::matmul(be, &mut cache.nz,&act, [tokens, inter],
        tensor_buf(&weights.fc2.weight)?, [inter, width])?;
    let out = bias_add(be, &out, weights.fc2.bias.as_ref(), tokens, width)?;
    let res2 = aq::add(be,&out, &res1, &[tokens, width])?;
    Ok(buf_tensor(be, res2, tokens as usize, width as usize))
}

/// Patch embedding: projection + bias + position table. The position
/// table has `patches_per_view` rows and repeats cyclically per view
/// (cuda kernel semantics: row r reads table[r % tpv]) -- any view count.
pub fn vision_patch_embed_ascend(
    be: &AscendBackend,
    cache: &mut AscendCaches,
    weights: &Bf16LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let tokens = patches.shape().dims()[0] as i64;
    let width = patches.shape().dims()[1] as i64;
    let out_w = weights.weight.shape().dims()[1] as i64;
    let proj = aq::matmul(be, &mut cache.nz, tensor_buf(patches)?, [tokens, width],
        tensor_buf(&weights.weight)?, [width, out_w])?;
    let proj = bias_add(be, &proj, weights.bias.as_ref(), tokens, out_w)?;
    let p = patches_per_view as i64;
    // cyclic per-view repeat of the [p, width] table; index row cached by
    // (tokens, tpv) -- the per-call h2d was a sync memcpy in capture windows
    let di = {
        let key = (tokens, p);
        match cache.pos_idx.get(&(-key.0, key.1, 0)) {
            Some(v) => v.clone(),
            None => {
                let idx: Vec<i32> = (0..tokens as usize).map(|r| (r % p as usize) as i32).collect();
                let idx_bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
                let buf = acl(be.ctx().malloc(idx_bytes.len()))?;
                acl(be.ctx().copy_h2d(&buf, &idx_bytes))?;
                let buf = Arc::new(buf);
                cache.pos_idx.insert((-key.0, key.1, 0), buf.clone());
                buf
            }
        }
    };
    let pos_rep = aq::gather_rows(be, tensor_buf(position_embedding)?, p, out_w, &di, tokens)?;
    let out = acl(aops::add_fp16(be.ctx(), be.stream(), &proj, &pos_rep, &[tokens, out_w]))?;
    Ok(be.wrap_fp16(out, vec![tokens as usize, out_w as usize]))
}

/// ada-norm, cuda kernel semantics (normalization.cuh ada_rms_norm_bf16):
/// y = rms(x) * (1 + style[0:cols]) + style[cols:2*cols]. The style
/// projection outputs [3*cols]; the third segment is unused here.
/// (1+scale)/shift rows are materialized once per style tensor and
/// cached in `cache.style_rows` -- steady-state steps run device-only.
fn adaptive_rms(be: &AscendBackend, cache: &mut AscendCaches, x: &DeviceBuffer, style: &Tensor, rows: i64, cols: i64, eps: f32) -> Result<DeviceBuffer> {
    // rms with gamma = ones (ada-norm has no gamma)
    let zeros = be.zeros(rows as usize * cols as usize * 2)?;
    let c = cols as usize;
    let ones = zeros_scale_buf(be, cache, c)?;
    let normed = acl(aops::add_rms_norm_fp16(be.ctx(), be.stream(), x, &zeros, &ones, &[rows, cols], eps as f64))?.0;
    let key = tensor_buf(style)?.as_ptr() as usize;
    let (scale_b, shift_b) = match cache.style_rows.get(&key) {
        Some(v) => (v.0.clone(), v.1.clone()),
        None => {
            let style_h = host_f16_row(be, style, 3 * c)?;
            let mut scale_row = vec![0u16; c];
            let mut shift_row = vec![0u16; c];
            for i in 0..c {
                scale_row[i] = f32_to_f16_bits(f16_bits_to_f32(style_h[i]) + 1.0);
                shift_row[i] = style_h[c + i];
            }
            let mut upload = |vals: &[u16]| -> Result<Arc<DeviceBuffer>> {
                let buf = acl(be.ctx().malloc(vals.len() * 2))?;
                let bytes: &[u8] = unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 2) };
                acl(be.ctx().copy_h2d(&buf, bytes))?;
                Ok(Arc::new(buf))
            };
            let entry = (upload(&scale_row)?, upload(&shift_row)?);
            cache.style_rows.insert(key, entry.clone());
            entry
        }
    };
    let scale_mat = acl(aops::row_replicate_fp16(be.ctx(), be.stream(), &scale_b, rows, cols))?;
    let shift_mat = acl(aops::row_replicate_fp16(be.ctx(), be.stream(), &shift_b, rows, cols))?;
    let scaled = acl(aops::mul_fp16(be.ctx(), be.stream(), &normed, &scale_mat, &[rows, cols]))?;
    acl(aops::add_fp16(be.ctx(), be.stream(), &scaled, &shift_mat, &[rows, cols]))
}

fn zeros_scale_buf(be: &AscendBackend, cache: &mut AscendCaches, cols: usize) -> Result<Arc<DeviceBuffer>> {
    // ada-norm's rms carries no gamma (the scale lives in the style
    // projection) -- gamma = ones of exactly `cols`, cached per size:
    // the per-call h2d was a sync memcpy inside capture windows.
    if let Some(v) = cache.ones_rows.get(&cols) {
        return Ok(v.clone());
    }
    let ones = vec![0x3c00u16; cols]; // 1.0 fp16
    let buf = acl(be.ctx().malloc(cols * 2))?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 2) };
    acl(be.ctx().copy_h2d(&buf, bytes))?;
    let buf = Arc::new(buf);
    cache.ones_rows.insert(cols, buf.clone());
    Ok(buf)
}

fn host_f16_row(be: &AscendBackend, t: &Tensor, len: usize) -> Result<Vec<u16>> {
    let b = tensor_buf(t)?;
    // `t` may be the async product of a matmul on our stream (e.g. the
    // style projections); synchronous aclrtMemcpy does NOT wait for it.
    acl(be.stream().synchronize())?;
    let mut host = vec![0u8; len * 2];
    acl(be.ctx().copy_d2h(b, &mut host))?;
    Ok(host.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 { return f32::from_bits(sign); }
        let e = (mant.leading_zeros() - 22) as i32;
        let norm = (mant << (e + 1)) & 0x03ff;
        let reb = (127 - 15 + 1 - e - 1) as u32;
        return f32::from_bits(sign | (reb << 23) | (norm << 13));
    }
    if exp == 0x1f { return f32::from_bits(sign | 0x7f80_0000 | (mant << 13)); }
    f32::from_bits(sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13))
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if ((bits >> 23) & 0xff) == 0 || exp <= 0 { return sign; }
    if exp >= 0x1f { return sign | 0x7c00; }
    let m = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let mut out = ((exp as u16) << 10) | m;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) { out += 1; }
    sign | out
}

/// Host-side kv bias slice (same pass-through as language path).
fn vision_kv_bias(be: &AscendBackend, cache: &mut AscendCaches, bias: &Option<Tensor>, start: i64, len: i64) -> Result<Option<Arc<DeviceBuffer>>> {
    kv_bias(be, cache, bias, start, len)
}

/// Action expert layer with ada-norm styles and prefix KV — mirror of
/// action_layer_bf16. prefix_k/v hold the language prefix keys/values
/// [prefix_tokens, kv_heads*d]; attention runs over prefix + this chunk.
#[allow(clippy::too_many_arguments)]
pub fn action_layer_ascend(
    be: &AscendBackend,
    cache: &mut AscendCaches,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceActionLayer,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_style: &Tensor,
    mlp_style: &Tensor,
    next_norm_style: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<ActionLayerOutput> {
    let ctx = be.ctx();
    let stream = be.stream();
    let tokens = input.shape().dims()[0] as i64;
    let width = input.shape().dims()[1] as i64;
    let input_b = tensor_buf(input)?;
    let trace = std::env::var("APXINF_ASCEND_TRACE").is_ok();
    macro_rules! mark {
        ($tag:expr) => {
            if trace {
                be.synchronize()?;
                eprintln!("[trace] action: {}", $tag);
            }
        };
    }

    let _ = (ctx, stream);
    let normalized = match attention_normalized {
        Some(v) => aq::take_rows(be, tensor_buf(v)?, 0, tokens, width)?,
        None => adaptive_rms(be, cache, input_b, attention_style, tokens, width, rms_eps)?,
    };
    mark!("ada-norm");

    let qd = config.num_heads as i64 * config.head_dim as i64;
    let kv_d = config.num_kv_heads as i64 * config.head_dim as i64;
    let fused_w = tensor_buf(&weights.qkv.weight)?;
    let out_w = qd + 2 * kv_d;
    let in_w = (fused_w.len() / 2) as i64 / out_w;
    let qkv = aq::matmul(be, &mut cache.nz, &normalized, [tokens, in_w], fused_w, [in_w, out_w])?;
    mark!("qkv matmul");
    let q_raw = aq::take_rows(be, &qkv, 0, tokens, qd)?;
    let k_raw = aq::take_rows(be, &qkv, tokens, tokens, kv_d)?;
    let v_raw = aq::take_rows(be, &qkv, tokens * 2, tokens, kv_d)?;
    let q_bias = kv_bias(be, cache, &weights.qkv.bias, 0, qd)?;
    let k_bias = kv_bias(be, cache, &weights.qkv.bias, qd, kv_d)?;
    let v_bias = kv_bias(be, cache, &weights.qkv.bias, qd + kv_d, kv_d)?;
    let q = aq::bias(be, &q_raw, q_bias.as_deref(), tokens, qd)?;
    let k = aq::bias(be, &k_raw, k_bias.as_deref(), tokens, kv_d)?;
    let v = aq::bias(be, &v_raw, v_bias.as_deref(), tokens, kv_d)?;

    let q_rope = apply_rope_rows(be, cache, &q, tokens, config.num_heads as i64, rope_theta, position_offset)?;
    let k_rope = apply_rope_rows(be, cache, &k, tokens, config.num_kv_heads as i64, rope_theta, position_offset)?;
    mark!("ropes");

    // concat prefix k/v with this chunk (rows), then full attention.
    let pk = tensor_buf(prefix_k)?;
    let pv = tensor_buf(prefix_v)?;
    let prefix_tokens = (pk.len() as i64 / 2) / kv_d;
    let total = prefix_tokens + tokens;
    let k_all = acl(aops::cat_fp16(ctx, stream, &[pk, &k_rope], &[vec![prefix_tokens, kv_d], vec![tokens, kv_d]], 0, &[total, kv_d]))?;
    let v_all = acl(aops::cat_fp16(ctx, stream, &[pv, &v], &[vec![prefix_tokens, kv_d], vec![tokens, kv_d]], 0, &[total, kv_d]))?;
    let attn = acl(aops::prompt_flash_attention_cross_bsh_fp16(
        ctx, stream, &q_rope, &k_all, &v_all, tokens, total,
        config.num_heads as i64, config.num_kv_heads as i64, config.head_dim as i64, None,
    ))?;
    mark!("cross pfa");

    let proj = aq::matmul(be, &mut cache.nz, &attn, [tokens, qd], tensor_buf(&weights.output.weight)?, [qd, width])?;
    mark!("output proj");
    let proj = bias_add(be, &proj, weights.output.bias.as_ref(), tokens, width)?;
    let res = aq::add(be, &proj, &normalized, &[tokens, width])?;
    let normed = adaptive_rms(be, cache, &res, mlp_style, tokens, width, rms_eps)?;
    mark!("mlp ada-norm");

    let gw = tensor_buf(&weights.gate_up.weight)?;
    // fused weight is [width, 2*inter] -- divide the doubled width out
    // (the old `... / width` read the FUSED width as inter, then doubled
    // it again in bsh -> 2x-OOB b descriptor -> aicore MTE fault)
    let inter = (gw.len() as i64 / 2) / (width * 2);
    let gate_up = aq::matmul(be, &mut cache.nz, &normed, [tokens, width], gw, [width, inter * 2])?;
    mark!("gate_up matmul");
    // [gate; up] concat order (from_host_parts) -- gelu on the first half.
    let gate = aq::take_rows(be, &gate_up, 0, tokens, inter)?;
    let up = aq::take_rows(be, &gate_up, tokens, tokens, inter)?;
    let gate_g = aq::gelu(be, &gate, &[tokens, inter], true)?;
    let act = aq::mul(be, &gate_g, &up, &[tokens, inter])?;
    mark!("geglu");
    let proj2 = aq::matmul(be, &mut cache.nz, &act, [tokens, inter], tensor_buf(&weights.down.weight)?, [inter, width])?;
    mark!("down matmul");
    let proj2 = bias_add(be, &proj2, weights.down.bias.as_ref(), tokens, width)?;
    let hidden = aq::add(be, &proj2, &res, &[tokens, width])?;
    let next_normalized = adaptive_rms(be, cache, &hidden, next_norm_style, tokens, width, rms_eps)?;
    Ok(ActionLayerOutput {
        hidden: be.wrap_fp16(hidden, vec![tokens as usize, width as usize]),
        next_normalized: be.wrap_fp16(next_normalized, vec![tokens as usize, width as usize]),
    })
}

