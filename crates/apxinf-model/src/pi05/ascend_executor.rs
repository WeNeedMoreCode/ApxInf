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
fn acl<T>(r: std::result::Result<T, AclError>) -> Result<T> {
    r.map_err(|e| Error::Other(format!("aclnn: {e}")))
}

/// acl()-bridged mirrors of the apxinf-ascend op functions so call sites
/// can use `?` directly against apxinf_core::Error.
mod aq {
    use super::*;
    use apxinf_ascend::AscendStream;

    pub fn matmul(be: &AscendBackend, a: &DeviceBuffer, ash: [i64; 2], b: &DeviceBuffer, bsh: [i64; 2]) -> Result<DeviceBuffer> {
        acl(aops::matmul_fp16(be.ctx(), be.stream(), a, ash, b, bsh))
    }
    pub fn bias(be: &AscendBackend, x: &DeviceBuffer, bias: Option<&DeviceBuffer>, rows: i64, cols: i64) -> Result<DeviceBuffer> {
        acl(aops::bias_add_fp16(be.ctx(), be.stream(), x, bias.ok_or_else(|| Error::Other("bias required in aq::bias".into()))?, rows, cols))
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
    #[allow(dead_code)]
    fn _unused(_: &AscendStream) {}
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
    cache: &mut AscendRopeCache,
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
    // materialize per-row cos/sin: gather table rows by position index,
    // each position repeated `heads` times.
    let mut pos_idx = Vec::with_capacity((tokens * heads) as usize);
    for t in 0..tokens {
        for _ in 0..heads {
            pos_idx.push((pos_offset as i64 + t) as i32);
        }
    }
    let pos_buf = acl(ctx.malloc(pos_idx.len() * 4))?;
    let pb = unsafe { std::slice::from_raw_parts(pos_idx.as_ptr() as *const u8, pos_idx.len() * 4) };
    acl(ctx.copy_h2d(&pos_buf, pb))?;

    let tables: AscendRopeTableArcs = {
        let t = cache.tables(be, theta, d, (pos_offset + tokens as usize).max(1))?;
        AscendRopeTableArcs { cos: t.cos.clone(), sin_signed: t.sin_signed.clone(), rot_idx: t.rot_idx.clone(), max_pos: t.max_pos }
    };
    let rows = tokens * heads;
    let max_pos = tables.max_pos as i64;
    let t_cos_tab = acl(apxinf_ascend::AclTensor::fp16_nd(&tables.cos, &[max_pos, d]))?;
    let t_sin_tab = acl(apxinf_ascend::AclTensor::fp16_nd(&tables.sin_signed, &[max_pos, d]))?;
    let t_pos = acl(apxinf_ascend::AclTensor::i32_nd(&pos_buf, &[rows]))?;
    let cos_pos = acl(ctx.malloc((rows * d * 2) as usize))?;
    let sin_pos = acl(ctx.malloc((rows * d * 2) as usize))?;
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
    acl(stream.synchronize())?;
    acl(aops::rope_rotate_half_fp16(ctx, stream, x, rows, d, &cos_pos, &sin_pos, &tables.rot_idx))
}

/// bias + reshape helper: y = x + bias (row broadcast), staying in buffers.
fn bias_add(be: &AscendBackend, x: &DeviceBuffer, bias: Option<&Tensor>, rows: i64, cols: i64) -> Result<DeviceBuffer> {
    let Some(bias) = bias else { return acl(aops::take_rows_fp16(be.ctx(), be.stream(), x, 0, (x.len() / 2) as i64, 2)) };
    let b = tensor_buf(bias)?;
    acl(aops::bias_add_fp16(be.ctx(), be.stream(), x, b, rows, cols))
}

fn tensor_buf(t: &Tensor) -> Result<&DeviceBuffer> {
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
    cache: &mut AscendRopeCache,
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
    let qkv = aq::matmul(be,norm_b, [tokens, hidden],
        tensor_buf(&weights.qkv.weight)?, [hidden, config.num_heads as i64 * config.head_dim as i64 + 2 * config.num_kv_heads as i64 * config.head_dim as i64])?;

    let qd = config.num_heads as i64 * config.head_dim as i64;
    let kv_d = config.num_kv_heads as i64 * config.head_dim as i64;
    let q_raw = aq::take_rows(be,&qkv, 0, tokens, qd)?;
    let k_raw = aq::take_rows(be,&qkv, tokens, tokens, kv_d)?;
    let v_raw = aq::take_rows(be,&qkv, tokens * 2, tokens, kv_d)?;
    // qkv bias spans the fused width; slice per part.
    let q_bias = tensor_opt(&weights.qkv.bias)?;
    let k_bias = kv_bias(be, &weights.qkv.bias, qd, kv_d)?;
    let v_bias = kv_bias(be, &weights.qkv.bias, qd + kv_d, kv_d)?;
    let q = aq::bias(be,&q_raw, q_bias, tokens, qd)?;
    let k = aq::bias(be,&k_raw, k_bias.as_ref(), tokens, kv_d)?;
    let v = aq::bias(be,&v_raw, v_bias.as_ref(), tokens, kv_d)?;

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
    let projected = aq::matmul(be,&attn, [tokens, config.num_heads as i64 * config.head_dim as i64],
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
    let gate_up = aq::matmul(be,fused_norm_b, [tokens, width],
        tensor_buf(&weights.gate_up.weight)?, [width, inter * 2])?;
    let up = aq::take_rows(be,&gate_up, 0, tokens, inter)?;
    let gate = aq::take_rows(be,&gate_up, tokens, tokens, inter)?;
    let gate_g = aq::gelu(be,&gate, &[tokens, inter], true)?;
    let activated = aq::mul(be,&up, &gate_g, &[tokens, inter])?;
    let projected2 = aq::matmul(be,&activated, [tokens, inter],
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

/// Slice a fused qkv bias into a kv-slice device row (host pass-through).
fn kv_bias(be: &AscendBackend, bias: &Option<Tensor>, start: i64, len: i64) -> Result<Option<DeviceBuffer>> {
    let Some(bias) = bias else { return Ok(None) };
    let b = tensor_buf(bias)?;
    let bytes = (len * 2) as usize;
    let off = (start * 2) as usize;
    // full d2h then host-side slice (bias is tiny and static)
    let mut full = vec![0u8; b.len()];
    acl(be.ctx().copy_d2h(b, &mut full))?;
    let out = acl(be.ctx().malloc(bytes))?;
    acl(be.ctx().copy_h2d(&out, &full[off..off + bytes]))?;
    acl(be.stream().synchronize())?;
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
    let zeros = be.zeros(width as usize * tokens as usize)?;

    let normalized = aq::layer_norm(be,input_b, &zeros,
        tensor_buf(&weights.norm1.weight)?, tensor_buf(&weights.norm1.bias)?,
        tokens, width, layer_norm_eps as f64)?;
    let qkv = aq::matmul(be,&normalized, [tokens, width],
        tensor_buf(&weights.qkv.weight)?, [width, (heads + 2 * heads) as i64 * head_dim as i64])?;
    let hd = head_dim as i64;
    let qd = heads as i64 * hd;
    let q_raw = aq::take_rows(be,&qkv, 0, tokens, qd)?;
    let k_raw = aq::take_rows(be,&qkv, tokens, tokens, qd)?;
    let v_raw = aq::take_rows(be,&qkv, tokens * 2, tokens, qd)?;
    let k_bias = vision_kv_bias(be, &weights.qkv.bias, qd, qd)?;
    let v_bias = vision_kv_bias(be, &weights.qkv.bias, qd * 2, qd)?;
    let q = aq::bias(be,&q_raw, tensor_opt(&weights.qkv.bias)?, tokens, qd)?;
    let k = aq::bias(be,&k_raw, k_bias.as_ref(), tokens, qd)?;
    let v = aq::bias(be,&v_raw, v_bias.as_ref(), tokens, qd)?;

    let attn = pfa_full(be, &q, &k, &v, tokens, heads as i64, heads as i64, hd)?;
    let proj = aq::matmul(be,&attn, [tokens, qd],
        tensor_buf(&weights.output.weight)?, [qd, width])?;
    let proj = bias_add(be, &proj, weights.output.bias.as_ref(), tokens, width)?;
    let res1 = aq::add(be,&proj, input_b, &[tokens, width])?;
    let norm2 = aq::layer_norm(be,&res1, &zeros,
        tensor_buf(&weights.norm2.weight)?, tensor_buf(&weights.norm2.bias)?, tokens, width, layer_norm_eps as f64)?;

    let fc1_w = tensor_buf(&weights.fc1.weight)?;
    let inter = (fc1_w.len() / 2) as i64 / width;
    let act = aq::matmul(be,&norm2, [tokens, width], fc1_w, [width, inter])?;
    let act = bias_add(be, &act, weights.fc1.bias.as_ref(), tokens, inter)?;
    let act = aq::gelu(be,&act, &[tokens, inter], false)?; // siglip uses exact gelu
    let out = aq::matmul(be,&act, [tokens, inter],
        tensor_buf(&weights.fc2.weight)?, [inter, width])?;
    let out = bias_add(be, &out, weights.fc2.bias.as_ref(), tokens, width)?;
    let res2 = aq::add(be,&out, &res1, &[tokens, width])?;
    Ok(buf_tensor(be, res2, tokens as usize, width as usize))
}

/// Patch embedding: projection + bias + position table (the position
/// rows repeat per view: cat(position, position) covers both views).
pub fn vision_patch_embed_ascend(
    be: &AscendBackend,
    weights: &Bf16LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let tokens = patches.shape().dims()[0] as i64;
    let width = patches.shape().dims()[1] as i64;
    let proj = aq::matmul(be, tensor_buf(patches)?, [tokens, width],
        tensor_buf(&weights.weight)?, [width, weights.weight.shape().dims()[1] as i64])?;
    let proj = bias_add(be, &proj, weights.bias.as_ref(), tokens, width)?;
    let pos = tensor_buf(position_embedding)?;
    let p = patches_per_view as i64;
    let pos2 = acl(aops::cat_fp16(be.ctx(), be.stream(), &[pos, pos],
        &[vec![p, width], vec![p, width]], 0, &[2 * p, width]))?;
    let out = acl(aops::add_fp16(be.ctx(), be.stream(), &proj, &pos2, &[tokens, width]))?;
    Ok(be.wrap_fp16(out, vec![tokens as usize, width as usize]))
}

/// ada-norm: y = rms_norm(x) * (1 + style), style is a [width] row.
fn adaptive_rms(be: &AscendBackend, x: &DeviceBuffer, style: &Tensor, rows: i64, cols: i64, eps: f32) -> Result<DeviceBuffer> {
    // rms with gamma = ones (ada-norm has no gamma), then scale by
    // row-replicated (1 + style).
    let zeros = be.zeros(rows as usize * cols as usize * 2)?;
    let ones = zeros_scale_buf(be, style)?;
    let normed = acl(aops::add_rms_norm_fp16(be.ctx(), be.stream(), x, &zeros, &ones, &[rows, cols], eps as f64))?.0;
    let mut one_plus = vec![0u16; cols as usize];
    let style_h = host_f16_row(be, style, cols as usize)?;
    for i in 0..cols as usize {
        let v = f16_bits_to_f32(style_h[i]);
        one_plus[i] = f32_to_f16_bits(v + 1.0);
    }
    let row = acl(be.ctx().malloc(cols as usize * 2))?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(one_plus.as_ptr() as *const u8, one_plus.len() * 2) };
    acl(be.ctx().copy_h2d(&row, bytes))?;
    let scale_mat = acl(aops::row_replicate_fp16(be.ctx(), be.stream(), &row, rows, cols))?;
    acl(aops::mul_fp16(be.ctx(), be.stream(), &normed, &scale_mat, &[rows, cols]))
}

fn zeros_scale_buf(be: &AscendBackend, style: &Tensor) -> Result<DeviceBuffer> {
    // gamma for rms = style itself in pi0.5's ada-norm? The cuda path uses
    // adaptive_rms(x, style): norm scale comes from the style projection.
    // Gemma3 ada-norm: normalized = rms(x) * (1 + style); rms has NO gamma.
    // So gamma = ones.
    let width = style.shape().dims()[0];
    let ones = vec![0x3c00u16; width]; // 1.0 fp16
    let buf = acl(be.ctx().malloc(width * 2))?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 2) };
    acl(be.ctx().copy_h2d(&buf, bytes))?;
    Ok(buf)
}

fn host_f16_row(be: &AscendBackend, t: &Tensor, len: usize) -> Result<Vec<u16>> {
    let b = tensor_buf(t)?;
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
fn vision_kv_bias(be: &AscendBackend, bias: &Option<Tensor>, start: i64, len: i64) -> Result<Option<DeviceBuffer>> {
    kv_bias(be, bias, start, len)
}

/// Action expert layer with ada-norm styles and prefix KV — mirror of
/// action_layer_bf16. prefix_k/v hold the language prefix keys/values
/// [prefix_tokens, kv_heads*d]; attention runs over prefix + this chunk.
#[allow(clippy::too_many_arguments)]
pub fn action_layer_ascend(
    be: &AscendBackend,
    cache: &mut AscendRopeCache,
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

    let _ = (ctx, stream);
    let normalized = match attention_normalized {
        Some(v) => aq::take_rows(be, tensor_buf(v)?, 0, tokens, width)?,
        None => adaptive_rms(be, input_b, attention_style, tokens, width, rms_eps)?,
    };

    let qd = config.num_heads as i64 * config.head_dim as i64;
    let kv_d = config.num_kv_heads as i64 * config.head_dim as i64;
    let fused_w = tensor_buf(&weights.qkv.weight)?;
    let out_w = qd + 2 * kv_d;
    let in_w = (fused_w.len() / 2) as i64 / out_w;
    let qkv = aq::matmul(be, &normalized, [tokens, in_w], fused_w, [in_w, out_w])?;
    let q_raw = aq::take_rows(be, &qkv, 0, tokens, qd)?;
    let k_raw = aq::take_rows(be, &qkv, tokens, tokens, kv_d)?;
    let v_raw = aq::take_rows(be, &qkv, tokens * 2, tokens, kv_d)?;
    let q_bias = tensor_opt(&weights.qkv.bias)?;
    let k_bias = kv_bias(be, &weights.qkv.bias, qd, kv_d)?;
    let v_bias = kv_bias(be, &weights.qkv.bias, qd + kv_d, kv_d)?;
    let q = acl(aops::bias_add_fp16(ctx, stream, &q_raw, q_bias.ok_or_else(|| Error::Other("q bias".into()))?, tokens, qd))?;
    let k = acl(aops::bias_add_fp16(ctx, stream, &k_raw, k_bias.as_ref().ok_or_else(|| Error::Other("k bias".into()))?, tokens, kv_d))?;
    let v = acl(aops::bias_add_fp16(ctx, stream, &v_raw, v_bias.as_ref().ok_or_else(|| Error::Other("v bias".into()))?, tokens, kv_d))?;

    let q_rope = apply_rope_rows(be, cache, &q, tokens, config.num_heads as i64, rope_theta, position_offset)?;
    let k_rope = apply_rope_rows(be, cache, &k, tokens, config.num_kv_heads as i64, rope_theta, position_offset)?;

    // concat prefix k/v with this chunk (rows), then full attention.
    let pk = tensor_buf(prefix_k)?;
    let pv = tensor_buf(prefix_v)?;
    let prefix_tokens = (pk.len() as i64 / 2) / kv_d;
    let total = prefix_tokens + tokens;
    let k_all = acl(aops::cat_fp16(ctx, stream, &[pk, &k_rope], &[vec![prefix_tokens, kv_d], vec![tokens, kv_d]], 0, &[total, kv_d]))?;
    let v_all = acl(aops::cat_fp16(ctx, stream, &[pv, &v], &[vec![prefix_tokens, kv_d], vec![tokens, kv_d]], 0, &[total, kv_d]))?;
    let attn = acl(aops::prompt_flash_attention_bsh_fp16(ctx, stream, &q_rope, &k_all, &v_all, total, config.num_heads as i64, config.num_kv_heads as i64, config.head_dim as i64, None))?;

    let proj = aq::matmul(be, &attn, [tokens, qd], tensor_buf(&weights.output.weight)?, [qd, width])?;
    let proj = bias_add(be, &proj, weights.output.bias.as_ref(), tokens, width)?;
    let res = aq::add(be, &proj, &normalized, &[tokens, width])?;
    let normed = adaptive_rms(be, &res, mlp_style, tokens, width, rms_eps)?;

    let gw = tensor_buf(&weights.gate_up.weight)?;
    let inter = (gw.len() as i64 / 2) / width;
    let gate_up = aq::matmul(be, &normed, [tokens, width], gw, [width, inter * 2])?;
    let up = aq::take_rows(be, &gate_up, 0, tokens, inter)?;
    let gate = aq::take_rows(be, &gate_up, tokens, tokens, inter)?;
    let gate_g = aq::gelu(be, &gate, &[tokens, inter], true)?;
    let act = aq::mul(be, &up, &gate_g, &[tokens, inter])?;
    let proj2 = aq::matmul(be, &act, [tokens, inter], tensor_buf(&weights.down.weight)?, [inter, width])?;
    let proj2 = bias_add(be, &proj2, weights.down.bias.as_ref(), tokens, width)?;
    let hidden = aq::add(be, &proj2, &res, &[tokens, width])?;
    let next_normalized = adaptive_rms(be, &hidden, next_norm_style, tokens, width, rms_eps)?;
    Ok(ActionLayerOutput {
        hidden: be.wrap_fp16(hidden, vec![tokens as usize, width as usize]),
        next_normalized: be.wrap_fp16(next_normalized, vec![tokens as usize, width as usize]),
    })
}

