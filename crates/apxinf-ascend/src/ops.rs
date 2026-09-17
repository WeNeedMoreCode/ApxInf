//! aclnn compute ops, starting with matmul (roadmap divergence point #1).

use crate::context::AscendContext;
use crate::ffi;
use crate::stream::AscendStream;
use crate::tensor::AclTensor;
use crate::{AclError, Result};

/// cubeMathType = KEEP_DTYPE: compute in the tensors' own dtype (fp16 in,
/// fp16 out on 310P3, which has no bf16 unit).
const CUBE_MATH_KEEP_DTYPE: i8 = 1;

/// M alignment for the aclnn matmul kernels on 310P3: with wide N
/// (>=12288), any M that is not a multiple of 16 trips an aicore tiling
/// fault (507015 at sync) on BOTH descriptors (m-ladder probe 2026-09-18:
/// 812/820/828 x N16384/32768 all fault, 832 all correct; N<=8192 is
/// immune). Zero-pad M up to the next multiple, slice the real rows back.
const MATMUL_M_ALIGN: i64 = 16;

/// Run `f` on a zero-padded [target_m, k] copy of `a`, slice the real
/// `m` rows back out. Padding appends whole rows so the real data is one
/// contiguous leading block -- a single D2D copy after a stream-ordered
/// memset.
fn with_m_padded(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    m: i64,
    k: i64,
    n: i64,
    target_m: i64,
    f: impl FnOnce(&crate::DeviceBuffer, i64) -> Result<crate::DeviceBuffer>,
) -> Result<crate::DeviceBuffer> {
    // pooled scratch: padded feeds async memset/copy/matmul and must
    // outlive the stream queue (aclrtFree is not stream-ordered)
    let padded = ctx.scratch_buf((target_m * k * 2) as usize)?;
    let padded: &crate::DeviceBuffer = &padded;
    ctx.memset_async(&padded, 0, stream)?;
    let bytes = (m * k * 2) as usize;
    let code = unsafe {
        ffi::aclrtMemcpyAsync(
            padded.as_ptr(),
            bytes,
            a.as_ptr() as *const std::ffi::c_void,
            bytes,
            ffi::ACL_MEMCPY_DEVICE_TO_DEVICE,
            stream.handle(),
        )
    };
    if code != 0 {
        return Err(AclError { code, op: "aclrtMemcpyAsync(m-pad)" });
    }
    let full = f(&padded, target_m)?;
    take_rows_fp16(ctx, stream, &full, 0, m, n)
}

/// Pad `a`'s row count up to a multiple of MATMUL_M_ALIGN when needed.
fn matmul_m_pad(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    m: i64,
    k: i64,
    n: i64,
    f: impl FnOnce(&crate::DeviceBuffer, i64) -> Result<crate::DeviceBuffer>,
) -> Result<crate::DeviceBuffer> {
    let target = (m + MATMUL_M_ALIGN - 1) / MATMUL_M_ALIGN * MATMUL_M_ALIGN;
    if target == m {
        return f(a, m);
    }
    with_m_padded(ctx, stream, a, m, k, n, target, f)
}

/// out = a @ b for fp16 ND matrices, executed on `stream`.
///
/// Shapes: a = [m, k], b = [k, n], out = [m, n] (all fp16, row-major).
/// Returns the output device buffer; sync via `stream.synchronize()`.
pub fn matmul_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    a_shape: [i64; 2],
    b: &crate::DeviceBuffer,
    b_shape: [i64; 2],
) -> Result<crate::DeviceBuffer> {
    let [m, k] = a_shape;
    let [k2, n] = b_shape;
    assert_eq!(k, k2, "inner dims must match: {a_shape:?} x {b_shape:?}");
    if m % MATMUL_M_ALIGN != 0 {
        return matmul_m_pad(ctx, stream, a, m, k, n, |a2, m2| {
            matmul_fp16(ctx, stream, a2, [m2, k], b, b_shape)
        });
    }

    let ta = AclTensor::fp16_nd(a, &a_shape)?;
    let tb = AclTensor::fp16_nd(b, &b_shape)?;
    let out = ctx.malloc((m * n * 2) as usize)?;
    let t_out = AclTensor::fp16_nd(&out, &[m, n])?;

    // Stage 1: plan -- workspace size + op executor.
    let mut ws_size: u64 = 0;
    let mut executor: *mut std::ffi::c_void = std::ptr::null_mut();
    let code = unsafe {
        ffi::aclnnMatmulGetWorkspaceSize(
            ta.handle(),
            tb.handle(),
            t_out.handle(),
            CUBE_MATH_KEEP_DTYPE,
            &mut ws_size,
            &mut executor,
        )
    };
    if code != 0 {
        return Err(AclError { code, op: "aclnnMatmulGetWorkspaceSize" });
    }

    // Stage 2: run. Workspace is device memory; 0-size is allowed with null.
    let ws = if ws_size > 0 { Some(ctx.malloc(ws_size as usize)?) } else { None };
    let code = unsafe {
        ffi::aclnnMatmul(
            ws.as_ref().map(|w| w.as_ptr()).unwrap_or(std::ptr::null_mut()),
            ws_size,
            executor,
            stream.handle(),
        )
    };
    if code != 0 {
        return Err(AclError { code, op: "aclnnMatmul" });
    }
    Ok(out)
}

/// Shared two-stage runner: plan (ws size + executor), malloc ws, run.
/// Closures receive (workspace_ptr, workspace_size, executor).
#[allow(clippy::too_many_arguments)]
fn two_stage(
    ctx: &AscendContext,
    stream: &AscendStream,
    op: &'static str,
    plan: impl FnOnce(&mut u64, &mut *mut std::ffi::c_void) -> i32,
    run: impl FnOnce(*mut std::ffi::c_void, u64, *mut std::ffi::c_void) -> i32,
) -> Result<()> {
    let mut ws_size: u64 = 0;
    let mut executor: *mut std::ffi::c_void = std::ptr::null_mut();
    let code = plan(&mut ws_size, &mut executor);
    if code != 0 {
        return Err(AclError { code, op });
    }
    let ws = if ws_size > 0 { Some(ctx.malloc(ws_size as usize)?) } else { None };
    let code = run(
        ws.as_ref().map(|w| w.as_ptr()).unwrap_or(std::ptr::null_mut()),
        ws_size,
        executor,
    );
    if code != 0 {
        return Err(AclError { code, op });
    }
    Ok(())
}

/// Concatenate fp16 tensors along `dim` (0-based, ND descriptors).
/// All inputs share the shape except along `dim`.
pub fn cat_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    parts: &[&crate::DeviceBuffer],
    shapes: &[Vec<i64>],
    dim: i64,
    out_shape: &[i64],
) -> Result<crate::DeviceBuffer> {
    assert_eq!(parts.len(), shapes.len());
    let out_elems: i64 = out_shape.iter().product();
    let out = ctx.malloc((out_elems * 2) as usize)?;

    let descs: Vec<AclTensor> = parts
        .iter()
        .zip(shapes)
        .map(|(b, s)| AclTensor::fp16_nd(b, s))
        .collect::<Result<Vec<_>>>()?;
    let t_out = AclTensor::fp16_nd(&out, out_shape)?;

    let handles: Vec<*mut std::ffi::c_void> = descs.iter().map(|d| d.handle()).collect();
    let list = unsafe {
        let l = ffi::aclCreateTensorList(handles.as_ptr(), handles.len() as u64);
        if l.is_null() {
            return Err(AclError { code: -1, op: "aclCreateTensorList" });
        }
        l
    };

    two_stage(
        ctx,
        stream,
        "aclnnCat",
        |ws, ex| unsafe { ffi::aclnnCatGetWorkspaceSize(list, dim, t_out.handle(), ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnCat(ws, size, ex, stream.handle()) },
    )?;
    // Ownership: aclDestroyTensorList releases the child descriptors too;
    // dropping our AclTensor wrappers as well would double-destroy and
    // corrupt the process's acl tensor registry (later ops segfault in
    // their plan phase -- bisected 2026-09-17).
    std::mem::forget(descs);
    unsafe { ffi::aclDestroyTensorList(list) };
    Ok(out)
}

/// Fused attention (PFA V3) for full (non-causal, unmasked) attention in
/// BNSD layout: q/k/v/out all [b, n, s, d] fp16, contiguous.
///
/// scale = 1/sqrt(d) unless overridden. 310P: fp16 only; the legacy
/// non-V3 PFA entry deprecates 2026-12 so we bind V3 from day one.
#[allow(clippy::too_many_arguments)]
pub fn prompt_flash_attention_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    q: &crate::DeviceBuffer,
    k: &crate::DeviceBuffer,
    v: &crate::DeviceBuffer,
    shape: [i64; 4], // [b, n, s, d]
    scale: Option<f64>,
) -> Result<crate::DeviceBuffer> {
    let [b, n, s, d] = shape;
    let elems = b * n * s * d;
    let out = ctx.malloc((elems * 2) as usize)?;

    let dims = [b, n, s, d];
    let tq = AclTensor::fp16_nd(q, &dims)?;
    let tk = AclTensor::fp16_nd(k, &dims)?;
    let tv = AclTensor::fp16_nd(v, &dims)?;
    let tout = AclTensor::fp16_nd(&out, &dims)?;

    let mut layout: [u8; 5] = *b"BNSD\0";
    let scale_value = scale.unwrap_or(1.0 / (d as f64).sqrt());
    let null = std::ptr::null_mut::<std::ffi::c_void>();

    two_stage(
        ctx,
        stream,
        "aclnnPromptFlashAttentionV3",
        |ws, ex| unsafe {
            ffi::aclnnPromptFlashAttentionV3GetWorkspaceSize(
                tq.handle(),
                tk.handle(),
                tv.handle(),
                null, // pseShift
                null, // attenMask (full attention)
                null, // actualSeqLengths
                null, // actualSeqLengthsKv
                null, // deqScale1
                null, // quantScale1
                null, // deqScale2
                null, // quantScale2
                null, // quantOffset2
                n,
                scale_value,
                i64::MAX, // preTokens: unrestricted
                0,        // nextTokens
                layout.as_mut_ptr(),
                n,        // numKeyValueHeads: no GQA
                0,        // sparseMode
                0,        // innerPrecise
                tout.handle(),
                ws,
                ex,
            )
        },
        |ws, size, ex| unsafe { ffi::aclnnPromptFlashAttentionV3(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// out = x + bias (fp16): x is [rows, cols], bias is [cols] -- broadcast
/// via a zero-stride descriptor, no materialized expansion.
pub fn bias_add_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    x: &crate::DeviceBuffer,
    bias: &crate::DeviceBuffer,
    rows: i64,
    cols: i64,
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc((rows * cols * 2) as usize)?;
    let tx = AclTensor::fp16_nd(x, &[rows, cols])?;
    let tb = AclTensor::fp16_row_broadcast(bias, rows, cols)?;
    let tout = AclTensor::fp16_nd(&out, &[rows, cols])?;
    let mut one: f32 = 1.0;
    let alpha = ScalarFp32::new(&mut one);
    two_stage(
        ctx,
        stream,
        "aclnnAdd(bias)",
        |ws, ex| unsafe { ffi::aclnnAddGetWorkspaceSize(tx.handle(), tb.handle(), alpha.handle(), tout.handle(), ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnAdd(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// Flow-matching Euler step: x = x0 + sigma * (x1 - x0), elementwise fp16.
/// Two composed aclnn calls (sub via mul(-1)+add avoided; muls+add used):
/// dx = (x1 - x0) needs a sub op -- express as add(x1, muls(x0, -1)).
pub fn euler_update_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    x0: &crate::DeviceBuffer,
    x1: &crate::DeviceBuffer,
    sigma: f32,
    shape: &[i64],
) -> Result<crate::DeviceBuffer> {
    let neg = muls_fp16(ctx, stream, x0, -1.0, shape)?;
    let dx = add_fp16(ctx, stream, x1, &neg, shape)?;
    let scaled = muls_fp16(ctx, stream, &dx, sigma, shape)?;
    add_fp16(ctx, stream, x0, &scaled, shape)
}

/// Copy out a row slice of a device buffer (D2D): rows
/// [row_offset, row_offset+rows) of a [total, cols] layout. Used for
/// QKV splits -- the aclTensor offset-view semantics proved
/// unreliable on device, so we pay a microsecond copy for certainty.
pub fn take_rows_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    buf: &crate::DeviceBuffer,
    row_offset: i64,
    rows: i64,
    cols: i64,
) -> Result<crate::DeviceBuffer> {
    let bytes = (rows * cols * 2) as usize;
    let off = (row_offset * cols * 2) as usize;
    assert!(off + bytes <= buf.len(), "take_rows out of bounds");
    let out = ctx.malloc(bytes)?;
    let code = unsafe {
        ffi::aclrtMemcpyAsync(
            out.as_ptr(),
            bytes,
            (buf.as_ptr() as *const u8).add(off) as *const std::ffi::c_void,
            bytes,
            ffi::ACL_MEMCPY_DEVICE_TO_DEVICE,
            stream.handle(),
        )
    };
    if code != 0 {
        return Err(AclError { code, op: "aclrtMemcpyAsync(take_rows)" });
    }
    Ok(out)
}

/// LayerNorm(x) via the fused aclnnAddLayerNorm(x, zeros, gamma, beta).
/// All three extra outputs (mean/rstd/xOut) are allocated here; `zeros`
/// must be a device-side zero buffer of x's size (caller-cached).
pub fn layer_norm_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    x: &crate::DeviceBuffer,
    zeros: &crate::DeviceBuffer,
    gamma: &crate::DeviceBuffer,
    beta: &crate::DeviceBuffer,
    rows: i64,
    cols: i64,
    eps: f64,
) -> Result<crate::DeviceBuffer> {
    let y = ctx.malloc((rows * cols * 2) as usize)?;
    let mean = ctx.malloc((rows * 4) as usize)?;
    let rstd = ctx.malloc((rows * 4) as usize)?;
    let x_out = ctx.malloc((rows * cols * 2) as usize)?;
    let dims = [rows, cols];
    let tx = AclTensor::fp16_nd(x, &dims)?;
    let tz = AclTensor::fp16_nd(zeros, &dims)?;
    let tg = AclTensor::fp16_nd(gamma, &[cols])?;
    let tb = AclTensor::fp16_nd(beta, &[cols])?;
    let ty = AclTensor::fp16_nd(&y, &dims)?;
    let tmean = AclTensor::fp32_nd(&mean, &[rows, 1])?;
    let trstd = AclTensor::fp32_nd(&rstd, &[rows, 1])?;
    let txo = AclTensor::fp16_nd(&x_out, &dims)?;
    two_stage(
        ctx,
        stream,
        "aclnnAddLayerNorm",
        |ws, ex| unsafe {
            ffi::aclnnAddLayerNormGetWorkspaceSize(
                tx.handle(), tz.handle(), tg.handle(), tb.handle(),
                std::ptr::null_mut(), eps, false as i32 as u8 != 0,
                ty.handle(), tmean.handle(), trstd.handle(), txo.handle(), ws, ex,
            )
        },
        |ws, size, ex| unsafe { ffi::aclnnAddLayerNorm(ws, size, ex, stream.handle()) },
    )?;
    Ok(y)
}

/// Replicate one fp16 row [d] into [rows, d] via a dim-0 gather -- the
/// broadcast substitute for ops that reject zero-stride descriptors
/// (aclnnMul et al). Row indices are rebuilt per call; cache at the
/// executor layer when rows is stable.
pub fn row_replicate_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    row: &crate::DeviceBuffer,
    rows: i64,
    d: i64,
) -> Result<crate::DeviceBuffer> {
    let idx: Vec<i32> = vec![0; rows as usize];
    let di = ctx.malloc(rows as usize * 4)?;
    ctx.copy_h2d(&di, unsafe { std::slice::from_raw_parts(idx.as_ptr() as *const u8, rows as usize * 4) })?;
    let out = ctx.malloc((rows * d * 2) as usize)?;
    let tr = AclTensor::fp16_nd(row, &[1, d])?;
    let ti = AclTensor::i32_nd(&di, &[rows])?;
    let to = AclTensor::fp16_nd(&out, &[rows, d])?;
    two_stage(
        ctx,
        stream,
        "aclnnGatherV2(replicate)",
        |ws, ex| unsafe { ffi::aclnnGatherV2GetWorkspaceSize(tr.handle(), 0, ti.handle(), to.handle(), ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnGatherV2(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// Rotate-half RoPE for one [rows, d] tensor, composed from verified ops
/// (gather + mul + add). The fused aclnn rope entries hard-reject pi0.5's
/// head_dims on 310P3 (ApplyRotaryPosEmb: d must be 64/128;
/// RotaryPositionEmbedding: d must be 32/64/96/128 -- pi0.5 uses 256/72),
/// so the composition is the primary path, not a fallback.
///
/// Caller-managed constants (upload once at load):
/// - cos_pos/sin_pos: fp16 [rows, d] rows selected per position;
///   sin_pos must ALREADY carry the rotate-half sign (sin'[r,i] =
///   sin(r,i) * (-1 if i<d/2 else +1)) -- aclnnMul rejects zero-stride
///   broadcasts, so the sign is folded into the table host-side.
/// - rot_idx: i32 [d], rotate-half gather index (i<d/2 -> i+d/2, else i-d/2)
pub fn rope_rotate_half_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    x: &crate::DeviceBuffer,
    rows: i64,
    d: i64,
    cos_pos: &crate::DeviceBuffer,
    sin_signed: &crate::DeviceBuffer,
    rot_idx: &crate::DeviceBuffer,
) -> Result<crate::DeviceBuffer> {
    // q_rot = gather(x, rot_idx) along the channel dim
    let t_x = AclTensor::fp16_nd(x, &[rows, d])?;
    let t_idx = AclTensor::i32_nd(rot_idx, &[d])?;
    let q_rot = ctx.malloc((rows * d * 2) as usize)?;
    let t_qrot = AclTensor::fp16_nd(&q_rot, &[rows, d])?;
    two_stage(
        ctx,
        stream,
        "aclnnGatherV2(rope)",
        |ws, ex| unsafe {
            ffi::aclnnGatherV2GetWorkspaceSize(t_x.handle(), 1, t_idx.handle(), t_qrot.handle(), ws, ex)
        },
        |ws, size, ex| unsafe { ffi::aclnnGatherV2(ws, size, ex, stream.handle()) },
    )?;

    // out = x * cos + q_rot * sin_signed
    let t1 = mul_fp16(ctx, stream, x, cos_pos, &[rows, d])?;
    let t2 = mul_fp16(ctx, stream, &q_rot, sin_signed, &[rows, d])?;
    add_fp16(ctx, stream, &t1, &t2, &[rows, d])
}

/// out = gelu(a), elementwise fp16 via aclnnGeluV2. `tanh_approx` selects
/// the tanh approximation (PaliGemma/gelu_tanh semantics) vs exact erf.
pub fn gelu_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    shape: &[i64],
    tanh_approx: bool,
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc(a.len())?;
    let ta = AclTensor::fp16_nd(a, shape)?;
    let tout = AclTensor::fp16_nd(&out, shape)?;
    let approximate: i64 = if tanh_approx { 1 } else { 0 };
    two_stage(
        ctx,
        stream,
        "aclnnGeluV2",
        |ws, ex| unsafe { ffi::aclnnGeluV2GetWorkspaceSize(ta.handle(), approximate, tout.handle(), ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnGeluV2(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// Embedding lookup: rows of `table` (fp16 [vocab, dim]) selected by
/// device-resident i32 `indices` ([n]) -> out fp16 [n, dim].
pub fn gather_rows_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    table: &crate::DeviceBuffer,
    vocab: i64,
    dim: i64,
    indices: &crate::DeviceBuffer,
    n: i64,
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc((n * dim * 2) as usize)?;
    let tt = AclTensor::fp16_nd(table, &[vocab, dim])?;
    let ti = AclTensor::i32_nd(indices, &[n])?;
    let tout = AclTensor::fp16_nd(&out, &[n, dim])?;
    two_stage(
        ctx,
        stream,
        "aclnnGatherV2",
        |ws, ex| unsafe { ffi::aclnnGatherV2GetWorkspaceSize(tt.handle(), 0, ti.handle(), tout.handle(), ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnGatherV2(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// Full attention in BSH layout: q [tokens, heads*d], k/v [tokens,
/// kv_heads*d] (GQA supported via num_kv_heads), out [tokens, heads*d].
/// Avoids any transpose between the [tokens, hidden] linear world and
/// attention.
pub fn prompt_flash_attention_bsh_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    q: &crate::DeviceBuffer,
    k: &crate::DeviceBuffer,
    v: &crate::DeviceBuffer,
    tokens: i64,
    heads: i64,
    kv_heads: i64,
    head_dim: i64,
    scale: Option<f64>,
) -> Result<crate::DeviceBuffer> {
    let qd = heads * head_dim;
    let kvd = kv_heads * head_dim;
    let out = ctx.malloc((tokens * qd * 2) as usize)?;
    let t1 = AclTensor::fp16_nd(q, &[1, tokens, qd])?;
    let t2 = AclTensor::fp16_nd(k, &[1, tokens, kvd])?;
    let t3 = AclTensor::fp16_nd(v, &[1, tokens, kvd])?;
    let tout = AclTensor::fp16_nd(&out, &[1, tokens, qd])?;

    let mut layout: [u8; 4] = *b"BSH\0";
    let scale_value = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());
    let null = std::ptr::null_mut::<std::ffi::c_void>();

    two_stage(
        ctx,
        stream,
        "aclnnPromptFlashAttentionV3(BSH)",
        |ws, ex| unsafe {
            ffi::aclnnPromptFlashAttentionV3GetWorkspaceSize(
                t1.handle(), t2.handle(), t3.handle(),
                null, null, null, null, null, null, null, null, null,
                heads, scale_value, i64::MAX, 0,
                layout.as_mut_ptr(),
                kv_heads, 0, 0,
                tout.handle(), ws, ex,
            )
        },
        |ws, size, ex| unsafe { ffi::aclnnPromptFlashAttentionV3(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// Batched BSH PFA: q/k/v are [batch * seq, heads*d] contiguous, viewed
/// as [batch, seq, heads*d]. Vision attends within each view's seq
/// window (SigLIP per-image attention), unlike the batch=1 entry above
/// which spans the whole row range.
#[allow(clippy::too_many_arguments)]
pub fn prompt_flash_attention_bsh_batch_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    q: &crate::DeviceBuffer,
    k: &crate::DeviceBuffer,
    v: &crate::DeviceBuffer,
    batch: i64,
    seq: i64,
    heads: i64,
    kv_heads: i64,
    head_dim: i64,
    scale: Option<f64>,
) -> Result<crate::DeviceBuffer> {
    let qd = heads * head_dim;
    let kvd = kv_heads * head_dim;
    let out = ctx.malloc((batch * seq * qd * 2) as usize)?;
    let t1 = AclTensor::fp16_nd(q, &[batch, seq, qd])?;
    let t2 = AclTensor::fp16_nd(k, &[batch, seq, kvd])?;
    let t3 = AclTensor::fp16_nd(v, &[batch, seq, kvd])?;
    let tout = AclTensor::fp16_nd(&out, &[batch, seq, qd])?;

    let mut layout: [u8; 4] = *b"BSH\0";
    let scale_value = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());
    let null = std::ptr::null_mut::<std::ffi::c_void>();

    two_stage(
        ctx,
        stream,
        "aclnnPromptFlashAttentionV3(BSH,batch)",
        |ws, ex| unsafe {
            ffi::aclnnPromptFlashAttentionV3GetWorkspaceSize(
                t1.handle(), t2.handle(), t3.handle(),
                null, null, null, null, null, null, null, null, null,
                heads, scale_value, i64::MAX, 0,
                layout.as_mut_ptr(),
                kv_heads, 0, 0,
                tout.handle(), ws, ex,
            )
        },
        |ws, size, ex| unsafe { ffi::aclnnPromptFlashAttentionV3(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// Cross-length BSH PFA for prefix attention: q covers `q_tokens` rows
/// while k/v cover `kv_tokens` (prefix + chunk). Same contiguous
/// [rows, heads*d] buffers, described with their own sequence lengths.
#[allow(clippy::too_many_arguments)]
pub fn prompt_flash_attention_cross_bsh_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    q: &crate::DeviceBuffer,
    k: &crate::DeviceBuffer,
    v: &crate::DeviceBuffer,
    q_tokens: i64,
    kv_tokens: i64,
    heads: i64,
    kv_heads: i64,
    head_dim: i64,
    scale: Option<f64>,
) -> Result<crate::DeviceBuffer> {
    let qd = heads * head_dim;
    let kvd = kv_heads * head_dim;
    let out = ctx.malloc((q_tokens * qd * 2) as usize)?;
    let t1 = AclTensor::fp16_nd(q, &[1, q_tokens, qd])?;
    let t2 = AclTensor::fp16_nd(k, &[1, kv_tokens, kvd])?;
    let t3 = AclTensor::fp16_nd(v, &[1, kv_tokens, kvd])?;
    let tout = AclTensor::fp16_nd(&out, &[1, q_tokens, qd])?;

    let mut layout: [u8; 4] = *b"BSH\0";
    let scale_value = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());
    let null = std::ptr::null_mut::<std::ffi::c_void>();

    two_stage(
        ctx,
        stream,
        "aclnnPromptFlashAttentionV3(BSH,cross)",
        |ws, ex| unsafe {
            ffi::aclnnPromptFlashAttentionV3GetWorkspaceSize(
                t1.handle(), t2.handle(), t3.handle(),
                null, null, null, null, null, null, null, null, null,
                heads, scale_value, i64::MAX, 0,
                layout.as_mut_ptr(),
                kv_heads, 0, 0,
                tout.handle(), ws, ex,
            )
        },
        |ws, size, ex| unsafe { ffi::aclnnPromptFlashAttentionV3(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// Host-side FRACTAL_NZ reordering: pull the weight back, rearrange into
/// 16x16 blocks on the CPU, upload once. Replaces aclnnNpuFormatCast,
/// whose descriptor expectations (undocumented ori-shape semantics) we
/// could not satisfy from the public aclCreateTensor -- weights load once
/// per process, so a host pass is free. Returns (nz_bytes, dims, strides).
pub fn host_nz_reorder(host: &[u8], rows: i64, cols: i64) -> (Vec<u8>, Vec<i64>, Vec<i64>) {
    let h1 = (rows as usize + 15) / 16;
    let w1 = (cols as usize + 15) / 16;
    let mut out = vec![0u16; h1 * w1 * 256];
    let src: Vec<u16> = host.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    for hb in 0..h1 {
        for wb in 0..w1 {
            for i in 0..16 {
                for j in 0..16 {
                    let r = hb * 16 + i;
                    let c = wb * 16 + j;
                    let v = if r < rows as usize && c < cols as usize { src[r * cols as usize + c] } else { 0 };
                    // block order W-major: [wb][hb][i][j]
                    out[(wb * h1 + hb) * 256 + i * 16 + j] = v;
                }
            }
        }
    }
    let bytes: Vec<u8> = out.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let dims = vec![w1 as i64, h1 as i64, 16, 16];
    let strides = vec![h1 as i64 * 256, 256, 16, 1];
    (bytes, dims, strides)
}

/// Build an FRACTAL_NZ descriptor over an already-NZ buffer (see
/// host_nz_reorder for the layout contract).
pub fn nz_tensor(buf: &crate::DeviceBuffer, dims: &[i64], strides: &[i64]) -> Result<AclTensor> {
    let raw = unsafe {
        ffi::aclCreateTensor(
            dims.as_ptr(), dims.len() as u64, ffi::ACL_FLOAT16,
            strides.as_ptr(), 0,
            ffi::ACL_FORMAT_FRACTAL_NZ,
            dims.as_ptr(), dims.len() as u64,
            buf.as_ptr(),
        )
    };
    if raw.is_null() {
        return Err(AclError { code: -1, op: "aclCreateTensor(NZ)" });
    }
    Ok(AclTensor::from_raw(raw))
}

/// out = a @ b_nz with the weight held in FRACTAL_NZ (host_nz_reorder
/// layout). Strides describe the NZ block order on the b buffer.
pub fn matmul_weight_nz_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    ash: [i64; 2],
    b_nz: &crate::DeviceBuffer,
    b_dims: &[i64],
    b_strides: &[i64],
    out_cols: i64,
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc((ash[0] * out_cols * 2) as usize)?;
    let ta = AclTensor::fp16_nd(a, &ash)?;
    let tb = nz_tensor(b_nz, b_dims, b_strides)?;
    let tout = AclTensor::fp16_nd(&out, &[ash[0], out_cols])?;
    let mut ws_size: u64 = 0;
    let mut executor: *mut std::ffi::c_void = std::ptr::null_mut();
    let code = unsafe {
        ffi::aclnnMatmulWeightNzGetWorkspaceSize(ta.handle(), tb.handle(), tout.handle(), 1, &mut ws_size, &mut executor)
    };
    if code != 0 {
        return Err(AclError { code, op: "aclnnMatmulWeightNzGetWorkspaceSize" });
    }
    let ws = if ws_size > 0 { Some(ctx.malloc(ws_size as usize)?) } else { None };
    let code = unsafe {
        ffi::aclnnMatmulWeightNz(
            ws.as_ref().map(|w| w.as_ptr()).unwrap_or(std::ptr::null_mut()),
            ws_size,
            executor,
            stream.handle(),
        )
    };
    if code != 0 {
        return Err(AclError { code, op: "aclnnMatmulWeightNz" });
    }
    Ok(out)
}

/// EXPERIMENTAL: matmul where `b` physically holds the [n, k] (transposed)
/// weight row-major and is described as a [k, n] view with transpose
/// strides [1, k] -- the layout torch's `a @ w.t()` feeds aclnnMatmul.
/// Hypothesis: the plain [k, n] row-major descriptor trips the
/// MatMulV2_NZ_ND kernel's tiling on 310P3 for large shapes.
pub fn matmul_b_t_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    ash: [i64; 2],
    b_t: &crate::DeviceBuffer, // physical [n, k] row-major
    k: i64,
    n: i64,
) -> Result<crate::DeviceBuffer> {
    if ash[0] % MATMUL_M_ALIGN != 0 {
        return matmul_m_pad(ctx, stream, a, ash[0], ash[1], n, |a2, m2| {
            matmul_b_t_fp16(ctx, stream, a2, [m2, ash[1]], b_t, k, n)
        });
    }
    let out = ctx.malloc((ash[0] * n * 2) as usize)?;
    let ta = AclTensor::fp16_nd(a, &ash)?;
    // view [k, n] over physical [n, k]: element (i,j) at j*k + i
    let dims = [k, n];
    let stride = [1, k];
    let raw = unsafe {
        ffi::aclCreateTensor(
            dims.as_ptr(), 2, ffi::ACL_FLOAT16,
            stride.as_ptr(), 0, ffi::ACL_FORMAT_ND,
            std::ptr::null(), 0, // storage dims unknown to the view
            b_t.as_ptr(),
        )
    };
    if raw.is_null() {
        return Err(AclError { code: -1, op: "aclCreateTensor(transposed b)" });
    }
    let tb = AclTensor::from_raw(raw);
    let tout = AclTensor::fp16_nd(&out, &[ash[0], n])?;
    let mut ws_size: u64 = 0;
    let mut executor: *mut std::ffi::c_void = std::ptr::null_mut();
    let code = unsafe {
        ffi::aclnnMatmulGetWorkspaceSize(ta.handle(), tb.handle(), tout.handle(), 1, &mut ws_size, &mut executor)
    };
    if code != 0 {
        return Err(AclError { code, op: "aclnnMatmul(t-b) plan" });
    }
    let ws = if ws_size > 0 { Some(ctx.malloc(ws_size as usize)?) } else { None };
    let code = unsafe {
        ffi::aclnnMatmul(
            ws.as_ref().map(|w| w.as_ptr()).unwrap_or(std::ptr::null_mut()),
            ws_size,
            executor,
            stream.handle(),
        )
    };
    if code != 0 {
        return Err(AclError { code, op: "aclnnMatmul(t-b) run" });
    }
    Ok(out)
}

/// aten::mm mirror via aclnnMm: plain [k, n] row-major b. torch_npu's mm
/// computes the wide-N shapes that crash aclnnMatmul on 310P3 -- test
/// whether this entry avoids the cliff. cubeMathType 2 = KEEP_DTYPE.
pub fn mm_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    ash: [i64; 2],
    b: &crate::DeviceBuffer,
    bsh: [i64; 2],
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc((ash[0] * bsh[1] * 2) as usize)?;
    let ta = AclTensor::fp16_nd(a, &ash)?;
    let tb = AclTensor::fp16_nd(b, &bsh)?;
    let tout = AclTensor::fp16_nd(&out, &[ash[0], bsh[1]])?;
    two_stage(
        ctx,
        stream,
        "aclnnMm",
        |ws, ex| unsafe { ffi::aclnnMmGetWorkspaceSize(ta.handle(), tb.handle(), tout.handle(), 2, ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnMm(ws, size, ex, stream.handle()) },)?;
    Ok(out)
}

/// aclnnMm over the transposed-b view ([k, n] shape, stride [1, k] over
/// physical [n, k]).
pub fn mm_b_t_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    ash: [i64; 2],
    b_t: &crate::DeviceBuffer, // physical [n, k] row-major
    k: i64,
    n: i64,
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc((ash[0] * n * 2) as usize)?;
    let ta = AclTensor::fp16_nd(a, &ash)?;
    let dims = [k, n];
    let stride = [1, k];
    let raw = unsafe {
        ffi::aclCreateTensor(
            dims.as_ptr(), 2, ffi::ACL_FLOAT16,
            stride.as_ptr(), 0, ffi::ACL_FORMAT_ND,
            std::ptr::null(), 0,
            b_t.as_ptr(),
        )
    };
    if raw.is_null() {
        return Err(AclError { code: -1, op: "aclCreateTensor(mm t-b)" });
    }
    let tb = AclTensor::from_raw(raw);
    let tout = AclTensor::fp16_nd(&out, &[ash[0], n])?;
    two_stage(
        ctx,
        stream,
        "aclnnMm(t-b)",
        |ws, ex| unsafe { ffi::aclnnMmGetWorkspaceSize(ta.handle(), tb.handle(), tout.handle(), 2, ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnMm(ws, size, ex, stream.handle()) },)?;
    Ok(out)
}

/// BLAS gemm with native transB: A [m, k] (transA=0) x B physical [n, k]
/// row-major with transB=1. out = 1 * A * B^T + 0 * C.
pub fn gemm_b_t_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    ash: [i64; 2],
    b_t: &crate::DeviceBuffer, // physical [n, k] row-major
    k: i64,
    n: i64,
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc((ash[0] * n * 2) as usize)?;
    let ta = AclTensor::fp16_nd(a, &ash)?;
    let tb = AclTensor::fp16_nd(b_t, &[n, k])?;
    let tc = AclTensor::fp16_nd(&out, &[ash[0], n])?;
    let tout = AclTensor::fp16_nd(&out, &[ash[0], n])?;
    two_stage(
        ctx,
        stream,
        "aclnnGemm(tB)",
        |ws, ex| unsafe {
            ffi::aclnnGemmGetWorkspaceSize(ta.handle(), tb.handle(), tc.handle(), 1.0, 0.0, 0, 1, tout.handle(), 2, ws, ex)
        },
        |ws, size, ex| unsafe { ffi::aclnnGemm(ws, size, ex, stream.handle()) },)?;
    Ok(out)
}

/// Host transpose [rows, cols] -> [cols, rows] (fp16 bytes).
pub fn host_transpose(host: &[u8], rows: i64, cols: i64) -> Vec<u8> {
    let src: Vec<u16> = host.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut out = vec![0u16; src.len()];
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            out[c * rows as usize + r] = src[r * cols as usize + c];
        }
    }
    out.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

pub fn add_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    b: &crate::DeviceBuffer,
    shape: &[i64],
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc(a.len())?;
    let ta = AclTensor::fp16_nd(a, shape)?;
    let tb = AclTensor::fp16_nd(b, shape)?;
    let tout = AclTensor::fp16_nd(&out, shape)?;
    let mut alpha: f32 = 1.0;
    let scalar = ScalarFp32::new(&mut alpha);
    two_stage(
        ctx,
        stream,
        "aclnnAdd",
        |ws, ex| unsafe {
            ffi::aclnnAddGetWorkspaceSize(ta.handle(), tb.handle(), scalar.handle(), tout.handle(), ws, ex)
        },
        |ws, size, ex| unsafe { ffi::aclnnAdd(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// out = a * b (fp16, same shape).
pub fn mul_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    b: &crate::DeviceBuffer,
    shape: &[i64],
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc(a.len())?;
    let ta = AclTensor::fp16_nd(a, shape)?;
    let tb = AclTensor::fp16_nd(b, shape)?;
    let tout = AclTensor::fp16_nd(&out, shape)?;
    two_stage(
        ctx,
        stream,
        "aclnnMul",
        |ws, ex| unsafe { ffi::aclnnMulGetWorkspaceSize(ta.handle(), tb.handle(), tout.handle(), ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnMul(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// out = a * scalar (fp16 elementwise; scalar passed as fp32).
pub fn muls_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    scalar: f32,
    shape: &[i64],
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc(a.len())?;
    let ta = AclTensor::fp16_nd(a, shape)?;
    let tout = AclTensor::fp16_nd(&out, shape)?;
    let mut s = scalar;
    let sc = ScalarFp32::new(&mut s);
    two_stage(
        ctx,
        stream,
        "aclnnMuls",
        |ws, ex| unsafe { ffi::aclnnMulsGetWorkspaceSize(ta.handle(), sc.handle(), tout.handle(), ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnMuls(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// out = silu(a) = a * sigmoid(a) (fp16 elementwise).
pub fn silu_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    a: &crate::DeviceBuffer,
    shape: &[i64],
) -> Result<crate::DeviceBuffer> {
    let out = ctx.malloc(a.len())?;
    let ta = AclTensor::fp16_nd(a, shape)?;
    let tout = AclTensor::fp16_nd(&out, shape)?;
    two_stage(
        ctx,
        stream,
        "aclnnSilu",
        |ws, ex| unsafe { ffi::aclnnSiluGetWorkspaceSize(ta.handle(), tout.handle(), ws, ex) },
        |ws, size, ex| unsafe { ffi::aclnnSilu(ws, size, ex, stream.handle()) },
    )?;
    Ok(out)
}

/// Fused y = rmsnorm(x1 + x2) * gamma over the last dim.
/// Returns (y, rstd) -- buffers for the op's extra outputs are allocated
/// here so callers don't care about them (rstd: fp32 [rows]).
pub fn add_rms_norm_fp16(
    ctx: &AscendContext,
    stream: &AscendStream,
    x1: &crate::DeviceBuffer,
    x2: &crate::DeviceBuffer,
    gamma: &crate::DeviceBuffer,
    shape: &[i64], // [rows, cols]
    epsilon: f64,
) -> Result<(crate::DeviceBuffer, crate::DeviceBuffer)> {
    let elems: i64 = shape.iter().product();
    let y = ctx.malloc((elems * 2) as usize)?;
    // rstd fp32 [rows]; xOut same as y -- op requires all three outputs.
    let rows = shape[0];
    let rstd = ctx.malloc((rows * 4) as usize)?;
    let x_out = ctx.malloc((elems * 2) as usize)?;

    let t1 = AclTensor::fp16_nd(x1, shape)?;
    let t2 = AclTensor::fp16_nd(x2, shape)?;
    let tg = AclTensor::fp16_nd(gamma, &[shape[1]])?;
    let ty = AclTensor::fp16_nd(&y, shape)?;
    // rstd must be 2-D [rows, 1] -- a 1-D [rows] descriptor is rejected by
    // the planner with a misleading ACLNN_ERR_INNER_NULLPTR (561103).
    let trstd = AclTensor::fp32_nd(&rstd, &[rows, 1])?;
    let tx = AclTensor::fp16_nd(&x_out, shape)?;
    two_stage(
        ctx,
        stream,
        "aclnnAddRmsNorm",
        |ws, ex| unsafe {
            ffi::aclnnAddRmsNormGetWorkspaceSize(
                t1.handle(),
                t2.handle(),
                tg.handle(),
                epsilon,
                ty.handle(),
                trstd.handle(),
                tx.handle(),
                ws,
                ex,
            )
        },
        |ws, size, ex| unsafe { ffi::aclnnAddRmsNorm(ws, size, ex, stream.handle()) },
    )?;
    Ok((y, rstd))
}

/// RAII aclScalar holding one fp16-compatible scalar. aclnnAdd's alpha is
/// documented fp32-capable; we pass fp32 by pointer.
struct ScalarFp32<'a> {
    raw: *mut std::ffi::c_void,
    _marker: std::marker::PhantomData<&'a mut f32>,
}

impl<'a> ScalarFp32<'a> {
    fn new(v: &'a mut f32) -> Self {
        let raw = unsafe { ffi::aclCreateScalar(v as *mut f32 as *mut std::ffi::c_void, ffi::ACL_FLOAT) };
        Self { raw, _marker: std::marker::PhantomData }
    }
    fn handle(&self) -> *mut std::ffi::c_void {
        self.raw
    }
}

impl Drop for ScalarFp32<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::aclDestroyScalar(self.raw) };
        }
    }
}
