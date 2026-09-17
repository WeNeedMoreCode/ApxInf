//! aclnn compute ops, starting with matmul (roadmap divergence point #1).

use crate::context::AscendContext;
use crate::ffi;
use crate::stream::AscendStream;
use crate::tensor::AclTensor;
use crate::{AclError, Result};

/// cubeMathType = KEEP_DTYPE: compute in the tensors' own dtype (fp16 in,
/// fp16 out on 310P3, which has no bf16 unit).
const CUBE_MATH_KEEP_DTYPE: i8 = 1;

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
