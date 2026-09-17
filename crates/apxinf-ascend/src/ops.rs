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

/// out = a + b (fp16, same shape).
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
