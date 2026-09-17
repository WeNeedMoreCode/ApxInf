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
