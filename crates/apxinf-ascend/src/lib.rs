//! Ascend (Huawei aclrt) backend for ApxInf — stage-2 scaffolding.
//!
//! Current scope: hand-written `aclrt` FFI ([ffi]) and RAII ownership types
//! ([context::AscendContext], [context::DeviceBuffer]) proving the Rust ↔
//! libascendcl link and device-memory round-trips on 310P3. The
//! `apxinf_core::Backend` implementation (aclnnMatmul, rms_norm, rope, …)
//! lands kernel by kernel on the stage-2 roadmap; until then
//! `Device::Ascend` arm in `create_backend` returns a "not yet implemented"
//! error by design, keeping the seam honest.
//!
//! FFI declarations are transcribed by hand from the CANN 8.5.1 headers
//! (`acl_base.h`, `acl_rt.h`) — no bindgen/clang dependency. Keep them in
//! sync manually when the toolkit moves.

pub mod context;
pub mod ffi;

pub use context::{AscendContext, DeviceBuffer};

/// ACL error code (aclError, i32). 0 = ACL_SUCCESS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("aclError {code} at {op}")]
pub struct AclError {
    pub code: i32,
    pub op: &'static str,
}

pub type Result<T> = std::result::Result<T, AclError>;
