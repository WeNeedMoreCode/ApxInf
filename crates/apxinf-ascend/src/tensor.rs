//! RAII wrapper over aclnn's opaque `aclTensor` descriptors.
//!
//! An `AclTensor` is a view (shape/stride/format/dtype) bound to a device
//! pointer -- it borrows the memory of a [`DeviceBuffer`](crate::DeviceBuffer)
//! and must not outlive it. Destruction frees only the descriptor.

use std::ffi::c_void;

use crate::ffi;
use crate::{AclError, Result};

/// Continuous ND (row-major) tensor descriptor over device memory.
pub struct AclTensor {
    raw: *mut c_void,
}

impl AclTensor {
    /// Build an ND fp16 descriptor for `buf` with the given shape.
    /// `buf.len` must equal `shape.iter().product() * 2`.
    pub fn fp16_nd(buf: &crate::DeviceBuffer, shape: &[i64]) -> Result<Self> {
        let elems: i64 = shape.iter().product();
        let expect = (elems * 2) as usize;
        assert_eq!(buf.len(), expect, "buffer/device size mismatch for fp16 shape {shape:?}");
        // row-major contiguous strides
        let mut stride = vec![0i64; shape.len()];
        let mut acc = 1i64;
        for i in (0..shape.len()).rev() {
            stride[i] = acc;
            acc *= shape[i].max(1);
        }
        let raw = unsafe {
            ffi::aclCreateTensor(
                shape.as_ptr(),
                shape.len() as u64,
                ffi::ACL_FLOAT16,
                stride.as_ptr(),
                0,
                ffi::ACL_FORMAT_ND,
                shape.as_ptr(), // storage shape == view shape for ND contiguous
                shape.len() as u64,
                buf.as_ptr(),
            )
        };
        if raw.is_null() {
            return Err(AclError { code: -1, op: "aclCreateTensor" });
        }
        Ok(Self { raw })
    }

    pub fn handle(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for AclTensor {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::aclDestroyTensor(self.raw) };
        }
    }
}
