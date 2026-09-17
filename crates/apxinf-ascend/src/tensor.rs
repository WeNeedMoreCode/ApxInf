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
    /// ND INT32 descriptor (gather indices on device).
    pub fn i32_nd(buf: &crate::DeviceBuffer, shape: &[i64]) -> Result<Self> {
        let elems: i64 = shape.iter().product();
        assert_eq!(buf.len(), (elems * 4) as usize, "size mismatch for i32 shape {shape:?}");
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
                ffi::ACL_INT32,
                stride.as_ptr(),
                0,
                ffi::ACL_FORMAT_ND,
                shape.as_ptr(),
                shape.len() as u64,
                buf.as_ptr(),
            )
        };
        if raw.is_null() {
            return Err(AclError { code: -1, op: "aclCreateTensor(i32)" });
        }
        Ok(Self { raw })
    }

    // NOTE: no offset-view constructor. aclCreateTensor's `offset`
    // semantics proved unreliable on device (neither element- nor
    // byte-units produced the expected slice through a gather), so row
    // splits use ops::take_rows_fp16 -- a D2D memcpy with explicit,
    // verified semantics. Do not reintroduce views without a device
    // probe proving the offset unit.

    /// fp16 [rows, cols] descriptor whose row stride is 0 -- broadcasts a
    /// [cols] bias across rows in aclnnAdd without materializing.
    pub fn fp16_row_broadcast(buf: &crate::DeviceBuffer, rows: i64, cols: i64) -> Result<Self> {
        assert_eq!(buf.len(), (cols * 2) as usize, "bias buffer must hold one fp16 row");
        let dims = [rows, cols];
        let stride = [0i64, 1];
        let storage = [cols]; // backing buffer physically holds one row
        let raw = unsafe {
            ffi::aclCreateTensor(
                dims.as_ptr(),
                dims.len() as u64,
                ffi::ACL_FLOAT16,
                stride.as_ptr(),
                0,
                ffi::ACL_FORMAT_ND,
                storage.as_ptr(),
                storage.len() as u64,
                buf.as_ptr(),
            )
        };
        if raw.is_null() {
            return Err(AclError { code: -1, op: "aclCreateTensor(row_broadcast)" });
        }
        Ok(Self { raw })
    }

    /// Build an ND fp32 descriptor (elementwise output stats, rstd etc.).
    pub fn fp32_nd(buf: &crate::DeviceBuffer, shape: &[i64]) -> Result<Self> {
        let elems: i64 = shape.iter().product();
        assert_eq!(buf.len(), (elems * 4) as usize, "size mismatch for fp32 shape {shape:?}");
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
                ffi::ACL_FLOAT,
                stride.as_ptr(),
                0,
                ffi::ACL_FORMAT_ND,
                shape.as_ptr(),
                shape.len() as u64,
                buf.as_ptr(),
            )
        };
        if raw.is_null() {
            return Err(AclError { code: -1, op: "aclCreateTensor(fp32)" });
        }
        Ok(Self { raw })
    }

    /// Build an ND fp16 descriptor for `buf` with the given shape.
    /// `buf.len` must equal `shape.iter().product() * 2`.
    /// Wrap a raw aclTensor handle (crate-internal constructor for
    /// custom-descriptor builders like the NZ path).
    pub(crate) fn from_raw(raw: *mut std::ffi::c_void) -> Self {
        Self { raw }
    }

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
