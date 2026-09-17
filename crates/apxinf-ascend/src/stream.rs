//! Minimal aclrt stream wrapper.

use std::ffi::c_void;

use crate::ffi;
use crate::{AclError, Result};

/// An aclrt stream, destroyed on drop.
pub struct AscendStream {
    raw: *mut c_void,
}

unsafe impl Send for AscendStream {}

impl AscendStream {
    pub fn new() -> Result<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let code = unsafe { ffi::aclrtCreateStream(&mut raw) };
        if code != 0 {
            return Err(AclError { code, op: "aclrtCreateStream" });
        }
        Ok(Self { raw })
    }

    pub fn handle(&self) -> *mut c_void {
        self.raw
    }

    pub fn synchronize(&self) -> Result<()> {
        let code = unsafe { ffi::aclrtSynchronizeStream(self.raw) };
        if code != 0 {
            return Err(AclError { code, op: "aclrtSynchronizeStream" });
        }
        Ok(())
    }
}

impl Default for AscendStream {
    fn default() -> Self {
        Self::new().expect("AscendStream::new")
    }
}

impl Drop for AscendStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::aclrtDestroyStream(self.raw) };
        }
    }
}
