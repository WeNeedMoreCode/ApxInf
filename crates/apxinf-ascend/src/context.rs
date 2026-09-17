//! RAII ownership over aclrt device/context/memory.

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::ffi;
use crate::{AclError, Result};

/// `aclInit` is process-global and must run exactly once; a second call
/// returns an error even though the runtime is up. `get_or_init` gives us an
/// idempotent, race-free "first caller's code sticks" semantic.
static ACL_INIT: OnceLock<i32> = OnceLock::new();

fn ensure_init() -> Result<()> {
    let &code = ACL_INIT.get_or_init(|| unsafe { ffi::aclInit(std::ptr::null()) });
    if code == 0 {
        Ok(())
    } else {
        Err(AclError { code, op: "aclInit" })
    }
}

/// An Ascend device context: SetDevice + explicit context, torn down in Drop.
///
/// All `DeviceBuffer`s must be dropped before their creating context (ACL
/// frees device memory through the device's context). Buffers here do not
/// hold a reference — scope them tighter than the context.
pub struct AscendContext {
    device_id: i32,
    raw: *mut c_void,
}

// The raw context handle is opaque; ACL APIs are thread-safe for concurrent
// use of one context (synchronize on the calling thread).
unsafe impl Send for AscendContext {}
unsafe impl Sync for AscendContext {}

impl AscendContext {
    pub fn new(device_id: usize) -> Result<Self> {
        ensure_init()?;
        let id = i32::try_from(device_id)
            .map_err(|_| AclError { code: -1, op: "device_id overflow" })?;
        let code = unsafe { ffi::aclrtSetDevice(id) };
        if code != 0 {
            return Err(AclError { code, op: "aclrtSetDevice" });
        }
        let mut raw: *mut c_void = std::ptr::null_mut();
        let code = unsafe { ffi::aclrtCreateContext(&mut raw, id) };
        if code != 0 {
            unsafe { ffi::aclrtResetDevice(id) };
            return Err(AclError { code, op: "aclrtCreateContext" });
        }
        Ok(Self { device_id: id, raw })
    }

    pub fn device_id(&self) -> usize {
        self.device_id as usize
    }

    fn check(&self, code: i32, op: &'static str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(AclError { code, op })
        }
    }

    /// Make this context current on the calling thread.
    pub fn set_current(&self) -> Result<()> {
        self.check(unsafe { ffi::aclrtSetCurrentContext(self.raw) }, "aclrtSetCurrentContext")
    }

    /// Allocate `len` bytes of device memory.
    pub fn malloc(&self, len: usize) -> Result<DeviceBuffer> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let code =
            unsafe { ffi::aclrtMalloc(&mut ptr, len, ffi::ACL_MEM_MALLOC_HUGE_FIRST) };
        self.check(code, "aclrtMalloc")?;
        Ok(DeviceBuffer { ptr, len })
    }

    /// Host → device copy.
    pub fn copy_h2d(&self, buf: &DeviceBuffer, host: &[u8]) -> Result<()> {
        assert_eq!(host.len(), buf.len, "h2d length mismatch");
        self.check(
            unsafe {
                ffi::aclrtMemcpy(
                    buf.ptr,
                    buf.len,
                    host.as_ptr() as *const c_void,
                    host.len(),
                    ffi::ACL_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "aclrtMemcpy h2d",
        )
    }

    /// Device → host copy.
    pub fn copy_d2h(&self, buf: &DeviceBuffer, host: &mut [u8]) -> Result<()> {
        assert_eq!(host.len(), buf.len, "d2h length mismatch");
        self.check(
            unsafe {
                ffi::aclrtMemcpy(
                    host.as_mut_ptr() as *mut c_void,
                    host.len(),
                    buf.ptr,
                    buf.len,
                    ffi::ACL_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "aclrtMemcpy d2h",
        )
    }

    /// Stream-ordered memset (capturable: used as ACLGraph payload).
    pub fn memset_async(&self, buf: &DeviceBuffer, value: u8, stream: &crate::AscendStream) -> Result<()> {
        self.check(
            unsafe {
                ffi::aclrtMemsetAsync(
                    buf.ptr,
                    buf.len,
                    value as i32,
                    buf.len,
                    stream.handle(),
                )
            },
            "aclrtMemsetAsync",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        self.check(unsafe { ffi::aclrtSynchronizeDevice() }, "aclrtSynchronizeDevice")
    }
}

impl Drop for AscendContext {
    fn drop(&mut self) {
        unsafe {
            ffi::aclrtDestroyContext(self.raw);
            ffi::aclrtResetDevice(self.device_id);
        }
    }
}

/// Device memory owned by an [`AscendContext`], freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for DeviceBuffer {}
// Shared read-only access to a raw pointer is sound: ops take &self and ACL
// memory APIs accept const device pointers for reads.
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::aclrtFree(self.ptr) };
        }
    }
}
