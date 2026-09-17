//! Raw Ascend CL (aclrt) FFI — transcribed from CANN 8.5.1 headers.
//!
//! `aclError` is `int32_t`; enums cross the boundary as `u32` with the
//! constants below named after their C counterparts.

use std::ffi::c_void;

extern "C" {
    // acl_base.h
    pub fn aclInit(configPath: *const i8) -> i32;
    pub fn aclFinalize() -> i32;
    pub fn aclrtGetVersion(major: *mut u32, minor: *mut u32, patch: *mut u32) -> i32;

    // acl_rt.h — device & context
    pub fn aclrtSetDevice(deviceId: i32) -> i32;
    pub fn aclrtResetDevice(deviceId: i32) -> i32;
    pub fn aclrtGetDevice(deviceId: *mut i32) -> i32;
    pub fn aclrtCreateContext(context: *mut *mut c_void, deviceId: i32) -> i32;
    pub fn aclrtDestroyContext(context: *mut c_void) -> i32;
    pub fn aclrtSetCurrentContext(context: *mut c_void) -> i32;
    pub fn aclrtGetCurrentContext(context: *mut *mut c_void) -> i32;

    // acl_rt.h — stream
    pub fn aclrtCreateStream(stream: *mut *mut c_void) -> i32;
    pub fn aclrtDestroyStream(stream: *mut c_void) -> i32;
    pub fn aclrtSynchronizeStream(stream: *mut c_void) -> i32;
    pub fn aclrtSynchronizeDevice() -> i32;

    // acl_rt.h — memory
    pub fn aclrtMalloc(devPtr: *mut *mut c_void, size: usize, policy: u32) -> i32;
    pub fn aclrtFree(devPtr: *mut c_void) -> i32;
    pub fn aclrtMemcpy(
        dst: *mut c_void,
        destMax: usize,
        src: *const c_void,
        count: usize,
        kind: u32,
    ) -> i32;
}

/// aclrtMemcpyKind.
pub const ACL_MEMCPY_HOST_TO_HOST: u32 = 0;
pub const ACL_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub const ACL_MEMCPY_DEVICE_TO_HOST: u32 = 2;
pub const ACL_MEMCPY_DEVICE_TO_DEVICE: u32 = 3;

/// aclrtMemMallocPolicy.
pub const ACL_MEM_MALLOC_HUGE_FIRST: u32 = 0;
