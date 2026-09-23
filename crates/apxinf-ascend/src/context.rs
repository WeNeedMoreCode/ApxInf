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
    arena: std::sync::Mutex<Option<Arena>>,
    /// Reuse pool for short-lived scratch buffers that feed async ops.
    /// aclrtFree does not wait for the stream, so dropping a buffer whose
    /// async consumers are still queued is a use-after-free (sdma copy
    /// error 0x217 under deep queues, 2026-09-18 bench). Pooling by size
    /// keeps every buffer alive forever; single-stream execution makes
    /// same-size reuse sequentially ordered (a kernel's reads finish
    /// before the next same-size writer's writes begin).
    scratch: std::sync::Mutex<std::collections::HashMap<usize, Vec<std::sync::Arc<DeviceBuffer>>>>,
}

impl AscendContext {
    /// Get a pooled scratch buffer of exactly `len` bytes (zeroed by the
    /// caller before use as an accumulator input; contents are undefined).
    /// Same-size requests share ONE buffer for the process lifetime --
    /// single-stream ordering makes that sequentially safe, and inside a
    /// capture arena it means one slice per size (not one per op; the
    /// full-depth graph would otherwise bump ~8GB of duplicated
    /// workspaces into the arena).
    pub fn scratch_buf(&self, len: usize) -> Result<std::sync::Arc<DeviceBuffer>> {
        let mut pool = self.scratch.lock().unwrap();
        if let Some(buf) = pool.get(&len).and_then(|v| v.last()) {
            return Ok(buf.clone());
        }
        let buf = std::sync::Arc::new(self.malloc(len)?);
        pool.entry(len).or_default().push(buf.clone());
        Ok(buf)
    }
}

/// Active capture-allocation arena: `malloc` bumps from `base` until
/// `off` reaches `cap`. The backing buffer is owned by the caller that
/// entered the arena (stored with the captured graph).
struct Arena {
    base: *mut c_void,
    cap: usize,
    off: usize,
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
        Ok(Self { device_id: id, raw, scratch: Default::default(), arena: None.into() })
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

    /// Allocate `len` bytes of device memory. While an arena is active
    /// (see [`Self::enter_arena`]) this bumps from the arena instead --
    /// addresses stay stable for the graph's lifetime.
    pub fn malloc(&self, len: usize) -> Result<DeviceBuffer> {
        if self.arena.lock().unwrap().is_some() {
            return self.arena_alloc(len).ok_or_else(|| {
                AclError { code: -1, op: "capture arena exhausted (grow enter_arena size)" }
            });
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let code =
            unsafe { ffi::aclrtMalloc(&mut ptr, len, ffi::ACL_MEM_MALLOC_HUGE_FIRST) };
        self.check(code, "aclrtMalloc")?;
        Ok(DeviceBuffer { ptr, len, owned: true })
    }

    /// Enter capture-allocation mode: subsequent `malloc`/`scratch_buf`
    /// bump from one pre-allocated arena whose addresses the captured
    /// graph bakes in. No allocation happens on the driver inside the
    /// capture window. Returns nothing; pair with `exit_arena` (after
    /// the graph is built) which reports bytes used.
    /// Returns the arena's backing buffer -- the CALLER must keep it (and
    /// store it with the captured graph): every bump is a slice of it, so
    /// dropping the owner frees what the graph bakes in.
    pub fn enter_arena(&self, bytes: usize) -> Result<std::sync::Arc<DeviceBuffer>> {
        let owner = std::sync::Arc::new(self.malloc_outside_arena(bytes)?);
        let mut guard = self.arena.lock().unwrap();
        *guard = Some(Arena {
            base: owner.as_ptr(),
            cap: bytes,
            off: 0,
        });
        Ok(owner)
    }

    /// Leave capture-allocation mode; returns bytes consumed.
    pub fn exit_arena(&self) -> usize {
        let guard = self.arena.lock().unwrap();
        match guard.as_ref() {
            Some(a) => a.off,
            None => 0,
        }
        // the arena itself stays alive via the Arc held by captured
        // resources; the flag must be cleared by the caller pairing --
        // see `clear_arena`.
    }

    /// Drop the arena flag (the backing buffer dies when its last Arc
    /// does -- capture outputs and the graph hold slices' parents).
    pub fn clear_arena(&self) {
        self.arena.lock().unwrap().take();
    }

    fn malloc_outside_arena(&self, len: usize) -> Result<DeviceBuffer> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let code =
            unsafe { ffi::aclrtMalloc(&mut ptr, len, ffi::ACL_MEM_MALLOC_HUGE_FIRST) };
        self.check(code, "aclrtMalloc")?;
        Ok(DeviceBuffer { ptr, len, owned: true })
    }

    fn arena_alloc(&self, len: usize) -> Option<DeviceBuffer> {
        const ALIGN: usize = 512;
        let mut guard = self.arena.lock().unwrap();
        let arena = guard.as_mut()?;
        let start = (arena.off + ALIGN - 1) / ALIGN * ALIGN;
        let end = start.checked_add(len)?;
        if end > arena.cap {
            return None; // exhausted: caller falls back or errors upstream
        }
        arena.off = end;
        Some(DeviceBuffer {
            ptr: unsafe { arena.base.add(start) },
            len,
            owned: false,
        })
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

    /// Stream-ordered device → device copy (async, host returns immediately;
    /// ordered against compute on the same stream). The host-synchronous
    /// counterpart of copy_h2d/copy_d2h for inter-segment relays that must
    /// never touch host memory.
    pub fn copy_d2d_async(
        &self,
        dst: &DeviceBuffer,
        src: &DeviceBuffer,
        stream: &crate::AscendStream,
    ) -> Result<()> {
        assert_eq!(dst.len(), src.len(), "d2d length mismatch");
        self.check(
            unsafe {
                ffi::aclrtMemcpyAsync(
                    dst.as_ptr(),
                    dst.len(),
                    src.as_ptr(),
                    src.len(),
                    ffi::ACL_MEMCPY_DEVICE_TO_DEVICE,
                    stream.handle(),
                )
            },
            "aclrtMemcpyAsync d2d",
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
    // arena bumps are slices of a caller-owned backing buffer: their
    // "free" is a no-op (the owner frees the whole arena)
    owned: bool,
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

    /// Unowned view into `backing`'s memory at a byte offset (no free on
    /// drop — the backing buffer owns the allocation and must outlive the
    /// view). For device-resident constant tables shared across runs.
    pub fn view_of(backing: &DeviceBuffer, offset_bytes: usize, len_bytes: usize) -> DeviceBuffer {
        assert!(offset_bytes + len_bytes <= backing.len, "view_of 越界");
        DeviceBuffer {
            ptr: unsafe { backing.ptr.add(offset_bytes) },
            len: len_bytes,
            owned: false,
        }
    }
}

/// Buffers whose async consumers may still sit in the stream queue.
/// aclrtFree is NOT stream-ordered: freeing immediately lets a later
/// malloc reuse the address while a kernel/memcpy still touches it
/// (sdma copy error 0x217 under deep queues, bench 2026-09-18). Drops
/// park here; the next successful stream synchronize flushes for real.
/// Raw device pointer + length, parked until the stream drains. The
/// wrapper makes the static's Send requirement honest (ACL device
/// pointers are plain addresses; freeing on any thread after the
/// stream drained is safe).
struct PendingPtr(*mut std::ffi::c_void, usize);
unsafe impl Send for PendingPtr {}

static PENDING_FREES: std::sync::Mutex<Vec<PendingPtr>> = std::sync::Mutex::new(Vec::new());

/// Really free everything parked by DeviceBuffer drops. Only safe after
/// the stream has drained (callers: stream/context synchronize).
pub fn flush_pending_frees() {
    let mut pending = PENDING_FREES.lock().unwrap();
    for entry in pending.drain(..) {
        unsafe { ffi::aclrtFree(entry.0) };
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            PENDING_FREES.lock().unwrap().push(PendingPtr(self.ptr, self.len));
        }
    }
}
