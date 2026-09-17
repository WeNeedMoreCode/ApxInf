//! ACLGraph capture/replay — the Ascend counterpart of apxinf-cuda's graph.rs.
//!
//! The aclmdlRI (model Runtime Instance) flow maps 1:1 onto CUDA Graphs
//! minus instantiation: capture on a non-default stream, end produces an
//! executable RI handle, replay is one async launch. Verified end-to-end on
//! 310P3 / CANN 9.0.1 (see dev_logs/aclgraph_probe/): all three capture
//! modes work; `memsetAsync` is a capturable payload. CANN 8.5.1 rejects
//! `aclmdlRICaptureBegin` with 207000, so this module is only usable in the
//! 9.0.1+ container.

use std::ffi::c_void;

use crate::ffi;
use crate::stream::AscendStream;
use crate::{AclError, Result};

/// Capture scoping, mirroring apxinf-cuda's `CaptureMode`.
#[derive(Clone, Copy, Debug)]
pub enum CaptureMode {
    Global,
    ThreadLocal,
    Relaxed,
}

impl CaptureMode {
    fn as_acl(self) -> i32 {
        match self {
            CaptureMode::Global => ffi::ACL_MODEL_RI_CAPTURE_MODE_GLOBAL,
            CaptureMode::ThreadLocal => ffi::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL,
            CaptureMode::Relaxed => ffi::ACL_MODEL_RI_CAPTURE_MODE_RELAXED,
        }
    }
}

/// A captured graph; replays on the stream it was captured on.
pub struct AscendGraph {
    ri: *mut c_void,
    stream: *mut c_void,
}

impl AscendGraph {
    /// Launch the whole captured graph asynchronously.
    pub fn replay(&self) -> Result<()> {
        let code = unsafe { ffi::aclmdlRIExecuteAsync(self.ri, self.stream) };
        if code != 0 {
            return Err(AclError { code, op: "aclmdlRIExecuteAsync" });
        }
        Ok(())
    }

    /// Launch and wait for completion.
    pub fn replay_sync(&self) -> Result<()> {
        self.replay()?;
        let code = unsafe { ffi::aclrtSynchronizeStream(self.stream) };
        if code != 0 {
            return Err(AclError { code, op: "aclrtSynchronizeStream" });
        }
        Ok(())
    }
}

impl Drop for AscendGraph {
    fn drop(&mut self) {
        if !self.ri.is_null() {
            unsafe { ffi::aclmdlRIDestroy(self.ri) };
        }
    }
}

/// Begin capture on `stream`. The ops issued on the stream until `end`
/// become the graph. Must run outside any other capture.
pub fn begin(stream: &AscendStream, mode: CaptureMode) -> Result<()> {
    let code = unsafe { ffi::aclmdlRICaptureBegin(stream.handle(), mode.as_acl()) };
    if code != 0 {
        return Err(AclError { code, op: "aclmdlRICaptureBegin" });
    }
    Ok(())
}

/// End capture and return the executable graph.
pub fn end(stream: &AscendStream) -> Result<AscendGraph> {
    let mut ri: *mut c_void = std::ptr::null_mut();
    let code = unsafe { ffi::aclmdlRICaptureEnd(stream.handle(), &mut ri) };
    if code != 0 {
        return Err(AclError { code, op: "aclmdlRICaptureEnd" });
    }
    Ok(AscendGraph { ri, stream: stream.handle() })
}

/// Abort a capture in progress (e.g. after a payload op failed mid-capture);
/// leaves the stream usable.
pub fn abort(stream: &AscendStream) -> Result<()> {
    let code = unsafe { ffi::aclmdlRIAbort(stream.handle()) };
    if code != 0 {
        return Err(AclError { code, op: "aclmdlRIAbort" });
    }
    Ok(())
}
