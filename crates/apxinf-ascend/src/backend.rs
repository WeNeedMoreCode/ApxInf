//! Ascend device backend: context + stream + zero-cache carrier.
//!
//! Device tensors are fp16 (310P3 has no bf16/fp8 unit). The inference
//! path (GE OM building/serving in apxinf-model) consumes these accessors
//! directly; no aclnn trait surface lives here.

use std::sync::{Arc, Mutex};

use apxinf_core::{Device, Error, Result, Storage, Tensor};

use crate::context::{AscendContext, DeviceBuffer};
use crate::stream::AscendStream;
use crate::AclError;

pub struct AscendBackend {
    ctx: Arc<AscendContext>,
    stream: Arc<AscendStream>,
    /// Zero-filled buffer reused by the rms_norm-via-add_rms_norm shim.
    zeros: Mutex<std::collections::HashMap<usize, Arc<DeviceBuffer>>>,
}

fn acl_err(e: AclError) -> Error {
    Error::Other(format!("aclnn: {e}"))
}

impl AscendBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = AscendContext::new(device_id).map_err(acl_err)?;
        let stream = AscendStream::new().map_err(acl_err)?;
        Ok(Self { ctx: Arc::new(ctx), stream: Arc::new(stream), zeros: Mutex::new(Default::default()) })
    }

    fn dev(&self) -> Device {
        Device::Ascend(self.ctx.device_id())
    }

    fn zeros_like(&self, len: usize) -> Result<Arc<DeviceBuffer>> {
        let mut guard = self.zeros.lock().unwrap();
        if let Some(z) = guard.get(&len) {
            return Ok(z.clone());
        }
        let z = Arc::new(self.ctx.malloc(len).map_err(acl_err)?);
        // Device memory is not guaranteed zeroed; host-zero it once.
        let zeros_bytes = vec![0u8; len];
        self.ctx.copy_h2d(&z, &zeros_bytes).map_err(acl_err)?;
        guard.insert(len, z.clone());
        Ok(z)
    }

    // Accessors for the pi05 ascend executor layer (crate-public seam).
    pub fn ctx(&self) -> &Arc<AscendContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<AscendStream> {
        &self.stream
    }

    /// Cached zero buffer of `len` bytes (shared with rms_norm shim).
    pub fn zeros(&self, len: usize) -> Result<Arc<DeviceBuffer>> {
        self.zeros_like(len)
    }

    /// Wrap a device buffer as an fp16 [rows, cols] tensor.
    pub fn wrap_fp16(&self, buf: DeviceBuffer, dims: impl Into<Vec<usize>>) -> Tensor {
        let dims = dims.into();
        let len = buf.len();
        Tensor::from_raw_parts(
            dims.into(),
            apxinf_core::DType::F16,
            self.dev(),
            Storage::Gpu {
                device: self.dev(),
                handle: apxinf_core::storage::GpuStorageHandle {
                    ptr: buf.as_ptr() as usize,
                    len,
                    _prevent_leak: Some(Arc::new(buf)),
                },
            },
        )
    }
}

