//! `apxinf_core::Backend` implementation over the aclnn ops.
//!
//! Device tensors are fp16 (310P3 has no bf16/fp8 unit); `to_device`
//! converts f32 → f16 and `to_cpu` converts back, mirroring the ModelZoo
//! torch_npu semantics this port calibrates against. Storage reuses the
//! backend-neutral `Storage::Gpu` slot: `_prevent_leak` carries an
//! `Arc<DeviceBuffer>`, which we borrow back via `downcast_ref` — no raw
//! views that could free on drop.
//!
//! Not yet implemented (stage-2 queue): rope, embedding, KV-cache sdpa.
//! Those arms return `Error::Other` with a clear marker until landed.

use std::sync::{Arc, Mutex};

use apxinf_core::{Backend, Device, Error, Graph, Result, Storage, Tensor};

use crate::context::{AscendContext, DeviceBuffer};
use crate::graph::{self, AscendGraph};
use crate::ops;
use crate::stream::AscendStream;
use crate::AclError;

pub struct AscendBackend {
    ctx: Arc<AscendContext>,
    stream: Arc<AscendStream>,
    /// Zero-filled buffer reused by the rms_norm-via-add_rms_norm shim.
    zeros: Mutex<Option<Arc<DeviceBuffer>>>,
}

fn acl_err(e: AclError) -> Error {
    Error::Other(format!("aclnn: {e}"))
}

impl AscendBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = AscendContext::new(device_id).map_err(acl_err)?;
        let stream = AscendStream::new().map_err(acl_err)?;
        Ok(Self { ctx: Arc::new(ctx), stream: Arc::new(stream), zeros: Mutex::new(None) })
    }

    fn dev(&self) -> Device {
        Device::Ascend(self.ctx.device_id())
    }

    /// Borrow the `DeviceBuffer` behind a device tensor.
    fn buf(t: &Tensor) -> Result<&DeviceBuffer> {
        match t.storage() {
            Storage::Gpu { device: Device::Ascend(_), handle } => handle
                ._prevent_leak
                .as_ref()
                .and_then(|any| any.downcast_ref::<DeviceBuffer>())
                .ok_or_else(|| Error::Other("ascend tensor storage missing DeviceBuffer".into())),
            _ => Err(Error::UnsupportedDevice(t.device())),
        }
    }

    /// Wrap a fresh device buffer as an fp16 tensor of `shape`.
    fn wrap(&self, buf: DeviceBuffer, shape: impl Into<apxinf_core::Shape>) -> Tensor {
        let len = buf.len();
        Tensor::from_raw_parts(
            shape.into(),
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

    /// Shape as i64 dims for the aclnn descriptors.
    fn dims(t: &Tensor) -> Vec<i64> {
        t.shape().dims().iter().map(|&d| d as i64).collect()
    }

    fn zeros_like(&self, len: usize) -> Result<Arc<DeviceBuffer>> {
        let mut guard = self.zeros.lock().unwrap();
        if let Some(z) = guard.as_ref() {
            if z.len() == len {
                return Ok(z.clone());
            }
        }
        let z = Arc::new(self.ctx.malloc(len).map_err(acl_err)?);
        // Device memory is not guaranteed zeroed; host-zero it once.
        let zeros_bytes = vec![0u8; len];
        self.ctx.copy_h2d(&z, &zeros_bytes).map_err(acl_err)?;
        *guard = Some(z.clone());
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

struct GraphBox(AscendGraph);
impl Graph for GraphBox {
    fn replay(&self) -> Result<()> {
        self.0.replay_sync().map_err(acl_err)
    }
}

impl Backend for AscendBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let shape = Self::dims(input);
        if shape.len() != 2 {
            return Err(Error::Other(format!(
                "ascend rms_norm expects 2-D [seq, hidden], got {shape:?} (flatten first)"
            )));
        }
        let zeros = self.zeros_like(input.storage().len())?;
        let (y, _rstd) = ops::add_rms_norm_fp16(
            &self.ctx,
            &self.stream,
            Self::buf(input)?,
            &zeros,
            Self::buf(weight)?,
            &shape,
            eps as f64,
        )
        .map_err(acl_err)?;
        Ok(self.wrap(y, input.shape().clone()))
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        let out = ops::silu_fp16(&self.ctx, &self.stream, Self::buf(input)?, &Self::dims(input))
            .map_err(acl_err)?;
        Ok(self.wrap(out, input.shape().clone()))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out = ops::add_fp16(
            &self.ctx,
            &self.stream,
            Self::buf(a)?,
            Self::buf(b)?,
            &Self::dims(a),
        )
        .map_err(acl_err)?;
        Ok(self.wrap(out, a.shape().clone()))
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out = ops::mul_fp16(
            &self.ctx,
            &self.stream,
            Self::buf(a)?,
            Self::buf(b)?,
            &Self::dims(a),
        )
        .map_err(acl_err)?;
        Ok(self.wrap(out, a.shape().clone()))
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let out =
            ops::muls_fp16(&self.ctx, &self.stream, Self::buf(input)?, factor, &Self::dims(input))
                .map_err(acl_err)?;
        Ok(self.wrap(out, input.shape().clone()))
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let da = Self::dims(a);
        let db = Self::dims(b);
        if da.len() != 2 || db.len() != 2 || da[1] != db[0] {
            return Err(Error::Other(format!(
                "ascend matmul expects 2-D [m,k]x[k,n], got {da:?}x{db:?}"
            )));
        }
        let out = ops::matmul_fp16(
            &self.ctx,
            &self.stream,
            Self::buf(a)?,
            [da[0], da[1]],
            Self::buf(b)?,
            [db[0], db[1]],
        )
        .map_err(acl_err)?;
        Ok(self.wrap(out, vec![da[0] as usize, db[1] as usize]))
    }

    fn rope(&self, _input: &Tensor, _n_heads: usize, _head_dim: usize, _theta: f32, _pos_offset: u32) -> Result<Tensor> {
        Err(Error::Other("ascend rope pending (aclnnApplyRotaryPosEmbV2 queued)".into()))
    }

    fn embedding(&self, _table: &Tensor, _ids: &[u32]) -> Result<Tensor> {
        Err(Error::Other("ascend embedding pending (gather op queued)".into()))
    }

    fn sdpa_decode(&self, _q: &Tensor, _kv: &mut dyn apxinf_core::KvCache, _layer_idx: usize,
                   _n_heads: usize, _n_kv_heads: usize, _head_dim: usize,
                   _kv_len: usize, _max_seq_len: usize) -> Result<Tensor> {
        Err(Error::Other("ascend sdpa_decode pending (kv cache queued)".into()))
    }

    fn sdpa_prefill(&self, _q: &Tensor, _kv: &mut dyn apxinf_core::KvCache, _layer_idx: usize,
                    _n_heads: usize, _n_kv_heads: usize, _head_dim: usize,
                    _kv_len: usize, _max_seq_len: usize) -> Result<Tensor> {
        Err(Error::Other("ascend sdpa_prefill pending (kv cache queued)".into()))
    }

    fn create_kv_cache(&self, _n_layers: usize, _n_kv_heads: usize,
                       _head_dim: usize, _max_seq_len: usize) -> Box<dyn apxinf_core::KvCache> {
        unimplemented!("ascend kv cache is queued for stage-2; not reachable until the PI0.5 executor lands")
    }

    fn kv_append(&self, _kv: &mut dyn apxinf_core::KvCache, _layer_idx: usize,
                 _k: &Tensor, _v: &Tensor, _append_len: usize) -> Result<()> {
        Err(Error::Other("ascend kv_append pending".into()))
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(acl_err)
    }

    fn begin_capture(&self) -> Result<()> {
        graph::begin(&self.stream, graph::CaptureMode::ThreadLocal).map_err(acl_err)
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        let g = graph::end(&self.stream).map_err(acl_err)?;
        Ok(Box::new(GraphBox(g)))
    }

    fn device(&self) -> Device {
        self.dev()
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device() == self.dev() {
            return Ok(tensor.clone());
        }
        let f32s = tensor.to_f32_vec()?;
        let f16s: Vec<half::f16> = f32s.iter().map(|&x| half::f16::from_f32(x)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&f16s);
        let buf = self.ctx.malloc(bytes.len()).map_err(acl_err)?;
        self.ctx.copy_h2d(&buf, bytes).map_err(acl_err)?;
        Ok(self.wrap(buf, tensor.shape().clone()))
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device() == Device::Cpu {
            return Ok(tensor.clone());
        }
        let buf = Self::buf(tensor)?;
        let mut bytes = vec![0u8; buf.len()];
        self.ctx.copy_d2h(buf, &mut bytes).map_err(acl_err)?;
        let f16s: Vec<half::f16> = bytemuck::cast_slice(&bytes).to_vec();
        let f32s: Vec<f32> = f16s.iter().map(|x| x.to_f32()).collect();
        Tensor::from_f32(tensor.shape().dims().to_vec(), &f32s)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl apxinf_core::SamplingBackend for AscendBackend {
    fn create_token_sampler(&self, _spec: apxinf_core::TokenSamplingSpec) -> Result<Box<dyn apxinf_core::TokenSampler>> {
        Err(Error::Other("ascend token sampler pending (flow sampling lands with the PI0.5 executor)".into()))
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn apxinf_core::NormalGenerator>> {
        // Validate ownership + fp16, matching the CPU/CUDA contracts.
        if output.device() != self.dev() {
            return Err(Error::UnsupportedDevice(output.device()));
        }
        if output.dtype() != apxinf_core::DType::F16 {
            return Err(Error::Other(format!(
                "ascend normal generator expects F16 output, got {:?}",
                output.dtype()
            )));
        }
        let _ = Self::buf(&output)?; // ensure ascend storage is present
        Ok(Box::new(AscendNormalGenerator {
            output,
            ctx: self.ctx.clone(),
        }))
    }
}

/// Flow-matching noise via the deterministic core RNG, uploaded host-side.
/// Same semantics as the CPU backend's generator; generation cost is one
/// h2d copy of numel*2 bytes, hidden behind the model's async dispatch.
struct AscendNormalGenerator {
    output: Tensor,
    ctx: Arc<AscendContext>,
}

impl apxinf_core::NormalGenerator for AscendNormalGenerator {
    fn output(&self) -> &Tensor {
        &self.output
    }

    fn generate(&mut self, rng: apxinf_core::RngKey) -> Result<&Tensor> {
        let n = self.output.numel();
        let f32s = apxinf_core::standard_normal_f32(n, rng);
        let f16s: Vec<half::f16> = f32s.iter().map(|&x| half::f16::from_f32(x)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&f16s);
        let buf = match self.output.storage() {
            Storage::Gpu { handle, .. } => handle
                ._prevent_leak
                .as_ref()
                .and_then(|any| any.downcast_ref::<DeviceBuffer>())
                .ok_or_else(|| Error::Other("ascend tensor storage missing DeviceBuffer".into()))?,
            _ => return Err(Error::UnsupportedDevice(self.output.device())),
        };
        self.ctx.copy_h2d(buf, bytes).map_err(acl_err)?;
        Ok(&self.output)
    }
}
