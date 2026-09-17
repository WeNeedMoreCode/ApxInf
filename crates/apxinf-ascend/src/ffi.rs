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
    pub fn aclrtMemcpyAsync(
        dst: *mut c_void,
        destMax: usize,
        src: *const c_void,
        count: usize,
        kind: u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn aclrtMemsetAsync(
        dst: *mut c_void,
        destMax: usize,
        value: i32,
        count: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// aclrtMemcpyKind.
pub const ACL_MEMCPY_HOST_TO_HOST: u32 = 0;
pub const ACL_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub const ACL_MEMCPY_DEVICE_TO_HOST: u32 = 2;
pub const ACL_MEMCPY_DEVICE_TO_DEVICE: u32 = 3;

/// aclrtMemMallocPolicy.
pub const ACL_MEM_MALLOC_HUGE_FIRST: u32 = 0;

// aclDataType (acl_base_rt.h).
pub const ACL_FLOAT: u32 = 0;
pub const ACL_FLOAT16: u32 = 1;
pub const ACL_INT32: u32 = 3;

// aclFormat (acl_base_rt.h).
pub const ACL_FORMAT_ND: u32 = 2;
pub const ACL_FORMAT_FRACTAL_NZ: u32 = 29;

// aclnnStatus (aclnn/acl_meta.h): int32, OK = 0.

// aclnn tensors (aclnn/acl_meta.h): opaque descriptor handles that bind a
// device pointer to shape/stride/format for the aclnn op entry points.
extern "C" {
    pub fn aclCreateTensor(
        viewDims: *const i64,
        viewDimsNum: u64,
        dataType: u32,
        stride: *const i64,
        offset: i64,
        format: u32,
        storageDims: *const i64,
        storageDimsNum: u64,
        tensorData: *mut c_void,
    ) -> *mut c_void;
    pub fn aclDestroyTensor(tensor: *mut c_void) -> i32;

    // aclnnMatmul (aclnnop/aclnn_matmul.h), two-stage API: plan then run.
    // cubeMathType: 1 = KEEP_DTYPE (compute in the tensors' dtype).
    pub fn aclnnMatmulGetWorkspaceSize(
        selfT: *mut c_void,
        mat2: *mut c_void,
        out: *mut c_void,
        cubeMathType: i8,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnMatmul(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_mm.h: aten::mm mirror, 2x2 pattern only.
    pub fn aclnnMmGetWorkspaceSize(
        selfT: *mut c_void,
        mat2: *mut c_void,
        out: *mut c_void,
        cubeMathType: i8,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnMm(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/level2/aclnn_gemm.h: BLAS gemm with explicit transA/transB.
    // out = alpha * op(A) * op(B) + beta * C.
    pub fn aclnnGemmGetWorkspaceSize(
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
        alpha: f32,
        beta: f32,
        transA: i64,
        transB: i64,
        out: *mut c_void,
        cubeMathType: i8,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnGemm(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_add.h: out = self + alpha * other.
    pub fn aclnnAddGetWorkspaceSize(
        selfT: *mut c_void,
        other: *mut c_void,
        alpha: *mut c_void,
        out: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnAdd(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_silu.h: out = silu(self), elementwise.
    pub fn aclnnSiluGetWorkspaceSize(
        selfT: *mut c_void,
        out: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnSilu(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_add_rms_norm.h: fused y = rmsnorm(x1 + x2) * gamma,
    // plus optional rstd/x outputs (buffers must be provided).
    pub fn aclnnAddRmsNormGetWorkspaceSize(
        x1: *mut c_void,
        x2: *mut c_void,
        gamma: *mut c_void,
        epsilon: f64,
        yOut: *mut c_void,
        rstdOut: *mut c_void,
        xOut: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnAddRmsNorm(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnn/acl_meta.h scalars and lists.
    pub fn aclCreateScalar(value: *mut c_void, dataType: u32) -> *mut c_void;
    pub fn aclDestroyScalar(scalar: *mut c_void) -> i32;
    pub fn aclCreateTensorList(value: *const *mut c_void, size: u64) -> *mut c_void;
    pub fn aclDestroyTensorList(list: *mut c_void) -> i32;

    // aclnnop/aclnn_cat.h: concatenate tensors along `dim`.
    pub fn aclnnCatGetWorkspaceSize(
        tensors: *mut c_void,
        dim: i64,
        out: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnCat(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_apply_rotary_pos_emb.h: fused in-place RoPE for q/k,
    // rotate-half formula (== Gemma semantics), layout=1 means BSND.
    // Supported on Atlas inference cards per cann-ops-adv docs.
    pub fn aclnnApplyRotaryPosEmbGetWorkspaceSize(
        queryRef: *mut c_void,
        keyRef: *mut c_void,
        cos: *mut c_void,
        sin: *mut c_void,
        layout: i64,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnApplyRotaryPosEmb(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_add_layer_norm.h: y = LayerNorm(x1 + x2) * gamma + beta,
    // with optional bias and extra outputs (mean/rstd/x). Used as plain
    // LayerNorm by passing a zeros second input.
    pub fn aclnnAddLayerNormGetWorkspaceSize(
        x1: *mut c_void,
        x2: *mut c_void,
        gamma: *mut c_void,
        beta: *mut c_void,
        biasOptional: *mut c_void,
        epsilon: f64,
        additionalOutput: bool,
        yOut: *mut c_void,
        meanOut: *mut c_void,
        rstdOut: *mut c_void,
        xOut: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnAddLayerNorm(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_npu_format_cast.h: ND -> FRACTAL_NZ weight conversion
    // (torch_npu's npu_format_cast(w, 29) at the aclnn layer).
    pub fn aclnnNpuFormatCastCalculateSizeAndFormat(
        srcTensor: *mut c_void,
        dstFormat: i32,
        additionalDtype: i32,
        dstShape: *mut *mut i64,
        dstShapeSize: *mut u64,
        actualFormat: *mut i32,
    ) -> i32;
    pub fn aclnnNpuFormatCastGetWorkspaceSize(
        srcTensor: *mut c_void,
        dstTensor: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnNpuFormatCast(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_matmul.h (WeightNz variant): mat2 held in FRACTAL_NZ.
    pub fn aclnnMatmulWeightNzGetWorkspaceSize(
        selfT: *mut c_void,
        mat2: *mut c_void,
        out: *mut c_void,
        cubeMathType: i8,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnMatmulWeightNz(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_gelu_v2.h: gelu with an `approximate` selector
    // (0 = erf, 1 = tanh), the PyTorch-compatible semantics torch_npu
    // dispatches to. This is the ONLY working gelu entry on 310P3:
    // aclnnGelu core-dumps and aclnnFastGelu returns 161002
    // ("not implemented, dtype support list []") on every dtype and both
    // CANN 8.5.1/9.0.1 -- those op binaries simply do not ship for this
    // SoC (proven via CANN debug logs, 2026-09-17).
    pub fn aclnnGeluV2GetWorkspaceSize(
        x: *mut c_void,
        approximate: i64,
        y: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnGeluV2(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_gather_v2.h: out[i..] = self[index[i]..] along `dim`.
    pub fn aclnnGatherV2GetWorkspaceSize(
        selfT: *mut c_void,
        dim: i64,
        index: *mut c_void,
        out: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnGatherV2(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/aclnn_mul.h: out = self * other / out = self * scalar.
    pub fn aclnnMulGetWorkspaceSize(
        selfT: *mut c_void,
        other: *mut c_void,
        out: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnMul(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    pub fn aclnnMulsGetWorkspaceSize(
        selfT: *mut c_void,
        other: *mut c_void,
        out: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnMuls(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // aclnnop/level2/aclnn_prompt_flash_attention_v3.h -- PFA, the fused
    // attention for 310P-class inference cards (fp16 only). The legacy
    // aclnnPromptFlashAttention deprecates 2026-12; V3 is the long-term API.
    pub fn aclnnPromptFlashAttentionV3GetWorkspaceSize(
        query: *mut c_void,
        key: *mut c_void,
        value: *mut c_void,
        pseShift: *mut c_void,
        attenMask: *mut c_void,
        actualSeqLengths: *mut c_void,
        actualSeqLengthsKv: *mut c_void,
        deqScale1: *mut c_void,
        quantScale1: *mut c_void,
        deqScale2: *mut c_void,
        quantScale2: *mut c_void,
        quantOffset2: *mut c_void,
        numHeads: i64,
        scaleValue: f64,
        preTokens: i64,
        nextTokens: i64,
        inputLayout: *mut u8,
        numKeyValueHeads: i64,
        sparseMode: i64,
        innerPrecise: i64,
        attentionOut: *mut c_void,
        workspaceSize: *mut u64,
        executor: *mut *mut c_void,
    ) -> i32;
    pub fn aclnnPromptFlashAttentionV3(
        workspace: *mut c_void,
        workspaceSize: u64,
        executor: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

// aclmdlRICaptureMode (acl_rt.h): capture scoping, mirrors CUDA's
// cudaStreamCaptureMode.
pub const ACL_MODEL_RI_CAPTURE_MODE_GLOBAL: i32 = 0;
pub const ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL: i32 = 1;
pub const ACL_MODEL_RI_CAPTURE_MODE_RELAXED: i32 = 2;

extern "C" {
    // acl_rt.h -- ACLGraph (model Runtime Instance): capture/replay, the
    // Ascend counterpart of CUDA Graphs. Runtime-gated by CANN version:
    // symbols exist in 8.5.1 but capture is rejected (aclError 207000);
    // verified working on 310P3 under CANN 9.0.1 (probe
    // dev_logs/aclgraph_probe/, 2026-09-17).
    pub fn aclmdlRICaptureBegin(stream: *mut c_void, mode: i32) -> i32;
    pub fn aclmdlRICaptureEnd(stream: *mut c_void, modelRI: *mut *mut c_void) -> i32;
    pub fn aclmdlRICaptureGetInfo(
        stream: *mut c_void,
        status: *mut i32,
        modelRI: *mut *mut c_void,
    ) -> i32;
    pub fn aclmdlRIExecuteAsync(modelRI: *mut c_void, stream: *mut c_void) -> i32;
    pub fn aclmdlRIExecute(modelRI: *mut c_void, timeout_ms: i32) -> i32;
    pub fn aclmdlRIDestroy(modelRI: *mut c_void) -> i32;
    pub fn aclmdlRIAbort(stream: *mut c_void) -> i32;
}
