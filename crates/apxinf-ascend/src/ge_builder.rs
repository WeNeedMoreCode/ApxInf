//! Generic GE graph-builder FFI client (C side in ascendc/ge_builder).
//!
//! Model definitions stay in Rust — which operators, which descs/attrs,
//! how things wire — expressed by this module's callers; the C++ side is
//! a thin manipulation layer. Loads via dlopen and links aarch64
//! artifacts, so the crate compiles on non-Ascend machines.
//!
//! Session semantics: `init` must run before the process's
//! `aclInit`/`SetDevice` (early POC measured GRAPH_FAILED on reverse
//! init order); `GeGraph::begin` opens a graph slot (several models can
//! coexist — vision/prefix/flow OMs); `build` materializes + compiles +
//! loads; `load` loads an on-disk OM directly (no compile). Compiled
//! artifacts cache through `save`, keyed by (shape signature, engine
//! version).
//!
//! Pitfalls inherited from the ge-offline-om skill: Data nodes need the
//! ghost input desc 0 and index attr (handled C++-side); build options
//! must carry input_shape+input_format (fail-fast C++-side); graph
//! building is pure operator flow — no AddNodeByOp.

use std::ffi::{c_void, CString};
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use crate::{AclError, AscendStream, DeviceBuffer, Result};

static LIB: OnceLock<std::result::Result<Library, String>> = OnceLock::new();

fn lib() -> std::result::Result<&'static Library, String> {
    LIB.get_or_init(|| {
        // bare name: resolved via the loader search path (LD_LIBRARY_PATH);
        // override with APXINF_GE_BUILDER_LIB for an explicit path
        let path = std::env::var("APXINF_GE_BUILDER_LIB")
            .unwrap_or_else(|_| "libge_builder.so".into());
        unsafe { Library::new(&path) }.map_err(|e| format!("dlopen {path}: {e}"))
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn cerr(op: &'static str, rc: i32) -> AclError {
    eprintln!("[ge_builder] {op} rc={rc}");
    AclError { code: rc, op }
}

/// Tensor dtype (FFI-side string mapping, C++ ParseDtype).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Fp16,
    Fp32,
    Int32,
    Int64,
    Int8,
    /// QuantBatchMatmulV3 packed scale (TransQuantParamV2's uint64
    /// output).
    Uint64,
}

impl Dtype {
    fn as_c(&self) -> CString {
        CString::new(match self {
            Dtype::Fp16 => "fp16",
            Dtype::Fp32 => "fp32",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Int8 => "int8",
            Dtype::Uint64 => "uint64",
        })
        .expect("dtype str")
    }
}

/// One GE model slot: the build-time handle and the post-load execution
/// handle are the same value (the C++-side slot index).
#[derive(Debug, Clone, Copy)]
pub struct GeGraph {
    handle: i64,
}

/// Process-wide GE build-session init (call before aclInit/SetDevice).
pub fn init(soc_version: &str) -> Result<()> {
    type F = unsafe extern "C" fn(*const std::os::raw::c_char) -> i32;
    let l = lib().map_err(|e| {
        eprintln!("[ge_builder] {e}");
        AclError { code: -1, op: "ge load" }
    })?;
    let f: Symbol<F> = unsafe { l.get(b"geb_init") }.map_err(|e| {
        eprintln!("[ge_builder] dlsym geb_init: {e}");
        AclError { code: -1, op: "ge dlsym" }
    })?;
    let soc = CString::new(soc_version).expect("soc");
    let rc = unsafe { f(soc.as_ptr()) };
    if rc != 0 {
        return Err(cerr("geb_init", rc));
    }
    Ok(())
}

/// Session teardown (unloads all models + aclgrphBuildFinalize).
pub fn fini() -> Result<()> {
    call0(b"geb_fini", "geb_fini")
}

/// Loads an on-disk OM (no GE compile session; pure ACL runtime).
pub fn load(path: &str) -> Result<GeGraph> {
    type F = unsafe extern "C" fn(*const std::os::raw::c_char) -> i64;
    let l = lib().map_err(|e| {
        eprintln!("[ge_builder] {e}");
        AclError { code: -1, op: "ge load" }
    })?;
    let f: Symbol<F> = unsafe { l.get(b"geb_model_load") }.map_err(|e| {
        eprintln!("[ge_builder] dlsym geb_model_load: {e}");
        AclError { code: -1, op: "ge dlsym" }
    })?;
    let p = CString::new(path).expect("path");
    let h = unsafe { f(p.as_ptr()) };
    if h < 0 {
        return Err(cerr("geb_model_load", h as i32));
    }
    Ok(GeGraph { handle: h })
}

fn call0(name: &[u8], op: &'static str) -> Result<()> {
    type F = unsafe extern "C" fn() -> i32;
    let l = lib().map_err(|e| {
        eprintln!("[ge_builder] {e}");
        AclError { code: -1, op: "ge load" }
    })?;
    let f: Symbol<F> = unsafe { l.get(name) }.map_err(|e| {
        eprintln!("[ge_builder] dlsym {}: {e}", String::from_utf8_lossy(name));
        AclError { code: -1, op: "ge dlsym" }
    })?;
    let rc = unsafe { f() };
    if rc != 0 {
        return Err(cerr(op, rc));
    }
    Ok(())
}

impl GeGraph {
    /// Opens a new graph slot and makes it current (add_*/set_*/link act
    /// on the most recently begun slot).
    pub fn begin(name: &str) -> Result<GeGraph> {
        type F = unsafe extern "C" fn(*const std::os::raw::c_char) -> i64;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_model_begin") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_model_begin: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(name).expect("name");
        let h = unsafe { f(n.as_ptr()) };
        if h < 0 {
            return Err(cerr("geb_model_begin", h as i32));
        }
        Ok(GeGraph { handle: h })
    }

    /// atc-style build option (passthrough for input_shape and friends).
    pub fn set_option(&self, key: &str, value: &str) -> Result<()> {
        type F = unsafe extern "C" fn(*const std::os::raw::c_char, *const std::os::raw::c_char) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_option") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_option: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let k = CString::new(key).expect("key");
        let v = CString::new(value).expect("value");
        let rc = unsafe { f(k.as_ptr(), v.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_set_option", rc));
        }
        Ok(())
    }

    /// ND input-shape convenience: input_format=ND +
    /// input_shape="n:d0,d1;...".
    pub fn set_nd_input_shape(&self, shapes: &[(&str, &[i64])]) -> Result<()> {
        let mut s = String::new();
        for (i, (name, dims)) in shapes.iter().enumerate() {
            if i > 0 {
                s.push(';');
            }
            s.push_str(name);
            s.push(':');
            for (d, dim) in dims.iter().enumerate() {
                if d > 0 {
                    s.push(',');
                }
                s.push_str(&dim.to_string());
            }
        }
        self.set_option("input_format", "ND")?;
        self.set_option("input_shape", &s)
    }

    /// Data input node (ghost desc 0 + index attr handled C++-side).
    pub fn add_data(&self, name: &str, index: i64, dims: &[i64], dtype: Dtype) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            i64,
            *const i64,
            i32,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_add_data") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_add_data: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(name).expect("name");
        let dt = dtype.as_c();
        let rc = unsafe { f(n.as_ptr(), index, dims.as_ptr(), dims.len() as i32, dt.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_add_data", rc));
        }
        Ok(())
    }

    /// Data input with an explicit desc format (FRACTAL_NZ weight
    /// passthrough experiments).
    pub fn add_data_fmt(
        &self,
        name: &str,
        index: i64,
        dims: &[i64],
        dtype: Dtype,
        fmt: &str,
    ) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            i64,
            *const i64,
            i32,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_add_data_fmt") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_add_data_fmt: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(name).expect("name");
        let dt = dtype.as_c();
        let fm = CString::new(fmt).expect("fmt");
        let rc =
            unsafe { f(n.as_ptr(), index, dims.as_ptr(), dims.len() as i32, dt.as_ptr(), fm.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_add_data_fmt", rc));
        }
        Ok(())
    }

    /// Operator node (GE IR registered name, e.g. MatMulV2 /
    /// PromptFlashAttention).
    pub fn add_op(&self, name: &str, ty: &str) -> Result<()> {
        type F = unsafe extern "C" fn(*const std::os::raw::c_char, *const std::os::raw::c_char) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_add_op") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_add_op: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(name).expect("name");
        let t = CString::new(ty).expect("type");
        let rc = unsafe { f(n.as_ptr(), t.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_add_op", rc));
        }
        Ok(())
    }

    /// Const node (1-D int32) — shape-class inputs (Reshape's shape,
    /// LayerNormV4's normalized_shape) must be compile-time constants: as
    /// Data inputs they turn consumer descs unknown, forcing dynamic
    /// shape splits and host-scheduled stalls.
    pub fn add_const_i32(&self, name: &str, vals: &[i32]) -> Result<()> {
        type F = unsafe extern "C" fn(*const std::os::raw::c_char, *const i32, i32) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_add_const_i32") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_add_const_i32: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(name).expect("name");
        let rc = unsafe { f(n.as_ptr(), vals.as_ptr(), vals.len() as i32) };
        if rc != 0 {
            return Err(cerr("geb_add_const_i32", rc));
        }
        Ok(())
    }

    /// Const node (arbitrary dtype/shape, raw bytes) — weights baked
    /// into the graph (if the compile folds the ND->NZ conversion, the
    /// OM carries NZ weights with zero runtime tax).
    pub fn add_const_raw(&self, name: &str, dims: &[i64], dtype: Dtype, bytes: &[u8]) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const i64,
            i32,
            *const std::os::raw::c_char,
            *const u8,
            i64,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_add_const_raw") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_add_const_raw: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(name).expect("name");
        let dt = dtype.as_c();
        let rc = unsafe {
            f(n.as_ptr(), dims.as_ptr(), dims.len() as i32, dt.as_ptr(), bytes.as_ptr(),
              bytes.len() as i64)
        };
        if rc != 0 {
            return Err(cerr("geb_add_const_raw", rc));
        }
        Ok(())
    }

    /// Dynamic-input port registration (DYNAMIC_INPUT operators —
    /// CreateOperatorByName pre-creates none; ge_builder links
    /// libgraph_base symbols directly for
    /// OpDescUtils::GetOpDescFromOperator + OpDesc::AddDynamicInputDesc,
    /// the same path the TF parser takes). Port names are base+index
    /// (x0/x1/...). Call after add_op and before descs/links (e.g.
    /// ConcatD's dyn_inputs("x", 2)).
    pub fn dyn_inputs(&self, op: &str, base: &str, n: i32) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            i32,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_dyn_inputs") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_dyn_inputs: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let o = CString::new(op).expect("op");
        let b = CString::new(base).expect("base");
        let rc = unsafe { f(o.as_ptr(), b.as_ptr(), n) };
        if rc != 0 {
            return Err(cerr("geb_dyn_inputs", rc));
        }
        Ok(())
    }

    /// Dynamic-output port allocation (multi-output legacy operators'
    /// same-named y x N ports — CreateOperator pre-creates no dynamic
    /// outputs; allocate here before setting descs/wiring).
    pub fn dyn_outputs(&self, op: &str, base: &str, n: i32) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            i32,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_dyn_outputs") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_dyn_outputs: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let o = CString::new(op).expect("op");
        let b = CString::new(base).expect("base");
        let rc = unsafe { f(o.as_ptr(), b.as_ptr(), n) };
        if rc != 0 {
            return Err(cerr("geb_dyn_outputs", rc));
        }
        Ok(())
    }

    /// Dynamic-input port count probe (diagnostics).
    pub fn dyn_probe(&self, op: &str, base: &str) -> Result<i32> {
        type F = unsafe extern "C" fn(*const std::os::raw::c_char, *const std::os::raw::c_char) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_dyn_probe") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_dyn_probe: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let o = CString::new(op).expect("op");
        let b = CString::new(base).expect("base");
        let rc = unsafe { f(o.as_ptr(), b.as_ptr()) };
        if rc < 0 {
            return Err(cerr("geb_dyn_probe", rc));
        }
        Ok(rc)
    }

    /// Indexed edge: SetInput(dst_index, src, src_index).
    pub fn link_idx(&self, dst: &str, dst_index: i32, src: &str, src_index: i32) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            i32,
            *const std::os::raw::c_char,
            i32,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_link_idx") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_link_idx: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let d = CString::new(dst).expect("dst");
        let sc = CString::new(src).expect("src");
        let rc = unsafe { f(d.as_ptr(), dst_index, sc.as_ptr(), src_index) };
        if rc != 0 {
            return Err(cerr("geb_link_idx", rc));
        }
        Ok(())
    }

    fn set_desc(&self, sym: &[u8], op: &'static str, op_name: &str, port: &str, dims: &[i64],
                dtype: Dtype) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const i64,
            i32,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(sym) }.map_err(|e| {
            eprintln!("[ge_builder] dlsym {}: {e}", String::from_utf8_lossy(sym));
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op_name).expect("op name");
        let p = CString::new(port).expect("port");
        let dt = dtype.as_c();
        let rc = unsafe { f(n.as_ptr(), p.as_ptr(), dims.as_ptr(), dims.len() as i32, dt.as_ptr()) };
        if rc != 0 {
            return Err(cerr(op, rc));
        }
        Ok(())
    }

    pub fn set_input_desc(&self, op: &str, port: &str, dims: &[i64], dtype: Dtype) -> Result<()> {
        self.set_desc(b"geb_set_input_desc", "geb_set_input_desc", op, port, dims, dtype)
    }

    pub fn set_output_desc(&self, op: &str, port: &str, dims: &[i64], dtype: Dtype) -> Result<()> {
        self.set_desc(b"geb_set_output_desc", "geb_set_output_desc", op, port, dims, dtype)
    }

    /// Per-index output desc for dynamic multi-output operators (the
    /// same-named "y" x N ports go through C++
    /// UpdateDynamicOutputDesc(name, index) — the flat-index
    /// UpdateOutputDesc is rejected for dynamic output ports).
    pub fn set_output_desc_idx(&self, op: &str, port: &str, out_idx: i32, dims: &[i64], dtype: Dtype) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            i32,
            *const i64,
            i32,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_output_desc_idx") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_output_desc_idx: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op).expect("op");
        let p = CString::new(port).expect("port");
        let dt = dtype.as_c();
        let rc = unsafe { f(n.as_ptr(), p.as_ptr(), out_idx, dims.as_ptr(), dims.len() as i32, dt.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_set_output_desc_idx", rc));
        }
        Ok(())
    }

    /// Consumer input desc with an explicit format (ND/FRACTAL_NZ) —
    /// for MatMulV2's x2 in NZ weight-passthrough experiments (dims are
    /// NZ 4-D, matching the upstream NZ Data desc).
    pub fn set_input_desc_fmt(
        &self,
        op: &str,
        port: &str,
        dims: &[i64],
        dtype: Dtype,
        fmt: &str,
    ) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const i64,
            i32,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_input_desc_fmt") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_input_desc_fmt: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op).expect("op");
        let p = CString::new(port).expect("port");
        let dt = dtype.as_c();
        let fm = CString::new(fmt).expect("fmt");
        let rc = unsafe {
            f(n.as_ptr(), p.as_ptr(), dims.as_ptr(), dims.len() as i32, dt.as_ptr(), fm.as_ptr())
        };
        if rc != 0 {
            return Err(cerr("geb_set_input_desc_fmt", rc));
        }
        Ok(())
    }

    pub fn set_attr_bool(&self, op: &str, attr: &str, v: bool) -> Result<()> {
        type F = unsafe extern "C" fn(*const std::os::raw::c_char, *const std::os::raw::c_char, i32) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_attr_bool") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_attr_bool: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op).expect("op");
        let a = CString::new(attr).expect("attr");
        let rc = unsafe { f(n.as_ptr(), a.as_ptr(), v as i32) };
        if rc != 0 {
            return Err(cerr("geb_set_attr_bool", rc));
        }
        Ok(())
    }

    pub fn set_attr_int(&self, op: &str, attr: &str, v: i64) -> Result<()> {
        type F = unsafe extern "C" fn(*const std::os::raw::c_char, *const std::os::raw::c_char, i64) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_attr_int") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_attr_int: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op).expect("op");
        let a = CString::new(attr).expect("attr");
        let rc = unsafe { f(n.as_ptr(), a.as_ptr(), v) };
        if rc != 0 {
            return Err(cerr("geb_set_attr_int", rc));
        }
        Ok(())
    }

    pub fn set_attr_float(&self, op: &str, attr: &str, v: f64) -> Result<()> {
        type F = unsafe extern "C" fn(*const std::os::raw::c_char, *const std::os::raw::c_char, f64) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_attr_float") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_attr_float: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op).expect("op");
        let a = CString::new(attr).expect("attr");
        let rc = unsafe { f(n.as_ptr(), a.as_ptr(), v) };
        if rc != 0 {
            return Err(cerr("geb_set_attr_float", rc));
        }
        Ok(())
    }

    pub fn set_attr_str(&self, op: &str, attr: &str, v: &str) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_attr_str") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_attr_str: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op).expect("op");
        let a = CString::new(attr).expect("attr");
        let s = CString::new(v).expect("value");
        let rc = unsafe { f(n.as_ptr(), a.as_ptr(), s.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_set_attr_str", rc));
        }
        Ok(())
    }

    /// DataType-typed attr (Cast.dst_type etc.) — parsed C++-side by
    /// ParseDtype.
    pub fn set_attr_dtype(&self, op: &str, attr: &str, v: &str) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_attr_dtype") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_attr_dtype: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op).expect("op");
        let a = CString::new(attr).expect("attr");
        let s = CString::new(v).expect("value");
        let rc = unsafe { f(n.as_ptr(), a.as_ptr(), s.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_set_attr_dtype", rc));
        }
        Ok(())
    }

    pub fn set_attr_int_list(&self, op: &str, attr: &str, v: &[i64]) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const i64,
            i32,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_set_attr_int_list") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_set_attr_int_list: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = CString::new(op).expect("op");
        let a = CString::new(attr).expect("attr");
        let rc = unsafe { f(n.as_ptr(), a.as_ptr(), v.as_ptr(), v.len() as i32) };
        if rc != 0 {
            return Err(cerr("geb_set_attr_int_list", rc));
        }
        Ok(())
    }

    /// dst.SetInput(port, src, src_port) — explicit source output port.
    /// Multi-output operators (AddRmsNorm's y/rstd/x_out, ARPE's q/k)
    /// MUST use this: default output resolution fails silently, the edge
    /// never lands in the graph (dumps show in0 <- <none>; compilation
    /// dies on the dangling edge).
    pub fn link_out(&self, dst: &str, port: &str, src: &str, src_port: &str) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_link_out") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_link_out: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let d = CString::new(dst).expect("dst");
        let p = CString::new(port).expect("port");
        let s = CString::new(src).expect("src");
        let sp = CString::new(src_port).expect("src port");
        let rc = unsafe { f(d.as_ptr(), p.as_ptr(), s.as_ptr(), sp.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_link_out", rc));
        }
        Ok(())
    }

    /// dst.SetInput(port, src) — the edge registers on both
    /// OperatorImpls and materializes at build.
    pub fn link(&self, dst: &str, port: &str, src: &str) -> Result<()> {
        type F = unsafe extern "C" fn(
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_link") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_link: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let d = CString::new(dst).expect("dst");
        let p = CString::new(port).expect("port");
        let s = CString::new(src).expect("src");
        let rc = unsafe { f(d.as_ptr(), p.as_ptr(), s.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_link", rc));
        }
        Ok(())
    }

    fn name_list(&self, sym: &[u8], op: &'static str, names: &[&str]) -> Result<()> {
        type F = unsafe extern "C" fn(*const *const std::os::raw::c_char, i32) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(sym) }.map_err(|e| {
            eprintln!("[ge_builder] dlsym {}: {e}", String::from_utf8_lossy(sym));
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let cstrs: Vec<CString> = names.iter().map(|n| CString::new(*n).expect("name")).collect();
        let ptrs: Vec<*const std::os::raw::c_char> = cstrs.iter().map(|c| c.as_ptr()).collect();
        let rc = unsafe { f(ptrs.as_ptr(), ptrs.len() as i32) };
        if rc != 0 {
            return Err(cerr(op, rc));
        }
        Ok(())
    }

    /// Graph inputs (Data nodes in bind order).
    pub fn graph_inputs(&self, names: &[&str]) -> Result<()> {
        self.name_list(b"geb_graph_inputs", "geb_graph_inputs", names)
    }

    /// Graph outputs (located by operator name in the materialized
    /// graph).
    pub fn graph_outputs(&self, names: &[&str]) -> Result<()> {
        self.name_list(b"geb_graph_outputs", "geb_graph_outputs", names)
    }

    /// Graph outputs with explicit output port indices (required for
    /// multi-output operators: RmsNorm's REQUIRED rstd=1, ARPE's key=1 —
    /// a required output left as a dead intermediate silently kills
    /// compilation).
    pub fn graph_outputs_idx(&self, names: &[&str], idxs: &[i32]) -> Result<()> {
        type F = unsafe extern "C" fn(*const *const std::os::raw::c_char, *const i32, i32) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_graph_outputs_idx") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_graph_outputs_idx: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let cstrs: Vec<CString> = names.iter().map(|n| CString::new(*n).expect("name")).collect();
        let ptrs: Vec<*const std::os::raw::c_char> = cstrs.iter().map(|c| c.as_ptr()).collect();
        let rc = unsafe { f(ptrs.as_ptr(), idxs.as_ptr(), ptrs.len() as i32) };
        if rc != 0 {
            return Err(cerr("geb_graph_outputs_idx", rc));
        }
        Ok(())
    }

    /// Materialize + aclgrphBuildModel in-memory compile + load (first
    /// compile takes seconds to minutes; artifacts cache via save).
    pub fn build(&self) -> Result<()> {
        type F = unsafe extern "C" fn(i64) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_model_build") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_model_build: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let rc = unsafe { f(self.handle) };
        if rc != 0 {
            return Err(cerr("geb_model_build", rc));
        }
        Ok(())
    }

    fn io_i64(&self, sym: &[u8], idx: i32) -> Result<i64> {
        type F = unsafe extern "C" fn(i64, i32) -> i64;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(sym) }.map_err(|e| {
            eprintln!("[ge_builder] dlsym {}: {e}", String::from_utf8_lossy(sym));
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let v = unsafe { f(self.handle, idx) };
        if v < 0 {
            return Err(cerr("io query", v as i32));
        }
        Ok(v)
    }

    fn io_count(&self, sym: &[u8]) -> Result<usize> {
        type F = unsafe extern "C" fn(i64) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(sym) }.map_err(|e| {
            eprintln!("[ge_builder] dlsym {}: {e}", String::from_utf8_lossy(sym));
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let v = unsafe { f(self.handle) };
        if v < 0 {
            return Err(cerr("io count", v));
        }
        Ok(v as usize)
    }

    pub fn num_inputs(&self) -> Result<usize> {
        self.io_count(b"geb_num_inputs")
    }

    pub fn num_outputs(&self) -> Result<usize> {
        self.io_count(b"geb_num_outputs")
    }

    /// Model-reported input byte size (alignment included — do not
    /// compute it yourself).
    pub fn input_size(&self, idx: usize) -> Result<usize> {
        self.io_i64(b"geb_input_size", idx as i32).map(|v| v as usize)
    }

    pub fn output_size(&self, idx: usize) -> Result<usize> {
        self.io_i64(b"geb_output_size", idx as i32).map(|v| v as usize)
    }

    fn io_dims(&self, sym: &[u8], idx: usize) -> Result<Vec<i64>> {
        type F = unsafe extern "C" fn(i64, i32, *mut i64, i32) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(sym) }.map_err(|e| {
            eprintln!("[ge_builder] dlsym {}: {e}", String::from_utf8_lossy(sym));
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let mut buf = [0i64; 16];
        let rc = unsafe { f(self.handle, idx as i32, buf.as_mut_ptr(), buf.len() as i32) };
        if rc < 0 {
            return Err(cerr("io dims", rc));
        }
        Ok(buf[..rc as usize].to_vec())
    }

    pub fn input_dims(&self, idx: usize) -> Result<Vec<i64>> {
        self.io_dims(b"geb_input_dims", idx)
    }

    pub fn output_dims(&self, idx: usize) -> Result<Vec<i64>> {
        self.io_dims(b"geb_output_dims", idx)
    }

    /// Async execution (caller owns the stream sync; datasets assemble
    /// per the model-reported sizes).
    pub fn run(&self, inputs: &[&DeviceBuffer], outputs: &[&DeviceBuffer],
               stream: &AscendStream) -> Result<()> {
        let ins: Vec<*mut c_void> = inputs.iter().map(|b| b.as_ptr()).collect();
        let outs: Vec<*mut c_void> = outputs.iter().map(|b| b.as_ptr()).collect();
        self.run_raw(&ins, &outs, stream.handle() as *mut c_void)
    }

    /// Sync execution (bisect variant for debugging; when semantics are
    /// fine, suspect data/dtype before scheduling).
    pub fn run_sync(&self, inputs: &[&DeviceBuffer], outputs: &[&DeviceBuffer]) -> Result<()> {
        let ins: Vec<*mut c_void> = inputs.iter().map(|b| b.as_ptr()).collect();
        let outs: Vec<*mut c_void> = outputs.iter().map(|b| b.as_ptr()).collect();
        self.run_raw(&ins, &outs, std::ptr::null_mut())
    }

    pub fn run_raw(&self, inputs: &[*mut c_void], outputs: &[*mut c_void], stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(i64, *const *mut c_void, i32, *const *mut c_void, i32, *mut c_void) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_run") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_run: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let rc = unsafe {
            f(self.handle, inputs.as_ptr(), inputs.len() as i32, outputs.as_ptr(),
              outputs.len() as i32, stream)
        };
        if rc != 0 {
            return Err(cerr("geb_run", rc));
        }
        Ok(())
    }

    /// Save the OM to disk (cache key = shape signature + engine
    /// version).
    pub fn save(&self, path: &str) -> Result<()> {
        type F = unsafe extern "C" fn(i64, *const std::os::raw::c_char) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_model_save") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_model_save: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let p = CString::new(path).expect("path");
        let rc = unsafe { f(self.handle, p.as_ptr()) };
        if rc != 0 {
            return Err(cerr("geb_model_save", rc));
        }
        Ok(())
    }

    // ---- Dynamic gears (ge.dynamicDims OMs) ----

    /// Gear count (0 for a static OM).
    pub fn gear_count(&self) -> Result<usize> {
        type F = unsafe extern "C" fn(i64) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_gear_count") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_gear_count: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let v = unsafe { f(self.handle) };
        if v < 0 {
            return Err(cerr("geb_gear_count", v));
        }
        Ok(v as usize)
    }

    /// All dynamic-input dims of one gear (concatenated in registration
    /// order — same source as the S2 payload).
    pub fn gear_dims(&self, gear: usize) -> Result<Vec<i64>> {
        type F = unsafe extern "C" fn(i64, i32, *mut i64, i32) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_gear_dims") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_gear_dims: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        // Capacity probe (returns dimCount) -> re-invoke.
        let n = unsafe { f(self.handle, gear as i32, std::ptr::null_mut(), 0) };
        if n <= 0 {
            return Err(cerr("geb_gear_dims", n));
        }
        let mut buf = vec![0i64; n as usize];
        let rc = unsafe { f(self.handle, gear as i32, buf.as_mut_ptr(), n) };
        if rc < 0 {
            return Err(cerr("geb_gear_dims", rc));
        }
        buf.truncate(rc as usize);
        Ok(buf)
    }

    /// Gear execution: dyn_dims = the dims of ALL dynamic inputs
    /// (concatenation order; they must go in one call — per-input calls
    /// fail with 145012). Equivalent to run() on a static OM (gear count
    /// 0; dyn_dims ignored), so callers can switch without branching.
    pub fn run_gear(
        &self,
        inputs: &[&DeviceBuffer],
        outputs: &[&DeviceBuffer],
        stream: &AscendStream,
        dyn_dims: &[i64],
    ) -> Result<()> {
        let ins: Vec<*mut c_void> = inputs.iter().map(|b| b.as_ptr()).collect();
        let outs: Vec<*mut c_void> = outputs.iter().map(|b| b.as_ptr()).collect();
        type F = unsafe extern "C" fn(
            i64,
            *const *mut c_void,
            i32,
            *const *mut c_void,
            i32,
            *mut c_void,
            *const i64,
            i32,
        ) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_run_gear") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_run_gear: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let rc = unsafe {
            f(
                self.handle,
                ins.as_ptr(),
                ins.len() as i32,
                outs.as_ptr(),
                outs.len() as i32,
                stream.handle() as *mut c_void,
                if dyn_dims.is_empty() {
                    std::ptr::null()
                } else {
                    dyn_dims.as_ptr()
                },
                dyn_dims.len() as i32,
            )
        };
        if rc != 0 {
            return Err(cerr("geb_run_gear", rc));
        }
        Ok(())
    }

    /// Output dims of the most recently executed gear (for rank>=3 with
    /// a non-trailing dynamic dim, consumers see max-gear strides —
    /// measured).
    pub fn cur_output_dims(&self, idx: usize) -> Result<Vec<i64>> {
        type F = unsafe extern "C" fn(i64, i32, *mut i64, i32) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_cur_output_dims") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_cur_output_dims: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let n = unsafe { f(self.handle, idx as i32, std::ptr::null_mut(), 0) };
        if n <= 0 {
            return Err(cerr("geb_cur_output_dims", n));
        }
        let mut buf = vec![0i64; n as usize];
        let rc = unsafe { f(self.handle, idx as i32, buf.as_mut_ptr(), n) };
        if rc < 0 {
            return Err(cerr("geb_cur_output_dims", rc));
        }
        buf.truncate(rc as usize);
        Ok(buf)
    }

    pub fn unload(&self) -> Result<()> {
        type F = unsafe extern "C" fn(i64) -> i32;
        let l = lib().map_err(|e| {
            eprintln!("[ge_builder] {e}");
            AclError { code: -1, op: "ge load" }
        })?;
        let f: Symbol<F> = unsafe { l.get(b"geb_model_unload") }.map_err(|e| {
            eprintln!("[ge_builder] dlsym geb_model_unload: {e}");
            AclError { code: -1, op: "ge dlsym" }
        })?;
        let rc = unsafe { f(self.handle) };
        if rc != 0 {
            return Err(cerr("geb_model_unload", rc));
        }
        Ok(())
    }
}
