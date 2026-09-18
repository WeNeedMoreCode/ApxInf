//! Generic GE graph-builder FFI client (C 路线 C1，ascendc/ge_builder)。
//!
//! 模型定义留在 Rust——哪些算子、什么 desc/attrs、怎么连线都由本模块的
//! 调用方（未来是 executor）表达；C++ 侧只是薄操纵层。dlopen 方式与
//! ada_rms 相同（aarch64 产物，非昇腾机器可编译本 crate）。
//!
//! 会话语义：`init` 必须在进程里 `aclInit`/`SetDevice` 之前跑（ge_poc
//! 实测 GE 反向初始化会 GRAPH_FAILED）；`GeGraph::begin` 开一个构图槽
//! （可并存多个模型——C2 需要 vision/prefix/flow 三段 OM）；`build`
//! 物化+编译+加载；`load` 直接装载落盘 OM（跳过编译）。编译产物缓存
//! 走 `save`，键为 (shape 签名, 引擎版本)（C2 落地）。
//!
//! 陷阱继承自 ge-offline-om skill：Data 必设幽灵 input desc 0 与 index
//! attr（C++ 侧已代劳）；build options 必传 input_shape+input_format
//! （C++ 侧 fail-fast）；构图纯 operator 流，禁 AddNodeByOp。

use std::ffi::{c_void, CString};
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use crate::{AclError, AscendStream, DeviceBuffer, Result};

static LIB: OnceLock<std::result::Result<Library, String>> = OnceLock::new();

fn lib() -> std::result::Result<&'static Library, String> {
    LIB.get_or_init(|| {
        let path = std::env::var("APXINF_GE_BUILDER_LIB").unwrap_or_else(|_| {
            "/data/apxinf/ascendc/ge_builder/build/libge_builder.so".into()
        });
        unsafe { Library::new(&path) }.map_err(|e| format!("dlopen {path}: {e}"))
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn cerr(op: &'static str, rc: i32) -> AclError {
    eprintln!("[ge_builder] {op} rc={rc}");
    AclError { code: rc, op }
}

/// Tensor dtype（FFI 侧字符串映射，C++ ParseDtype）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Fp16,
    Fp32,
    Int32,
    Int64,
}

impl Dtype {
    fn as_c(&self) -> CString {
        CString::new(match self {
            Dtype::Fp16 => "fp16",
            Dtype::Fp32 => "fp32",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
        })
        .expect("dtype str")
    }
}

/// 一个 GE 模型槽：构图期句柄与加载后的执行句柄同值（C++ 侧 slot 索引）。
#[derive(Debug, Clone, Copy)]
pub struct GeGraph {
    handle: i64,
}

/// 进程级 GE 构图会话初始化（在 aclInit/SetDevice 之前调用）。
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

/// 会话收尾（卸载全部模型 + aclgrphBuildFinalize）。
pub fn fini() -> Result<()> {
    type F = unsafe extern "C" fn() -> i32;
    call0(b"geb_fini", "geb_fini")
}

/// 装载落盘 OM（不经 GE 编译会话，纯 ACL runtime）。
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
    /// 开一个新构图槽并成为当前槽（add_*/set_*/link 都作用于最近 begin 的槽）。
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

    /// atc 风格编译选项（input_shape 等的直通口）。
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

    /// ND 输入 shape 便捷口：input_format=ND + input_shape="n:d0,d1;..."。
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

    /// Data 输入节点（幽灵 desc 0 + index attr 由 C++ 侧代劳）。
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

    /// 算子节点（GE IR 注册名，如 MatMulV2 / PromptFlashAttention）。
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

    /// dst.SetInput(port, src)——连线登记在 OperatorImpl 双侧，build 时物化。
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

    /// 图输入（Data 节点按绑定顺序）。
    pub fn graph_inputs(&self, names: &[&str]) -> Result<()> {
        self.name_list(b"geb_graph_inputs", "geb_graph_inputs", names)
    }

    /// 图输出（按算子名在物化图中定位）。
    pub fn graph_outputs(&self, names: &[&str]) -> Result<()> {
        self.name_list(b"geb_graph_outputs", "geb_graph_outputs", names)
    }

    /// 物化 + aclgrphBuildModel 内存编译 + 加载（首编秒级~分钟级，产物走 save 缓存）。
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

    /// 模型报告的输入字节数（含对齐，勿自算）。
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

    /// 异步执行（调用方负责 stream sync；dataset 按模型报告大小逐次组装）。
    pub fn run(&self, inputs: &[&DeviceBuffer], outputs: &[&DeviceBuffer],
               stream: &AscendStream) -> Result<()> {
        let ins: Vec<*mut c_void> = inputs.iter().map(|b| b.as_ptr()).collect();
        let outs: Vec<*mut c_void> = outputs.iter().map(|b| b.as_ptr()).collect();
        self.run_raw(&ins, &outs, stream.handle() as *mut c_void)
    }

    /// 同步执行（排障二分变体：语义正常时优先怀疑数据/dtype 而非调度）。
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

    /// OM 落盘（缓存键 = shape 签名 + 引擎版本，C2 定式）。
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
