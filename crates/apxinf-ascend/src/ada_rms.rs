//! Runtime-dlopen bridge to the AscendC fused ada-rms-norm kernel
//! (ascendc/ada_rms_norm/): y[r,:] = x[r,:]/rms(x[r,:]) * scale[:] +
//! shift[:] in ONE kernel. Replaces the add_rms_norm(x,zeros)+mul+add
//! composition that msprof pinned at 2 of 3 big-matrix kernels per
//! ada-norm call (~400 calls per full-depth inference).
//!
//! The host wrapper `.so` is an aarch64 build artifact (compiled on the
//! server via run.sh); loading is lazy and env-overridable so non-Ascend
//! machines still compile the crate.

use std::ffi::c_void;
use std::sync::OnceLock;

use libloading::{Library, Symbol};

use crate::{AclError, Result};

type RunFn = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    i32,
    i32,
    f32,
    i32,
    *mut c_void,
    i32,
) -> i32;

static LIB: OnceLock<std::result::Result<Library, String>> = OnceLock::new();

fn lib() -> std::result::Result<&'static Library, String> {
    LIB.get_or_init(|| {
        let path = std::env::var("APXINF_ADA_RMS_LIB").unwrap_or_else(|_| {
            "/data/apxinf/ascendc/ada_rms_norm/out/lib/libada_rms_norm_host.so".into()
        });
        unsafe { Library::new(&path) }.map_err(|e| format!("dlopen {path}: {e}"))
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Whether the fused kernel is loadable (feature-gates the call site).
pub fn available() -> bool {
    lib().is_ok()
}

/// Run the fused kernel. `x`/`y` are [rows, cols] fp16 device buffers;
/// `scale`/`shift` are [cols] fp16 rows (scale = 1+style[0:w] folded
/// host-side). `diag` 1/2 are kernel bring-up passthrough probes.
/// Streams-only: no sync inside, capture-window safe.
#[allow(clippy::too_many_arguments)]
pub fn run(
    stream: &crate::AscendStream,
    x: &crate::DeviceBuffer,
    scale: &crate::DeviceBuffer,
    shift: &crate::DeviceBuffer,
    y: &crate::DeviceBuffer,
    rows: i32,
    cols: i32,
    eps: f32,
    cores: i32,
    diag: i32,
) -> Result<()> {
    let lib = lib().map_err(|e| {
        eprintln!("[ada_rms] {e}");
        AclError { code: -1, op: "ada_rms load" }
    })?;
    let run: Symbol<RunFn> = unsafe { lib.get(b"ada_rms_norm_run") }.map_err(|e| {
        eprintln!("[ada_rms] dlsym ada_rms_norm_run: {e}");
        AclError { code: -1, op: "ada_rms dlsym" }
    })?;
    let rc = unsafe {
        run(
            x.as_ptr(),
            scale.as_ptr(),
            shift.as_ptr(),
            y.as_ptr(),
            rows,
            cols,
            eps,
            cores,
            stream.handle(),
            diag,
        )
    };
    if rc != 0 {
        return Err(AclError { code: rc, op: "ada_rms_norm_run" });
    }
    Ok(())
}
