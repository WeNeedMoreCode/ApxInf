//! Link the Ascend CL runtime (libascendcl.so).
//!
//! Toolkit location: $ASCEND_TOOLKIT_HOME (lib64/) or the standard
//! /usr/local/Ascend/ascend-toolkit/latest install. Override with
//! ASCEND_TOOLKIT_HOME when the toolkit lives elsewhere.
//!
//! Only final artifacts (examples/bins/tests) actually link ascendcl; rlib
//! builds pass through untouched, so non-Ascend machines can still build.

use std::path::PathBuf;

fn main() {
    let home = std::env::var_os("ASCEND_TOOLKIT_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/Ascend/ascend-toolkit/latest"));
    println!("cargo:rustc-link-search=native={}", home.join("lib64").display());
    println!("cargo:rustc-link-lib=dylib=ascendcl");
    println!("cargo:rerun-if-env-changed=ASCEND_TOOLKIT_HOME");
}
