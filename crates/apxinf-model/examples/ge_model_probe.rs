//! GE three-segment OM probe driver (thin shell): env dispatch +
//! GEB_CKPT loading + ge_builder/ACL init ordering. All implementation
//! lives in apxinf-model/src/pi05/ascend_ge.rs (the GeServe in-process
//! facade and serve_loop share one frame implementation). The full env
//! set and protocol are documented at the top of ascend_ge.rs.
//!
//! Run (rust toolchain under a CANN toolkit env):
//!   GEB_SEG=vision GEB_DEPTH=2 cargo run --example ge_model_probe --features ascend --release -p apxinf-model
//!   GEB_SEG=e2e GEB_E2E_SERVE=<spool dir> GEB_TOKENS=144 GEB_CKPT=<ckpt>
//!   GEB_OM_DIR=<om_cache>/tl144 (starts a serve bucket — the
//!   supervisor script does this)

use apxinf_ascend::ge_builder;
use apxinf_ascend::AscendBackend;
use apxinf_model::pi05::{seg_e2e, seg_flow, seg_prefix, seg_vision, Pi05Config, Pi05Weights};

fn main() {
    let seg = std::env::var("GEB_SEG").unwrap_or_else(|_| "prefix".into());
    let bench = std::env::var("GEB_BENCH").is_ok();
    ge_builder::init("Ascend310P3").expect("geb init (before acl runtime)");
    let be = AscendBackend::new(0).expect("be");
    // GEB_PRELOAD_EARLY=1 (e2e serve cold-start lazy): ckpt parsing
    // (~63s full) runs in parallel with the OM preload — the thread
    // holds a backend context reference (process-lifetime, one-shot).
    if std::env::var("GEB_PRELOAD_EARLY").is_ok() && seg == "e2e" {
        // Activate the context on the main thread first (a bare
        // SetCurrentContext on a never-activated context fails with
        // 107000 across threads — measured on tl152; only after
        // activation does the early thread's bind hold).
        let _ = be.ctx().set_current();
        let ctx2: &apxinf_ascend::AscendContext = unsafe { std::mem::transmute(be.ctx()) };
        std::thread::spawn(move || {
            apxinf_model::pi05::preload_e2e_oms(ctx2);
        });
    }
    // GEB_CKPT: real checkpoint (LeRobot safetensors dir or single
    // file). The host parse pipeline does LeRobot prefix normalization,
    // [in,out] physical transposes, and the Gemma 1+w scale fold (g1/g2
    // become ones, folded into q/k/v/gate/up — the probe graph consumes
    // that convention as-is).
    let real = std::env::var("GEB_CKPT").ok().map(|path| {
        let t0 = std::time::Instant::now();
        // Two-phase timing: mmap+Tensor copy vs the from_map math
        // (normalization/transposes/fold) — locates the stub win.
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(std::path::Path::new(&path))
            .expect("GEB_CKPT load_native_path");
        let t1 = std::time::Instant::now();
        println!("GEB_CKPT io/copy: {:?} ({} tensors)", t1 - t0, tensors.len());
        if std::env::var("GEB_CKPT_SERVE_MIN").is_ok() {
            // serve minimal face (styles chain + embedding table),
            // everything else synthetic — placeholders cannot enter the
            // graph: the bake (GEB_SAVE) path refuses outright.
            assert!(
                std::env::var("GEB_SAVE").is_err(),
                "GEB_CKPT_SERVE_MIN placeholder weights cannot be baked — GEB_SAVE requires the full load"
            );
            let w =
                Pi05Weights::from_map_serve(&Pi05Config::default(), tensors)
                    .expect("GEB_CKPT from_map_serve");
            println!(
                "GEB_CKPT from_map_serve: {:?} (total {:?})",
                t0.elapsed() - (t1 - t0),
                t0.elapsed()
            );
            w
        } else {
            let w = Pi05Weights::from_map(&Pi05Config::default(), tensors).expect("GEB_CKPT from_map");
            println!("GEB_CKPT from_map: {:?} (total {:?})", t0.elapsed() - (t1 - t0), t0.elapsed());
            w
        }
    });
    match seg.as_str() {
        "vision" => seg_vision(&be, real.as_ref(), None),
        "prefix" => seg_prefix(&be, real.as_ref(), None),
        "flow" => seg_flow(&be, real.as_ref(), None),
        "e2e" => seg_e2e(&be, bench, real.as_ref()),
        other => panic!("GEB_SEG: vision|prefix|flow|e2e, got {other}"),
    }
}
