//! GE 三段 OM 探针驱动（薄壳）：env 分发 + GEB_CKPT 加载 + ge_builder/ACL
//! 初始化序。全部实现已下沉 apxinf-model/src/pi05/ascend_ge.rs（crate 库面；
//! GeServe 进程内直调门面与 serve_loop 共用同一帧实现）。env 全集与协议
//! 见 ascend_ge.rs 头部注释。
//!
//! 运行（rust 容器，先 source /data/apxinf/rust_env.sh）：
//!   GEB_SEG=vision GEB_DEPTH=2 cargo run --example ge_model_probe --features ascend --release -p apxinf-model
//!   GEB_SEG=e2e GEB_E2E_SERVE=<spool 目录> GEB_TOKENS=144 GEB_CKPT=<ckpt>
//!   GEB_OM_DIR=/data/apxinf/om_cache/tl144（起 serve 桶——supervisor.sh 代劳）

use apxinf_ascend::ge_builder;
use apxinf_ascend::AscendBackend;
use apxinf_model::pi05::{optest, seg_e2e, seg_flow, seg_prefix, seg_vision, Pi05Config, Pi05Weights};

fn main() {
    let seg = std::env::var("GEB_SEG").unwrap_or_else(|_| "prefix".into());
    let bench = std::env::var("GEB_BENCH").is_ok();
    ge_builder::init("Ascend310P3").expect("geb init (before acl runtime)");
    let be = AscendBackend::new(0).expect("be");
    // GEB_CKPT：真 checkpoint（LeRobot safetensors 目录/单文件）。host 解析
    // 管线含 LeRobot 前缀归一、[in,out] 物理转置、Gemma 1+w scale 折叠
    // （g1/g2 变 ones，折叠进 q/k/v/gate/up——probe 图直接消费该约定）
    let real = std::env::var("GEB_CKPT").ok().map(|path| {
        let t0 = std::time::Instant::now();
        let w = Pi05Weights::from_safetensors(&Pi05Config::default(), std::path::Path::new(&path))
            .expect("GEB_CKPT load");
        println!("GEB_CKPT: {path} loaded in {:?}", t0.elapsed());
        w
    });
    if let Ok(t) = std::env::var("GEB_OPTEST") {
        optest(&be, &t, real.as_ref());
        return;
    }
    match seg.as_str() {
        "vision" => seg_vision(&be, bench, real.as_ref(), None),
        "prefix" => seg_prefix(&be, bench, real.as_ref(), None),
        "flow" => seg_flow(&be, bench, real.as_ref(), None),
        "e2e" => seg_e2e(&be, bench, real.as_ref()),
        other => panic!("GEB_SEG: vision|prefix|flow|e2e, got {other}"),
    }
}
