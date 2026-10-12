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
    // GEB_PRELOAD_EARLY=1（e2e serve 冷启动 lazy）：ckpt 解析（~63s 全量）
    // 与 OM 预载并行——线程持 backend context 引用（进程存续，一次性）
    if std::env::var("GEB_PRELOAD_EARLY").is_ok() && seg == "e2e" {
        // 主线程先激活 context（裸 SetCurrentContext 从未激活的 context 跨
        // 线程会 107000——tl152 实测；激活后 early 线程绑定才成立）
        let _ = be.ctx().set_current();
        let ctx2: &apxinf_ascend::AscendContext = unsafe { std::mem::transmute(be.ctx()) };
        std::thread::spawn(move || {
            apxinf_model::pi05::preload_e2e_oms(ctx2);
        });
    }
    // GEB_CKPT：真 checkpoint（LeRobot safetensors 目录/单文件）。host 解析
    // 管线含 LeRobot 前缀归一、[in,out] 物理转置、Gemma 1+w scale 折叠
    // （g1/g2 变 ones，折叠进 q/k/v/gate/up——probe 图直接消费该约定）
    let real = std::env::var("GEB_CKPT").ok().map(|path| {
        let t0 = std::time::Instant::now();
        // 两段计时（lazy 施工取证）：mmap+Tensor 拷贝 vs from_map 数学
        // （前缀归一/[in,out] 转置/Gemma fold）——决定 stub 化收益落点
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(std::path::Path::new(&path))
            .expect("GEB_CKPT load_native_path");
        let t1 = std::time::Instant::now();
        println!("GEB_CKPT io/copy: {:?}（{} tensors）", t1 - t0, tensors.len());
        if std::env::var("GEB_CKPT_SERVE_MIN").is_ok() {
            // serve 最小消费面（styles 链 + 嵌入表），其余 synthetic 占位——
            // 占位不可入图：bake（GEB_SAVE）路径直接拒绝
            assert!(
                std::env::var("GEB_SAVE").is_err(),
                "GEB_CKPT_SERVE_MIN 占位权重不可入图——bake（GEB_SAVE）必须全量加载"
            );
            let w =
                Pi05Weights::from_map_serve(&Pi05Config::default(), tensors)
                    .expect("GEB_CKPT from_map_serve");
            println!(
                "GEB_CKPT from_map_serve: {:?}（total {:?}）",
                t0.elapsed() - (t1 - t0),
                t0.elapsed()
            );
            w
        } else {
            let w = Pi05Weights::from_map(&Pi05Config::default(), tensors).expect("GEB_CKPT from_map");
            println!("GEB_CKPT from_map: {:?}（total {:?}）", t0.elapsed() - (t1 - t0), t0.elapsed());
            w
        }
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
