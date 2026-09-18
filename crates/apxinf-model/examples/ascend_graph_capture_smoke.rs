//! ACLGraph capture smoke (perf phase, 2026-09-18): warm the caches
//! eagerly, capture a full inference into a graph with all allocations
//! bumping from one arena, then replay twice and diff against the eager
//! output. Depth 2/2/2 keeps the turn-around fast.
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ascend_graph_capture_smoke --features ascend --release -p apxinf-model
use std::sync::Arc;

use apxinf_core::{Backend as _, Graph as _, Tensor};
use half::bf16;

use apxinf_model::pi05::{
    Pi05AscendRuntime, Pi05Config, Pi05Weights, StaticBf16Pi05Weights, GemmaVariantConfig,
};

fn rand_host(rows: usize, cols: usize, seed: &mut u32) -> Tensor {
    let mut v = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push(bf16::from_f32(((*seed >> 16) as i32 % 200 - 100) as f32 / 400.0));
    }
    Tensor::from_bf16(vec![rows, cols], &v).unwrap()
}

fn main() {
    let be = Arc::new(apxinf_ascend::AscendBackend::new(0).expect("backend"));
    let mut config = Pi05Config::default();
    config.vision_depth = 2;
    config.vocab_size = 2048;
    config.language = GemmaVariantConfig { depth: 2, ..GemmaVariantConfig::GEMMA_2B };
    config.action_expert = GemmaVariantConfig { depth: 2, ..GemmaVariantConfig::GEMMA_300M };
    let host_weights = Pi05Weights::synthetic(&config, 1234).expect("synthetic weights");
    let weights = Arc::new(StaticBf16Pi05Weights::from_host(&host_weights, be.as_ref(), false).unwrap());
    let runtime = Pi05AscendRuntime::new(be.clone(), Arc::new(config.clone()), weights).unwrap();

    let patch_rows = config.num_views * config.patches_per_view();
    let patch_w = 3 * config.patch_size * config.patch_size;
    let mut seed = 7u32;
    let patches = rand_host(patch_rows, patch_w, &mut seed);
    let patches_d = be.to_device(&patches).unwrap();
    let token_ids: Vec<u32> = (0..60u32).map(|i| (i * 37) % config.vocab_size as u32).collect();
    // fixed noise (deterministic input; replay output must match eager)
    let noise = be.to_device(&rand_host(config.action_horizon, config.action_dim, &mut seed)).unwrap();
    let time_embeddings = (0..config.num_flow_steps)
        .map(|step| {
            let time = config.flow_start_time * (1.0 - step as f32 / config.num_flow_steps as f32);
            let values = apxinf_model::pi05::sinusoidal_time_embedding(
                time, config.action_expert.width, config.time_min_period, config.time_max_period,
            )
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
            be.to_device(&Tensor::from_bf16(vec![1, config.action_expert.width], &values).unwrap()).unwrap()
        })
        .collect::<Vec<_>>();

    // 1) eager warmup x2 (fills nz/rope/style/kv-bias/pos-idx caches)
    let eager = runtime.infer(&patches_d, &token_ids, &noise, &time_embeddings).expect("eager");
    be.synchronize().expect("sync");
    let eager2 = runtime.infer(&patches_d, &token_ids, &noise, &time_embeddings).expect("eager2");
    be.synchronize().expect("sync");
    let mut eager_h = vec![0f32; eager2.numel()];
    {
        let host = be.to_cpu(&eager2).unwrap();
        eager_h.copy_from_slice(&host.to_f32_vec().unwrap());
    }
    drop(eager);
    drop(eager2);
    be.synchronize().unwrap();
    apxinf_ascend::flush_pending_frees();
    println!("warmup done");

    // 2) styles precomputed OUTSIDE the window (their first-use row
    //    materialization is host work); pre-warm them + arena capture
    let styles = runtime.prepare_all_styles(&time_embeddings).expect("styles");
    let _prewarm = runtime
        .infer_with_styles(&patches_d, &token_ids, &noise, &styles)
        .expect("style prewarm");
    be.synchronize().unwrap();
    drop(_prewarm);
    apxinf_ascend::flush_pending_frees();

    const ARENA: usize = 2 * 1024 * 1024 * 1024;
    let arena_owner = be.ctx().enter_arena(ARENA).expect("arena");
    be.begin_capture().expect("begin capture");
    let captured = match runtime.infer_with_styles(&patches_d, &token_ids, &noise, &styles) {
        Ok(t) => t,
        Err(e) => {
            let _ = be.end_capture();
            be.ctx().clear_arena();
            panic!("capture-window infer failed: {e:?}");
        }
    };
    let graph = be.end_capture().expect("end capture");
    let used = be.ctx().exit_arena();
    be.ctx().clear_arena();
    println!("captured graph built (arena used {used} bytes)");

    // 3) replay: warm once, then time it and diff against eager
    let mut timings = Vec::new();
    for i in 0..12 {
        let t0 = std::time::Instant::now();
        graph.replay().expect("replay");
        be.synchronize().expect("sync after replay");
        if i >= 2 {
            timings.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let host = be.to_cpu(&captured).unwrap();
        let vals = host.to_f32_vec().unwrap();
        let finite = vals.iter().filter(|v| v.is_finite()).count();
        let max_diff = vals.iter().zip(&eager_h).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        if i < 2 {
            println!("replay {i}: finite={}/{} max_diff_vs_eager={max_diff:.6}", finite, vals.len());
            assert_eq!(finite, vals.len());
            assert!(max_diff < 0.05, "replay diverged from eager: {max_diff}");
        }
    }
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // eager comparison timing (same shapes, caches warm)
    let t0 = std::time::Instant::now();
    let eager3 = runtime.infer_with_styles(&patches_d, &token_ids, &noise, &styles).expect("eager3");
    be.synchronize().unwrap();
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0;
    drop(eager3);
    println!(
        "REPLAY_BENCH: graph p50={:.1}ms (n={}) vs eager {:.1}ms -- {:.2}x",
        timings[timings.len() / 2],
        timings.len(),
        eager_ms,
        eager_ms / timings[timings.len() / 2]
    );
    drop(graph);
    drop(captured);
    drop(arena_owner); // frees the arena backing once the graph is gone
    println!("ASCEND_GRAPH_CAPTURE_SMOKE_OK");
}
