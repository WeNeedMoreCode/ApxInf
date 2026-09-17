//! Random-weights π0.5 bench through the VlaRuntime trait (D+F stage,
//! 2026-09-18). Depth-reduced weights, real widths; eager path; measures
//! per-call latency of the full model forward (vision + prefix + 10 flow
//! steps) end to end.
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ascend_random_bench --features ascend --release -p apxinf-model
use std::sync::Arc;

use apxinf_core::{Backend as _, Tensor};
use half::bf16;

use apxinf_model::vla::{InitialLatent, Observation, VlaRequest, VlaRuntime, VisionObservation};
use apxinf_model::pi05::{
    Pi05AscendRuntime, Pi05AscendVlaRuntime, Pi05Config, Pi05Weights, StaticBf16Pi05Weights,
    GemmaVariantConfig,
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

    // random weights via the engine's own synthetic generator
    let host_weights = Pi05Weights::synthetic(&config, 1234).expect("synthetic weights");
    let weights = Arc::new(
        StaticBf16Pi05Weights::from_host(&host_weights, be.as_ref(), false).expect("upload"),
    );
    let runtime = Pi05AscendRuntime::new(be.clone(), Arc::new(config.clone()), weights).expect("runtime");
    let vla = Pi05AscendVlaRuntime::new(be.clone(), Arc::new(config.clone()), runtime).expect("vla");

    let patch_rows = config.num_views * config.patches_per_view();
    let patch_w = 3 * config.patch_size * config.patch_size;
    let mut seed = 7u32;
    let patches = rand_host(patch_rows, patch_w, &mut seed);
    let token_ids: Vec<u32> = (0..60u32).map(|i| (i * 37) % config.vocab_size as u32).collect();
    let observation = Observation {
        vision: VisionObservation::Patches(patches),
        token_ids,
        state: None,
        action_mask: None,
    };

    let prepared = vla.prepare(&observation.inference_spec()).expect("prepare");
    let rng = apxinf_core::RngKey::new(1, 2, 3);

    // warmup (fills weight-transpose caches)
    for _ in 0..3 {
        let request = VlaRequest { observation: &observation, initial_latent: InitialLatent::Generate { rng } };
        prepared.run(&request).expect("warmup run");
    }
    be.synchronize().expect("sync");

    // timed samples
    let samples = 20;
    let mut times = Vec::with_capacity(samples);
    for i in 0..samples {
        let request = VlaRequest { observation: &observation, initial_latent: InitialLatent::Generate { rng } };
        let t0 = std::time::Instant::now();
        let action = prepared.run(&request).expect("bench run");
        be.synchronize().expect("sync");
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        if i == 0 {
            let host = be.to_cpu(action.tensor()).expect("to_cpu");
            let vals = host.to_f32_vec().unwrap();
            let finite = vals.iter().filter(|v| v.is_finite()).count();
            println!("sanity: {:?} finite={}/{}", host.shape().dims(), finite, vals.len());
            assert_eq!(finite, vals.len());
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[times.len() / 2];
    let p90 = times[times.len() * 9 / 10];
    println!(
        "ASCEND_RANDOM_BENCH: n={samples} p50={p50:.1}ms p90={p90:.1}ms min={:.1}ms max={:.1}ms (depth 2/2/2, real widths)",
        times[0], times[times.len() - 1]
    );
}
