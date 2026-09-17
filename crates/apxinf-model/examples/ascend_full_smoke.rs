//! Full random-weights π0.5 forward through Pi05AscendRuntime (2026-09-18).
//! Depth-reduced (vision 2 / language 2 / action 2) with REAL widths so
//! every node class executes on device: wide gate_up matmuls (N=32768),
//! batched per-view vision attention, cross-length prefix attention,
//! ada-norm styles, and the flow-denoise loop.
//!
//!   source /data/apxinf/rust_env.sh
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ascend_full_smoke --features ascend --release -p apxinf-model
use std::sync::Arc;

use apxinf_core::{Backend as _, Tensor};
use half::bf16;

use apxinf_model::pi05::{
    sinusoidal_time_embedding, ActionLayerWeights, AdaRmsNormWeights, GemmaAttentionWeights,
    GemmaMlpWeights, GemmaVariantConfig, LanguageLayerWeights, LayerNormWeights, LinearWeights,
    Pi05AscendRuntime, Pi05Config, Pi05Weights, StaticBf16Pi05Weights, VisionBlockWeights,
    VisionWeights,
};

fn rand_host(rows: usize, cols: usize, seed: &mut u32) -> Tensor {
    let mut v = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push(bf16::from_f32(((*seed >> 16) as i32 % 200 - 100) as f32 / 400.0));
    }
    Tensor::from_bf16(vec![rows, cols], &v).unwrap()
}

fn rand_linear(rows: usize, cols: usize, seed: &mut u32, bias: bool) -> LinearWeights {
    LinearWeights {
        weight: rand_host(rows, cols, seed),
        bias: bias.then(|| rand_host(1, cols, seed).reshape(vec![cols]).unwrap()),
    }
}

fn rand_ln(cols: usize, seed: &mut u32) -> LayerNormWeights {
    LayerNormWeights {
        weight: rand_host(1, cols, seed).reshape(vec![cols]).unwrap(),
        bias: rand_host(1, cols, seed).reshape(vec![cols]).unwrap(),
    }
}

fn rand_ada(cols: usize, seed: &mut u32) -> AdaRmsNormWeights {
    AdaRmsNormWeights { style: rand_linear(cols, cols, seed, true) }
}

fn rand_attn(width: usize, qd: usize, kvd: usize, seed: &mut u32) -> GemmaAttentionWeights {
    GemmaAttentionWeights {
        q: rand_linear(width, qd, seed, true),
        k: rand_linear(width, kvd, seed, true),
        v: rand_linear(width, kvd, seed, true),
        output: rand_linear(qd, width, seed, true),
    }
}

fn rand_mlp(width: usize, inter: usize, seed: &mut u32) -> GemmaMlpWeights {
    GemmaMlpWeights {
        gate: rand_linear(width, inter, seed, false),
        up: rand_linear(width, inter, seed, false),
        down: rand_linear(inter, width, seed, false),
    }
}

fn main() {
    let be = Arc::new(apxinf_ascend::AscendBackend::new(0).expect("backend"));
    let mut config = Pi05Config::default();
    config.vision_depth = 2;
    config.vocab_size = 2048; // random-weights smoke: tiny fake vocab
    config.language = GemmaVariantConfig { depth: 2, ..GemmaVariantConfig::GEMMA_2B };
    config.action_expert = GemmaVariantConfig { depth: 2, ..GemmaVariantConfig::GEMMA_300M };

    let vw = config.vision_width; // 1152
    let vinter = config.vision_mlp_dim; // 4304
    let vqd = config.vision_heads * config.vision_head_dim; // 1152
    let patch_w = 3 * config.patch_size * config.patch_size; // 588
    let tpv = config.patches_per_view(); // 256
    let lw = config.language.width; // 2048
    let lqd = config.language.num_heads * config.language.head_dim; // 2048
    let lkvd = config.language.num_kv_heads * config.language.head_dim; // 256
    let linter = config.language.mlp_dim; // 16384
    let aw = config.action_expert.width; // 1024
    let aqd = config.action_expert.num_heads * config.action_expert.head_dim; // 2048
    let akvd = config.action_expert.num_kv_heads * config.action_expert.head_dim; // 256
    let ainter = config.action_expert.mlp_dim; // 4096

    let mut seed = 0xc0ffeeu32;
    let vision = VisionWeights {
        patch_embedding: rand_linear(patch_w, vw, &mut seed, true),
        position_embedding: rand_host(tpv, vw, &mut seed),
        blocks: (0..config.vision_depth)
            .map(|_| VisionBlockWeights {
                norm1: rand_ln(vw, &mut seed),
                q: rand_linear(vw, vqd, &mut seed, true),
                k: rand_linear(vw, vqd, &mut seed, true),
                v: rand_linear(vw, vqd, &mut seed, true),
                output: rand_linear(vqd, vw, &mut seed, true),
                norm2: rand_ln(vw, &mut seed),
                fc1: rand_linear(vw, vinter, &mut seed, true),
                fc2: rand_linear(vinter, vw, &mut seed, true),
            })
            .collect(),
        post_layer_norm: rand_ln(vw, &mut seed),
        multimodal_projector: rand_linear(vw, lw, &mut seed, true),
        token_embedding: rand_host(config.vocab_size, lw, &mut seed),
    };
    let language_layers = (0..config.language.depth)
        .map(|_| LanguageLayerWeights {
            input_norm_scale: rand_host(1, lw, &mut seed).reshape(vec![lw]).unwrap(),
            attention: rand_attn(lw, lqd, lkvd, &mut seed),
            post_attention_norm_scale: rand_host(1, lw, &mut seed).reshape(vec![lw]).unwrap(),
            mlp: rand_mlp(lw, linter, &mut seed),
        })
        .collect();
    let action_layers = (0..config.action_expert.depth)
        .map(|_| ActionLayerWeights {
            input_norm: rand_ada(aw, &mut seed),
            attention: rand_attn(aw, aqd, akvd, &mut seed),
            post_attention_norm: rand_ada(aw, &mut seed),
            mlp: rand_mlp(aw, ainter, &mut seed),
        })
        .collect();
    let host_weights = Pi05Weights {
        vision,
        language_layers,
        language_final_norm_scale: rand_host(1, lw, &mut seed).reshape(vec![lw]).unwrap(),
        action_layers,
        action_final_norm: rand_ada(aw, &mut seed),
        action_in: rand_linear(config.action_dim, aw, &mut seed, true),
        action_out: rand_linear(aw, config.action_dim, &mut seed, true),
        time_mlp_in: rand_linear(aw, aw, &mut seed, true),
        time_mlp_out: rand_linear(aw, aw, &mut seed, true),
    };
    println!("random host weights built (depth-reduced, real widths)");

    let weights = Arc::new(
        StaticBf16Pi05Weights::from_host(&host_weights, be.as_ref(), false).expect("weight upload"),
    );
    println!("weights on device (bf16 host -> f16 via to_device)");

    let runtime = Pi05AscendRuntime::new(be.clone(), Arc::new(config.clone()), weights)
        .expect("runtime");

    let patch_rows = config.num_views * tpv;
    let patches = be
        .to_device(&rand_host(patch_rows, patch_w, &mut seed))
        .expect("patches");
    let token_ids: Vec<u32> = (0..60u32).map(|i| (i * 37) % config.vocab_size as u32).collect();
    let noise = be
        .to_device(&rand_host(config.action_horizon, config.action_dim, &mut seed))
        .expect("noise");
    // inline upload_time_embeddings_bf16 (its lib definition sits behind
    // the cuda feature; the math is host-side and backend-agnostic)
    let time_embeddings = (0..config.num_flow_steps)
        .map(|step| {
            let time =
                config.flow_start_time * (1.0 - step as f32 / config.num_flow_steps as f32);
            let values = sinusoidal_time_embedding(
                time,
                config.action_expert.width,
                config.time_min_period,
                config.time_max_period,
            )
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
            be.to_device(&Tensor::from_bf16(vec![1, config.action_expert.width], &values).unwrap())
                .expect("time embedding upload")
        })
        .collect::<Vec<_>>();

    let t0 = std::time::Instant::now();
    // stage-by-stage with syncs between -- first fault names its stage
    let vision = runtime.encode_vision(&patches).expect("encode_vision");
    be.synchronize().expect("sync vision");
    println!("stage vision ok {:?}", t0.elapsed());
    let prefix_emb = runtime.embed_prefix(&vision, &token_ids).expect("embed_prefix");
    be.synchronize().expect("sync embed");
    println!("stage embed ok {:?}", t0.elapsed());
    let prefix = runtime.prefix_forward(&prefix_emb).expect("prefix_forward");
    be.synchronize().expect("sync prefix");
    println!("stage prefix ok {:?}", t0.elapsed());
    let dt = -config.flow_start_time / config.num_flow_steps as f32;
    let mut state = noise.clone();
    for (step, time_embedding) in time_embeddings.iter().enumerate() {
        state = runtime
            .denoise_step(&state, time_embedding, &prefix, dt)
            .unwrap_or_else(|e| panic!("denoise step {step}: {e:?}"));
        be.synchronize().unwrap_or_else(|e| panic!("sync step {step}: {e:?}"));
        println!("denoise step {step} ok {:?}", t0.elapsed());
    }
    let out = state;
    be.synchronize().expect("sync");
    println!("full random-weights infer in {:?} (first call incl. caches)", t0.elapsed());

    let hidden = be.to_cpu(&out).expect("to_cpu");
    let vals = hidden.to_f32_vec().unwrap();
    let finite = vals.iter().filter(|v| v.is_finite()).count();
    println!(
        "actions: {:?} finite={}/{} max_abs={:.4}",
        hidden.shape().dims(),
        finite,
        vals.len(),
        vals.iter().fold(0f32, |m, v| m.max(v.abs()))
    );
    assert_eq!(finite, vals.len(), "non-finite values in actions");
    assert_eq!(hidden.shape().dims(), &[config.action_horizon, config.action_dim]);
    println!("ASCEND_FULL_SMOKE_OK");
}
