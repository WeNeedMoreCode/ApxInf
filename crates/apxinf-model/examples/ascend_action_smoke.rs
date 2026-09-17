//! Action-expert miniature smoke: replicate denoise step 0 in isolation
//! (styles + action_in + action layers + action_out + euler over a fake
//! prefix KV) to corner the in-graph down-matmul fault into a
//! self-contained reproducer.
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ascend_action_smoke --features ascend --release -p apxinf-model
use std::sync::Arc;

use apxinf_core::{Backend as _, Tensor};
use half::bf16;

use apxinf_model::pi05::{
    action_layer_ascend, AscendCaches, Bf16DeviceActionLayer, GemmaVariantConfig, LinearWeights,
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

fn main() {
    let be = Arc::new(apxinf_ascend::AscendBackend::new(0).expect("backend"));
    let config = GemmaVariantConfig { depth: 2, ..GemmaVariantConfig::GEMMA_300M };
    let width = config.width; // 1024
    let qd = config.num_heads * config.head_dim; // 2048
    let kvd = config.num_kv_heads * config.head_dim; // 256
    let inter = config.mlp_dim; // 4096
    let tokens = match std::env::var("APXINF_ACTION_TOKENS").as_deref() {
        Ok(v) => v.parse().unwrap(),
        Err(_) => 50usize,
    };
    let prefix_tokens = 828usize; // 768 vision + 60 text
    let action_dim = 32usize;

    let mut seed = 0x9e37u32;
    let lin = |rows: usize, cols: usize, seed: &mut u32, bias: bool| {
        apxinf_model::pi05::Bf16LinearWeights::from_host(&rand_linear(rows, cols, seed, bias), be.as_ref())
            .unwrap()
    };
    let mk_layer = |seed: &mut u32| Bf16DeviceActionLayer {
        input_style: lin(width, width, seed, true),
        // fused [q; k; v] along the output dim, matching from_host_parts
        qkv: lin(width, qd + 2 * kvd, seed, true),
        output: lin(qd, width, seed, true),
        post_attention_style: lin(width, width, seed, true),
        // fused [gate; up]
        gate_up: lin(width, inter * 2, seed, false),
        down: lin(inter, width, seed, false),
    };
    let layers: Vec<Bf16DeviceActionLayer> = (0..config.depth)
        .map(|_| mk_layer(&mut seed))
        .collect();
    let final_style = {
        let w = rand_linear(width, width, &mut seed, true);
        apxinf_model::pi05::Bf16LinearWeights::from_host(&w, be.as_ref()).unwrap()
    };
    let action_in = {
        let w = rand_linear(action_dim, width, &mut seed, true);
        apxinf_model::pi05::Bf16LinearWeights::from_host(&w, be.as_ref()).unwrap()
    };
    let action_out = {
        let w = rand_linear(width, action_dim, &mut seed, true);
        apxinf_model::pi05::Bf16LinearWeights::from_host(&w, be.as_ref()).unwrap()
    };
    println!("action weights on device");

    let state = be.to_device(&rand_host(tokens, action_dim, &mut seed)).unwrap();
    let prefix_k = be.to_device(&rand_host(prefix_tokens, kvd, &mut seed)).unwrap();
    let prefix_v = be.to_device(&rand_host(prefix_tokens, kvd, &mut seed)).unwrap();
    let styles: Vec<Tensor> = (0..2 * config.depth + 1)
        .map(|_| be.to_device(&rand_host(1, width, &mut seed).reshape(vec![width]).unwrap()).unwrap())
        .collect();

    let mut cache = AscendCaches::new();
    let trace = std::env::var("APXINF_ASCEND_TRACE").is_ok();
    let mut hidden = be.to_device(&rand_host(tokens, width, &mut seed)).unwrap();
    let _ = &state;
    let _ = &action_in;
    if trace {
        be.synchronize().unwrap();
        eprintln!("[trace] hidden staged");
    }
    let mut attention_normalized: Option<Tensor> = None;
    for index in 0..config.depth {
        let next_norm_style = if index + 1 < config.depth {
            &styles[index + 1]
        } else {
            &styles[2 * config.depth]
        };
        let out = action_layer_ascend(
            &be,
            &mut cache,
            config,
            &layers[index],
            &hidden,
            attention_normalized.as_ref(),
            &styles[index],                 // attention style
            &styles[config.depth + index],  // mlp style
            next_norm_style,
            &prefix_k,
            &prefix_v,
            prefix_tokens,
            1e-6,
            10_000.0,
        )
        .unwrap_or_else(|e| panic!("action layer {index}: {e:?}"));
        hidden = out.hidden;
        attention_normalized = Some(out.next_normalized);
        if trace {
            be.synchronize().unwrap();
            eprintln!("[trace] action layer {index} done");
        }
    }
    be.synchronize().unwrap();
    println!("both action layers OK");
    let _ = final_style;
    let _ = action_out;
    println!("ASCEND_ACTION_SMOKE_OK");
}
