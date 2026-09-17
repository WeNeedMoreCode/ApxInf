//! First real-weight-path smoke for the ascend executor (2026-09-17).
//! Random host weights go through the REAL upload path
//! (Bf16LinearWeights::from_host -> backend.to_device -> F16 on NPU),
//! then one full language layer forward runs on 310P3. Verifies the
//! wiring end to end (shapes, layer composition, no device errors);
//! numeric parity vs the torch path lands with the C-stage runtime.
//!
//!   source /data/apxinf/rust_env.sh
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ascend_layer_smoke --features ascend --release -p apxinf-model

use apxinf_core::{Backend as _, Tensor};
use half::bf16;

use apxinf_model::pi05::{bf16_to_device, AscendCaches, Bf16DeviceLanguageLayer, Bf16LinearWeights, language_layer_ascend, LinearWeights, Pi05Config};

fn rand_host(rows: usize, cols: usize, seed: &mut u32) -> Tensor {
    let mut v = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push(bf16::from_f32(((*seed >> 16) as i32 % 2000 - 1000) as f32 / 2000.0));
    }
    Tensor::from_bf16(vec![rows, cols], &v).unwrap()
}

fn main() {
    let be = apxinf_ascend::AscendBackend::new(0).expect("backend");
    let config = Pi05Config::default();
    let lang = config.language;
    println!(
        "language: heads={} kv={} head_dim={} width={}",
        lang.num_heads, lang.num_kv_heads, lang.head_dim, lang.head_dim * lang.num_heads
    );

    let width = lang.num_heads * lang.head_dim;
    let kv_width = lang.num_kv_heads * lang.head_dim;
    let qkv_out = width + 2 * kv_width;
    let inter = width * 4; // gemma-ish intermediate
    let mut seed = 0x1234u32;

    let lin = |rows: usize, cols: usize, seed: &mut u32| -> Bf16LinearWeights {
        Bf16LinearWeights::from_host(
            &LinearWeights {
                weight: rand_host(rows, cols, seed),
                bias: Some(rand_host(1, cols, seed).reshape(vec![cols]).unwrap()),
            },
            &be,
        )
        .unwrap()
    };

    let layer = Bf16DeviceLanguageLayer {
        input_norm_scale: bf16_to_device(&rand_host(1, width, &mut seed).reshape(vec![width]).unwrap(), &be).unwrap(),
        qkv: lin(width, qkv_out, &mut seed),
        output: lin(width, width, &mut seed),
        post_attention_norm_scale: bf16_to_device(&rand_host(1, width, &mut seed).reshape(vec![width]).unwrap(), &be).unwrap(),
        gate_up: lin(width, inter * 2, &mut seed),
        down: lin(inter, width, &mut seed),
    };
    println!("weights uploaded (bf16 host -> f16 device via from_host)");

    let tokens = 8usize;
    let input = be.to_device(&rand_host(tokens, width, &mut seed)).unwrap();
    println!("input on device: {:?} {:?}", input.shape().dims(), input.dtype());

    let mut cache = AscendCaches::new();
    let t0 = std::time::Instant::now();
    let out = language_layer_ascend(
        &be, &mut cache, lang, &layer, &input, true, 0, 1e-6, config.rope_theta,
    )
    .expect("language layer forward");
    be.synchronize().expect("sync");
    println!("one language layer forward in {:?} (first call incl. op warmup)", t0.elapsed());

    let hidden = be.to_cpu(&out.hidden).expect("to_cpu");
    let vals = hidden.to_f32_vec().unwrap();
    let finite = vals.iter().filter(|v| v.is_finite()).count();
    println!(
        "hidden: {:?} finite={}/{} max_abs={:.4}",
        hidden.shape().dims(),
        finite,
        vals.len(),
        vals.iter().fold(0f32, |m, v| m.max(v.abs()))
    );
    assert_eq!(finite, vals.len(), "non-finite values in hidden");
    assert_eq!(hidden.shape().dims(), &[tokens, width]);
    let kv_tokens = out.key.shape().dims()[0];
    assert_eq!(kv_tokens, tokens);
    println!("ASCEND_LAYER_SMOKE_OK");
}
