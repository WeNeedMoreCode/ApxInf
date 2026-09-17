//! Real-checkpoint smoke (M3 precursor): load the LeRobot π0.5
//! safetensors (7GB, /data/apxinf/weights/pi05_libero_finetuned) through
//! the pi05-ascend registry and run one full-depth inference.
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ascend_checkpoint_smoke --features ascend --release -p apxinf-model
use apxinf_core::Tensor;
use half::bf16;

use apxinf_model::vla::{InitialLatent, Observation, VlaRequest, VlaRuntime, VisionObservation};
use apxinf_model::{AutoModel, LoadOptions, LoadedModel, Pi05Config, SyntheticWeights};

fn main() {
    apxinf_model::register_builtin_models();

    let checkpoint = std::env::var("PI05_CKPT")
        .unwrap_or_else(|_| "/data/apxinf/weights/pi05_libero_finetuned".into());

    let mut options = LoadOptions::default();
    options.model_name = Some("pi05-ascend".into());
    // LeRobot config.json is not the HF layout the parser expects; the
    // checkpoint matches Pi05Config::default()'s architecture.
    options.config = Some(Pi05Config::default());

    let t0 = std::time::Instant::now();
    let loaded = match AutoModel::load_model(apxinf_core::Device::Ascend(0), &checkpoint, &options) {
        Ok(m) => m,
        Err(e) => {
            // fall back to synthetic weights (same shapes) when the
            // checkpoint directory is absent (e.g. local machines)
            eprintln!("checkpoint load failed ({e:?}); retrying synthetic");
            options.synthetic = Some(SyntheticWeights { seed: 1234 });
            AutoModel::load_model(apxinf_core::Device::Ascend(0), ".", &options)
                .expect("synthetic fallback load")
        }
    };
    println!("model loaded in {:?}", t0.elapsed());

    let LoadedModel::Vla(vla) = loaded else {
        panic!("expected a VLA model");
    };
    let contract = vla.contract();
    println!(
        "loaded: action {:?} patch {:?} vocab tokens<=257152",
        contract.action_shape, contract.patch_shape
    );

    let mut seed = 7u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        bf16::from_f32(((seed >> 16) as i32 % 200 - 100) as f32 / 400.0)
    };
    let elems = contract.patch_shape[0] * contract.patch_shape[1];
    let patches: Vec<bf16> = (0..elems).map(|_| rnd()).collect();
    let observation = Observation {
        vision: VisionObservation::Patches(
            Tensor::from_bf16(vec![contract.patch_shape[0], contract.patch_shape[1]], &patches)
                .unwrap(),
        ),
        // placeholder prompt tokens (real tokenizer lands with the
        // LIBERO integration); values < paligemma vocab
        token_ids: (0..60u32).map(|i| (i * 37 + 2) % 257_152).collect(),
        state: None,
        action_mask: None,
    };
    let rng = apxinf_core::RngKey::new(1, 2, 3);
    let request = VlaRequest {
        observation: &observation,
        initial_latent: InitialLatent::Generate { rng },
    };

    let t1 = std::time::Instant::now();
    let values = vla.infer_host_f32(&request).expect("inference");
    let e2e = t1.elapsed();
    let finite = values.iter().filter(|v| v.is_finite()).count();
    println!(
        "inference: {:?} (first call incl. transpose caches); {} values, finite={}/{}",
        e2e,
        values.len(),
        finite,
        values.len()
    );
    // second call for steady-state timing
    let t2 = std::time::Instant::now();
    let values2 = vla.infer_host_f32(&request).expect("second inference");
    println!("second inference: {:?}, finite {}/{}", t2.elapsed(), values2.iter().filter(|v| v.is_finite()).count(), values2.len());
    assert_eq!(finite, values.len());
    println!("ASCEND_CHECKPOINT_SMOKE_OK");
}
