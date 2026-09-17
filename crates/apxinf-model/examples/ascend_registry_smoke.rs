//! Registry smoke for the ascend path (E stage): AutoModel loads
//! "pi05-ascend" with synthetic weights end to end and runs one
//! inference through the trait object.
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example ascend_registry_smoke --features ascend --release -p apxinf-model
use apxinf_core::Tensor;
use half::bf16;

use apxinf_model::vla::{InitialLatent, Observation, VlaRequest, VlaRuntime, VisionObservation};
use apxinf_model::{AutoModel, LoadOptions, LoadedModel, SyntheticWeights};

fn main() {
    apxinf_model::register_builtin_models();

    let mut options = LoadOptions::default();
    options.model_name = Some("pi05-ascend".into());
    options.synthetic = Some(SyntheticWeights { seed: 1234 });
    let loaded = AutoModel::load_model(
        apxinf_core::Device::Ascend(0),
        ".",
        &options,
    )
    .expect("registry load");

    let LoadedModel::Vla(vla) = loaded else {
        panic!("expected a VLA model");
    };
    println!(
        "loaded: action {:?} patch {:?} rgb_u8={}",
        vla.action_shape(),
        vla.contract().patch_shape,
        vla.contract().accepts_rgb_u8
    );

    let contract = vla.contract();
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
        token_ids: (0..60u32).map(|i| (i * 37) % 2048).collect(),
        state: None,
        action_mask: None,
    };
    let rng = apxinf_core::RngKey::new(1, 2, 3);
    let request = VlaRequest {
        observation: &observation,
        initial_latent: InitialLatent::Generate { rng },
    };
    let values = vla.infer_host_f32(&request).expect("inference");
    let finite = values.iter().filter(|v| v.is_finite()).count();
    println!("actions: {} values, finite={}/{}", values.len(), finite, values.len());
    assert_eq!(finite, values.len());
    println!("ASCEND_REGISTRY_SMOKE_OK");
}
