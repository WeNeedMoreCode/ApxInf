//! Minimal Ascend VLA frontend for π0.5 — eager-only mirror of the cuda
//! vla_runtime's eager strategy (D stage). Tuning, graph capture, and
//! RGB-u8 device patchify are stubbed: patches arrive preprocessed, noise
//! comes from the deterministic core RNG (same math as the ascend normal
//! generator), and time embeddings are built once at load.

use std::sync::Arc;

use apxinf_core::{Backend as _, Error, Result, Tensor};
use half::bf16;

use crate::vla::{
    Action, InitialLatent, InferenceSpec, PreparedInference, VlaContract, VlaRequest,
    VlaRuntime, VisionObservation,
};

use super::{sinusoidal_time_embedding, Pi05AscendRuntime, Pi05Config};

#[derive(Clone)]
pub struct Pi05AscendVlaRuntime {
    backend: Arc<apxinf_ascend::AscendBackend>,
    config: Arc<Pi05Config>,
    runtime: Arc<Pi05AscendRuntime>,
    time_embeddings: Arc<Vec<Tensor>>,
}

impl Pi05AscendVlaRuntime {
    pub fn new(
        backend: Arc<apxinf_ascend::AscendBackend>,
        config: Arc<Pi05Config>,
        runtime: Pi05AscendRuntime,
    ) -> Result<Self> {
        // timestep embeddings are request-independent: build once
        let mut time_embeddings = Vec::with_capacity(config.num_flow_steps);
        for step in 0..config.num_flow_steps {
            let time = config.flow_start_time * (1.0 - step as f32 / config.num_flow_steps as f32);
            let values = sinusoidal_time_embedding(
                time,
                config.action_expert.width,
                config.time_min_period,
                config.time_max_period,
            )
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
            time_embeddings
                .push(backend.to_device(&Tensor::from_bf16(vec![1, config.action_expert.width], &values)?)?);
        }
        Ok(Self {
            backend,
            config,
            runtime: Arc::new(runtime),
            time_embeddings: Arc::new(time_embeddings),
        })
    }

    fn patches(&self, vision: &VisionObservation) -> Result<Tensor> {
        match vision {
            VisionObservation::Patches(patches) => self.backend.to_device(patches),
            VisionObservation::RgbU8 { .. } => Err(Error::Other(
                "π0.5 ascend path expects preprocessed patches (rgb-u8 patchify pending)".into(),
            )),
        }
    }

    fn noise(&self, latent: InitialLatent<'_>) -> Result<Tensor> {
        let (horizon, dim) = (self.config.action_horizon, self.config.action_dim);
        match latent {
            InitialLatent::Provided(t) => self.backend.to_device(t),
            InitialLatent::Generate { rng } => {
                let f32s = apxinf_core::standard_normal_f32(horizon * dim, rng);
                let values = f32s.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>();
                self.backend
                    .to_device(&Tensor::from_bf16(vec![horizon, dim], &values)?)
            }
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let patches = self.patches(&request.observation.vision)?;
        let noise = self.noise(request.initial_latent)?;
        let output = self.runtime.infer(
            &patches,
            &request.observation.token_ids,
            &noise,
            &self.time_embeddings,
        )?;
        Ok(Action::new(output))
    }
}

impl VlaRuntime for Pi05AscendVlaRuntime {
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [self.config.action_horizon, self.config.action_dim],
            patch_shape: [
                self.config.num_views * self.config.patches_per_view(),
                3 * self.config.patch_size * self.config.patch_size,
            ],
            max_token_len: self.config.max_token_len,
            num_views: self.config.num_views,
            image_size: self.config.image_size,
            patch_size: self.config.patch_size,
            accepts_rgb_u8: false,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        request.observation.validate()?;
        request.observation.inference_spec().validate()?;
        let action = self.run(request)?;
        self.backend.synchronize()?;
        Ok(action)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        Ok(Box::new(AscendEagerPrepared {
            vla: Arc::new(self.clone()),
            spec: *spec,
        }))
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        self.backend.to_cpu(action.tensor())?.to_f32_vec()
    }
}

struct AscendEagerPrepared {
    vla: Arc<Pi05AscendVlaRuntime>,
    spec: InferenceSpec,
}

impl PreparedInference for AscendEagerPrepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "π0.5 ascend prepared spec does not match the request".into(),
            ));
        }
        let action = self.vla.run(request)?;
        self.vla.backend.synchronize()?;
        Ok(action)
    }
}
