//! Ascend π0.5 eager runtime — mirror of bf16_runtime.rs over the aclnn
//! compositions in `ascend_executor` (transposed-b matmul, M-padded,
//! readback-safe). Weight tensors reuse StaticBf16Pi05Weights (F16 on
//! device through AscendBackend::to_device).
//!
//! ACLGraph capture is deliberately deferred past eager parity: the
//! ada-norm host reads (host_f16_row of style tensors) must be
//! pre-resolved into device rows before a capture window opens.

use std::cell::RefCell;
use std::sync::Arc;

use apxinf_ascend::AscendBackend;
use apxinf_core::{Backend as _, Error, Result, Tensor};

use super::ascend_executor::{aq, bias_add, tensor_buf, AscendCaches};
use super::{
    action_layer_ascend, language_layer_ascend, vision_layer_ascend,
    vision_patch_embed_ascend, Pi05Config, StaticBf16Pi05Weights,
};

pub struct AscendPrefixKvCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub tokens: usize,
}

pub struct AscendStepStyles {
    pub attention: Vec<Tensor>,
    pub mlp: Vec<Tensor>,
    pub final_norm: Tensor,
}

pub struct Pi05AscendRuntime {
    pub backend: Arc<AscendBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<StaticBf16Pi05Weights>,
    caches: RefCell<AscendCaches>,
}

impl Pi05AscendRuntime {
    pub fn new(
        backend: Arc<AscendBackend>,
        config: Arc<Pi05Config>,
        weights: Arc<StaticBf16Pi05Weights>,
    ) -> Result<Self> {
        config.validate()?;
        if weights.vision_layers.len() != config.vision_depth
            || weights.language_layers.len() != config.language.depth
            || weights.action_layers.len() != config.action_expert.depth
        {
            return Err(Error::Other(
                "π0.5 Ascend device weight depth mismatch".into(),
            ));
        }
        // The ascend executor halves the [gate; up] concat -- reject the
        // SM89 interleaved tactic layouts at the door.
        let plain_gate_up = |w: &super::Bf16LinearWeights| !w.bf16_dual_geglu_interleaved;
        if !weights.language_layers.iter().all(|l| plain_gate_up(&l.gate_up))
            || !weights.action_layers.iter().all(|l| plain_gate_up(&l.gate_up))
        {
            return Err(Error::Other(
                "π0.5 Ascend path requires plain [gate; up] gate_up weights (interleaved tactic layout is CUDA-only)".into(),
            ));
        }
        Ok(Self {
            backend,
            config,
            weights,
            caches: RefCell::new(AscendCaches::new()),
        })
    }

    pub fn config(&self) -> &Pi05Config {
        &self.config
    }

    pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
        let be = &*self.backend;
        let weights = &self.weights;
        let config = &self.config;
        let mut hidden = vision_patch_embed_ascend(
            be,
            &mut self.caches.borrow_mut(),
            &weights.patch_embedding,
            &weights.position_embedding,
            patches,
            config.patches_per_view(),
        )?;
        for layer in &weights.vision_layers {
            hidden = vision_layer_ascend(
                be,
                &mut self.caches.borrow_mut(),
                layer,
                &hidden,
                config.patches_per_view(),
                config.vision_heads,
                config.vision_head_dim,
                config.layer_norm_eps,
            )?;
        }
        let tokens = hidden.shape().dims()[0] as i64;
        let width = hidden.shape().dims()[1] as i64;
        let hb = tensor_buf(&hidden)?;
        let zeros = be.zeros((tokens * width * 2) as usize)?;
        let normed = aq::layer_norm(
            be,
            hb,
            &zeros,
            tensor_buf(&weights.vision_post_norm.weight)?,
            tensor_buf(&weights.vision_post_norm.bias)?,
            tokens,
            width,
            config.layer_norm_eps as f64,
        )?;
        let out_w = weights.multimodal_projector.weight.shape().dims()[1] as i64;
        let proj = aq::matmul(
            be,
            &mut self.caches.borrow_mut().nz,
            &normed,
            [tokens, width],
            tensor_buf(&weights.multimodal_projector.weight)?,
            [width, out_w],
        )?;
        let out = bias_add(be, &proj, weights.multimodal_projector.bias.as_ref(), tokens, out_w)?;
        Ok(be.wrap_fp16(out, vec![tokens as usize, out_w as usize]))
    }

    /// token ids are tiny; take them as a host slice (once per infer).
    pub fn embed_prefix(&self, vision_tokens: &Tensor, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() || token_ids.len() > self.config.max_token_len {
            return Err(Error::Other(format!(
                "π0.5 token count must be in 1..={}, got {}",
                token_ids.len(),
                self.config.max_token_len
            )));
        }
        let be = &*self.backend;
        let width = vision_tokens.shape().dims()[1] as i64;
        let vision_rows = vision_tokens.shape().dims()[0] as i64;
        let vocab = self.weights.token_embedding.shape().dims()[0] as i64;
        let count = token_ids.len() as i64;

        // token ids are effectively static per deployment: cache the
        // device index row by the id sequence (the per-call h2d was a
        // sync memcpy inside capture windows)
        let di = {
            let mut caches = self.caches.borrow_mut();
            match caches.token_idx.get(token_ids) {
                Some(v) => v.clone(),
                None => {
                    let idx: Vec<i32> = token_ids.iter().map(|&v| v as i32).collect();
                    let idx_bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let buf = be
                        .ctx()
                        .malloc(idx_bytes.len())
                        .map_err(|e| Error::Other(format!("aclnn: {e}")))?;
                    be.ctx()
                        .copy_h2d(&buf, &idx_bytes)
                        .map_err(|e| Error::Other(format!("aclnn: {e}")))?;
                    let buf = std::sync::Arc::new(buf);
                    caches.token_idx.insert(token_ids.to_vec(), buf.clone());
                    buf
                }
            }
        };

        let language = aq::gather_rows(
            be,
            tensor_buf(&self.weights.token_embedding)?,
            vocab,
            width,
            &di,
            count,
        )?;
        let out = aq::cat2(
            be,
            tensor_buf(vision_tokens)?,
            [vision_rows, width],
            &language,
            [count, width],
        )?;
        Ok(be.wrap_fp16(out, vec![(vision_rows + count) as usize, width as usize]))
    }

    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<AscendPrefixKvCache> {
        let be = &*self.backend;
        let mut hidden = prefix.clone();
        let mut keys = Vec::with_capacity(self.config.language.depth);
        let mut values = Vec::with_capacity(self.config.language.depth);
        for (index, layer) in self.weights.language_layers.iter().enumerate() {
            let output = language_layer_ascend(
                be,
                &mut self.caches.borrow_mut(),
                self.config.language,
                layer,
                &hidden,
                index + 1 < self.config.language.depth,
                0,
                self.config.rms_norm_eps,
                self.config.rope_theta,
            )?;
            hidden = output.hidden;
            keys.push(output.key);
            values.push(output.value);
        }
        Ok(AscendPrefixKvCache {
            keys,
            values,
            tokens: prefix.shape().dims()[0],
        })
    }

    fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
        let be = &*self.backend;
        let rows = time_embedding.shape().dims()[0] as i64;
        let width = time_embedding.shape().dims()[1] as i64;
        let mlp = |x: &Tensor, lin: &super::Bf16LinearWeights| -> Result<Tensor> {
            let out_w = lin.weight.shape().dims()[1] as i64;
            let proj = aq::matmul(
                be,
                &mut self.caches.borrow_mut().nz,
                tensor_buf(x)?,
                [rows, width],
                tensor_buf(&lin.weight)?,
                [width, out_w],
            )?;
            let biased = bias_add(be, &proj, lin.bias.as_ref(), rows, out_w)?;
            let activated = aq::silu(be, &biased, &[rows, out_w])?;
            Ok(be.wrap_fp16(activated, vec![rows as usize, out_w as usize]))
        };
        let hidden = mlp(time_embedding, &self.weights.time_mlp_in)?;
        mlp(&hidden, &self.weights.time_mlp_out)
    }

    fn style(&self, conditioning: &Tensor, weights: &super::Bf16LinearWeights) -> Result<Tensor> {
        let be = &*self.backend;
        let rows = conditioning.shape().dims()[0] as i64;
        let width = conditioning.shape().dims()[1] as i64;
        let out_w = weights.weight.shape().dims()[1] as i64;
        let proj = aq::matmul(
            be,
            &mut self.caches.borrow_mut().nz,
            tensor_buf(conditioning)?,
            [rows, width],
            tensor_buf(&weights.weight)?,
            [width, out_w],
        )?;
        let style = bias_add(be, &proj, weights.bias.as_ref(), rows, out_w)?;
        Ok(be.wrap_fp16(style, vec![(rows * out_w) as usize]))
    }

    fn prepare_step_styles(&self, time_embedding: &Tensor) -> Result<AscendStepStyles> {
        let conditioning = self.conditioning(time_embedding)?;
        let mut attention = Vec::with_capacity(self.config.action_expert.depth);
        let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
        for layer in &self.weights.action_layers {
            attention.push(self.style(&conditioning, &layer.input_style)?);
            mlp.push(self.style(&conditioning, &layer.post_attention_style)?);
        }
        let final_norm = self.style(&conditioning, &self.weights.action_final_style)?;
        Ok(AscendStepStyles {
            attention,
            mlp,
            final_norm,
        })
    }

    pub fn prepare_all_styles(&self, time_embeddings: &[Tensor]) -> Result<Vec<AscendStepStyles>> {
        if time_embeddings.len() != self.config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} timestep embeddings, got {}",
                self.config.num_flow_steps,
                time_embeddings.len()
            )));
        }
        time_embeddings
            .iter()
            .map(|embedding| self.prepare_step_styles(embedding))
            .collect()
    }

    fn denoise_step_with_styles(
        &self,
        state: &Tensor,
        styles: &AscendStepStyles,
        prefix: &AscendPrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        if prefix.keys.len() != self.config.action_expert.depth
            || prefix.values.len() != self.config.action_expert.depth
            || styles.attention.len() != self.config.action_expert.depth
            || styles.mlp.len() != self.config.action_expert.depth
        {
            return Err(Error::Other("π0.5 Ascend prefix/style depth mismatch".into()));
        }
        let be = &*self.backend;
        let trace = std::env::var("APXINF_ASCEND_TRACE").is_ok();
        let mut mark = |tag: &str| -> Result<()> {
            // print-only: a stream sync inside a capture window is rejected
            if trace {
                eprintln!("[trace] {tag}");
            }
            Ok(())
        };
        mark("styles prepared")?;
        let rows = state.shape().dims()[0] as i64;
        let state_w = state.shape().dims()[1] as i64;
        let in_w = self.weights.action_in.weight.shape().dims()[0] as i64;
        let out_w = self.weights.action_in.weight.shape().dims()[1] as i64;
        let projected = aq::matmul(
            be,
            &mut self.caches.borrow_mut().nz,
            tensor_buf(state)?,
            [rows, state_w],
            tensor_buf(&self.weights.action_in.weight)?,
            [in_w, out_w],
        )?;
        mark("action_in matmul")?;
        let biased = bias_add(
            be,
            &projected,
            self.weights.action_in.bias.as_ref(),
            rows,
            out_w,
        )?;
        let mut hidden = be.wrap_fp16(biased, vec![rows as usize, out_w as usize]);
        let mut attention_normalized: Option<Tensor> = None;
        for index in 0..self.config.action_expert.depth {
            let layer = &self.weights.action_layers[index];
            let next_norm_style = if index + 1 < self.config.action_expert.depth {
                &styles.attention[index + 1]
            } else {
                &styles.final_norm
            };
            let output = action_layer_ascend(
                be,
                &mut self.caches.borrow_mut(),
                self.config.action_expert,
                layer,
                &hidden,
                attention_normalized.as_ref(),
                &styles.attention[index],
                &styles.mlp[index],
                next_norm_style,
                &prefix.keys[index],
                &prefix.values[index],
                prefix.tokens,
                self.config.rms_norm_eps,
                self.config.rope_theta,
            )?;
            hidden = output.hidden;
            attention_normalized = Some(output.next_normalized);
            mark(&format!("action layer {index} done"))?;
        }
        let hidden = attention_normalized.ok_or_else(|| {
            Error::Other("π0.5 action expert must contain at least one layer".into())
        })?;
        let rows = hidden.shape().dims()[0] as i64;
        let width = hidden.shape().dims()[1] as i64;
        let out_w = self.weights.action_out.weight.shape().dims()[1] as i64;
        let velocity = aq::matmul(
            be,
            &mut self.caches.borrow_mut().nz,
            tensor_buf(&hidden)?,
            [rows, width],
            tensor_buf(&self.weights.action_out.weight)?,
            [width, out_w],
        )?;
        let velocity = bias_add(
            be,
            &velocity,
            self.weights.action_out.bias.as_ref(),
            rows,
            out_w,
        )?;
        mark("action_out matmul")?;
        let updated = super::ascend_executor::euler_mul(
            be,
            &mut self.caches.borrow_mut(),
            tensor_buf(state)?,
            &velocity,
            dt,
            rows,
            out_w,
        )?;
        mark("euler")?;
        Ok(be.wrap_fp16(updated, vec![rows as usize, out_w as usize]))
    }

    pub fn denoise_step(
        &self,
        state: &Tensor,
        time_embedding: &Tensor,
        prefix: &AscendPrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        let styles = self.prepare_step_styles(time_embedding)?;
        self.denoise_step_with_styles(state, &styles, prefix, dt)
    }

    pub fn denoise_all_steps_with_styles(
        &self,
        noise: &Tensor,
        styles: &[AscendStepStyles],
        prefix: &AscendPrefixKvCache,
    ) -> Result<Tensor> {
        if styles.len() != self.config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} precomputed style sets, got {}",
                self.config.num_flow_steps,
                styles.len()
            )));
        }
        let mut state = noise.clone();
        let dt = -self.config.flow_start_time / self.config.num_flow_steps as f32;
        for step_styles in styles {
            state = self.denoise_step_with_styles(&state, step_styles, prefix, dt)?;
        }
        Ok(state)
    }

    pub fn denoise_all_steps(
        &self,
        noise: &Tensor,
        time_embeddings: &[Tensor],
        prefix: &AscendPrefixKvCache,
    ) -> Result<Tensor> {
        let styles = self.prepare_all_styles(time_embeddings)?;
        self.denoise_all_steps_with_styles(noise, &styles, prefix)
    }

    pub fn infer_with_styles(
        &self,
        patches: &Tensor,
        token_ids: &[u32],
        noise: &Tensor,
        styles: &[AscendStepStyles],
    ) -> Result<Tensor> {
        let vision = self.encode_vision(patches)?;
        let prefix = self.embed_prefix(&vision, token_ids)?;
        let prefix = self.prefix_forward(&prefix)?;
        self.denoise_all_steps_with_styles(noise, styles, &prefix)
    }

    pub fn infer(
        &self,
        patches: &Tensor,
        token_ids: &[u32],
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        let styles = self.prepare_all_styles(time_embeddings)?;
        self.infer_with_styles(patches, token_ids, noise, &styles)
    }
}
