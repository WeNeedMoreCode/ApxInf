//! Typed π0.5 checkpoint weights.
//!
//! LeRobot checkpoints use Hugging Face's `[out, in]` linear layout. ApxInf's
//! row-major GEMMs consume `[in, out]`, so every projection is physically
//! transposed while loading. Gemma language RMSNorm parameters are learned
//! offsets and are converted once to their final `1 + weight` scale.

use std::collections::HashMap;
use std::path::Path;

use half::bf16;
use apxinf_core::{DType, Error, Result, Tensor};

use super::{GemmaVariantConfig, Pi05Config};

const ROOT: &str = "paligemma_with_expert";

#[derive(Debug)]
pub struct LinearWeights {
    /// Physical `[in, out]` matrix used by row-major GEMM.
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

#[derive(Debug)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct AdaRmsNormWeights {
    /// Conditioning projection `[width, 3 * width]`.
    pub style: LinearWeights,
}

#[derive(Debug)]
pub struct VisionBlockWeights {
    pub norm1: LayerNormWeights,
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
    pub norm2: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

#[derive(Debug)]
pub struct VisionWeights {
    /// Flattened patch convolution `[3 * patch * patch, vision_width]`.
    pub patch_embedding: LinearWeights,
    pub position_embedding: Tensor,
    pub blocks: Vec<VisionBlockWeights>,
    pub post_layer_norm: LayerNormWeights,
    pub multimodal_projector: LinearWeights,
    /// Tied PaliGemma token embedding `[vocab, language_width]`.
    pub token_embedding: Tensor,
}

#[derive(Debug)]
pub struct GemmaAttentionWeights {
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
}

#[derive(Debug)]
pub struct GemmaMlpWeights {
    pub gate: LinearWeights,
    pub up: LinearWeights,
    pub down: LinearWeights,
}

#[derive(Debug)]
pub struct LanguageLayerWeights {
    /// Final multiplicative scale, after applying Gemma's `1 + weight` rule.
    pub input_norm_scale: Tensor,
    pub attention: GemmaAttentionWeights,
    pub post_attention_norm_scale: Tensor,
    pub mlp: GemmaMlpWeights,
}

#[derive(Debug)]
pub struct ActionLayerWeights {
    pub input_norm: AdaRmsNormWeights,
    pub attention: GemmaAttentionWeights,
    pub post_attention_norm: AdaRmsNormWeights,
    pub mlp: GemmaMlpWeights,
}

#[derive(Debug)]
pub struct Pi05Weights {
    pub vision: VisionWeights,
    pub language_layers: Vec<LanguageLayerWeights>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<ActionLayerWeights>,
    pub action_final_norm: AdaRmsNormWeights,
    pub action_in: LinearWeights,
    pub action_out: LinearWeights,
    pub time_mlp_in: LinearWeights,
    pub time_mlp_out: LinearWeights,
}

impl Pi05Weights {
    /// Load a single-file or sharded LeRobot/OpenPI SafeTensors checkpoint.
    pub fn from_safetensors(config: &Pi05Config, path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load π0.5 SafeTensors: {error}")))?;
        Self::from_map(config, tensors)
    }

    /// Serve-only minimal load straight from a checkpoint dir (see `from_map_serve`).
    pub fn from_safetensors_serve(config: &Pi05Config, path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load π0.5 SafeTensors: {error}")))?;
        Self::from_map_serve(config, tensors)
    }

    /// Consume a LeRobot/OpenPI π0.5 safetensors map and validate every tensor
    /// required by the inference graph. Unused action LM-head weights may remain.
    pub fn from_map(config: &Pi05Config, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        config.validate()?;
        normalize_lerobot_prefix(&mut tensors);

        let vision_prefix = format!("{ROOT}.paligemma.model.vision_tower.vision_model");
        let mut vision_blocks = Vec::with_capacity(config.vision_depth);
        for layer in 0..config.vision_depth {
            let p = format!("{vision_prefix}.encoder.layers.{layer}");
            vision_blocks.push(VisionBlockWeights {
                norm1: take_layer_norm(&mut tensors, &format!("{p}.layer_norm1"))?,
                q: take_linear(&mut tensors, &format!("{p}.self_attn.q_proj"), true)?,
                k: take_linear(&mut tensors, &format!("{p}.self_attn.k_proj"), true)?,
                v: take_linear(&mut tensors, &format!("{p}.self_attn.v_proj"), true)?,
                output: take_linear(&mut tensors, &format!("{p}.self_attn.out_proj"), true)?,
                norm2: take_layer_norm(&mut tensors, &format!("{p}.layer_norm2"))?,
                fc1: take_linear(&mut tensors, &format!("{p}.mlp.fc1"), true)?,
                fc2: take_linear(&mut tensors, &format!("{p}.mlp.fc2"), true)?,
            });
        }

        let patch_weight_name = format!("{vision_prefix}.embeddings.patch_embedding.weight");
        let patch_weight = take(&mut tensors, &patch_weight_name)?;
        let expected_patch_elements =
            config.vision_width * 3 * config.patch_size * config.patch_size;
        if patch_weight.numel() != expected_patch_elements {
            return Err(Error::Other(format!(
                "{patch_weight_name}: expected {expected_patch_elements} elements, got {}",
                patch_weight.numel()
            )));
        }
        let patch_weight = patch_weight.reshape(vec![
            config.vision_width,
            3 * config.patch_size * config.patch_size,
        ])?;
        let patch_embedding = LinearWeights {
            weight: transpose_2d(&patch_weight)?,
            bias: Some(take(
                &mut tensors,
                &format!("{vision_prefix}.embeddings.patch_embedding.bias"),
            )?),
        };

        let vision = VisionWeights {
            patch_embedding,
            position_embedding: take(
                &mut tensors,
                &format!("{vision_prefix}.embeddings.position_embedding.weight"),
            )?,
            blocks: vision_blocks,
            post_layer_norm: take_layer_norm(
                &mut tensors,
                &format!("{vision_prefix}.post_layernorm"),
            )?,
            multimodal_projector: take_linear(
                &mut tensors,
                &format!("{ROOT}.paligemma.model.multi_modal_projector.linear"),
                true,
            )?,
            token_embedding: take_any(
                &mut tensors,
                &[
                    format!("{ROOT}.paligemma.model.language_model.embed_tokens.weight"),
                    format!("{ROOT}.paligemma.lm_head.weight"),
                ],
            )?,
        };

        let language_prefix = format!("{ROOT}.paligemma.model.language_model");
        let mut language_layers = Vec::with_capacity(config.language.depth);
        for layer in 0..config.language.depth {
            let p = format!("{language_prefix}.layers.{layer}");
            // Fold Gemma's learned RMSNorm multiplier into the consuming
            // weights in FP32 before FP8 quantization.  Keeping it as an FP16
            // activation multiply can round channels near -1 to zero and also
            // gives the packed QKV/gate-up matrix a much worse tensor scale.
            let attention_scale =
                add_one(take(&mut tensors, &format!("{p}.input_layernorm.weight"))?)?;
            let mlp_scale = add_one(take(
                &mut tensors,
                &format!("{p}.post_attention_layernorm.weight"),
            )?)?;
            let mut attention = take_attention(&mut tensors, &p)?;
            attention.q = fold_input_scale(attention.q, &attention_scale)?;
            attention.k = fold_input_scale(attention.k, &attention_scale)?;
            attention.v = fold_input_scale(attention.v, &attention_scale)?;
            let mut mlp = take_mlp(&mut tensors, &p)?;
            mlp.gate = fold_input_scale(mlp.gate, &mlp_scale)?;
            mlp.up = fold_input_scale(mlp.up, &mlp_scale)?;
            language_layers.push(LanguageLayerWeights {
                input_norm_scale: ones(attention_scale.shape().dims())?,
                attention,
                post_attention_norm_scale: ones(mlp_scale.shape().dims())?,
                mlp,
            });
        }

        let action_prefix = format!("{ROOT}.gemma_expert.model");
        let mut action_layers = Vec::with_capacity(config.action_expert.depth);
        for layer in 0..config.action_expert.depth {
            let p = format!("{action_prefix}.layers.{layer}");
            action_layers.push(ActionLayerWeights {
                input_norm: take_ada_norm(&mut tensors, &format!("{p}.input_layernorm"))?,
                attention: take_attention(&mut tensors, &p)?,
                post_attention_norm: take_ada_norm(
                    &mut tensors,
                    &format!("{p}.post_attention_layernorm"),
                )?,
                mlp: take_mlp(&mut tensors, &p)?,
            });
        }

        let weights = Self {
            vision,
            language_layers,
            language_final_norm_scale: add_one(take(
                &mut tensors,
                &format!("{language_prefix}.norm.weight"),
            )?)?,
            action_layers,
            action_final_norm: take_ada_norm(&mut tensors, &format!("{action_prefix}.norm"))?,
            action_in: take_linear(&mut tensors, "action_in_proj", true)?,
            action_out: take_linear(&mut tensors, "action_out_proj", true)?,
            time_mlp_in: take_linear(&mut tensors, "time_mlp_in", true)?,
            time_mlp_out: take_linear(&mut tensors, "time_mlp_out", true)?,
        };

        validate_shapes(config, &weights)?;
        Ok(weights)
    }

    /// Serve-only minimal load: synthetic skeleton (shape-correct placeholders)
    /// plus the host-consumed subset — `time_mlp_{in,out}`, the 37 AdaRMS style
    /// projections (`gemma_expert` per-layer input/post norms + final norm) and
    /// the tied token embedding for host lookup. Everything else (vision tower,
    /// language layers, expert projections) is baked into the prebaked OMs a
    /// serve loads, so placeholders are never consumed: serve skips graph
    /// compilation, and the probe driver refuses `GEB_CKPT_SERVE_MIN` on a bake
    /// (`GEB_SAVE`) path so placeholder weights can't silently reach an OM.
    /// Cuts the 61s single-threaded from_map math over 812 tensors to ~2s
    /// (cold-start breakdown: io/copy 5.5s is untouched). The
    /// follow-up zeroes the skeleton fill (`synthetic_skeleton`:
    /// placeholders are never read, and untouched zero pages cost no RSS)
    /// and parallelizes the 39 style transposes that then dominate.
    pub fn from_map_serve(config: &Pi05Config, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        config.validate()?;
        normalize_lerobot_prefix(&mut tensors);
        let mut weights = Self::synthetic_skeleton(config)?;
        // styles 链（18 层 ×2 + final + time_mlp ×2 = 39 件 [3w,w]/[w,w] 转置）
        // 是零物化骨架后数学段的大头：先顺序 take（纯 move），再按线程分块
        // 并行转置——chunk 独占所有权无锁；不引 rayon（引擎 crate 零新依赖）。
        let action_prefix = format!("{ROOT}.gemma_expert.model");
        let mut srcs: Vec<(Tensor, Option<Tensor>)> = Vec::with_capacity(39);
        for layer in 0..config.action_expert.depth {
            let p = format!("{action_prefix}.layers.{layer}");
            for half in ["input_layernorm", "post_attention_layernorm"] {
                let d = format!("{p}.{half}.dense");
                srcs.push((
                    take(&mut tensors, &format!("{d}.weight"))?,
                    Some(take(&mut tensors, &format!("{d}.bias"))?),
                ));
            }
        }
        srcs.push((
            take(&mut tensors, &format!("{action_prefix}.norm.dense.weight"))?,
            Some(take(&mut tensors, &format!("{action_prefix}.norm.dense.bias"))?),
        ));
        for t in ["time_mlp_in", "time_mlp_out"] {
            srcs.push((
                take(&mut tensors, &format!("{t}.weight"))?,
                Some(take(&mut tensors, &format!("{t}.bias"))?),
            ));
        }
        let mut lin = parallel_transpose(srcs)?.into_iter();
        for layer in 0..config.action_expert.depth {
            weights.action_layers[layer].input_norm =
                AdaRmsNormWeights { style: lin.next().expect("style slots") };
            weights.action_layers[layer].post_attention_norm =
                AdaRmsNormWeights { style: lin.next().expect("style slots") };
        }
        weights.action_final_norm =
            AdaRmsNormWeights { style: lin.next().expect("style slots") };
        weights.time_mlp_in = lin.next().expect("style slots");
        weights.time_mlp_out = lin.next().expect("style slots");
        // flow 段 Data bias 槽（ainb/aoutb 图输入——binds 物化消费；mm
        // 权重 wt() 在 WCONST 下 Const 烤入 OM 不占输入，占位无害）
        weights.action_in.bias = take_opt(&mut tensors, "action_in_proj.bias");
        weights.action_out.bias = take_opt(&mut tensors, "action_out_proj.bias");
        // vision 段 Data 输入源（binds 物化消费——mm 权重仍占位：WCONST
        // 烤入 OM 不走 binds；tmp 桶 vis_head 分岔定位的补遗：
        // patch/pos/LN/qkv/fc bias 全部经 s.data() 进 binds）
        let vp = format!("{ROOT}.paligemma.model.vision_tower.vision_model");
        weights.vision.patch_embedding.bias =
            Some(take(&mut tensors, &format!("{vp}.embeddings.patch_embedding.bias"))?);
        weights.vision.position_embedding =
            take(&mut tensors, &format!("{vp}.embeddings.position_embedding.weight"))?;
        for layer in 0..config.vision_depth {
            let p = format!("{vp}.encoder.layers.{layer}");
            let blk = &mut weights.vision.blocks[layer];
            blk.norm1 = take_layer_norm(&mut tensors, &format!("{p}.layer_norm1"))?;
            blk.norm2 = take_layer_norm(&mut tensors, &format!("{p}.layer_norm2"))?;
            blk.q.bias = take_opt(&mut tensors, &format!("{p}.self_attn.q_proj.bias"));
            blk.k.bias = take_opt(&mut tensors, &format!("{p}.self_attn.k_proj.bias"));
            blk.v.bias = take_opt(&mut tensors, &format!("{p}.self_attn.v_proj.bias"));
            blk.output.bias = take_opt(&mut tensors, &format!("{p}.self_attn.out_proj.bias"));
            blk.fc1.bias = take_opt(&mut tensors, &format!("{p}.mlp.fc1.bias"));
            blk.fc2.bias = take_opt(&mut tensors, &format!("{p}.mlp.fc2.bias"));
        }
        weights.vision.post_layer_norm =
            take_layer_norm(&mut tensors, &format!("{vp}.post_layernorm"))?;
        weights.vision.multimodal_projector.bias = take_opt(
            &mut tensors,
            &format!("{ROOT}.paligemma.model.multi_modal_projector.linear.bias"),
        );
        weights.vision.token_embedding = take_any(
            &mut tensors,
            &[
                format!("{ROOT}.paligemma.model.language_model.embed_tokens.weight"),
                format!("{ROOT}.paligemma.lm_head.weight"),
            ],
        )?;
        Ok(weights)
    }

    /// Build deterministic random host weights for a config, with no checkpoint.
    ///
    /// Latency depends only on tensor shape and dtype, so a benchmark that only
    /// measures the engine (L0/L1) needs no trained weights. Every tensor is
    /// produced directly at ApxInf's final `[in, out]` orientation (no transpose
    /// or scale fold), filled by a seeded LCG in a small range so deep BF16/FP8
    /// stacks stay finite. RMSNorm scales are ones (the language folder already
    /// folds the learned offset into the projections, so the runtime multiplies
    /// by unit scale); LayerNorm gamma is one and beta is zero.
    pub fn synthetic(config: &Pi05Config, seed: u64) -> Result<Self> {
        let mut rng = SyntheticRng::new(seed);
        Self::build_skeleton(config, &mut SkeletonFill::Random(&mut rng))
    }

    /// Shape-only zero skeleton: every placeholder `synthetic` would fill with
    /// RNG values is allocated zeroed instead. For serve-min (`from_map_serve`)
    /// the placeholders are provably never read — the prebaked OM carries the
    /// Const/WCONST faces — so the ~7GB LCG fill is pure cold-start waste, and
    /// zero pages that are never touched cost neither time nor RSS. Benchmarks
    /// and bakes must keep `synthetic`: they compute with the values.
    pub fn synthetic_skeleton(config: &Pi05Config) -> Result<Self> {
        // GEB_SM_ZSCOPE：bisect/debug 旋钮。默认空 = 全零（生产）；rand =
        // 全随机占位（旧行为单变量对照——隔离「零填充」与「重构本身」）
        if std::env::var("GEB_SM_ZSCOPE").as_deref() == Ok("rand") {
            let mut rng = SyntheticRng::new(0);
            return Self::build_skeleton(config, &mut SkeletonFill::Random(&mut rng));
        }
        Self::build_skeleton(config, &mut SkeletonFill::Zero)
    }

    fn build_skeleton(config: &Pi05Config, fill: &mut SkeletonFill) -> Result<Self> {
        config.validate()?;

        let lang = &config.language;
        let action = &config.action_expert;
        let vision_qkv = config.vision_width; // SigLIP heads*head_dim == width
        let patch_width = 3 * config.patch_size * config.patch_size;

        let mut vision_blocks = Vec::with_capacity(config.vision_depth);
        for _ in 0..config.vision_depth {
            vision_blocks.push(VisionBlockWeights {
                norm1: synthetic_layer_norm(config.vision_width)?,
                q: fill.linear(config.vision_width, vision_qkv, true)?,
                k: fill.linear(config.vision_width, vision_qkv, true)?,
                v: fill.linear(config.vision_width, vision_qkv, true)?,
                output: fill.linear(vision_qkv, config.vision_width, true)?,
                norm2: synthetic_layer_norm(config.vision_width)?,
                fc1: fill.linear(config.vision_width, config.vision_mlp_dim, true)?,
                fc2: fill.linear(config.vision_mlp_dim, config.vision_width, true)?,
            });
        }

        let vision = VisionWeights {
            patch_embedding: fill.linear(patch_width, config.vision_width, true)?,
            position_embedding: fill
                .tensor(vec![config.patches_per_view(), config.vision_width])?,
            blocks: vision_blocks,
            post_layer_norm: synthetic_layer_norm(config.vision_width)?,
            multimodal_projector: fill.linear(config.vision_width, lang.width, true)?,
            token_embedding: fill.tensor(vec![config.vocab_size, lang.width])?,
        };

        let mut language_layers = Vec::with_capacity(lang.depth);
        for _ in 0..lang.depth {
            language_layers.push(LanguageLayerWeights {
                input_norm_scale: ones(&[lang.width])?,
                attention: synthetic_attention(fill, lang)?,
                post_attention_norm_scale: ones(&[lang.width])?,
                mlp: synthetic_mlp(fill, lang)?,
            });
        }

        let mut action_layers = Vec::with_capacity(action.depth);
        for _ in 0..action.depth {
            action_layers.push(ActionLayerWeights {
                input_norm: synthetic_ada_norm(fill, action.width)?,
                attention: synthetic_attention(fill, action)?,
                post_attention_norm: synthetic_ada_norm(fill, action.width)?,
                mlp: synthetic_mlp(fill, action)?,
            });
        }

        let weights = Self {
            vision,
            language_layers,
            language_final_norm_scale: ones(&[lang.width])?,
            action_layers,
            action_final_norm: synthetic_ada_norm(fill, action.width)?,
            action_in: fill.linear(config.action_dim, action.width, true)?,
            action_out: fill.linear(action.width, config.action_dim, true)?,
            time_mlp_in: fill.linear(action.width, action.width, true)?,
            time_mlp_out: fill.linear(action.width, action.width, true)?,
        };

        validate_shapes(config, &weights)?;
        Ok(weights)
    }
}

/// Small deterministic LCG used only to fill synthetic benchmark weights.
struct SyntheticRng {
    state: u64,
}

impl SyntheticRng {
    /// Peak magnitude of every synthetic weight element. Small enough that deep
    /// BF16/FP8 stacks cannot overflow and trip the runtime's finite-output gate.
    const AMPLITUDE: f32 = 0.02;

    fn new(seed: u64) -> Self {
        // Offset the seed so seed=0 does not start the LCG at a fixed point.
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 33) as u32; // top 31 bits
        let unit = bits as f32 / (1u32 << 31) as f32; // [0, 1)
        (unit * 2.0 - 1.0) * Self::AMPLITUDE
    }
}

fn synthetic_tensor(rng: &mut SyntheticRng, shape: Vec<usize>) -> Result<Tensor> {
    let count: usize = shape.iter().product();
    let values: Vec<bf16> = (0..count).map(|_| bf16::from_f32(rng.next_f32())).collect();
    Tensor::from_bf16(shape, &values)
}

/// Fill policy for skeleton construction (see `synthetic_skeleton`).
enum SkeletonFill<'r> {
    Random(&'r mut SyntheticRng),
    Zero,
}

impl SkeletonFill<'_> {
    fn tensor(&mut self, shape: Vec<usize>) -> Result<Tensor> {
        match self {
            Self::Random(rng) => synthetic_tensor(rng, shape),
            // 大块 `vec![0u8]` 走 calloc → 懒 commit 零页：不 touch 不计费
            Self::Zero => Ok(Tensor::zeros(shape, DType::BF16)),
        }
    }

    fn linear(&mut self, in_dim: usize, out_dim: usize, has_bias: bool) -> Result<LinearWeights> {
        Ok(LinearWeights {
            weight: self.tensor(vec![in_dim, out_dim])?,
            bias: has_bias.then(|| self.tensor(vec![out_dim])).transpose()?,
        })
    }
}

/// Parallel `transpose_2d` over disjoint (weight, bias) pairs — the serve-min
/// style chain is 39 independent cache-hostile transposes (~925MB), the only
/// parallel-worthy math left after the zero-fill skeleton. Chunks own their
/// tensors outright: no locks, no shared state, `std::thread::scope` joins
/// before any result is consumed.
fn parallel_transpose(mut srcs: Vec<(Tensor, Option<Tensor>)>) -> Result<Vec<LinearWeights>> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8);
    let chunk = srcs.len().div_ceil(threads);
    let done: Vec<Result<LinearWeights>> = std::thread::scope(|s| {
        let handles: Vec<_> = srcs
            .chunks_mut(chunk)
            .map(|c| {
                s.spawn(move || {
                    c.iter_mut()
                        .map(|(w, b)| {
                            Ok(LinearWeights {
                                weight: transpose_2d(w)?,
                                bias: b.take(),
                            })
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut out = Vec::new();
        for h in handles {
            out.extend(h.join().expect("style transpose thread panicked"));
        }
        out
    });
    done.into_iter().collect()
}

fn synthetic_layer_norm(dim: usize) -> Result<LayerNormWeights> {
    let weight = vec![bf16::ONE; dim];
    let bias = vec![bf16::ZERO; dim];
    Ok(LayerNormWeights {
        weight: Tensor::from_bf16(vec![dim], &weight)?,
        bias: Tensor::from_bf16(vec![dim], &bias)?,
    })
}

fn synthetic_ada_norm(fill: &mut SkeletonFill, width: usize) -> Result<AdaRmsNormWeights> {
    Ok(AdaRmsNormWeights {
        style: fill.linear(width, 3 * width, true)?,
    })
}

fn synthetic_attention(
    fill: &mut SkeletonFill,
    config: &GemmaVariantConfig,
) -> Result<GemmaAttentionWeights> {
    let q = config.num_heads * config.head_dim;
    let kv = config.num_kv_heads * config.head_dim;
    Ok(GemmaAttentionWeights {
        q: fill.linear(config.width, q, false)?,
        k: fill.linear(config.width, kv, false)?,
        v: fill.linear(config.width, kv, false)?,
        output: fill.linear(q, config.width, false)?,
    })
}

fn synthetic_mlp(fill: &mut SkeletonFill, config: &GemmaVariantConfig) -> Result<GemmaMlpWeights> {
    Ok(GemmaMlpWeights {
        gate: fill.linear(config.width, config.mlp_dim, false)?,
        up: fill.linear(config.width, config.mlp_dim, false)?,
        down: fill.linear(config.mlp_dim, config.width, false)?,
    })
}

fn take_attention(
    tensors: &mut HashMap<String, Tensor>,
    layer: &str,
) -> Result<GemmaAttentionWeights> {
    let p = format!("{layer}.self_attn");
    Ok(GemmaAttentionWeights {
        q: take_linear(tensors, &format!("{p}.q_proj"), false)?,
        k: take_linear(tensors, &format!("{p}.k_proj"), false)?,
        v: take_linear(tensors, &format!("{p}.v_proj"), false)?,
        output: take_linear(tensors, &format!("{p}.o_proj"), false)?,
    })
}

fn take_mlp(tensors: &mut HashMap<String, Tensor>, layer: &str) -> Result<GemmaMlpWeights> {
    let p = format!("{layer}.mlp");
    Ok(GemmaMlpWeights {
        gate: take_linear(tensors, &format!("{p}.gate_proj"), false)?,
        up: take_linear(tensors, &format!("{p}.up_proj"), false)?,
        down: take_linear(tensors, &format!("{p}.down_proj"), false)?,
    })
}

fn take_layer_norm(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
) -> Result<LayerNormWeights> {
    Ok(LayerNormWeights {
        weight: take(tensors, &format!("{prefix}.weight"))?,
        bias: take(tensors, &format!("{prefix}.bias"))?,
    })
}

fn take_ada_norm(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<AdaRmsNormWeights> {
    Ok(AdaRmsNormWeights {
        style: take_linear(tensors, &format!("{prefix}.dense"), true)?,
    })
}

fn take_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    has_bias: bool,
) -> Result<LinearWeights> {
    let weight = transpose_2d(&take(tensors, &format!("{prefix}.weight"))?)?;
    let bias = has_bias
        .then(|| take(tensors, &format!("{prefix}.bias")))
        .transpose()?;
    Ok(LinearWeights { weight, bias })
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing π0.5 weight `{name}`")))
}

/// Optional take for serve-min overwrites: absent key keeps the placeholder.
fn take_opt(tensors: &mut HashMap<String, Tensor>, name: &str) -> Option<Tensor> {
    tensors.remove(name)
}

fn take_any(tensors: &mut HashMap<String, Tensor>, names: &[String]) -> Result<Tensor> {
    for name in names {
        if let Some(tensor) = tensors.remove(name) {
            return Ok(tensor);
        }
    }
    Err(Error::Other(format!(
        "missing π0.5 weight (accepted aliases: {})",
        names.join(", ")
    )))
}

fn normalize_lerobot_prefix(tensors: &mut HashMap<String, Tensor>) {
    let canonical_prefix = format!("{ROOT}.");
    let wrapped_prefix = format!("model.{ROOT}.");
    if tensors
        .keys()
        .any(|name| name.starts_with(&canonical_prefix))
        || !tensors.keys().any(|name| name.starts_with(&wrapped_prefix))
    {
        return;
    }
    *tensors = std::mem::take(tensors)
        .into_iter()
        .map(|(name, tensor)| {
            let name = name.strip_prefix("model.").unwrap_or(&name).to_owned();
            (name, tensor)
        })
        .collect();
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "π0.5 linear weight must be 2D, got shape {dims:?}"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    match tensor.dtype() {
        DType::F32 => {
            let src = tensor.as_f32()?;
            let mut dst = vec![0.0; src.len()];
            for row in 0..rows {
                for col in 0..cols {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
            Tensor::from_f32(vec![cols, rows], &dst)
        }
        DType::F16 => {
            let src = tensor.as_f16()?;
            let mut dst = vec![half::f16::ZERO; src.len()];
            for row in 0..rows {
                for col in 0..cols {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
            Tensor::from_f16(vec![cols, rows], &dst)
        }
        DType::BF16 => {
            let src = tensor.as_bf16()?;
            let mut dst = vec![bf16::ZERO; src.len()];
            for row in 0..rows {
                for col in 0..cols {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &dst)
        }
        DType::F8E4M3 => {
            let src = tensor.as_f8_e4m3()?;
            let mut dst = vec![0u8; src.len()];
            for row in 0..rows {
                for col in 0..cols {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
            Tensor::from_f8_e4m3(vec![cols, rows], &dst)
        }
    }
}

fn add_one(tensor: Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    match tensor.dtype() {
        DType::F32 => {
            let values = tensor.as_f32()?.iter().map(|x| x + 1.0).collect::<Vec<_>>();
            Tensor::from_f32(dims, &values)
        }
        DType::F16 => {
            let values = tensor
                .as_f16()?
                .iter()
                .map(|x| half::f16::from_f32(x.to_f32() + 1.0))
                .collect::<Vec<_>>();
            Tensor::from_f16(dims, &values)
        }
        DType::BF16 => {
            let values = tensor
                .as_bf16()?
                .iter()
                .map(|x| bf16::from_f32(x.to_f32() + 1.0))
                .collect::<Vec<_>>();
            Tensor::from_bf16(dims, &values)
        }
        DType::F8E4M3 => Err(Error::Other(
            "π0.5 RMSNorm parameters cannot be stored as unscaled FP8".into(),
        )),
    }
}

fn ones(shape: &[usize]) -> Result<Tensor> {
    Tensor::from_f32(shape.to_vec(), &vec![1.0; shape.iter().product()])
}

fn fold_input_scale(mut linear: LinearWeights, scale: &Tensor) -> Result<LinearWeights> {
    let dims = linear.weight.shape().dims();
    if dims.len() != 2 || scale.shape().dims() != [dims[0]] {
        return Err(Error::Other(format!(
            "π0.5 input-scale fold mismatch: weight {dims:?}, scale {:?}",
            scale.shape().dims()
        )));
    }
    let mut weight = linear.weight.to_f32_vec()?;
    let scale = scale.to_f32_vec()?;
    for (row, multiplier) in weight.chunks_exact_mut(dims[1]).zip(scale) {
        for value in row {
            *value *= multiplier;
        }
    }
    linear.weight = Tensor::from_f32(dims.to_vec(), &weight)?;
    Ok(linear)
}

fn validate_shapes(config: &Pi05Config, weights: &Pi05Weights) -> Result<()> {
    expect_shape(
        "token_embedding",
        &weights.vision.token_embedding,
        &[config.vocab_size, config.language.width],
    )?;
    expect_linear(
        "multimodal_projector",
        &weights.vision.multimodal_projector,
        config.vision_width,
        config.language.width,
    )?;
    expect_linear(
        "action_in_proj",
        &weights.action_in,
        config.action_dim,
        config.action_expert.width,
    )?;
    expect_linear(
        "action_out_proj",
        &weights.action_out,
        config.action_expert.width,
        config.action_dim,
    )?;
    expect_linear(
        "time_mlp_in",
        &weights.time_mlp_in,
        config.action_expert.width,
        config.action_expert.width,
    )?;
    expect_linear(
        "time_mlp_out",
        &weights.time_mlp_out,
        config.action_expert.width,
        config.action_expert.width,
    )?;
    validate_expert_shapes("language", &config.language, &weights.language_layers)?;
    validate_action_shapes(&config.action_expert, &weights.action_layers)?;
    Ok(())
}

fn validate_expert_shapes(
    name: &str,
    config: &GemmaVariantConfig,
    layers: &[LanguageLayerWeights],
) -> Result<()> {
    if layers.len() != config.depth {
        return Err(Error::Other(format!(
            "{name}: expected {} layers, got {}",
            config.depth,
            layers.len()
        )));
    }
    for (index, layer) in layers.iter().enumerate() {
        validate_attention(name, index, config, &layer.attention)?;
        validate_mlp(name, index, config, &layer.mlp)?;
    }
    Ok(())
}

fn validate_action_shapes(
    config: &GemmaVariantConfig,
    layers: &[ActionLayerWeights],
) -> Result<()> {
    if layers.len() != config.depth {
        return Err(Error::Other(format!(
            "action: expected {} layers, got {}",
            config.depth,
            layers.len()
        )));
    }
    for (index, layer) in layers.iter().enumerate() {
        validate_attention("action", index, config, &layer.attention)?;
        validate_mlp("action", index, config, &layer.mlp)?;
    }
    Ok(())
}

fn validate_attention(
    name: &str,
    index: usize,
    config: &GemmaVariantConfig,
    attention: &GemmaAttentionWeights,
) -> Result<()> {
    expect_linear(
        &format!("{name}.{index}.q"),
        &attention.q,
        config.width,
        config.num_heads * config.head_dim,
    )?;
    let kv = config.num_kv_heads * config.head_dim;
    expect_linear(&format!("{name}.{index}.k"), &attention.k, config.width, kv)?;
    expect_linear(&format!("{name}.{index}.v"), &attention.v, config.width, kv)?;
    expect_linear(
        &format!("{name}.{index}.output"),
        &attention.output,
        config.num_heads * config.head_dim,
        config.width,
    )
}

fn validate_mlp(
    name: &str,
    index: usize,
    config: &GemmaVariantConfig,
    mlp: &GemmaMlpWeights,
) -> Result<()> {
    expect_linear(
        &format!("{name}.{index}.gate"),
        &mlp.gate,
        config.width,
        config.mlp_dim,
    )?;
    expect_linear(
        &format!("{name}.{index}.up"),
        &mlp.up,
        config.width,
        config.mlp_dim,
    )?;
    expect_linear(
        &format!("{name}.{index}.down"),
        &mlp.down,
        config.mlp_dim,
        config.width,
    )
}

fn expect_linear(name: &str, linear: &LinearWeights, input: usize, output: usize) -> Result<()> {
    expect_shape(name, &linear.weight, &[input, output])?;
    if let Some(bias) = &linear.bias {
        expect_shape(&format!("{name}.bias"), bias, &[output])?;
    }
    Ok(())
}

fn expect_shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "π0.5 {name}: expected shape {expected:?}, got {:?}",
            tensor.shape().dims()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_f32_is_physical() {
        let tensor = Tensor::from_f32(vec![2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
        let transposed = transpose_2d(&tensor).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.as_f32().unwrap(), &[1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn transpose_bf16_preserves_dtype() {
        let values = [1., 2., 3., 4.].map(bf16::from_f32);
        let tensor = Tensor::from_bf16(vec![2, 2], &values).unwrap();
        let transposed = transpose_2d(&tensor).unwrap();
        assert_eq!(transposed.dtype(), DType::BF16);
        assert_eq!(
            transposed
                .as_bf16()
                .unwrap()
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>(),
            vec![1., 3., 2., 4.]
        );
    }

    #[test]
    fn gemma_norm_offset_is_finalized_once() {
        let tensor = Tensor::from_f32(vec![3], &[-0.5, 0.0, 0.5]).unwrap();
        assert_eq!(add_one(tensor).unwrap().as_f32().unwrap(), &[0.5, 1.0, 1.5]);
    }

    #[test]
    fn missing_key_names_the_checkpoint_tensor() {
        let error = take(&mut HashMap::new(), "a.b.weight").unwrap_err();
        assert!(error.to_string().contains("a.b.weight"));
    }

    #[test]
    fn normalizes_lerobot_model_namespace() {
        let mut tensors = HashMap::from([
            (
                format!("model.{ROOT}.paligemma.lm_head.weight"),
                Tensor::zeros(vec![1], DType::F32),
            ),
            (
                "model.action_in_proj.bias".to_string(),
                Tensor::zeros(vec![1], DType::F32),
            ),
        ]);
        normalize_lerobot_prefix(&mut tensors);
        assert!(tensors.contains_key(&format!("{ROOT}.paligemma.lm_head.weight")));
        assert!(tensors.contains_key("action_in_proj.bias"));
    }

    #[test]
    fn synthetic_weights_pass_shape_validation() {
        // validate_shapes runs inside synthetic(); reaching Ok proves every
        // validated linear/embedding shape is consistent with the config.
        for config in [Pi05Config::default(), Pi05Config::thor_two_view()] {
            let weights = Pi05Weights::synthetic(&config, 7).unwrap();
            assert_eq!(weights.language_layers.len(), config.language.depth);
            assert_eq!(weights.action_layers.len(), config.action_expert.depth);
            assert_eq!(
                weights.vision.position_embedding.shape().dims(),
                &[config.patches_per_view(), config.vision_width]
            );
        }
    }

    #[test]
    fn synthetic_weights_are_deterministic() {
        let a = Pi05Weights::synthetic(&Pi05Config::thor_two_view(), 3).unwrap();
        let b = Pi05Weights::synthetic(&Pi05Config::thor_two_view(), 3).unwrap();
        assert_eq!(
            a.action_in.weight.as_bf16().unwrap(),
            b.action_in.weight.as_bf16().unwrap()
        );
    }
}
