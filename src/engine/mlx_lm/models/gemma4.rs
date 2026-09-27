//! Gemma-4 (text-only) inference — `model_type = gemma4` / `gemma4_text`.
//!
//! Supports `mlx-community/gemma-4-{e2b,e4b}-it-*` checkpoints. The vision and
//! audio towers of `Gemma4ForConditionalGeneration` are skipped: safetensors
//! keys are stripped of the `language_model.` prefix (see `load_gemma4_any` in
//! `mlx_native.rs`) and the `vision_tower.*` / `audio_tower.*` / `embed_vision.*`
//! / `embed_audio.*` weights are simply left unmatched.
//!
//! ## What Gemma-4 does differently from Gemma-3 (why this is a separate module)
//!
//! Ported faithfully from Apple's `mlx_lm/models/gemma4_text.py` (© 2025 Apple).
//!
//! - **Standard RMSNorm** (`weight * normalize(x)`), NOT Gemma's historical
//!   `(1 + weight)` form. The reference uses `nn.RMSNorm` directly with no `+1`
//!   and no sanitize fix-up, so the checkpoints store norm weights centred at
//!   ~1.0. We therefore use [`mlx_rs::nn::RmsNorm`] (and a scale-free variant
//!   for the value norm), unlike `gemma3::GemmaRmsNorm`.
//! - **Per-Layer Embeddings (PLE)** — the "effective params" mechanism behind
//!   the `E2B` / `E4B` naming. A second embedding table (`embed_tokens_per_layer`)
//!   plus a model-level projection feed a per-layer side input that is gated
//!   into each block's residual stream. See [`Gemma4TextModel::per_layer_inputs`].
//! - **Cross-layer KV sharing** — the last `num_kv_shared_layers` layers carry
//!   **no** `k_proj` / `v_proj` / `k_norm`; they reuse the (already-RoPE'd,
//!   already-cached) keys/values produced by the most recent earlier layer of
//!   the same attention type. See [`Gemma4TextModel::forward`].
//! - **Per-attention-type head dim** — `full_attention` layers use
//!   `global_head_dim` (512); `sliding_attention` layers use `head_dim` (256).
//! - **Proportional / partial RoPE** — `full_attention` layers rotate only
//!   `partial_rotary_factor * head_dim` dims (theta = 1e6); the rest pass
//!   through unrotated (encoded as `inf` frequencies). `sliding_attention`
//!   layers use plain RoPE (theta = 1e4). See [`ProportionalRope`].
//! - **Double-wide MLP** on KV-shared layers (`intermediate_size * 2`).
//! - **GeGLU** MLP (`gelu_approx(gate) * up`) and **`layer_scalar`** post-block
//!   scaling.
//! - **Attention scale is `1.0`** (queries are pre-normalised by `q_norm`),
//!   not `1/sqrt(head_dim)`.
//! - **Final logit softcap** = 30.0.
//!
//! ## Caching (matches the `gemma3` approach in this crate)
//!
//! Non-shared layers each own an FP16 KV cache (`fp16_with_max`); KV-shared
//! layers hold `None` and reuse intermediates. Sliding-window attention is
//! enforced **only via the prefill mask** — the KV buffer is never shrunk to
//! `sliding_window` (that would break SDPA shapes on a single-pass prefill).
//! As in `gemma3`, this means single-token decode attends to the full cache
//! rather than a strict window; accepted as a known approximation.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        arange, argpartition_axis, concatenate_axis, full, indexing::IndexOp, ones, power,
        softmax_axis, zeros_dtype,
    },
    quantization::{MaybeQuantized, Quantizable as _},
    Array, Dtype,
};
use serde::Deserialize;
use serde_json::Value;

use super::super::{
    cache::{KeyValueCache, KvCache, KvFetchResult},
    error::Error,
    utils::{create_causal_mask, moe::gather_qmm, scaled_dot_product_attention},
};

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// One entry of `config.json::text_config::rope_parameters`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeParam {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

/// Text config schema for `gemma4` / `gemma4_text`. Multimodal checkpoints nest
/// these under `text_config`; [`get_gemma4_model_args`] handles both.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub head_dim: i32,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub num_kv_shared_layers: i32,
    #[serde(default)]
    pub hidden_size_per_layer_input: i32,
    pub vocab_size: i32,
    #[serde(default)]
    pub vocab_size_per_layer_input: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    /// Second KV head count, used by **full-attention** layers on the k-eq-V
    /// checkpoints (26B/31B). `None` → full layers reuse
    /// [`Self::num_key_value_heads`] like the sliding ones.
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    /// Full-attention layers derive V from the **raw K projection** instead of
    /// carrying their own `v_proj` (26B/31B). The two still diverge afterwards:
    /// K takes the scaled per-head norm and RoPE, V takes the no-scale norm and
    /// no RoPE, so the cache still stores them separately.
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// Replace the single dense FFN with `dense shared FFN ‖ 128 routed
    /// experts` (26B-A4B). The two branches run in **parallel** off the same
    /// residual and are summed — not chained.
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<i32>,
    #[serde(default)]
    pub top_k_experts: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    /// Per-layer attention type. Defaults to `sliding_window_pattern-1` sliding
    /// layers followed by one full layer, repeated.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, RopeParam>>,

    /// EOS token id(s). Gemma-4 instruct lists `[1, 106, 50]`. Folded in by
    /// [`get_gemma4_model_args`] from the outer config scope.
    #[serde(skip)]
    pub eos_token_ids: Vec<u32>,
    /// BOS token (default 2). The Gemma chat template starts with `{{ bos_token }}`.
    #[serde(skip)]
    pub bos_token_id: Option<u32>,
    /// Quantization of the stacked routed-expert weights, read from
    /// `config.json::quantization` by [`get_gemma4_model_args`].
    ///
    /// These weights arrive from safetensors **already quantized and already
    /// stacked**, so unlike every other module the runtime never quantizes
    /// them — it only needs the group size and bit width to interpret them in
    /// `gather_qmm`. Getting either wrong is not a load error; it silently
    /// misreads the packed words.
    #[serde(skip, default = "default_expert_quant")]
    pub expert_quant: (i32, i32),
}

fn default_expert_quant() -> (i32, i32) {
    (64, 4)
}

/// Fully-specified MoE geometry (see [`ModelArgs::moe`]).
#[derive(Debug, Clone, Copy)]
pub struct MoeArgs {
    pub num_experts: i32,
    pub top_k: i32,
    pub intermediate_size: i32,
}

fn default_global_head_dim() -> i32 {
    0
}
fn default_sliding_window() -> i32 {
    512
}
fn default_sliding_window_pattern() -> i32 {
    5
}
fn default_tie_word_embeddings() -> bool {
    true
}

impl ModelArgs {
    /// `first_kv_shared_layer_idx` — layers at or above this index reuse KV.
    pub fn first_kv_shared(&self) -> i32 {
        self.num_hidden_layers - self.num_kv_shared_layers
    }

    /// Resolved per-layer attention types (from config or the default pattern).
    pub fn resolved_layer_types(&self) -> Vec<String> {
        if let Some(lt) = &self.layer_types {
            if lt.len() == self.num_hidden_layers as usize {
                return lt.clone();
            }
        }
        let p = self.sliding_window_pattern.max(1) as usize;
        let n = self.num_hidden_layers as usize;
        (0..n)
            .map(|i| {
                if (i + 1) % p == 0 {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect()
    }

    fn rope_for(&self, layer_type: &str) -> RopeParam {
        if let Some(rp) = &self.rope_parameters {
            if let Some(p) = rp.get(layer_type) {
                return p.clone();
            }
        }
        // Reference defaults.
        if layer_type == "full_attention" {
            RopeParam {
                rope_type: Some("proportional".into()),
                rope_theta: Some(1_000_000.0),
                partial_rotary_factor: Some(0.25),
            }
        } else {
            RopeParam {
                rope_type: Some("default".into()),
                rope_theta: Some(10_000.0),
                partial_rotary_factor: Some(1.0),
            }
        }
    }

    /// Whether a layer of this type derives V from the raw K projection.
    /// Only full-attention layers ever do; sliding layers always keep their own
    /// `v_proj`.
    fn k_eq_v_for(&self, layer_type: &str) -> bool {
        self.attention_k_eq_v && layer_type == "full_attention"
    }

    /// KV head count for a given attention type. The k-eq-V checkpoints give
    /// full-attention layers their own (smaller) count.
    fn n_kv_heads_for(&self, layer_type: &str) -> i32 {
        match self.num_global_key_value_heads {
            Some(n) if n > 0 && self.k_eq_v_for(layer_type) => n,
            _ => self.num_key_value_heads,
        }
    }

    /// MoE geometry, present only when the block is enabled *and* fully
    /// specified. A partially-declared MoE is treated as absent rather than
    /// half-built: every field is load-bearing, and a missing one would
    /// otherwise surface as a shape error deep in the expert matmul.
    pub fn moe(&self) -> Option<MoeArgs> {
        if !self.enable_moe_block {
            return None;
        }
        Some(MoeArgs {
            num_experts: self.num_experts.filter(|n| *n > 0)?,
            top_k: self.top_k_experts.filter(|k| *k > 0)?,
            intermediate_size: self.moe_intermediate_size.filter(|s| *s > 0)?,
        })
    }

    /// Head dim used by a given attention type.
    fn head_dim_for(&self, layer_type: &str) -> i32 {
        if layer_type == "full_attention" && self.global_head_dim > 0 {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }
}

// -----------------------------------------------------------------------------
// RoPE — default (sliding) or proportional/partial (full)
// -----------------------------------------------------------------------------

/// Partial / proportional RoPE: rotate the first `rotated_dims` of each head and
/// leave the rest untouched. Mirrors `mlx_lm.models.rope_utils.ProportionalRoPE`
/// — the non-rotated tail is encoded as `inf` frequencies (rotation angle
/// `pos/inf = 0`, i.e. identity).
#[derive(Debug, Clone, ModuleParameters)]
pub struct ProportionalRope {
    dims: i32,
    /// Precomputed frequencies (not a trainable parameter).
    freqs: Array,
}

impl ProportionalRope {
    pub fn new(dims: i32, rotated_dims: i32, base: f32, factor: f32) -> Result<Self, Exception> {
        // freqs_rotated[i] = factor * base^(2i / dims)  for i in 0..rotated_dims/2
        let exps = arange::<_, f32>(0, rotated_dims, 2)?.divide(&array!(dims as f32))?;
        let rotated = power(&array!(base), &exps)?.multiply(&array!(factor))?;
        let pad_len = ((dims - rotated_dims) / 2).max(0);
        let freqs = if pad_len > 0 {
            let pad = full::<f32>(&[pad_len], &array!(f32::INFINITY))?;
            concatenate_axis(&[rotated, pad], 0)?
        } else {
            rotated
        };
        Ok(Self { dims, freqs })
    }

    /// `x`: `[B, n_heads, L, head_dim]`.
    fn forward(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let shape = x.shape().to_vec();
        let x3 = x.reshape(&[-1, x.dim(-2), x.dim(-1)])?;
        let out = mlx_rs::fast::rope(
            &x3,
            self.dims,
            false,
            None::<f32>,
            1.0,
            offset,
            Some(&self.freqs),
        )?;
        out.reshape(&shape)
    }
}

#[derive(Debug, Clone)]
pub enum Gemma4Rope {
    Default(nn::Rope),
    Proportional(ProportionalRope),
}

impl Gemma4Rope {
    fn new(head_dim: i32, param: &RopeParam) -> Result<Self, Exception> {
        let theta = param.rope_theta.unwrap_or(10_000.0);
        let kind = param.rope_type.as_deref().unwrap_or("default");
        if kind == "proportional" {
            let factor = param.partial_rotary_factor.unwrap_or(1.0);
            let rotated = ((head_dim as f32) * factor).round() as i32;
            Ok(Self::Proportional(ProportionalRope::new(
                head_dim, rotated, theta, 1.0,
            )?))
        } else {
            let rope = nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(theta)
                .scale(1.0)
                .build()
                .expect("Infallible");
            Ok(Self::Default(rope))
        }
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array, Exception> {
        match self {
            Self::Default(rope) => {
                let input = nn::RopeInputBuilder::new(x).offset(offset).build()?;
                rope.forward(input)
            }
            Self::Proportional(p) => p.forward(x, offset),
        }
    }
}

// `Gemma4Rope` carries no trainable parameters (RoPE is config-only), but the
// `#[param]` field on `Attention` needs it to be a `ModuleParameters`. Delegate
// to the active variant (both expose empty parameter sets).
impl mlx_rs::module::ModuleParameters for Gemma4Rope {
    fn num_parameters(&self) -> usize {
        0
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Default(r) => r.freeze_parameters(recursive),
            Self::Proportional(r) => r.freeze_parameters(recursive),
        }
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Default(r) => r.unfreeze_parameters(recursive),
            Self::Proportional(r) => r.unfreeze_parameters(recursive),
        }
    }
    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            Self::Default(r) => r.parameters(),
            Self::Proportional(r) => r.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        match self {
            Self::Default(r) => r.parameters_mut(),
            Self::Proportional(r) => r.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            Self::Default(r) => r.trainable_parameters(),
            Self::Proportional(r) => r.trainable_parameters(),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Default(r) => r.all_frozen(),
            Self::Proportional(r) => r.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Default(r) => r.any_frozen(),
            Self::Proportional(r) => r.any_frozen(),
        }
    }
}

// -----------------------------------------------------------------------------
// Scale-free RMSNorm (value norm) — `rms_norm(x, ones, eps)`
// -----------------------------------------------------------------------------

fn rms_norm_no_scale(x: &Array, eps: f32) -> Result<Array, Exception> {
    let dim = x.dim(-1);
    let ones = ones::<f32>(&[dim])?;
    mlx_rs::fast::rms_norm(x, &ones, eps)
}

/// `rms_norm` with an explicit weight vector. The router needs this because its
/// effective weight is `scale · hidden^-0.5` — a value that is *not* the stored
/// tensor, so `nn::RmsNorm` (which owns its weight) cannot express it.
fn rms_norm_weighted(x: &Array, weight: &Array, eps: f32) -> Result<Array, Exception> {
    mlx_rs::fast::rms_norm(x, weight, eps)
}

// -----------------------------------------------------------------------------
// Attention
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub is_full: bool,
    /// `false` for KV-shared layers (no k/v projections of their own).
    pub has_kv: bool,
    /// V comes from the raw `k_proj` output; there is no `v_proj` to load.
    pub k_eq_v: bool,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub v_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
    #[param]
    pub rope: Gemma4Rope,
}

impl Attention {
    pub fn new(args: &ModelArgs, layer_idx: usize, layer_type: &str) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.n_kv_heads_for(layer_type);
        let head_dim = args.head_dim_for(layer_type);
        let is_full = layer_type == "full_attention";
        let has_kv = (layer_idx as i32) < args.first_kv_shared();
        // k-eq-V layers carry no `v_proj` at all: V is the *raw* K projection,
        // which then takes the no-scale norm and skips RoPE.
        let k_eq_v = args.k_eq_v_for(layer_type);

        let mk_lin = |i: i32, o: i32| nn::LinearBuilder::new(i, o).bias(false).build();
        let q_proj = mk_lin(dim, n_heads * head_dim)?;
        let o_proj = mk_lin(n_heads * head_dim, dim)?;
        let (k_proj, v_proj, k_norm) = if has_kv {
            (
                Some(MaybeQuantized::Original(mk_lin(
                    dim,
                    n_kv_heads * head_dim,
                )?)),
                if k_eq_v {
                    None
                } else {
                    Some(MaybeQuantized::Original(mk_lin(
                        dim,
                        n_kv_heads * head_dim,
                    )?))
                },
                Some(
                    nn::RmsNormBuilder::new(head_dim)
                        .eps(args.rms_norm_eps)
                        .build()?,
                ),
            )
        } else {
            (None, None, None)
        };

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let rope = Gemma4Rope::new(head_dim, &args.rope_for(layer_type))?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0,
            is_full,
            has_kv,
            k_eq_v,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj,
            v_proj,
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
        })
    }

    /// Returns `(output, keys, values)` where `keys`/`values` are the full
    /// (RoPE'd, cached) tensors — captured so KV-shared layers can reuse them.
    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KvCache>,
        shared_kv: Option<(&Array, &Array)>,
        rope_offset: i32,
    ) -> Result<(Array, Array, Array), Exception> {
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let queries = self
            .q_norm
            .forward(&queries.reshape(&[B, L, self.n_heads, -1])?)?;
        let mut queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        queries = self.rope.forward(&queries, rope_offset)?;

        // Resolve keys/values: either reused from an earlier layer, or freshly
        // projected, normed and RoPE'd here.
        let (keys, values) = if let Some((k, v)) = shared_kv {
            (k.clone(), v.clone())
        } else {
            let k_proj = self
                .k_proj
                .as_mut()
                .ok_or_else(|| Exception::custom("non-shared layer missing k_proj"))?;
            let raw_k = k_proj.forward(x)?;
            // k-eq-V: the *raw* K projection is also the raw V. The paths
            // diverge immediately below — K gets the scaled per-head norm and
            // RoPE, V the no-scale norm and no RoPE — so the two still land in
            // the cache as distinct tensors.
            let raw_v = if self.k_eq_v {
                raw_k.clone()
            } else {
                let v_proj = self
                    .v_proj
                    .as_mut()
                    .ok_or_else(|| Exception::custom("non-shared layer missing v_proj"))?;
                v_proj.forward(x)?
            };

            let k_norm = self
                .k_norm
                .as_mut()
                .ok_or_else(|| Exception::custom("non-shared layer missing k_norm"))?;
            let keys = k_norm.forward(&raw_k.reshape(&[B, L, self.n_kv_heads, -1])?)?;
            let mut keys = keys.transpose_axes(&[0, 2, 1, 3])?;
            keys = self.rope.forward(&keys, rope_offset)?;

            let values = rms_norm_no_scale(&raw_v.reshape(&[B, L, self.n_kv_heads, -1])?, 1e-6)?;
            let values = values.transpose_axes(&[0, 2, 1, 3])?;
            (keys, values)
        };

        // Non-shared layers append to and fetch from their own cache; shared
        // layers reuse the (already full) tensors directly.
        let (k_full, v_full) = if let Some(cache) = cache {
            match cache.update_and_fetch(keys, values)? {
                KvFetchResult::Fp16(k, v) => (k, v),
                KvFetchResult::TurboQuant => {
                    return Err(Exception::custom(
                        "Gemma-4 KV cache must be FP16 (TurboQuant not wired)",
                    ));
                }
            }
        } else {
            (keys, values)
        };

        let output = scaled_dot_product_attention(
            queries,
            k_full.clone(),
            v_full.clone(),
            None::<&mut KvCache>,
            self.scale,
            mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        Ok((self.o_proj.forward(&output)?, k_full, v_full))
    }
}

// -----------------------------------------------------------------------------
// MLP — GeGLU
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let mk = |i: i32, o: i32| nn::LinearBuilder::new(i, o).bias(false).build();
        Ok(Self {
            gate_proj: MaybeQuantized::Original(mk(dim, hidden_dim)?),
            up_proj: MaybeQuantized::Original(mk(dim, hidden_dim)?),
            down_proj: MaybeQuantized::Original(mk(hidden_dim, dim)?),
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gated =
            nn::gelu_approximate(self.gate_proj.forward(x)?)?.multiply(self.up_proj.forward(x)?)?;
        self.down_proj.forward(&gated)
    }
}

// -----------------------------------------------------------------------------
// Decoder layer — 4 norms + attn + mlp + per-layer-input gate + layer scalar
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// MoE — router + stacked routed experts (26B-A4B)
// -----------------------------------------------------------------------------

/// Expert router: `rms_norm(x, scale · hidden⁻¹ᐟ²) → proj → top-k → softmax →
/// × per_expert_scale`.
///
/// Two details that are easy to get wrong and produce plausible-looking
/// garbage rather than an error:
///
/// 1. **The softmax is over the k selected scores only**, not the full expert
///    axis. Softmaxing all 128 and then gathering k would give weights that no
///    longer sum to 1, silently scaling every routed contribution down.
/// 2. **The norm weight folds in `hidden_size^-0.5`.** The checkpoint's
///    `router.scale` is the bare learned vector; the `1/√hidden` factor is part
///    of the op, not the tensor.
///
/// Note the router reads the *unnormalized* block input, the same tensor the
/// dense branch normalizes with its own `pre_feedforward_layernorm`.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Router {
    pub top_k: i32,
    pub eps: f32,
    /// `hidden_size^-0.5`, folded into the norm weight at call time.
    pub root_size: f32,

    /// 8-bit in the shipped checkpoint (per-module override in
    /// `config.json::quantization`), unlike the 4-bit experts around it.
    #[quantizable]
    #[param]
    pub proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub scale: Param<Array>,
    #[param]
    pub per_expert_scale: Param<Array>,
}

impl Router {
    pub fn new(args: &ModelArgs, moe: MoeArgs) -> Result<Self, Exception> {
        Ok(Self {
            top_k: moe.top_k,
            eps: args.rms_norm_eps,
            root_size: (args.hidden_size as f32).powf(-0.5),
            proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, moe.num_experts)
                    .bias(false)
                    .build()?,
            ),
            scale: Param::new(ones::<f32>(&[args.hidden_size])?),
            per_expert_scale: Param::new(ones::<f32>(&[moe.num_experts])?),
        })
    }

    /// `x [B, L, hidden]` → `(indices [B, L, K], weights [B, L, K])`.
    fn forward(&mut self, x: &Array) -> Result<(Array, Array), Exception> {
        let weight = self.scale.as_ref().multiply(&array!(self.root_size))?;
        let normed = rms_norm_weighted(x, &weight, self.eps)?;
        let scores = self.proj.forward(&normed)?;

        // Top-k experts. Order within the k is irrelevant — the combine is a
        // weighted sum — but the softmax and the `per_expert_scale` gather must
        // use the *same* index tensor, hence one `indices` for both.
        let k = self.top_k;
        let part = argpartition_axis(&scores.negative()?, k - 1, -1)?;
        let indices = part.index((.., .., 0..k));

        let picked = scores.take_along_axis(&indices, -1)?;
        let weights = softmax_axis(&picked, -1, Some(true))?;
        let per_expert = self.per_expert_scale.as_ref().take_along_axis(&indices, -1)?;
        Ok((indices, weights.multiply(&per_expert)?))
    }
}

/// One stacked quantized expert projection: `weight [E, out, in_packed]` with
/// `scales`/`biases [E, out, in/group_size]`.
///
/// Loaded verbatim from the checkpoint's `experts.switch_glu.*` tensors, which
/// ship already stacked *and* already quantized — so these are plain `#[param]`
/// arrays and must be excluded from the runtime per-module quantize pass.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchLinear {
    pub group_size: i32,
    pub bits: i32,
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,
}

impl SwitchLinear {
    fn placeholder(group_size: i32, bits: i32) -> Result<Self, Exception> {
        // Real shapes arrive from the weight loader; 1-element placeholders
        // just make the parameter keys exist for matching.
        let z = || zeros_dtype(&[1], Dtype::Float32);
        Ok(Self {
            group_size,
            bits,
            weight: Param::new(z()?),
            scales: Param::new(z()?),
            biases: Param::new(z()?),
        })
    }

    fn forward(&self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        gather_qmm(
            x,
            self.weight.as_ref(),
            self.scales.as_ref(),
            self.biases.as_ref(),
            indices,
            true,
            self.group_size,
            self.bits,
            false,
        )
    }
}

/// The 128 routed experts as three stacked matmuls, gated GeGLU-style.
///
/// Same shape as the DeepSeek switch-MLP, with **GeGLU** rather than SwiGLU:
/// `down(gelu_approx(gate(x)) · up(x))`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGlu {
    #[param]
    pub gate_proj: SwitchLinear,
    #[param]
    pub up_proj: SwitchLinear,
    #[param]
    pub down_proj: SwitchLinear,
}

impl SwitchGlu {
    fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: SwitchLinear::placeholder(group_size, bits)?,
            up_proj: SwitchLinear::placeholder(group_size, bits)?,
            down_proj: SwitchLinear::placeholder(group_size, bits)?,
        })
    }

    /// `x [B, L, hidden]`, `indices [B, L, K]` → `[B, L, K, hidden]`.
    #[allow(non_snake_case)]
    fn forward(&self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let s = x.shape();
        let (B, L, H) = (s[0], s[1], s[2]);
        // `[B, L, 1, 1, hidden]` so the batch dims broadcast against the
        // `[B, L, K]` index tensor and the matmul sees M = 1.
        let x_e = x.reshape(&[B, L, 1, 1, H])?;
        let gate = nn::gelu_approximate(self.gate_proj.forward(&x_e, indices)?)?;
        let up = self.up_proj.forward(&x_e, indices)?;
        let h = gate.multiply(&up)?;
        let out = self.down_proj.forward(&h, indices)?;
        let os = out.shape();
        out.reshape(&[os[0], os[1], os[2], os[4]])
    }
}

/// Routed-expert branch: run the selected experts and combine them by router
/// weight.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Experts {
    #[param]
    pub switch_glu: SwitchGlu,
}

impl Experts {
    pub fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        Ok(Self {
            switch_glu: SwitchGlu::new(group_size, bits)?,
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &self,
        x: &Array,
        indices: &Array,
        weights: &Array,
    ) -> Result<Array, Exception> {
        let s = weights.shape();
        let (B, L, K) = (s[0], s[1], s[2]);
        let y = self.switch_glu.forward(x, indices)?; // [B, L, K, hidden]
        y.multiply(&weights.reshape(&[B, L, K, 1])?)?
            .sum_axes(&[-2], false)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,

    // MoE (26B-A4B). Present together or not at all — when set, the dense
    // `mlp` stops being the whole FFN and becomes the *shared* branch of a
    // parallel pair. These are flat fields rather than a nested `MoeBlock` so
    // the flattened parameter keys match the checkpoint's tensor names
    // verbatim (`model.layers.N.router.proj.weight`, …) and the weight loader
    // needs no remapping.
    #[quantizable]
    #[param]
    pub router: Option<Router>,
    #[param]
    pub experts: Option<Experts>,
    /// Wraps the **dense** branch's output. Despite the name it is not a second
    /// copy of `post_feedforward_layernorm`, which still wraps the sum.
    #[param]
    pub post_feedforward_layernorm_1: Option<nn::RmsNorm>,
    #[param]
    pub post_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    pub pre_feedforward_layernorm_2: Option<nn::RmsNorm>,

    // Per-layer input gating (PLE). Present when `hidden_size_per_layer_input > 0`.
    #[quantizable]
    #[param]
    pub per_layer_input_gate: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    pub per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub post_per_layer_input_norm: Option<nn::RmsNorm>,

    #[param]
    pub layer_scalar: Param<Array>,
}

impl DecoderLayer {
    pub fn new(
        args: &ModelArgs,
        layer_idx: usize,
        expert_group_size: i32,
        expert_bits: i32,
    ) -> Result<Self, Exception> {
        let layer_types = args.resolved_layer_types();
        let layer_type = &layer_types[layer_idx];
        let self_attn = Attention::new(args, layer_idx, layer_type)?;

        let is_kv_shared =
            (layer_idx as i32) >= args.first_kv_shared() && args.first_kv_shared() > 0;
        let use_double_wide = args.use_double_wide_mlp && is_kv_shared;
        let inter = args.intermediate_size * if use_double_wide { 2 } else { 1 };
        let mlp = Mlp::new(args.hidden_size, inter)?;

        let mk_norm = || {
            nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()
        };

        let hpl = args.hidden_size_per_layer_input;
        let (gate, proj, norm) = if hpl > 0 {
            (
                Some(MaybeQuantized::Original(
                    nn::LinearBuilder::new(args.hidden_size, hpl)
                        .bias(false)
                        .build()?,
                )),
                Some(MaybeQuantized::Original(
                    nn::LinearBuilder::new(hpl, args.hidden_size)
                        .bias(false)
                        .build()?,
                )),
                Some(mk_norm()?),
            )
        } else {
            (None, None, None)
        };

        let (router, experts, post_ffn_1, post_ffn_2, pre_ffn_2) = match args.moe() {
            None => (None, None, None, None, None),
            Some(m) => (
                Some(Router::new(args, m)?),
                Some(Experts::new(expert_group_size, expert_bits)?),
                Some(mk_norm()?),
                Some(mk_norm()?),
                Some(mk_norm()?),
            ),
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: mk_norm()?,
            post_attention_layernorm: mk_norm()?,
            pre_feedforward_layernorm: mk_norm()?,
            post_feedforward_layernorm: mk_norm()?,
            router,
            experts,
            post_feedforward_layernorm_1: post_ffn_1,
            post_feedforward_layernorm_2: post_ffn_2,
            pre_feedforward_layernorm_2: pre_ffn_2,
            per_layer_input_gate: gate,
            per_layer_projection: proj,
            post_per_layer_input_norm: norm,
            layer_scalar: Param::new(ones::<f32>(&[1])?),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KvCache>,
        per_layer_input: Option<&Array>,
        shared_kv: Option<(&Array, &Array)>,
        rope_offset: i32,
    ) -> Result<(Array, Array, Array), Exception> {
        // h = x + post_attn(attn(input_norm(x)))
        let normed = self.input_layernorm.forward(x)?;
        let (attn_out, k_full, v_full) =
            self.self_attn
                .forward(&normed, mask, cache, shared_kv, rope_offset)?;
        let h = x.add(&self.post_attention_layernorm.forward(&attn_out)?)?;

        // Feed-forward. Dense checkpoints run one branch; the MoE checkpoint
        // runs the dense (shared) branch and the routed-expert branch **in
        // parallel** off the same `h` and sums them. Note the router reads the
        // raw `h`, not either branch's normalized input.
        let ffn_out = match (
            self.router.as_mut(),
            self.experts.as_ref(),
            self.post_feedforward_layernorm_1.as_mut(),
            self.post_feedforward_layernorm_2.as_mut(),
            self.pre_feedforward_layernorm_2.as_mut(),
        ) {
            (Some(router), Some(experts), Some(post_1), Some(post_2), Some(pre_2)) => {
                let shared_in = self.pre_feedforward_layernorm.forward(&h)?;
                let shared = post_1.forward(&self.mlp.forward(&shared_in)?)?;

                let (indices, weights) = router.forward(&h)?;
                let routed_in = pre_2.forward(&h)?;
                let routed = post_2.forward(&experts.forward(&routed_in, &indices, &weights)?)?;

                shared.add(&routed)?
            }
            _ => {
                let ffn_in = self.pre_feedforward_layernorm.forward(&h)?;
                self.mlp.forward(&ffn_in)?
            }
        };
        let mut h = h.add(&self.post_feedforward_layernorm.forward(&ffn_out)?)?;

        // Per-layer input gating.
        if let (Some(gate_lin), Some(proj_lin), Some(norm), Some(ple)) = (
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
            per_layer_input,
        ) {
            let residual = h.clone();
            let gate = nn::gelu_approximate(gate_lin.forward(&h)?)?;
            let gate = gate.multiply(ple)?;
            let gate = proj_lin.forward(&gate)?;
            let gate = norm.forward(&gate)?;
            h = residual.add(&gate)?;
        }

        // Learned per-layer scalar.
        h = h.multiply(self.layer_scalar.as_ref())?;

        Ok((h, k_full, v_full))
    }
}

// -----------------------------------------------------------------------------
// Backbone
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma4TextModel {
    pub args: ModelArgs,
    pub layer_types: Vec<String>,
    /// For each layer, the index of the layer whose KV it reuses (== own index
    /// for non-shared layers).
    pub previous_kvs: Vec<usize>,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub embed_tokens_per_layer: Option<MaybeQuantized<nn::Embedding>>,
    /// Quantization for this module differs by checkpoint flavour:
    /// - regular `gemma-4-*-it-4bit`: stored bf16 (NOT quantized) — keep
    ///   `MaybeQuantized::Original` and skip in the per-module quantize loop.
    /// - `gemma-4-*-OptiQ-4bit`: stored 8-bit per OptiQ's per-module overrides.
    ///
    /// The loader inspects `model.safetensors.index.json` and only quantizes
    /// the slots that ship `.scales` weights, so both layouts load correctly.
    #[quantizable]
    #[param]
    pub per_layer_model_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub per_layer_projection_norm: Option<nn::RmsNorm>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Gemma4TextModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n = args.num_hidden_layers as usize;
        let layer_types = args.resolved_layer_types();

        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let hpl = args.hidden_size_per_layer_input;
        let (eple, plmp, plpn) = if hpl > 0 {
            (
                Some(MaybeQuantized::Original(nn::Embedding::new(
                    args.vocab_size_per_layer_input,
                    args.num_hidden_layers * hpl,
                )?)),
                Some(MaybeQuantized::Original(
                    nn::LinearBuilder::new(args.hidden_size, args.num_hidden_layers * hpl)
                        .bias(false)
                        .build()?,
                )),
                Some(
                    nn::RmsNormBuilder::new(hpl)
                        .eps(args.rms_norm_eps)
                        .build()?,
                ),
            )
        } else {
            (None, None, None)
        };

        let (eg, eb) = args.expert_quant;
        let layers = (0..n)
            .map(|i| DecoderLayer::new(args, i, eg, eb))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        // Map each KV-shared layer to the most recent earlier layer of the same
        // attention type (mirrors the reference `previous_kvs` construction).
        let mut previous_kvs: Vec<usize> = (0..n).collect();
        let m = args.first_kv_shared();
        if args.num_kv_shared_layers > 0 && m > 0 {
            let m = m as usize;
            let mut last_of_type: HashMap<&str, usize> = HashMap::new();
            for (i, lt) in layer_types.iter().enumerate().take(m) {
                last_of_type.insert(lt.as_str(), i);
            }
            for (j, slot) in previous_kvs.iter_mut().enumerate().take(n).skip(m) {
                if let Some(src) = last_of_type.get(layer_types[j].as_str()) {
                    *slot = *src;
                }
            }
        }

        Ok(Self {
            args: args.clone(),
            layer_types,
            previous_kvs,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            embed_tokens_per_layer: eple,
            per_layer_model_projection: plmp,
            per_layer_projection_norm: plpn,
            layers,
            norm,
        })
    }

    fn embed(&mut self, inputs: &Array) -> Result<Array, Exception> {
        match &mut self.embed_tokens {
            MaybeQuantized::Original(e) => e.forward(inputs),
            MaybeQuantized::Quantized(q) => q.forward(inputs),
        }
    }

    /// Build the `[B, L, num_layers, hidden_per_layer]` per-layer side input.
    fn per_layer_inputs(
        &mut self,
        inputs: &Array,
        scaled_emb: &Array,
    ) -> Result<Option<Array>, Exception> {
        let hpl = self.args.hidden_size_per_layer_input;
        if hpl <= 0 {
            return Ok(None);
        }
        let n = self.args.num_hidden_layers;
        let shape = scaled_emb.shape();
        let (b, l) = (shape[0], shape[1]);

        // Lookup table contribution.
        let eple = self.embed_tokens_per_layer.as_mut().ok_or_else(|| {
            Exception::custom("PLE configured but embed_tokens_per_layer missing")
        })?;
        let ple = match eple {
            MaybeQuantized::Original(e) => e.forward(inputs)?,
            MaybeQuantized::Quantized(q) => q.forward(inputs)?,
        };
        let ple = ple
            .multiply(&array!((hpl as f32).sqrt()))?
            .reshape(&[b, l, n, hpl])?;

        // Model-projection contribution.
        let plmp = self.per_layer_model_projection.as_mut().ok_or_else(|| {
            Exception::custom("PLE configured but per_layer_model_projection missing")
        })?;
        let proj = plmp.forward(scaled_emb)?;
        let proj = proj
            .multiply(&array!((self.args.hidden_size as f32).powf(-0.5)))?
            .reshape(&[b, l, n, hpl])?;
        let norm = self.per_layer_projection_norm.as_mut().ok_or_else(|| {
            Exception::custom("PLE configured but per_layer_projection_norm missing")
        })?;
        let proj = norm.forward(&proj)?;

        // Combine: (proj + ple) * 2^-0.5.
        let combined = proj.add(&ple)?.multiply(&array!(2.0_f32.powf(-0.5)))?;
        Ok(Some(combined))
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut [Option<KvCache>],
        rope_offset: usize,
    ) -> Result<Array, Exception> {
        let n = self.layers.len();
        let rope_off = i32::try_from(rope_offset)
            .map_err(|_| Exception::custom("rope_offset exceeds i32::MAX"))?;

        let emb = self.embed(inputs)?;
        let scaled_emb = emb.multiply(&array!((self.args.hidden_size as f32).sqrt()))?;
        let per_layer = self.per_layer_inputs(inputs, &scaled_emb)?;

        // Per-layer masks. Full-attention layers need a causal mask only during
        // multi-token prefill — single-token decode (`seq == 1`) attends the
        // whole cache, which is correct for full attention, so `None`.
        //
        // Sliding-window layers need an explicit windowed causal mask during
        // multi-token prefill: `create_causal_mask(seq, offset, window)`
        // restricts each query to the last `sliding_window` keys.
        //
        // On single-token decode (`seq == 1`) sliding layers also use `None`:
        // their KV cache is itself bounded to `sliding_window` (see
        // `make_caches` → `with_decode_window`), so after the per-step eviction
        // the buffer holds *exactly* the in-window keys and the query attends
        // all of them — same as full-attention decode. This is equivalent to
        // the old `create_causal_mask(1, offset, window)` (which masked the
        // out-of-window keys to zero) but avoids both storing those keys and
        // rebuilding an O(context) mask on every decode step.
        //
        // (This is sound only because the sliding cache is windowed. If a
        // sliding layer ran on an unbounded cache, `None` here would let it
        // attend the full history — the latent gemma3 decode bug. `make_caches`
        // guarantees the windowing whenever `window < cap`, and when
        // `window >= cap` the buffer never exceeds the window anyway.)
        let seq = scaled_emb.dim(1);
        let full_mask = if seq <= 1 {
            None
        } else {
            Some(create_causal_mask(seq, Some(rope_off), None, None)?)
        };
        let sliding_mask = if seq <= 1 {
            None
        } else {
            Some(create_causal_mask(
                seq,
                Some(rope_off),
                Some(self.args.sliding_window),
                None,
            )?)
        };

        let mut h = scaled_emb;
        // Captured (keys, values) per layer for KV reuse.
        let mut intermediates: Vec<Option<(Array, Array)>> = vec![None; n];

        for idx in 0..n {
            let is_full = self.layer_types[idx] == "full_attention";
            let mask = if is_full {
                full_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            // Per-layer side input slice `[B, L, hpl]`.
            let ple_slice = match &per_layer {
                Some(p) => Some(p.index((.., .., idx as i32, ..))),
                None => None,
            };

            let prev_idx = self.previous_kvs[idx];
            let is_shared = prev_idx != idx;

            let (h_new, k_full, v_full) = if is_shared {
                let (k, v) = intermediates[prev_idx]
                    .clone()
                    .ok_or_else(|| Exception::custom("shared layer has no source KV"))?;
                self.layers[idx].forward(
                    &h,
                    mask,
                    None,
                    ple_slice.as_ref(),
                    Some((&k, &v)),
                    rope_off,
                )?
            } else {
                // SAFETY: caches and layers are disjoint structures; split the
                // borrow so we can hold `&mut KvCache` and `&mut DecoderLayer`.
                let cache = caches.get_mut(idx).and_then(|c| c.as_mut());
                self.layers[idx].forward(&h, mask, cache, ple_slice.as_ref(), None, rope_off)?
            };

            h = h_new;
            intermediates[idx] = Some((k_full, v_full));
        }

        self.norm.forward(&h)
    }
}

// -----------------------------------------------------------------------------
// Top-level model
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: Gemma4TextModel,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,

    /// Vision-side projection from the (Phase 2) vision tower's hidden space
    /// into the text decoder's embedding space. Always present when the
    /// Gemma-4 model is loadable so the safetensors loader can land the
    /// `embed_vision.embedding_projection.{weight,scales,biases}` keys (3
    /// keys in the e2b 4-bit checkpoint). Phase 1 loads it; Phase 3 wires it
    /// into `forward_with_images`.
    #[quantizable]
    #[param]
    pub embed_vision: Option<super::gemma4_vision::MultimodalEmbedder>,
}

/// Vision-tower hidden size for the Gemma-4 E2B / E4B checkpoints.
/// Hard-coded for Phase 1; Phase 2 will parse this from
/// `config.json::vision_config.hidden_size` when porting the vision tower.
const PHASE1_VISION_HIDDEN_SIZE: i32 = 768;

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Gemma4TextModel::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };

        let embed_vision = Some(super::gemma4_vision::MultimodalEmbedder::new(
            PHASE1_VISION_HIDDEN_SIZE,
            args.hidden_size,
            args.rms_norm_eps,
        )?);

        Ok(Self {
            args,
            model,
            lm_head,
            embed_vision,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// FP16 KV cache for non-shared layers only; KV-shared layers hold `None`
    /// and reuse intermediates. Length == `num_hidden_layers`.
    ///
    /// Sliding-window layers get a **decode-time windowed** cache capped at
    /// `sliding_window`: prefill stays unbounded (every query keeps its full
    /// window; the prefix-cache snapshot is exact), but once decode starts the
    /// KV is evicted to the last `sliding_window` keys. Since those layers only
    /// ever attend the last `sliding_window` positions, this changes nothing
    /// about the output (the dropped keys were already masked to zero) while
    /// bounding decode-phase KV memory and per-step attention cost to the window
    /// instead of the full sequence length. Full-attention layers keep the
    /// global `max_kv_tokens` cap.
    pub fn make_caches(&self, max_kv_tokens: i32) -> Vec<Option<KvCache>> {
        let cap = max_kv_tokens.max(1);
        let window = self.args.sliding_window.max(1);
        let first_shared = self.args.first_kv_shared();
        (0..self.args.num_hidden_layers)
            .map(|i| {
                let owns_kv = first_shared <= 0 || i < first_shared;
                if !owns_kv {
                    return None;
                }
                let is_sliding = self.model.layer_types[i as usize] != "full_attention";
                if is_sliding && window < cap {
                    Some(KvCache::fp16_decode_windowed(cap, window))
                } else {
                    // Full-attention layer, or window ≥ cap (the global cap is
                    // already ≤ window, so decode never exceeds the window).
                    Some(KvCache::fp16_with_max(cap))
                }
            })
            .collect()
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut [Option<KvCache>],
        rope_offset: usize,
    ) -> Result<Array, Exception> {
        let out = self.model.forward(inputs, caches, rope_offset)?;
        let logits = match self.lm_head.as_mut() {
            Some(lm) => lm.forward(&out)?,
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&out)?,
                MaybeQuantized::Quantized(q) => q.as_linear(&out)?,
            },
        };
        if let Some(cap) = self.args.final_logit_softcapping {
            let cap_a = array!(cap);
            return mlx_rs::ops::tanh(&logits.divide(&cap_a)?)?.multiply(&cap_a);
        }
        Ok(logits)
    }

    pub fn eval(&self) -> Result<(), Exception> {
        use mlx_rs::module::ModuleParameters;
        mlx_rs::transforms::eval(self.parameters().flatten().values().copied())?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Config loader
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: Option<HashMap<String, Value>>,
    pub weight_map: HashMap<String, String>,
}

/// Parse `config.json` for `gemma4` (multimodal wrapper, `text_config` nested)
/// or `gemma4_text` (top-level). Folds the chat-tuned `eos_token_id`
/// (`[1, 106, 50]`) and `bos_token_id` from the outer scope.
pub fn get_gemma4_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let path = model_dir.as_ref().join("config.json");
    let raw = std::fs::read_to_string(&path)?;
    let root: Value = serde_json::from_str(&raw)?;

    let text_obj = match root.get("text_config") {
        Some(inner) => inner.clone(),
        None => root.clone(),
    };
    let mut args: ModelArgs = serde_json::from_value(text_obj)?;

    let eos_value = root.get("eos_token_id").cloned().or_else(|| {
        root.get("text_config")
            .and_then(|t| t.get("eos_token_id"))
            .cloned()
    });
    args.eos_token_ids = match eos_value {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(arr)) => arr
            .into_iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => vec![],
    };
    args.bos_token_id = root
        .get("bos_token_id")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .or(Some(2));

    // Routed experts follow the checkpoint's top-level quantization, not any
    // per-module override (the overrides in this checkpoint target
    // `router.proj`, which is 8-bit and quantized the normal way).
    if let Some(q) = root
        .get("quantization")
        .or_else(|| root.get("quantization_config"))
    {
        let g = q.get("group_size").and_then(|v| v.as_i64());
        let b = q.get("bits").and_then(|v| v.as_i64());
        if let (Some(g), Some(b)) = (g, b) {
            args.expert_quant = (g as i32, b as i32);
        }
    }

    Ok(args)
}

// -----------------------------------------------------------------------------
// Output-stream parser — Gemma-4 harmony tool-call body
// -----------------------------------------------------------------------------

/// Parser primitives for Gemma-4's harmony output dialect — colocated with
/// the model so all Gemma-4-specific format knowledge (tool-call body
/// grammar, `<|"|>` string wrapper, argument serialization) lives in one
/// place. The generic state machine in
/// [`crate::engine::stream_parser`] dispatches to these via the
/// `ToolCallFormat::Gemma4Compact` discriminant.
///
/// Wire format (from the chat template):
///
/// ```text
///   <|tool_call>call:NAME{key:val,key:val,…}<tool_call|>
/// ```
///
/// Argument values:
/// - strings: `<|"|>some string<|"|>`
/// - bare scalars: `true`, `false`, `null`, integers, floats
/// - nested objects: `{key:val,…}` (keys unquoted under `escape_keys=False`)
/// - arrays: `[val,val,…]`
pub mod parser {
    use serde_json::{Map, Value};

    /// Quote wrapper for string values — `<|"|>foo<|"|>` → `foo`.
    pub const QUOTE: &str = "<|\"|>";

    /// Parse `call:NAME{args}` body into the OpenAI-shape tool_call object
    /// (`{id, type:"function", function:{name, arguments:String}}`).
    pub fn parse_tool_call_body(body: &str, idx: usize) -> Option<Value> {
        let body = body.trim();
        let body = body.strip_prefix("call:").unwrap_or(body).trim();
        let brace = body.find('{')?;
        let name = body[..brace].trim();
        if name.is_empty() {
            return None;
        }
        let (args_val, _) = parse_value(body[brace..].trim());
        let arguments = serde_json::to_string(&args_val).unwrap_or_else(|_| "{}".to_string());
        Some(serde_json::json!({
            "id": format!("local_tool_{idx}"),
            "type": "function",
            "function": { "name": name, "arguments": arguments }
        }))
    }

    /// Recursive-descent parser for one argument value. Returns the parsed
    /// `Value` and the unconsumed remainder. Handles strings (quote-wrapped),
    /// objects, arrays, and scalars (bool / int / float / null / fallback string).
    pub fn parse_value(s: &str) -> (Value, &str) {
        let s = s.trim_start();

        // String wrapped in the gemma4 quote token. Some quants additionally
        // wrap the inner value in standard JSON quotes (e.g. the 4-bit Gemma-4
        // emits `<|"|>"google"<|"|>`), so trim any redundant `"` from the span.
        if let Some(r) = s.strip_prefix(QUOTE) {
            return match r.find(QUOTE) {
                Some(end) => (
                    Value::String(r[..end].trim_matches('"').to_string()),
                    &r[end + QUOTE.len()..],
                ),
                None => (Value::String(r.trim_matches('"').to_string()), ""),
            };
        }

        // Plain JSON-style string `"value"` — model emitted standard double
        // quotes with no gemma4 quote token at all.
        if let Some(r) = s.strip_prefix('"') {
            return match r.find('"') {
                Some(end) => (Value::String(r[..end].to_string()), &r[end + 1..]),
                None => (Value::String(r.to_string()), ""),
            };
        }

        if let Some(r) = s.strip_prefix('{') {
            let mut obj = Map::new();
            let mut rest = r;
            loop {
                rest = rest.trim_start();
                if let Some(a) = rest.strip_prefix('}') {
                    return (Value::Object(obj), a);
                }
                let Some(colon) = rest.find(':') else {
                    return (Value::Object(obj), rest);
                };
                // Keys may arrive bare (`engine:`), quote-token-wrapped, or —
                // in the 4-bit Gemma-4 — JSON-quoted (`"engine":`). Strip both
                // the gemma4 quote token and standard `"` so the map key is clean.
                let key = rest[..colon]
                    .trim()
                    .trim_start_matches(QUOTE)
                    .trim_end_matches(QUOTE)
                    .trim()
                    .trim_matches('"')
                    .to_string();
                let (val, after) = parse_value(&rest[colon + 1..]);
                obj.insert(key, val);
                rest = after.trim_start();
                if let Some(a) = rest.strip_prefix(',') {
                    rest = a;
                } else if let Some(a) = rest.strip_prefix('}') {
                    return (Value::Object(obj), a);
                } else {
                    return (Value::Object(obj), rest);
                }
            }
        }

        if let Some(r) = s.strip_prefix('[') {
            let mut arr = Vec::new();
            let mut rest = r;
            loop {
                rest = rest.trim_start();
                if let Some(a) = rest.strip_prefix(']') {
                    return (Value::Array(arr), a);
                }
                let (val, after) = parse_value(rest);
                arr.push(val);
                rest = after.trim_start();
                if let Some(a) = rest.strip_prefix(',') {
                    rest = a;
                } else if let Some(a) = rest.strip_prefix(']') {
                    return (Value::Array(arr), a);
                } else {
                    return (Value::Array(arr), rest);
                }
            }
        }

        let end = s.find([',', '}', ']']).unwrap_or(s.len());
        let raw = s[..end].trim();
        let val = if raw == "true" {
            Value::Bool(true)
        } else if raw == "false" {
            Value::Bool(false)
        } else if raw == "null" {
            Value::Null
        } else if let Ok(i) = raw.parse::<i64>() {
            Value::from(i)
        } else if let Ok(f) = raw.parse::<f64>() {
            Value::from(f)
        } else {
            Value::String(raw.to_string())
        };
        (val, &s[end..])
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_simple_tool_call_with_string_arg() {
            let tc =
                parse_tool_call_body("call:Skill{skill:<|\"|>agent-browser<|\"|>}", 0).unwrap();
            assert_eq!(tc["function"]["name"], "Skill");
            let args: Value =
                serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(args["skill"], "agent-browser");
        }

        #[test]
        fn parses_mixed_typed_args() {
            let tc = parse_tool_call_body("call:search{q:<|\"|>vàng<|\"|>,limit:5,fresh:true}", 7)
                .unwrap();
            assert_eq!(tc["id"], "local_tool_7");
            let args: Value =
                serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(args["q"], "vàng");
            assert_eq!(args["limit"], 5);
            assert_eq!(args["fresh"], true);
        }

        #[test]
        fn parses_nested_object_and_array_args() {
            let tc = parse_tool_call_body("call:f{a:{b:1,c:[2,3]},d:<|\"|>x<|\"|>}", 0).unwrap();
            let args: Value =
                serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(args["a"]["b"], 1);
            assert_eq!(args["a"]["c"], serde_json::json!([2, 3]));
            assert_eq!(args["d"], "x");
        }

        #[test]
        fn returns_none_for_empty_or_nameless_body() {
            assert!(parse_tool_call_body("", 0).is_none());
            assert!(parse_tool_call_body("call:{x:1}", 0).is_none());
        }

        #[test]
        fn parses_json_quoted_keys_and_doubled_quote_values() {
            // Exact shape emitted by mlx-community/gemma-4-e2b-it-4bit: JSON-quoted
            // keys (`"engine"`) and values wrapped in BOTH the gemma4 quote token
            // and standard quotes (`<|"|>"google"<|"|>`). Previously the literal
            // `"` survived, so keys became `"query"` and the browser tool reported
            // `Missing required field: query`.
            let body = "call:mcp__browser__search{\"engine\":<|\"|>\"google\"<|\"|>,\
                \"language\":<|\"|>\"vi\"<|\"|>,\"num_results\":10,\
                \"query\":<|\"|>\"giá vàng hôm nay\"<|\"|>}";
            let tc = parse_tool_call_body(body, 0).unwrap();
            assert_eq!(tc["function"]["name"], "mcp__browser__search");
            let args: Value =
                serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(args["query"], "giá vàng hôm nay");
            assert_eq!(args["engine"], "google");
            assert_eq!(args["language"], "vi");
            assert_eq!(args["num_results"], 10);
        }

        #[test]
        fn parses_pure_json_string_args() {
            // No gemma4 quote token at all — plain JSON quotes on both sides.
            let tc = parse_tool_call_body("call:f{\"q\":\"vàng\",\"n\":3}", 0).unwrap();
            let args: Value =
                serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(args["q"], "vàng");
            assert_eq!(args["n"], 3);
        }
    }
}

// -----------------------------------------------------------------------------
// Chat template integration
// -----------------------------------------------------------------------------

impl crate::engine::chat_template_openai::ChatTemplateModel for Model {
    /// Gemma-4 harmony output: thinking is emitted in a `<|channel>thought\n…
    /// <channel|>` channel; tool calls in `<|tool_call>call:NAME{key:val,…}
    /// <tool_call|>` with `<|"|>`-wrapped string args. Authoritative for this
    /// arch — the engine uses these markers directly, no template scan needed.
    fn markers(&self) -> crate::engine::stream_parser::MarkerSet {
        crate::engine::stream_parser::MarkerSet::gemma4()
    }

    fn resolve_special_tokens(
        &self,
        template: &str,
        tokenizer: &crate::engine::mlx_lm_utils::tokenizer::Tokenizer,
    ) -> crate::engine::chat_template_openai::SpecialTokens {
        use crate::engine::chat_template_openai::{template_mentions, SpecialTokens};
        let need_bos = template_mentions(template, "bos_token");
        let need_eos = template_mentions(template, "eos_token");
        if !need_bos && !need_eos {
            return SpecialTokens::empty();
        }
        let bos = if need_bos {
            self.args
                .bos_token_id
                .and_then(|id| tokenizer.decode(std::slice::from_ref(&id), false).ok())
        } else {
            None
        };
        let eos = if need_eos {
            self.args
                .eos_token_ids
                .first()
                .copied()
                .and_then(|id| tokenizer.decode(std::slice::from_ref(&id), false).ok())
        } else {
            None
        };
        SpecialTokens { bos, eos }
    }

    fn stop_token_ids(
        &self,
        _tokenizer: &crate::engine::mlx_lm_utils::tokenizer::Tokenizer,
    ) -> Vec<u32> {
        self.args.eos_token_ids.clone()
    }
}

// -----------------------------------------------------------------------------
// Per-module quantization (handles both uniform 4-bit and OptiQ mixed-bit)
// -----------------------------------------------------------------------------

/// Top-level `config.json::quantization` defaults plus the per-module override
/// map (OptiQ checkpoints). Top-level `{bits, group_size}` apply when a slot
/// is quantized in storage but absent from the per-module overrides.
pub struct QuantPlan {
    pub default: (i32, i32), // (group_size, bits)
    pub overrides: HashMap<String, (i32, i32)>,
    /// Module paths (HF full form, e.g. `language_model.model.embed_tokens`)
    /// that ship `.scales` in safetensors — only those get quantized; modules
    /// without `.scales` stay as plain `nn::Linear` / `nn::Embedding`.
    pub quantized_paths: HashSet<String>,
}

impl QuantPlan {
    /// Build the plan from `config.json` + the safetensors weight-map keys.
    /// Caller supplies the keys verbatim (with the `language_model.` prefix
    /// they have on disk) so we can detect which modules are actually stored
    /// quantized — the OptiQ config alone isn't sufficient because a slot can
    /// be quantized via the top-level default with no per-module entry.
    pub fn from_config_and_index<'a>(
        cfg: &serde_json::Value,
        weight_keys: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        let q = cfg
            .get("quantization")
            .or_else(|| cfg.get("quantization_config"));

        let default = match q {
            Some(qv) => (
                qv.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32,
                qv.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32,
            ),
            None => (64, 4),
        };

        let mut overrides: HashMap<String, (i32, i32)> = HashMap::new();
        if let Some(qobj) = q.and_then(|v| v.as_object()) {
            for (k, v) in qobj {
                if matches!(k.as_str(), "group_size" | "bits" | "mode") {
                    continue;
                }
                let Some(obj) = v.as_object() else { continue };
                let bits = obj
                    .get("bits")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(default.1 as i64) as i32;
                let gs = obj
                    .get("group_size")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(default.0 as i64) as i32;
                overrides.insert(k.clone(), (gs, bits));
            }
        }

        // Scan weight keys for `.scales` suffix — that's the runtime signal
        // "this module is stored quantized." More reliable than relying on the
        // OptiQ override list alone (regular 4-bit checkpoints quantize most
        // modules via the top-level default without any per-module entries).
        let mut quantized_paths: HashSet<String> = HashSet::new();
        for k in weight_keys {
            if let Some(base) = k.strip_suffix(".scales") {
                quantized_paths.insert(base.to_string());
            }
        }

        Self {
            default,
            overrides,
            quantized_paths,
        }
    }

    /// Bits/group for `path` (HF full form) — None means "leave unquantized".
    pub fn lookup(&self, path: &str) -> Option<(i32, i32)> {
        if !self.quantized_paths.contains(path) {
            return None;
        }
        Some(*self.overrides.get(path).unwrap_or(&self.default))
    }
}

fn quantize_maybe_linear(
    slot: &mut MaybeQuantized<nn::Linear>,
    path: &str,
    plan: &QuantPlan,
) -> Result<(), Exception> {
    if slot.is_quantized() {
        return Ok(());
    }
    let Some((gs, bits)) = plan.lookup(path) else {
        return Ok(()); // stored bf16 — leave as Original
    };
    let placeholder = nn::LinearBuilder::new(1, 1).bias(false).build()?;
    match std::mem::replace(slot, MaybeQuantized::Original(placeholder)) {
        MaybeQuantized::Original(linear) => {
            *slot = MaybeQuantized::Quantized(linear.try_into_quantized(gs, bits)?);
        }
        MaybeQuantized::Quantized(q) => *slot = MaybeQuantized::Quantized(q),
    }
    Ok(())
}

fn quantize_maybe_embedding(
    slot: &mut MaybeQuantized<nn::Embedding>,
    path: &str,
    plan: &QuantPlan,
) -> Result<(), Exception> {
    if slot.is_quantized() {
        return Ok(());
    }
    let Some((gs, bits)) = plan.lookup(path) else {
        return Ok(());
    };
    let placeholder = nn::Embedding::new(1, 1)?;
    match std::mem::replace(slot, MaybeQuantized::Original(placeholder)) {
        MaybeQuantized::Original(embed) => {
            *slot = MaybeQuantized::Quantized(embed.try_into_quantized(gs, bits)?);
        }
        MaybeQuantized::Quantized(q) => *slot = MaybeQuantized::Quantized(q),
    }
    Ok(())
}

/// Walk every Quantizable slot in the model and apply per-module quantization
/// per the [`QuantPlan`]. Replaces the old uniform `nn::quantize(model, …)`
/// path so we correctly handle both uniform 4-bit (`gemma-4-*-it-4bit`) and
/// OptiQ mixed-bit (`gemma-4-*-it-OptiQ-4bit`) checkpoints.
pub fn apply_per_module_quantization(model: &mut Model, plan: &QuantPlan) -> Result<(), Exception> {
    // Top-level embeddings + per-layer model projection
    quantize_maybe_embedding(
        &mut model.model.embed_tokens,
        "language_model.model.embed_tokens",
        plan,
    )?;
    if let Some(eple) = model.model.embed_tokens_per_layer.as_mut() {
        quantize_maybe_embedding(eple, "language_model.model.embed_tokens_per_layer", plan)?;
    }
    if let Some(plmp) = model.model.per_layer_model_projection.as_mut() {
        quantize_maybe_linear(
            plmp,
            "language_model.model.per_layer_model_projection",
            plan,
        )?;
    }

    // Per-layer modules
    for (i, layer) in model.model.layers.iter_mut().enumerate() {
        let lp = format!("language_model.model.layers.{i}");
        // Attention (k/v/k_norm absent on KV-shared layers — see Attention::new)
        quantize_maybe_linear(
            &mut layer.self_attn.q_proj,
            &format!("{lp}.self_attn.q_proj"),
            plan,
        )?;
        if let Some(k) = layer.self_attn.k_proj.as_mut() {
            quantize_maybe_linear(k, &format!("{lp}.self_attn.k_proj"), plan)?;
        }
        if let Some(v) = layer.self_attn.v_proj.as_mut() {
            quantize_maybe_linear(v, &format!("{lp}.self_attn.v_proj"), plan)?;
        }
        quantize_maybe_linear(
            &mut layer.self_attn.o_proj,
            &format!("{lp}.self_attn.o_proj"),
            plan,
        )?;
        // MLP
        quantize_maybe_linear(
            &mut layer.mlp.gate_proj,
            &format!("{lp}.mlp.gate_proj"),
            plan,
        )?;
        quantize_maybe_linear(&mut layer.mlp.up_proj, &format!("{lp}.mlp.up_proj"), plan)?;
        quantize_maybe_linear(
            &mut layer.mlp.down_proj,
            &format!("{lp}.mlp.down_proj"),
            plan,
        )?;
        // MoE router. The stacked `experts.switch_glu.*` weights are
        // deliberately absent from this walk: they arrive from safetensors
        // already quantized and already stacked, so they are plain `#[param]`
        // arrays rather than `MaybeQuantized` slots. Quantizing them here would
        // re-quantize packed uint32 words as if they were floats.
        if let Some(router) = layer.router.as_mut() {
            quantize_maybe_linear(&mut router.proj, &format!("{lp}.router.proj"), plan)?;
        }
        // PLE projections (per-layer-input gate + output projection)
        if let Some(g) = layer.per_layer_input_gate.as_mut() {
            quantize_maybe_linear(g, &format!("{lp}.per_layer_input_gate"), plan)?;
        }
        if let Some(p) = layer.per_layer_projection.as_mut() {
            quantize_maybe_linear(p, &format!("{lp}.per_layer_projection"), plan)?;
        }
    }

    // Optional lm_head (when tie_word_embeddings = false)
    if let Some(lm_head) = model.lm_head.as_mut() {
        quantize_maybe_linear(lm_head, "language_model.lm_head", plan)?;
    }

    // Vision-side projection (Phase 1 of vision support). The checkpoint
    // stores this UNDER the multimodal wrapper, so the path has NO
    // `language_model.` prefix — quantization plan keys must match exactly.
    if let Some(ev) = model.embed_vision.as_mut() {
        quantize_maybe_linear(
            &mut ev.embedding_projection,
            "embed_vision.embedding_projection",
            plan,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal `config.json` mirroring `mlx-community/gemma-4-e2b-it-4bit`
    /// (multimodal wrapper with nested `text_config` + top-level eos array).
    fn write_e2b_config(dir: &std::path::Path) {
        let cfg = serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "eos_token_id": [1, 106, 50],
            "bos_token_id": 2,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 20,
                "hidden_size_per_layer_input": 256,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 131072,
                "sliding_window": 512,
                "sliding_window_pattern": 5,
                "final_logit_softcapping": 30.0,
                "use_double_wide_mlp": true,
                "tie_word_embeddings": true,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            }
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
    }

    /// The fields of `mlx-community/gemma-4-26b-a4b-it-4bit`'s real
    /// `config.json` that make it a structurally different model from E2B.
    fn write_26b_a4b_config(dir: &std::path::Path) {
        let layer_types: Vec<&str> = (0..30)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        let cfg = serde_json::json!({
            "model_type": "gemma4",
            "eos_token_id": [1, 106, 50],
            "bos_token_id": 2,
            "quantization": { "group_size": 64, "bits": 4 },
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "intermediate_size": 2112,
                "num_attention_heads": 16,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "attention_k_eq_v": true,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 262144,
                "sliding_window": 1024,
                "final_logit_softcapping": 30.0,
                "use_double_wide_mlp": false,
                "tie_word_embeddings": true,
                "enable_moe_block": true,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704,
                "layer_types": layer_types
            }
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
    }

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The 26B-A4B checkpoint differs from E2B in four ways that each silently
    /// build the wrong model rather than failing: MoE, k-eq-V full attention
    /// with its own KV head count, no PLE, and no cross-layer KV sharing.
    #[test]
    fn parses_26b_a4b_moe_shape() {
        let tmp = tmp_dir("gemma4_26b_args_test");
        write_26b_a4b_config(&tmp);
        let args = get_gemma4_model_args(&tmp).unwrap();

        let moe = args.moe().expect("MoE block declared and fully specified");
        assert_eq!(moe.num_experts, 128);
        assert_eq!(moe.top_k, 8);
        assert_eq!(moe.intermediate_size, 704);

        // Routed experts read the top-level quantization, not a per-module one.
        assert_eq!(args.expert_quant, (64, 4));

        // k-eq-V applies to full-attention layers only, and brings its own KV
        // head count with it.
        assert!(args.k_eq_v_for("full_attention"));
        assert!(!args.k_eq_v_for("sliding_attention"));
        assert_eq!(args.n_kv_heads_for("full_attention"), 2);
        assert_eq!(args.n_kv_heads_for("sliding_attention"), 8);

        // No PLE, no KV sharing: every layer owns its KV.
        assert_eq!(args.hidden_size_per_layer_input, 0);
        assert_eq!(args.first_kv_shared(), 30);
    }

    /// E2B must be untouched by any of it — the MoE path is opt-in through
    /// `enable_moe_block`, and `attention_k_eq_v` defaults false so E2B's full
    /// layers keep their own `v_proj` and shared KV head count.
    #[test]
    fn e2b_keeps_its_dense_k_neq_v_shape() {
        let tmp = tmp_dir("gemma4_e2b_shape_test");
        write_e2b_config(&tmp);
        let args = get_gemma4_model_args(&tmp).unwrap();

        assert!(args.moe().is_none(), "E2B has no MoE block");
        assert!(!args.k_eq_v_for("full_attention"));
        assert_eq!(args.n_kv_heads_for("full_attention"), 1);
        assert_eq!(args.n_kv_heads_for("sliding_attention"), 1);
        assert_eq!(args.first_kv_shared(), 15);
    }

    /// A half-declared MoE must read as "no MoE" rather than build a
    /// half-configured one: every field is load-bearing, and a missing one
    /// would otherwise surface as a shape error deep inside the expert matmul.
    #[test]
    fn partially_declared_moe_is_treated_as_absent() {
        let base = serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": 2816, "num_hidden_layers": 30, "intermediate_size": 2112,
            "num_attention_heads": 16, "head_dim": 256, "num_key_value_heads": 8,
            "vocab_size": 262144, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
            "enable_moe_block": true, "num_experts": 128, "top_k_experts": 8,
        });
        // `moe_intermediate_size` missing.
        let args: ModelArgs = serde_json::from_value(base.clone()).unwrap();
        assert!(args.moe().is_none());

        // Present but zero — equally unusable.
        let mut with_zero = base;
        with_zero["moe_intermediate_size"] = serde_json::json!(0);
        let args: ModelArgs = serde_json::from_value(with_zero).unwrap();
        assert!(args.moe().is_none());
    }

    #[test]
    fn parses_text_config_and_folds_eos_bos() {
        let tmp = std::env::temp_dir().join("gemma4_args_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        write_e2b_config(&tmp);

        let args = get_gemma4_model_args(&tmp).unwrap();
        assert_eq!(args.model_type, "gemma4_text");
        assert_eq!(args.num_hidden_layers, 35);
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.global_head_dim, 512);
        assert_eq!(args.num_kv_shared_layers, 20);
        assert_eq!(args.hidden_size_per_layer_input, 256);
        assert_eq!(args.final_logit_softcapping, Some(30.0));
        assert!(args.use_double_wide_mlp);
        // Chat-tuned EOS array + default BOS folded from outer scope.
        assert_eq!(args.eos_token_ids, vec![1, 106, 50]);
        assert_eq!(args.bos_token_id, Some(2));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn layer_types_pattern_and_head_dims() {
        let tmp = std::env::temp_dir().join("gemma4_layertypes_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        write_e2b_config(&tmp);
        let args = get_gemma4_model_args(&tmp).unwrap();

        let lt = args.resolved_layer_types();
        assert_eq!(lt.len(), 35);
        // Pattern = 4× sliding then 1× full (sliding_window_pattern = 5):
        // full layers at indices 4, 9, 14, 19, 24, 29, 34.
        for (i, t) in lt.iter().enumerate() {
            let want_full = (i + 1) % 5 == 0;
            assert_eq!(
                t == "full_attention",
                want_full,
                "layer {i} attention type mismatch"
            );
        }
        // Full layers use global_head_dim; sliding use head_dim.
        assert_eq!(args.head_dim_for("full_attention"), 512);
        assert_eq!(args.head_dim_for("sliding_attention"), 256);
        // RoPE: full = proportional/1e6, sliding = default/1e4.
        assert_eq!(
            args.rope_for("full_attention").rope_type.as_deref(),
            Some("proportional")
        );
        assert_eq!(
            args.rope_for("full_attention").rope_theta,
            Some(1_000_000.0)
        );
        assert_eq!(
            args.rope_for("sliding_attention").rope_theta,
            Some(10_000.0)
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `QuantPlan` must correctly resolve both flavours of Gemma-4
    /// quantization: uniform `gemma-4-*-it-4bit` (top-level `bits` only, no
    /// per-module overrides — every module quantized via the default) and
    /// OptiQ `gemma-4-*-it-OptiQ-4bit` (most modules overridden to 8-bit,
    /// including `per_layer_model_projection` which is bf16 in regular 4-bit).
    /// A module is considered quantized iff it has `.scales` weights in the
    /// safetensors index.
    #[test]
    fn quant_plan_resolves_regular_vs_optiq() {
        // ── Regular 4-bit checkpoint ─────────────────────────────────────
        let regular_cfg = serde_json::json!({
            "quantization": { "bits": 4, "group_size": 64, "mode": "affine" }
        });
        // Realistic key list — q_proj has scales (quantized), per_layer_model_projection has only weight (bf16).
        let regular_keys = vec![
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj.scales",
            "language_model.model.layers.0.self_attn.q_proj.biases",
            "language_model.model.per_layer_model_projection.weight",
        ];
        let plan_reg = QuantPlan::from_config_and_index(&regular_cfg, regular_keys.into_iter());
        assert_eq!(plan_reg.default, (64, 4));
        assert_eq!(
            plan_reg.overrides.len(),
            0,
            "regular has no per-module overrides"
        );
        assert_eq!(
            plan_reg.lookup("language_model.model.layers.0.self_attn.q_proj"),
            Some((64, 4)),
            "quantized at top-level default"
        );
        assert_eq!(
            plan_reg.lookup("language_model.model.per_layer_model_projection"),
            None,
            "bf16-stored module must NOT be quantized in regular 4-bit"
        );

        // ── OptiQ checkpoint ─────────────────────────────────────────────
        let optiq_cfg = serde_json::json!({
            "quantization": {
                "bits": 4, "group_size": 64, "mode": "affine",
                "language_model.model.layers.0.self_attn.q_proj": { "bits": 8, "group_size": 64 },
                "language_model.model.per_layer_model_projection": { "bits": 8, "group_size": 64 },
            }
        });
        // In OptiQ, per_layer_model_projection ALSO has scales (it's quantized).
        let optiq_keys = vec![
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj.scales",
            "language_model.model.layers.0.self_attn.q_proj.biases",
            "language_model.model.per_layer_model_projection.weight",
            "language_model.model.per_layer_model_projection.scales",
            "language_model.model.per_layer_model_projection.biases",
        ];
        let plan_optiq = QuantPlan::from_config_and_index(&optiq_cfg, optiq_keys.into_iter());
        assert_eq!(plan_optiq.default, (64, 4));
        assert_eq!(plan_optiq.overrides.len(), 2, "two per-module overrides");
        assert_eq!(
            plan_optiq.lookup("language_model.model.layers.0.self_attn.q_proj"),
            Some((64, 8)),
            "OptiQ 8-bit override applied"
        );
        assert_eq!(
            plan_optiq.lookup("language_model.model.per_layer_model_projection"),
            Some((64, 8)),
            "OptiQ quantizes per_layer_model_projection at 8-bit (unlike regular 4-bit)"
        );
    }

    /// KV-shared layers (idx ≥ 15) reuse the most recent earlier same-type
    /// layer: full → 14, sliding → 13. Mirrors the reference `previous_kvs`.
    #[test]
    fn previous_kvs_routing_matches_reference() {
        let tmp = std::env::temp_dir().join("gemma4_prevkv_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        write_e2b_config(&tmp);
        let args = get_gemma4_model_args(&tmp).unwrap();

        let n = args.num_hidden_layers as usize;
        let m = args.first_kv_shared() as usize; // 35 - 20 = 15
        assert_eq!(m, 15);
        let layer_types = args.resolved_layer_types();

        // Replicate the routing logic (pure, no MLX needed).
        let mut previous_kvs: Vec<usize> = (0..n).collect();
        let mut last_of_type: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for (i, lt) in layer_types.iter().enumerate().take(m) {
            last_of_type.insert(lt.as_str(), i);
        }
        for (j, slot) in previous_kvs.iter_mut().enumerate().take(n).skip(m) {
            *slot = last_of_type[layer_types[j].as_str()];
        }

        // Last full in 0..15 is idx 14; last sliding is idx 13.
        assert_eq!(previous_kvs[19], 14, "shared full layer reuses layer 14");
        assert_eq!(previous_kvs[34], 14);
        assert_eq!(previous_kvs[15], 13, "shared sliding layer reuses layer 13");
        // Non-shared layers point to themselves.
        for (i, &p) in previous_kvs.iter().enumerate().take(m) {
            assert_eq!(p, i);
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
