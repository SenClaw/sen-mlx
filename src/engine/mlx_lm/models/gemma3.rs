//! Gemma-3 (text-only) inference.
//!
//! Supports `mlx-community/gemma-3-*-it-4bit` / `*-bf16` checkpoints. Vision /
//! multimodal towers are skipped: when loading a 4B+ checkpoint (which wraps
//! the LM in a vision-language model), the `language_model.model.` prefix on
//! safetensors keys is stripped to `model.` before matching against this
//! crate's parameter tree.
//!
//! ## What Gemma-3 does differently
//!
//! - **Hybrid attention**: every `sliding_window_pattern`-th layer (default 6)
//!   uses *full* attention with `rope_theta = 1e6`; the remaining layers use
//!   *sliding-window* attention with `rope_theta = rope_local_base_freq` (10k)
//!   and window=512. Both still use causal masking.
//! - **Gemma RMSNorm** `(1 + weight) * normalize(x)` instead of `weight * x`,
//!   applied to: block input/output norms, Q/K per-head norms, final model
//!   norm. See [`GemmaRmsNorm`].
//! - **Four norms per block** (vs Llama's two): `input_layernorm`,
//!   `post_attention_layernorm`, `pre_feedforward_layernorm`,
//!   `post_feedforward_layernorm`. Residual flow:
//!   `h = x + post_attn(attn(input(x)));`
//!   `h = h + post_ffn(mlp(pre_ffn(h)))`.
//! - **Per-head Q/K RMSNorm** before RoPE (similar to Qwen3, but Gemma-style).
//! - **Query pre-scaling**: Q is divided by `sqrt(query_pre_attn_scalar)`
//!   rather than `sqrt(head_dim)` — separable from head_dim for cases like
//!   Gemma-3 4B where these dimensions are tied differently.
//! - **MLP activation**: `gelu(tanh-approx)` = mlx-rs [`nn::gelu_approximate`].
//! - **Embedding scaling**: token embeddings multiplied by `sqrt(hidden_size)`
//!   before the first block (preserved from original Gemma).
//! - **Final logit softcap** (optional): `logits = softcap * tanh(logits/softcap)`
//!   when `final_logit_softcapping` is set in the config.
//!
//! ## Caching
//!
//! Every layer uses the same FP16 KV cap (`max_kv_tokens`) so a **single prefill**
//! step can run with `L` queries and `L` keys (e.g. long chat templates). **Sliding
//! attention** is enforced only via the per-layer mask (`sliding_window` on
//! [`Attention::is_full`] = false), not by shrinking the KV buffer — a small
//! rotating cache would truncate K/V below `L` and break SDPA shapes.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{ones_dtype, tanh},
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use super::super::{
    cache::{KeyValueCache, KvCache, KvFetchResult},
    error::Error,
    utils::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        scaled_dot_product_attention,
    },
};
// Reuse the input plumbing from qwen3 — it's identical shape across all
// decoder-only transformer architectures in this crate.
pub use super::qwen3::{sample, AttentionInput, ModelInput};

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Top-level `config.json` schema. Multimodal Gemma-3 checkpoints (`gemma3`)
/// wrap the text model in a `text_config` sub-object; text-only checkpoints
/// (`gemma3_text`) put the same fields at the top level. [`load_gemma3_args`]
/// handles both.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,
    /// Q is divided by `sqrt(query_pre_attn_scalar)`. Falls back to `head_dim`
    /// when missing.
    #[serde(default)]
    pub query_pre_attn_scalar: i32,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub final_logit_softcapping: Option<f32>,
    /// EOS token id(s). Gemma instruct uses an array `[1, 106]`; raw config
    /// from some checkpoints is a single int. [`load_gemma3_args`] folds both
    /// into this `Vec`.
    #[serde(skip)]
    pub eos_token_ids: Vec<u32>,
    /// BOS token (default 2). The Gemma HF `chat_template` starts with `{{ bos_token }}`.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

fn default_rope_local_base_freq() -> f32 {
    10_000.0
}
fn default_sliding_window() -> i32 {
    512
}
fn default_sliding_window_pattern() -> i32 {
    6
}
fn default_tie_word_embeddings() -> bool {
    true
}

impl ModelArgs {
    pub fn normalize(&mut self) {
        if self.query_pre_attn_scalar <= 0 {
            self.query_pre_attn_scalar = self.head_dim.max(1);
        }
    }

    /// `true` for layers that use full attention (rope_theta=1e6, no sliding mask).
    /// Pattern matches HF reference: layer `i` is full iff
    /// `(i + 1) % sliding_window_pattern == 0`.
    pub fn is_full_attention_layer(&self, layer_idx: usize) -> bool {
        let p = self.sliding_window_pattern.max(1) as usize;
        (layer_idx + 1) % p == 0
    }
}

// -----------------------------------------------------------------------------
// Gemma RMSNorm — (1 + weight) * normalize(x)
// -----------------------------------------------------------------------------

/// Gemma's RMSNorm uses `(1 + weight)` as the scale, *not* `weight`. The
/// safetensors store the original (subtracted-by-one) form, so loading
/// works directly — the `+1` is applied at forward time.
#[derive(Debug, Clone, ModuleParameters)]
pub struct GemmaRmsNorm {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl GemmaRmsNorm {
    pub fn new(dim: i32, eps: f32) -> Result<Self, Exception> {
        // Initialise weight as zeros so `1 + weight` defaults to identity.
        let weight = mlx_rs::ops::zeros::<f32>(&[dim])?;
        Ok(Self {
            weight: Param::new(weight),
            eps,
        })
    }
}

impl Module<&Array> for GemmaRmsNorm {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // `(1 + w) * x * rsqrt(mean(x^2, axis=-1) + eps)` — computed in f32
        // for stability, then cast back to the input dtype.
        let input_dtype = x.dtype();
        let x32 = x.as_dtype(mlx_rs::Dtype::Float32)?;
        let var = x32.square()?.mean_axes(&[-1], true)?;
        let normed = x32.multiply(&var.add(&array!(self.eps))?.rsqrt()?)?;
        let scale = ones_dtype(&[1], mlx_rs::Dtype::Float32)?
            .add(&self.weight.as_ref().as_dtype(mlx_rs::Dtype::Float32)?)?;
        normed.multiply(&scale)?.as_dtype(input_dtype)
    }

    fn training_mode(&mut self, _: bool) {}
}

// -----------------------------------------------------------------------------
// Attention
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,
    pub is_full: bool,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: GemmaRmsNorm,
    #[param]
    pub k_norm: GemmaRmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(args: &ModelArgs, is_full: bool) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        // Gemma divides Q by sqrt(query_pre_attn_scalar) — usually = head_dim
        // but configurable.
        let scale = (args.query_pre_attn_scalar as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(args.attention_bias)
            .build()?;

        let q_norm = GemmaRmsNorm::new(head_dim, args.rms_norm_eps)?;
        let k_norm = GemmaRmsNorm::new(head_dim, args.rms_norm_eps)?;

        // Pick RoPE base per the hybrid attention pattern.
        let theta = if is_full {
            args.rope_theta
        } else {
            args.rope_local_base_freq
        };
        let rope = initialize_rope(
            head_dim,
            theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            is_full,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            mut cache,
            rope_offset,
        } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];
        let rope_off = i32::try_from(rope_offset)
            .map_err(|_| Exception::custom("rope_offset exceeds i32::MAX"))?;

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = self
            .q_norm
            .forward(&queries.reshape(&[B, L, self.n_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(&keys.reshape(&[B, L, self.n_kv_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let fetch = if let Some(cache) = cache.as_mut() {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(rope_off)
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys).offset(rope_off).build()?;
            keys = self.rope.forward(k_input)?;
            cache.update_and_fetch(keys, values)?
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
            KvFetchResult::Fp16(keys, values)
        };

        let output = match fetch {
            KvFetchResult::Fp16(keys, values) => {
                scaled_dot_product_attention(queries, keys, values, cache, self.scale, mask)?
            }
            KvFetchResult::TurboQuant => {
                let c = cache
                    .as_mut()
                    .ok_or_else(|| Exception::custom("TurboQuant fetch without cache"))?;
                if let Some(out) = c.turboquant_attention(
                    &queries,
                    self.scale,
                    mask,
                    self.n_heads,
                    self.n_kv_heads,
                )? {
                    out
                } else {
                    return Err(Exception::custom(
                        "TurboQuant path active but turboquant_attention returned None",
                    ));
                }
            }
        }
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// -----------------------------------------------------------------------------
// MLP — gate * up via gelu(tanh-approx), then down
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim).bias(bias).build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim).bias(bias).build()?;
        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        // gelu_pytorch_tanh ≡ mlx-rs `gelu_approximate` (tanh-based GELU).
        let gated = nn::gelu_approximate(self.gate_proj.forward(input)?)?
            .multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&gated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

// -----------------------------------------------------------------------------
// Decoder block — 4 norms + attn + mlp
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: GemmaRmsNorm,
    #[param]
    pub post_attention_layernorm: GemmaRmsNorm,
    #[param]
    pub pre_feedforward_layernorm: GemmaRmsNorm,
    #[param]
    pub post_feedforward_layernorm: GemmaRmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs, is_full: bool) -> Result<Self, Exception> {
        let self_attn = Attention::new(args, is_full)?;
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size, args.mlp_bias)?;
        let mk = || GemmaRmsNorm::new(args.hidden_size, args.rms_norm_eps);
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: mk()?,
            post_attention_layernorm: mk()?,
            pre_feedforward_layernorm: mk()?,
            post_feedforward_layernorm: mk()?,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            cache,
            rope_offset,
        } = input;

        // h = x + post_attn(attn(input_norm(x)))
        let attn_in = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
            rope_offset,
        };
        let attn_out = self.self_attn.forward(attn_in)?;
        let h = x.add(&self.post_attention_layernorm.forward(&attn_out)?)?;
        // h = h + post_ffn(mlp(pre_ffn(h)))
        let ffn_in = self.pre_feedforward_layernorm.forward(&h)?;
        let ffn_out = self.mlp.forward(&ffn_in)?;
        h.add(&self.post_feedforward_layernorm.forward(&ffn_out)?)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

// -----------------------------------------------------------------------------
// Backbone model
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma3Model {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    /// Local attention span for sliding-window layers (full layers ignore this).
    pub sliding_window: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: GemmaRmsNorm,
}

impl Gemma3Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers as usize)
            .map(|i| TransformerBlock::new(args, args.is_full_attention_layer(i)))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = GemmaRmsNorm::new(args.hidden_size, args.rms_norm_eps)?;
        Ok(Self {
            hidden_size: args.hidden_size,
            num_hidden_layers: args.num_hidden_layers,
            sliding_window: args.sliding_window,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for Gemma3Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
            rope_offset,
        } = input;

        // Embed and scale by sqrt(hidden_size) — vanilla Gemma normalisation.
        let h = self.embed_tokens.forward(inputs)?;
        let scale = array!((self.hidden_size as f32).sqrt());
        let mut h = h.multiply(&scale)?;

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| None).collect();
        }
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            // Hybrid Gemma-3: sliding layers need a local window mask tied to
            // that layer's KV cap; full-attention layers must *not* reuse the
            // sliding mask (see mlx_lm `language.py` global vs sliding masks).
            //
            // Full-attention layers need a causal mask only during multi-token
            // prefill — single-token decode (`seq <= 1`) attends the whole
            // cache, which is correct for full attention, so `None`.
            //
            // Sliding-window layers need a windowed mask at EVERY step,
            // *including* single-token decode: `create_causal_mask(1, offset,
            // window)` yields a `[1, offset+1]` row that restricts the query to
            // the last `sliding_window` keys. Returning `None` on decode (the
            // old behaviour) let sliding layers attend the full KV cache —
            // out-of-distribution for layers trained with a fixed 512 window,
            // which degrades long generations (gibberish past the window). The
            // KV buffer is still never shrunk (memory grows with the sequence,
            // same trade-off as the rest of this crate); only attention is
            // windowed. Mirrors `gemma4`.
            let layer_mask: Option<Array> = match mask {
                Some(m) => Some(m.clone()),
                None => {
                    let seq = h.shape()[1];
                    let window = if layer.self_attn.is_full {
                        None
                    } else {
                        Some(self.sliding_window)
                    };
                    if window.is_none() && seq <= 1 {
                        None
                    } else {
                        let mut offset = rope_offset as i32;
                        if window.is_some() {
                            if let Some(cc) = c.as_ref() {
                                if let Some(cap) = cc.max_size() {
                                    offset = offset.min(cap);
                                }
                            }
                        }
                        Some(create_causal_mask(seq, Some(offset), window, None)?)
                    }
                }
            };
            let layer_input = AttentionInput {
                x: &h,
                mask: layer_mask.as_ref(),
                cache: c.as_mut(),
                rope_offset,
            };
            h = layer.forward(layer_input)?;
        }
        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

// -----------------------------------------------------------------------------
// Top-level model + loader
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: Gemma3Model,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Gemma3Model::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };
        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Per-layer KV cache (same `max_kv_tokens` for every layer).
    ///
    /// Sliding-window layers still use [`ModelArgs::sliding_window`] in the
    /// **attention mask** only. Do not cap their KV at `sliding_window`: prefill
    /// runs one forward with `L` tokens and needs K/V length `L` from
    /// `update_and_fetch`, or SDPA broadcast fails (`L` vs `sliding_window`).
    pub fn make_caches(&self, max_kv_tokens: i32) -> Vec<Option<KvCache>> {
        let cap = max_kv_tokens.max(1);
        (0..self.args.num_hidden_layers as usize)
            .map(|_| Some(KvCache::fp16_with_max(cap)))
            .collect()
    }

    pub fn forward<C: KeyValueCache>(
        &mut self,
        input: ModelInput<'_, C>,
    ) -> Result<Array, Exception> {
        let out = self.model.forward(input)?;
        let logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out)?,
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&out)?,
                MaybeQuantized::Quantized(q) => q.as_linear(&out)?,
            },
        };
        // Optional final-logit softcapping: `cap * tanh(logits/cap)`.
        if let Some(cap) = self.args.final_logit_softcapping {
            let cap_a = array!(cap);
            return tanh(&logits.divide(&cap_a)?)?.multiply(&cap_a);
        }
        Ok(logits)
    }
}

pub fn load_gemma3_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Parse `config.json` for either a text-only (`gemma3_text`) or wrapped
/// (`gemma3`) checkpoint. EOS may be int or array — we collect into `Vec`.
pub fn get_gemma3_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let path = model_dir.as_ref().join("config.json");
    let raw = std::fs::read_to_string(&path)?;
    let root: Value = serde_json::from_str(&raw)?;

    // If a multimodal wrapper, dig into `text_config`. Otherwise the root *is*
    // the text config — but we still need to merge `eos_token_id` from the
    // outer scope (the wrapper config typically holds the chat-tuned EOS array
    // while `text_config` only carries dims).
    let (text_obj, eos_value) = match root.get("text_config") {
        Some(inner) => (inner.clone(), root.get("eos_token_id").cloned()),
        None => (root.clone(), root.get("eos_token_id").cloned()),
    };
    let mut args: ModelArgs = serde_json::from_value(text_obj)?;
    args.normalize();
    // Fold EOS into the args. Accept int or array.
    args.eos_token_ids = match eos_value {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(arr)) => arr
            .into_iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => vec![],
    };
    // Multimodal wrappers often omit `bos_token_id` inside `text_config`; prefer outer root.
    if args.bos_token_id.is_none() {
        args.bos_token_id = root
            .get("bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32);
    }
    Ok(args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

/// Strip the multimodal prefix (`language_model.model.X` → `model.X`,
/// `language_model.lm_head.X` → `lm_head.X`) so wrapped checkpoints load
/// against the same parameter tree as text-only ones.
fn strip_multimodal_prefix(key: &str) -> &str {
    key.strip_prefix("language_model.").unwrap_or(key)
}

pub fn load_gemma3_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    use mlx_rs::module::ModuleParameters;

    let model_dir = model_dir.as_ref();
    let args = get_gemma3_model_args(model_dir)?;
    let mut model = Model::new(args)?;

    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let files: HashSet<&String> = weight_map.weight_map.values().collect();
        for f in files {
            shard_files.push(model_dir.join(f));
        }
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing model.safetensors in {}", model_dir.display()),
            )
            .into());
        }
        shard_files.push(single);
    }

    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();
    let mut unfilled: HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };

    for shard in &shard_files {
        let loaded = Array::load_safetensors(shard)?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, value) in loaded {
            let key = strip_multimodal_prefix(raw_key.as_str());
            if let Some(slot) = params.get_mut(key) {
                **slot = value;
                total_loaded += 1;
                unfilled.remove(key);
                continue;
            }
            total_missed += 1;
            if unmatched_samples.len() < 5 {
                unmatched_samples.push(key.to_string());
            }
        }
    }

    tracing::info!("[gemma3] safetensor load: {total_loaded} matched, {total_missed} unmatched");
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[gemma3] sample unmatched keys: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled.is_empty() {
        let mut samples: Vec<&String> = unfilled.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[gemma3] {} parameter slot(s) NOT populated — first few: {}",
            unfilled.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        return Err(
            Exception::custom("no safetensor keys matched the Gemma-3 parameter tree").into(),
        );
    }

    model.eval()?;
    Ok(model)
}

/// Gemma‑3 chat template begins with `{{ bos_token }}` (token id `2`).
/// Without this injection, the rendered prompt loses the leading BOS piece
/// the model was trained on and quality regresses sharply.
impl crate::engine::chat_template_openai::ChatTemplateModel for Model {
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
        if need_bos {
            match &bos {
                Some(s) if !s.is_empty() => tracing::debug!(
                    "[local-mlx-native] chat_template bos_token injected (decoded len={})",
                    s.len()
                ),
                _ => tracing::warn!(
                    "[local-mlx-native] chat_template mentions bos_token — could not decode from bos_token_id={:?}",
                    self.args.bos_token_id
                ),
            }
        }
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
        // Gemma-3 lists its terminators in config (`<eos>`, `<end_of_turn>`).
        self.args.eos_token_ids.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On single-token decode (`seq == 1`), a sliding-window layer must still
    /// receive a windowed mask so the query attends only the last
    /// `sliding_window` keys — not the full KV cache. This is the exact
    /// `create_causal_mask(1, offset, window)` call the forward loop now makes
    /// for sliding layers (previously it returned `None` on decode, letting
    /// sliding layers attend everything — see the mask comment in
    /// `Gemma3Model::forward`).
    #[test]
    fn sliding_layer_decode_mask_windows_to_last_keys() {
        // Decode step: 1 query at absolute position `offset`, KV holds
        // `offset + 1` keys (positions 0..=offset). window = 4.
        let offset = 10;
        let window = 4;
        let mask = create_causal_mask(1, Some(offset), Some(window), None).unwrap();
        // `[1, offset+1]` row over keys 0..=offset.
        assert_eq!(mask.shape(), &[1, offset + 1]);
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        // Visible keys: `offset - window < j <= offset` -> j in {7,8,9,10}.
        let expected: Vec<bool> = (0..=offset).map(|j| j > offset - window).collect();
        assert_eq!(flat, expected);
        assert_eq!(flat.iter().filter(|&&v| v).count(), window as usize);
    }

    /// A full-attention layer's decode call (`window = None`) leaves every
    /// cached key visible — that's why the forward loop keeps `None` for full
    /// layers on `seq <= 1` rather than building a mask at all. Here we assert
    /// the windowless mask itself imposes no upper restriction.
    #[test]
    fn full_layer_decode_mask_sees_all_keys() {
        let offset = 10;
        let mask = create_causal_mask(1, Some(offset), None, None).unwrap();
        assert_eq!(mask.shape(), &[1, offset + 1]);
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        assert!(
            flat.iter().all(|&v| v),
            "full attention sees all cached keys"
        );
    }

    /// When fewer keys are cached than the window, the windowed decode mask
    /// degenerates to plain causal (all keys visible) — no negative indices.
    #[test]
    fn sliding_layer_decode_mask_shorter_than_window() {
        let offset = 2; // 3 cached keys
        let window = 8;
        let mask = create_causal_mask(1, Some(offset), Some(window), None).unwrap();
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        assert_eq!(flat, vec![true; (offset + 1) as usize]);
    }
}
