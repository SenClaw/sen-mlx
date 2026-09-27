//! Gemma 2 model implementation.
//!
//! Differs from the standard transformer in several ways:
//! - Explicit `head_dim` (not derived from `hidden_size / num_attention_heads`)
//! - 4 layer norms per block (pre/post attention + pre/post feedforward)
//! - Attention logit soft-capping via tanh
//! - Final logit soft-capping
//! - `GeGLU` activation (GELU-gated instead of SiLU-gated)
//! - `RMSNorm` with +1 convention (weights stored as w-1)
//! - Alternating sliding window / global attention layers

use std::{collections::HashSet, path::Path};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    fast,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters, ModuleParametersExt},
    nn, ops,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;

use super::super::cache::{KeyValueCache, KvCache, KvFetchResult};
use super::super::error::Error;
use super::super::utils::{create_attention_mask, AttentionMask};

/// RoPE bypassing the 3‑D reshape in `nn::Rope::forward` (same rationale as mlx‑lm / Higgs).
fn apply_rope(x: &Array, rope: &nn::Rope, offset: i32) -> Result<Array, Exception> {
    fast::rope(
        x,
        rope.dimensions,
        rope.traditional,
        rope.base,
        rope.scale,
        offset,
        None,
    )
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_rope_theta() -> f32 {
    10000.0
}

const fn default_sliding_window_pattern() -> i32 {
    2
}

// Gemma 2 uses tied word embeddings by default (no separate lm_head weight).
const fn default_tie_word_embeddings() -> bool {
    true
}

/// Quantization parameters from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

/// Gemma 2 model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma2ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,

    /// Scale factor for attention logits: `1 / sqrt(query_pre_attn_scalar)`.
    /// Defaults to `head_dim` if not present.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<i32>,

    /// Tanh soft-capping for attention logits before softmax.
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,

    /// Tanh soft-capping for final output logits.
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,

    /// Sliding window size for local attention layers.
    #[serde(default)]
    pub sliding_window: Option<i32>,

    /// How many layers between sliding window layers (default 2 = alternating).
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    /// Populated by [`get_gemma2_model_args`]; used by the native MLX engine for stopping.
    #[serde(skip)]
    pub eos_token_ids: Vec<u32>,

    /// BOS id for HF `chat_template` (`{{ bos_token }}`), when present in config.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

impl Gemma2ModelArgs {
    fn attn_scale(&self) -> f32 {
        let scalar = self.query_pre_attn_scalar.unwrap_or(self.head_dim);
        let scalar_f32 = f32::from(i16::try_from(scalar).unwrap_or(i16::MAX));
        scalar_f32.sqrt().recip()
    }

    /// Whether layer at `idx` uses sliding window attention.
    const fn is_sliding_window_layer(&self, layer_idx: i32) -> bool {
        if self.sliding_window.is_none() || self.sliding_window_pattern <= 0 {
            return false;
        }
        layer_idx % self.sliding_window_pattern == 0
    }
}

// ---------------------------------------------------------------------------
// Attention (manual with soft-capping, 5D broadcast GQA)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma2Attention {
    n_heads: i32,
    n_kv_heads: i32,
    n_rep: i32,
    scale: f32,
    attn_logit_softcapping: Option<f32>,
    sliding_window: Option<i32>,

    // Pre-cast scalars to avoid f32 dtype promotion in bfloat16 models
    cached_scale: Option<Array>,
    cached_inv_cap: Option<Array>,
    cached_cap: Option<Array>,
    cached_neg_inf: Option<Array>,

    #[quantizable]
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    rope: nn::Rope,
}

impl Gemma2Attention {
    fn new(args: &Gemma2ModelArgs, sliding_window: bool) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let scale = args.attn_scale();

        let bias = args.attention_bias;
        let q_proj = nn::LinearBuilder::new(args.hidden_size, n_heads * head_dim)
            .bias(bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(args.hidden_size, n_kv_heads * head_dim)
            .bias(bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(args.hidden_size, n_kv_heads * head_dim)
            .bias(bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, args.hidden_size)
            .bias(bias)
            .build()?;

        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)
            .base(args.rope_theta)
            .scale(1.0)
            .build()
            .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?;

        let window = if sliding_window {
            args.sliding_window
        } else {
            None
        };

        Ok(Self {
            n_heads,
            n_kv_heads,
            n_rep: n_heads / n_kv_heads,
            scale,
            attn_logit_softcapping: args.attn_logit_softcapping,
            sliding_window: window,
            cached_scale: None,
            cached_inv_cap: None,
            cached_cap: None,
            cached_neg_inf: None,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

struct Gemma2AttentionInput<'a, C> {
    x: &'a Array,
    mask: Option<&'a Array>,
    cache: Option<&'a mut C>,
}

impl<C> Module<Gemma2AttentionInput<'_, C>> for Gemma2Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case, clippy::too_many_lines)]
    fn forward(&mut self, input: Gemma2AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma2AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        let mut queries = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = self
            .v_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(ref mut kv_cache) = cache {
            let off = kv_cache.stored_len();
            queries = apply_rope(&queries, &self.rope, off)?;
            keys = apply_rope(&keys, &self.rope, off)?;

            match kv_cache.update_and_fetch(keys, values)? {
                KvFetchResult::Fp16(k, v) => {
                    keys = k;
                    values = v;
                }
                KvFetchResult::TurboQuant => {
                    return Err(Exception::custom(
                        "gemma2: TurboQuant KV is not wired for Gemma‑2 MLX native; disable kv_cache_bits",
                    ));
                }
            }
        } else {
            queries = apply_rope(&queries, &self.rope, 0)?;
            keys = apply_rope(&keys, &self.rope, 0)?;
        }

        // GQA via 5D broadcast: avoids physically copying KV heads.
        // queries: [B, n_heads, L, D] -> [B, n_kv, n_rep, L, D]
        // keys/values: [B, n_kv, S, D] -> [B, n_kv, 1, S, D]
        let kv_s = *keys
            .shape()
            .get(2)
            .ok_or_else(|| Exception::custom("keys must be 4D"))?;
        let head_d = *queries
            .shape()
            .get(3)
            .ok_or_else(|| Exception::custom("queries must be 4D"))?;
        let q5 = queries.reshape(&[B, self.n_kv_heads, self.n_rep, L, head_d])?;
        let k5 = keys.reshape(&[B, self.n_kv_heads, 1, kv_s, head_d])?;
        let v5 = values.reshape(&[B, self.n_kv_heads, 1, kv_s, head_d])?;

        // Manual attention with soft-capping
        // scores: [B, n_kv, n_rep, L, S]
        if self
            .cached_scale
            .as_ref()
            .is_none_or(|cached| cached.dtype() != q5.dtype())
        {
            self.cached_scale = Some(array!(self.scale).as_dtype(q5.dtype())?);
        }
        let scale = self
            .cached_scale
            .as_ref()
            .ok_or_else(|| Exception::custom("cached_scale not initialized"))?;
        let mut scores = q5
            .matmul(&k5.transpose_axes(&[0, 1, 2, 4, 3])?)?
            .multiply(scale)?;

        // Soft-capping: tanh(scores / cap) * cap
        if let Some(cap) = self.attn_logit_softcapping {
            let needs_cap_cache = self
                .cached_inv_cap
                .as_ref()
                .is_none_or(|cached| cached.dtype() != scores.dtype())
                || self
                    .cached_cap
                    .as_ref()
                    .is_none_or(|cached| cached.dtype() != scores.dtype());
            if needs_cap_cache {
                self.cached_inv_cap = Some(array!(1.0 / cap).as_dtype(scores.dtype())?);
                self.cached_cap = Some(array!(cap).as_dtype(scores.dtype())?);
            }
            let inv_cap = self
                .cached_inv_cap
                .as_ref()
                .ok_or_else(|| Exception::custom("cached_inv_cap not initialized"))?;
            let cap_arr = self
                .cached_cap
                .as_ref()
                .ok_or_else(|| Exception::custom("cached_cap not initialized"))?;
            scores = ops::tanh(&scores.multiply(inv_cap)?)?.multiply(cap_arr)?;
        }

        // Apply sliding window mask (additive: -inf for out-of-window positions)
        if let Some(window) = self.sliding_window {
            let s_len = *scores
                .shape()
                .last()
                .ok_or_else(|| Exception::custom("scores must have >= 1 dim"))?;
            if s_len > window {
                if self
                    .cached_neg_inf
                    .as_ref()
                    .is_none_or(|cached| cached.dtype() != scores.dtype())
                {
                    self.cached_neg_inf = Some(array!(f32::NEG_INFINITY).as_dtype(scores.dtype())?);
                }
                let window_mask = create_sliding_window_mask(L, s_len, window)?;
                let neg_inf = self
                    .cached_neg_inf
                    .as_ref()
                    .ok_or_else(|| Exception::custom("cached_neg_inf not initialized"))?;
                scores = ops::r#where(&window_mask, &scores, neg_inf)?;
            }
        }

        // Apply causal mask (boolean: true = attend, false = mask out)
        if let Some(m) = mask {
            if self
                .cached_neg_inf
                .as_ref()
                .is_none_or(|cached| cached.dtype() != scores.dtype())
            {
                self.cached_neg_inf = Some(array!(f32::NEG_INFINITY).as_dtype(scores.dtype())?);
            }
            let neg_inf = self
                .cached_neg_inf
                .as_ref()
                .ok_or_else(|| Exception::custom("cached_neg_inf not initialized"))?;
            scores = ops::r#where(m, &scores, neg_inf)?;
        }

        let weights = ops::softmax_axis(&scores, -1, None)?;
        // [B, n_kv, n_rep, L, D] -> [B, n_heads, L, D] -> [B, L, n_heads*D]
        let output = weights
            .matmul(&v5)?
            .reshape(&[B, self.n_heads, L, head_d])?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// Create a boolean mask for sliding window attention.
///
/// For each query position q (with absolute position = offset + `q_local`),
/// only keys within `[q_abs - window + 1, q_abs]` are visible.
#[allow(non_snake_case)]
fn create_sliding_window_mask(L: i32, S: i32, window: i32) -> Result<Array, Exception> {
    // Query positions: last L of the S total positions
    // offset = S - L
    let offset = S - L;
    // For query at local index i (absolute position = offset + i),
    // key at index j is visible if j >= (offset + i) - window + 1 AND j <= (offset + i)
    // The causal mask already handles j <= (offset + i).
    // We just need: j >= (offset + i) - window + 1

    let query_positions = mlx_rs::arange!(start = offset, stop = offset + L)?;
    let key_positions = mlx_rs::arange!(stop = S)?;

    // lower_bound[i] = query_pos[i] - window + 1
    let lower_bounds = query_positions.subtract(array!(window - 1))?;
    // Reshape for broadcasting: [L, 1] vs [1, S]
    let lower_expanded = lower_bounds.reshape(&[L, 1])?;
    let key_expanded = key_positions.reshape(&[1, S])?;

    // mask[i, j] = key_positions[j] >= lower_bound[i]
    key_expanded.ge(&lower_expanded)
}

// ---------------------------------------------------------------------------
// MLP (GeGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma2Mlp {
    #[quantizable]
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,
}

impl Gemma2Mlp {
    fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for Gemma2Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        // GeGLU: gelu(gate) * up, then down
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

// ---------------------------------------------------------------------------
// Block (4 norms)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma2Block {
    #[quantizable]
    #[param]
    self_attn: Gemma2Attention,
    #[quantizable]
    #[param]
    mlp: Gemma2Mlp,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    post_feedforward_layernorm: nn::RmsNorm,
}

impl Gemma2Block {
    fn new(args: &Gemma2ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let sliding = args.is_sliding_window_layer(layer_idx);
        Ok(Self {
            self_attn: Gemma2Attention::new(args, sliding)?,
            mlp: Gemma2Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_feedforward_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_feedforward_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<Gemma2AttentionInput<'_, C>> for Gemma2Block
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Gemma2AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma2AttentionInput { x, mask, cache } = input;

        // Pre-attention norm -> attention -> post-attention norm -> residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(Gemma2AttentionInput {
            x: &normed,
            mask,
            cache,
        })?;
        let attn_normed = self.post_attention_layernorm.forward(&attn_out)?;
        let h = x.add(attn_normed)?;

        // Pre-feedforward norm -> MLP -> post-feedforward norm -> residual
        let ff_normed = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&ff_normed)?;
        let ff_post_normed = self.post_feedforward_layernorm.forward(&mlp_out)?;
        h.add(ff_post_normed)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma2Attention as Module<Gemma2AttentionInput<'_, C>>>::training_mode(
            &mut self.self_attn,
            mode,
        );
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma2Model {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<Gemma2Block>,
    #[param]
    norm: nn::RmsNorm,

    hidden_size: i32,
    cached_embed_scale: Option<Array>,
}

struct Gemma2ModelInput<'a, C> {
    inputs: &'a Array,
    mask: Option<&'a Array>,
    cache: &'a mut Vec<Option<C>>,
    /// Absolute RoPE offset for causal mask alignment (caller-maintained).
    rope_offset: usize,
}

impl Gemma2Model {
    fn new(args: &Gemma2ModelArgs) -> Result<Self, Exception> {
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }

        let layers = (0..args.num_hidden_layers)
            .map(|i| Gemma2Block::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            hidden_size: args.hidden_size,
            cached_embed_scale: None,
        })
    }
}

impl<C> Module<Gemma2ModelInput<'_, C>> for Gemma2Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Gemma2ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma2ModelInput {
            inputs,
            mask,
            cache,
            rope_offset,
        } = input;

        // Gemma 2 scales embeddings by sqrt(hidden_size)
        let mut h = self.embed_tokens.forward(inputs)?;
        if self.cached_embed_scale.is_none() {
            let hidden_size_f32 = f32::from(
                i16::try_from(self.hidden_size)
                    .map_err(|_| Exception::custom("hidden_size out of i16 range"))?,
            );
            self.cached_embed_scale = Some(array!(hidden_size_f32.sqrt()).as_dtype(h.dtype())?);
        }
        let embed_scale = self
            .cached_embed_scale
            .as_ref()
            .ok_or_else(|| Exception::custom("cached_embed_scale not initialized"))?;
        h = h.multiply(embed_scale)?;

        let computed_mask = match mask {
            Some(m) => Some(m.clone()),
            None => match create_attention_mask(&h, cache, rope_offset, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only Array mask is supported"));
                }
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| None).collect();
        } else if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "kv_cache length ({}) must match num layers ({})",
                cache.len(),
                self.layers.len()
            )));
        }

        for (layer, layer_cache) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(Gemma2AttentionInput {
                x: &h,
                mask: computed_mask.as_ref(),
                cache: layer_cache.as_mut(),
            })?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <Gemma2Block as Module<Gemma2AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Causal LM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma2CausalLM {
    pub args: Gemma2ModelArgs,

    #[quantizable]
    #[param]
    model: Gemma2Model,

    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,

    cached_final_inv_cap: Option<Array>,
    cached_final_cap: Option<Array>,
}

impl Gemma2CausalLM {
    pub fn new(args: Gemma2ModelArgs) -> Result<Self, Exception> {
        let model = Gemma2Model::new(&args)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        };

        Ok(Self {
            args,
            model,
            lm_head,
            cached_final_inv_cap: None,
            cached_final_cap: None,
        })
    }

    pub fn forward_hidden<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
        rope_offset: usize,
    ) -> Result<Array, Exception> {
        self.model.forward(Gemma2ModelInput {
            inputs,
            mask,
            cache: kv_cache,
            rope_offset,
        })
    }
}

impl<C> Module<super::qwen3::ModelInput<'_, C>> for Gemma2CausalLM
where
    C: KeyValueCache,
{
    type Output = Array;

    type Error = Exception;

    fn forward(
        &mut self,
        input: super::qwen3::ModelInput<'_, C>,
    ) -> Result<Self::Output, Self::Error> {
        let super::qwen3::ModelInput {
            inputs,
            mask,
            cache,
            rope_offset,
        } = input;
        let hidden_all = Gemma2CausalLM::forward_hidden(self, inputs, mask, cache, rope_offset)?;

        let mut logits = match self.lm_head.as_mut() {
            Some(head) => head.forward(&hidden_all)?,
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed) => embed.as_linear(&hidden_all)?,
                MaybeQuantized::Quantized(q_embed) => q_embed.as_linear(&hidden_all)?,
            },
        };

        if let Some(cap) = self.args.final_logit_softcapping {
            let needs_refresh = self
                .cached_final_inv_cap
                .as_ref()
                .is_none_or(|cached| cached.dtype() != logits.dtype());
            if needs_refresh {
                self.cached_final_inv_cap = Some(array!(1.0 / cap).as_dtype(logits.dtype())?);
                self.cached_final_cap = Some(array!(cap).as_dtype(logits.dtype())?);
            }
            let final_inv_cap = self
                .cached_final_inv_cap
                .as_ref()
                .ok_or_else(|| Exception::custom("cached_final_inv_cap not initialized"))?;
            let final_cap = self
                .cached_final_cap
                .as_ref()
                .ok_or_else(|| Exception::custom("cached_final_cap not initialized"))?;
            logits = ops::tanh(&logits.multiply(final_inv_cap)?)?.multiply(final_cap)?;
        }

        Ok(logits)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma2Model as Module<Gemma2ModelInput<'_, KvCache>>>::training_mode(
            &mut self.model,
            mode,
        );
        if let Some(h) = self.lm_head.as_mut() {
            h.training_mode(mode);
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn fold_eos_token_ids(val: Option<Value>) -> Vec<u32> {
    match val {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(arr)) => arr
            .into_iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => vec![],
    }
}

fn merge_eos_bos_tokenizer_fallback(
    model_dir: &Path,
    args: &mut Gemma2ModelArgs,
) -> Result<(), Error> {
    let tok = model_dir.join("tokenizer_config.json");
    let Ok(raw) = std::fs::read_to_string(tok) else {
        return Ok(());
    };
    let v: Value = serde_json::from_str(&raw)?;
    if args.eos_token_ids.is_empty() {
        if let Some(id) = v.get("eos_token_id").and_then(|x| x.as_u64()) {
            args.eos_token_ids = vec![id as u32];
        }
    }
    if args.bos_token_id.is_none() {
        args.bos_token_id = v
            .get("bos_token_id")
            .and_then(|x| x.as_u64())
            .map(|x| x as u32);
    }
    Ok(())
}

/// Merge `eos_token_ids` / `bos_token_id` from `config.json` (same pattern as Gemma‑3).
pub fn get_gemma2_model_args(model_dir: impl AsRef<Path>) -> Result<Gemma2ModelArgs, Error> {
    let model_dir = model_dir.as_ref();
    let raw = std::fs::read_to_string(model_dir.join("config.json"))?;
    let root: Value = serde_json::from_str(&raw)?;
    let (cfg_val, eos_outer) = match root.get("text_config") {
        Some(inner) => (inner.clone(), root.get("eos_token_id").cloned()),
        None => (root.clone(), root.get("eos_token_id").cloned()),
    };
    let mut args: Gemma2ModelArgs = serde_json::from_value(cfg_val)?;
    args.eos_token_ids = fold_eos_token_ids(eos_outer);
    if args.bos_token_id.is_none() {
        args.bos_token_id = root
            .get("bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|x| x as u32);
    }
    merge_eos_bos_tokenizer_fallback(model_dir, &mut args)?;
    Ok(args)
}

/// Compatibility alias — prefer [`get_gemma2_model_args`].
pub fn load_gemma2_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Gemma2ModelArgs, Error> {
    get_gemma2_model_args(model_dir)
}

fn strip_language_model_prefix(key: &str) -> &str {
    key.strip_prefix("language_model.").unwrap_or(key)
}

fn remap_gemma2_quantized_key(key: &str) -> Option<String> {
    if let Some(prefix) = key.strip_suffix(".weight") {
        Some(format!("{prefix}.inner.weight"))
    } else if let Some(prefix) = key.strip_suffix(".scales") {
        Some(format!("{prefix}.inner.scales"))
    } else if let Some(prefix) = key.strip_suffix(".biases") {
        Some(format!("{prefix}.inner.biases"))
    } else if let Some(prefix) = key.strip_suffix(".bias") {
        Some(format!("{prefix}.inner.bias"))
    } else {
        None
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: std::collections::HashMap<String, Value>,
    pub weight_map: std::collections::HashMap<String, String>,
}

fn gemma2_safetensors_has_lm_head(model_dir: &Path) -> Result<bool, Error> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let json = std::fs::read_to_string(&index)?;
        let map: WeightMap = serde_json::from_str(&json)?;
        for raw in map.weight_map.keys() {
            let k = strip_language_model_prefix(raw);
            if k == "lm_head" || k.starts_with("lm_head.") {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        let loaded = Array::load_safetensors(&single)?;
        for raw in loaded.keys() {
            let k = strip_language_model_prefix(raw.as_str());
            if k == "lm_head" || k.starts_with("lm_head.") {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Load Gemma‑2 weights from a HuggingFace / MLX model directory.
pub fn load_gemma2_model<P: AsRef<Path>>(model_dir: P) -> Result<Gemma2CausalLM, Error> {
    use mlx_rs::module::ModuleParameters;
    use mlx_rs::quantization::MaybeQuantized;

    let model_path = model_dir.as_ref();
    let mut args = get_gemma2_model_args(model_path)?;
    match gemma2_safetensors_has_lm_head(model_path) {
        Ok(true) if args.tie_word_embeddings => {
            tracing::info!(
                "[gemma2] safetensors list `lm_head.*` — overriding tie_word_embeddings = false"
            );
            args.tie_word_embeddings = false;
        }
        Ok(_) => {}
        Err(e) => tracing::warn!("[gemma2] LM-head probe failed (continuing): {e}"),
    }

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        head_dim = args.head_dim,
        vocab_size = args.vocab_size,
        attn_softcap = ?args.attn_logit_softcapping,
        final_softcap = ?args.final_logit_softcapping,
        "Loading Gemma 2 model (MLX native)"
    );

    let quantization = args.quantization.clone();
    let raw_model = Gemma2CausalLM::new(args)?;

    let mut model = if let Some(ref qc) = quantization {
        tracing::info!(
            group_size = qc.group_size,
            bits = qc.bits,
            "Applying quantization structure to Gemma 2"
        );
        let m = mlx_rs::nn::quantize(raw_model, Some(qc.group_size), Some(qc.bits))
            .map_err(|e| Error::Other(format!("Gemma2 quantize: {e}").into()))?;
        m.eval()
            .map_err(|e| Error::Other(format!("Gemma2 post-quant eval: {e}").into()))?;
        m
    } else {
        raw_model
    };

    let is_quant = quantization.is_some();
    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let files: HashSet<&String> = weight_map.weight_map.values().collect();
        for f in files {
            shard_files.push(model_path.join(f));
        }
    } else {
        let single = model_path.join("model.safetensors");
        if !single.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing model.safetensors in {}", model_path.display()),
            )
            .into());
        }
        shard_files.push(single);
    }

    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();
    let mut embed_weight: Option<Array> = None;
    let mut embed_scales: Option<Array> = None;
    let mut embed_biases: Option<Array> = None;
    let mut unfilled: HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };

    for shard in &shard_files {
        let loaded = Array::load_safetensors(shard)?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, value) in loaded {
            let key = strip_language_model_prefix(raw_key.as_str());
            match key {
                "model.embed_tokens.weight" => {
                    embed_weight = Some(value);
                    total_loaded += 1;
                    continue;
                }
                "model.embed_tokens.scales" => {
                    embed_scales = Some(value);
                    total_loaded += 1;
                    continue;
                }
                "model.embed_tokens.biases" => {
                    embed_biases = Some(value);
                    total_loaded += 1;
                    continue;
                }
                _ => {}
            }
            if let Some(slot) = params.get_mut(key) {
                **slot = value;
                total_loaded += 1;
                unfilled.remove(key);
                continue;
            }
            if is_quant {
                if let Some(remapped) = remap_gemma2_quantized_key(key) {
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled.remove(&remapped);
                        continue;
                    }
                }
            }
            total_missed += 1;
            if unmatched_samples.len() < 5 {
                unmatched_samples.push(key.to_string());
            }
        }
    }

    if embed_weight.is_some() || embed_scales.is_some() || embed_biases.is_some() {
        match &mut model.model.embed_tokens {
            MaybeQuantized::Quantized(q) => {
                if let Some(w) = embed_weight {
                    q.inner.weight.value = w;
                }
                if let Some(s) = embed_scales {
                    q.scales.value = s;
                }
                if let Some(b) = embed_biases {
                    q.biases.value = b;
                }
            }
            MaybeQuantized::Original(e) => {
                if let Some(w) = embed_weight {
                    e.weight.value = w;
                }
            }
        }
    }

    tracing::info!(
        "[gemma2] safetensor load: {total_loaded} matched, {total_missed} unmatched (quantized={is_quant})"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[gemma2] sample unmatched keys: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled.is_empty() {
        let mut samples: Vec<&String> = unfilled.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[gemma2] {} parameter slot(s) NOT populated — first few: {}",
            unfilled.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        return Err(
            Exception::custom("no safetensor keys matched the Gemma-2 parameter tree").into(),
        );
    }

    apply_rmsnorm_plus_one(&mut model)
        .map_err(|e| Error::Other(format!("RMSNorm +1: {e}").into()))?;

    model
        .eval()
        .map_err(|e| Error::Other(format!("Gemma2 eval: {e}").into()))?;

    tracing::info!("Gemma 2 model loaded successfully");
    Ok(model)
}

/// Add 1.0 to all `RmsNorm` weight parameters.
///
/// Gemma 2 stores norm weights pre-shifted by -1. Standard `RmsNorm` computes
/// `weight * rms_norm(x)`, so adding 1.0 to the stored weights gives the
/// correct Gemma 2 behavior: `(stored_weight + 1) * rms_norm(x)`.
fn apply_rmsnorm_plus_one(model: &mut Gemma2CausalLM) -> Result<(), Exception> {
    use std::rc::Rc;

    let one = array!(1.0_f32);
    let mut params = model.parameters_mut().flatten();

    let norm_keys: Vec<Rc<str>> = params
        .keys()
        .filter(|k| k.ends_with(".weight") && k.contains("norm"))
        .cloned()
        .collect();

    for key in &norm_keys {
        if let Some(param) = params.get_mut(&**key) {
            let shifted = param.add(&one)?;
            **param = shifted;
        }
    }

    let eval_targets: Vec<&Array> = norm_keys
        .iter()
        .filter_map(|k| params.get(&**k).map(|p| &**p))
        .collect();

    mlx_rs::transforms::eval(eval_targets)?;

    Ok(())
}

/// Gemma‑2 chat template opens with `{{ bos_token }}` and stamps `{{ eos_token }}`
/// after each turn. Decode the literal token piece from the tokenizer so
/// minijinja's substitution matches what the BPE encoder produces.
impl crate::engine::chat_template_openai::ChatTemplateModel for Gemma2CausalLM {
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
                    "[local-mlx-native] chat_template bos_token injected for Gemma-2 (decoded len={})",
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
        // Gemma-2 lists all terminators in config (`eos_token_id` folded into
        // `eos_token_ids`, e.g. `<eos>`=1 and `<end_of_turn>`=107).
        self.args.eos_token_ids.clone()
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn default_gemma2_args() -> Gemma2ModelArgs {
        Gemma2ModelArgs {
            model_type: "gemma2".to_owned(),
            hidden_size: 256,
            num_hidden_layers: 2,
            intermediate_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            query_pre_attn_scalar: None,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: Some(128),
            sliding_window_pattern: 2,
            quantization: None,
            eos_token_ids: vec![1],
            bos_token_id: Some(2),
        }
    }

    #[test]
    fn config_deserialization() {
        let json = r#"{
            "model_type": "gemma2",
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "intermediate_size": 9216,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256,
            "sliding_window": 4096
        }"#;

        let args: Gemma2ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "gemma2");
        assert_eq!(args.hidden_size, 2304);
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.num_attention_heads, 8);
        assert_eq!(args.num_key_value_heads, 4);
        assert_eq!(args.attn_logit_softcapping, Some(50.0));
        assert_eq!(args.final_logit_softcapping, Some(30.0));
        assert_eq!(args.sliding_window, Some(4096));
    }

    #[test]
    fn attn_scale_uses_query_pre_attn_scalar() {
        let mut args = default_gemma2_args();
        args.query_pre_attn_scalar = Some(256);
        let expected = (256.0_f32).sqrt().recip();
        assert!((args.attn_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn attn_scale_defaults_to_head_dim() {
        let args = default_gemma2_args();
        let expected = (64.0_f32).sqrt().recip();
        assert!((args.attn_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_layer_pattern() {
        let args = default_gemma2_args();
        // pattern=2: layers 0, 2, 4... are sliding window
        assert!(args.is_sliding_window_layer(0));
        assert!(!args.is_sliding_window_layer(1));
        assert!(args.is_sliding_window_layer(2));
        assert!(!args.is_sliding_window_layer(3));
    }

    #[test]
    fn sliding_window_disabled_without_window() {
        let mut args = default_gemma2_args();
        args.sliding_window = None;
        assert!(!args.is_sliding_window_layer(0));
        assert!(!args.is_sliding_window_layer(1));
    }

    #[test]
    fn model_construction() {
        let args = default_gemma2_args();
        let model = Gemma2CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_none()); // tied embeddings
    }

    #[test]
    fn model_construction_untied_embeddings() {
        let mut args = default_gemma2_args();
        args.tie_word_embeddings = false;
        let model = Gemma2CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn model_rejects_zero_vocab_size() {
        let mut args = default_gemma2_args();
        args.vocab_size = 0;
        assert!(Gemma2CausalLM::new(args).is_err());
    }

    #[test]
    fn model_rejects_zero_layers() {
        let mut args = default_gemma2_args();
        args.num_hidden_layers = 0;
        assert!(Gemma2CausalLM::new(args).is_err());
    }

    #[test]
    fn sliding_window_mask_shape() {
        let mask = create_sliding_window_mask(4, 10, 3).unwrap();
        assert_eq!(mask.shape(), &[4, 10]);
    }

    #[test]
    fn sliding_window_mask_values() {
        // L=3, S=5, window=2
        // Query positions (absolute): 2, 3, 4 (offset = S-L = 2)
        // Key positions: 0, 1, 2, 3, 4
        // The mask only enforces the lower bound (j >= q - window + 1).
        // The causal mask separately enforces the upper bound (j <= q).
        // q=2: j >= 1 -> [F, T, T, T, T]
        // q=3: j >= 2 -> [F, F, T, T, T]
        // q=4: j >= 3 -> [F, F, F, T, T]
        let mask = create_sliding_window_mask(3, 5, 2).unwrap();
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        let expected = [
            false, true, true, true, true, false, false, true, true, true, false, false, false,
            true, true,
        ];
        assert_eq!(flat, expected);
    }

    #[test]
    fn config_defaults_without_optional_fields() {
        let json = r#"{
            "model_type": "gemma2",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1000,
            "max_position_embeddings": 512
        }"#;

        let args: Gemma2ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.attn_logit_softcapping.is_none());
        assert!(args.final_logit_softcapping.is_none());
        assert!(args.sliding_window.is_none());
        assert!(args.quantization.is_none());
        assert_eq!(args.sliding_window_pattern, 2);
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!(args.tie_word_embeddings); // Gemma 2 default: tied embeddings
    }
}
