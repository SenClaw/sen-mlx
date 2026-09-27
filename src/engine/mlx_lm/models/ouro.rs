//! Ouro looped language model (LoopLM) — ByteDance `Ouro-2.6B` /
//! `Ouro-2.6B-Thinking` (mlx-community 4-bit ports).
//!
//! Architecturally this is a Llama-style decoder (SwiGLU MLP, RoPE, multi-head
//! attention, **no** per-head Q/K norm) with two twists:
//!
//! 1. **Sandwich norm.** Each decoder layer carries FOUR RMSNorms — a pre- and
//!    post-norm around *both* the attention and the MLP sublayer:
//!    `input_layernorm` (pre-attn), `input_layernorm_2` (post-attn, pre-add),
//!    `post_attention_layernorm` (pre-mlp), `post_attention_layernorm_2`
//!    (post-mlp, pre-add). This double normalisation is what keeps the
//!    recurrent loop numerically stable.
//!
//! 2. **Universal-Transformer loop.** The same `num_hidden_layers` (48) blocks
//!    are applied `total_ut_steps` (4) times. After *each* full sweep the
//!    shared final `norm` is applied and its output feeds the next sweep. The
//!    final sweep's normed hidden state is what the LM head reads. The model
//!    also trains an `early_exit_gate`, but `early_exit_threshold` defaults to
//!    `1.0` ⇒ the gate always selects the last step, so it is a no-op at
//!    inference and is intentionally **not loaded** here.
//!
//! ## KV cache
//!
//! Each `(ut_step, layer)` pair attends with its own Q/K/V (the hidden state
//! differs every sweep), so the cache holds `total_ut_steps * num_hidden_layers`
//! (= 192) independent slots, indexed `ut_step * num_hidden_layers + layer` —
//! matching the reference `modeling_ouro.py`. The engine (`mlx_native.rs`)
//! sizes the cache vector to that product by reporting `n_layers = 192` from
//! `load_state`; this module just indexes into it.
//!
//! Param naming matches HF safetensors keys exactly:
//! - `model.embed_tokens.weight`
//! - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
//! - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
//! - `model.layers.{i}.input_layernorm.weight` / `.input_layernorm_2.weight`
//! - `model.layers.{i}.post_attention_layernorm.weight` / `…_2.weight`
//! - `model.norm.weight`
//! - `lm_head.weight` (Ouro ships `tie_word_embeddings == false`)

use std::{collections::HashMap, path::Path};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::super::{
    cache::{KeyValueCache, KvFetchResult},
    error::Error,
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        scaled_dot_product_attention, AttentionMask,
    },
};
// Reuse the data-only input carriers from qwen3 (same as llama does); they are
// generic over `C: KeyValueCache` so the engine drives every transformer arch
// through identical call sites.
pub use super::qwen3::{sample, AttentionInput, ModelInput};

fn default_total_ut_steps() -> i32 {
    4
}

fn default_eos() -> u32 {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    /// Optional in some configs (derived from `hidden_size / num_attention_heads`).
    #[serde(default)]
    pub head_dim: i32,
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Number of recurrent "Universal Transformer" sweeps over the shared
    /// layer stack. Ouro-2.6B uses 4 (`R4`).
    #[serde(default = "default_total_ut_steps")]
    pub total_ut_steps: i32,
    /// EOS id from `config.json` (Ouro: `2` = `<|im_end|>` in the SmolLM vocab).
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
}

impl ModelArgs {
    pub fn normalize(&mut self) {
        if self.head_dim <= 0 && self.num_attention_heads > 0 {
            self.head_dim = self.hidden_size / self.num_attention_heads;
        }
        if self.total_ut_steps <= 0 {
            self.total_ut_steps = 1;
        }
    }

    /// Total KV-cache slots required: one per `(ut_step, layer)` pair.
    pub fn total_cache_layers(&self) -> usize {
        (self.total_ut_steps.max(1) as usize) * (self.num_hidden_layers.max(0) as usize)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

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
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
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

        let mut queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
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
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

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
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
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

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gated =
            nn::silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&gated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

/// One Ouro decoder layer with the sandwich-norm residual flow:
/// ```text
///   r = x;  x = input_layernorm(x);  x = attn(x)
///   x = input_layernorm_2(x);  x = r + x
///   r = x;  x = post_attention_layernorm(x);  x = mlp(x)
///   x = post_attention_layernorm_2(x);  out = r + x
/// ```
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
    pub input_layernorm_2: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm_2: nn::RmsNorm,
}

impl DecoderLayer {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let self_attn = Attention::new(args)?;
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size)?;
        let mk_norm = || {
            nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()
        };
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: mk_norm()?,
            input_layernorm_2: mk_norm()?,
            post_attention_layernorm: mk_norm()?,
            post_attention_layernorm_2: mk_norm()?,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for DecoderLayer
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

        // Attention sublayer (pre- + post-norm around attn).
        let attn_in = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
            rope_offset,
        };
        let attn_out = self.self_attn.forward(attn_in)?;
        let h = x.add(self.input_layernorm_2.forward(&attn_out)?)?;

        // MLP sublayer (pre- + post-norm around mlp).
        let mlp_out = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(self.post_attention_layernorm_2.forward(&mlp_out)?)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.input_layernorm_2.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.post_attention_layernorm_2.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct OuroModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,
    /// Recurrent sweeps the checkpoint was trained with (`config.total_ut_steps`).
    pub total_ut_steps: i32,
    /// Runtime override for the number of sweeps actually run (see
    /// [`Model::set_active_ut_steps`]). `None` → use `total_ut_steps`. Not a
    /// parameter — purely an inference knob, never (de)serialized.
    pub active_ut_steps: Option<i32>,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl OuroModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|_| DecoderLayer::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            total_ut_steps: args.total_ut_steps.max(1),
            active_ut_steps: None,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }

    /// Effective recurrent sweeps for this forward — the runtime override
    /// clamped to `1..=total_ut_steps` (values above the trained count are
    /// capped; `None` / `0` falls back to the trained count).
    fn effective_ut_steps(&self) -> usize {
        let trained = self.total_ut_steps.max(1);
        match self.active_ut_steps {
            Some(n) if n >= 1 => n.min(trained) as usize,
            _ => trained as usize,
        }
    }

    /// Run the full recurrent body. Returns the final post-norm hidden state
    /// `[B, L, H]` (the last UT sweep's `norm` output) — no LM head applied.
    fn forward_body<C>(&mut self, input: ModelInput<'_, C>) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let ModelInput {
            inputs,
            mask,
            cache,
            rope_offset,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, rope_offset, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only `Array` mask is supported"));
                }
                None => None,
            },
        };

        let n_layers = self.layers.len();
        // Run the runtime-selected number of sweeps (≤ trained `total_ut_steps`).
        // Fewer sweeps = proportionally faster decode for a little quality (the
        // model is trained so every step's output is usable). Cache slots for
        // any skipped trailing sweeps simply stay empty — `ut < total_steps`
        // never indexes them, and their RoPE-independent early sweeps (ut 0..k)
        // write identical KV regardless of the total, so a full-depth prefix
        // snapshot stays valid when decoding at the same-or-lower depth.
        let total_steps = self.effective_ut_steps();

        // The engine pre-sizes `cache` to `total_ut_steps * n_layers` Some(KvCache)
        // slots. Guard the cacheless path (e.g. a fresh snapshot vector) by
        // sizing to the trained length so `ut * n_layers + l` always indexes in.
        if cache.is_empty() {
            *cache = (0..self.total_ut_steps.max(1) as usize * n_layers)
                .map(|_| None)
                .collect();
        }

        for ut in 0..total_steps {
            for l in 0..n_layers {
                let idx = ut * n_layers + l;
                let c = cache.get_mut(idx).and_then(|slot| slot.as_mut());
                let layer_input = AttentionInput {
                    x: &h,
                    mask: mask.as_ref(),
                    cache: c,
                    rope_offset,
                };
                h = self.layers[l].forward(layer_input)?;
            }
            // Shared final norm is applied after every sweep; its output feeds
            // the next sweep and (on the last sweep) the LM head.
            h = self.norm.forward(&h)?;
        }

        Ok(h)
    }

    fn training_mode_impl<C: KeyValueCache>(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <DecoderLayer as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: OuroModel,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = OuroModel::new(&args)?;
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

    /// Trained recurrent depth (`config.total_ut_steps`).
    pub fn total_ut_steps(&self) -> i32 {
        self.model.total_ut_steps
    }

    /// Override the number of recurrent sweeps run per forward. `Some(n)` runs
    /// `min(n, total_ut_steps)` sweeps (n ≥ 1); `None` or `Some(0)` restores the
    /// trained depth. Set per request from `LocalModelSettings::recurrence_steps`
    /// before prefill/decode. Persists on the loaded model until changed.
    pub fn set_active_ut_steps(&mut self, steps: Option<i32>) {
        self.model.active_ut_steps = match steps {
            Some(n) if n >= 1 => Some(n),
            _ => None,
        };
    }

    fn lm_head_forward(&mut self, hidden: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(hidden),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(hidden),
                MaybeQuantized::Quantized(q) => q.as_linear(hidden),
            },
        }
    }

    /// Recurrent body only — hidden states `[B, L, H]`, no LM head. Used for
    /// intermediate chunks of a chunked prefill (their KV writes are the only
    /// side-effect we need).
    pub fn forward_hidden<C: KeyValueCache>(
        &mut self,
        input: ModelInput<'_, C>,
    ) -> Result<Array, Exception> {
        self.model.forward_body(input)
    }

    /// Body + LM head on the **last position only** — final prefill chunk and
    /// decode-time single-token forward.
    pub fn forward_last_token<C: KeyValueCache>(
        &mut self,
        input: ModelInput<'_, C>,
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward_body(input)?;
        let last = hidden.index((.., -1.., ..));
        self.lm_head_forward(&last)
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward_body(input)?;
        self.lm_head_forward(&out)
    }

    fn training_mode(&mut self, mode: bool) {
        self.model.training_mode_impl::<C>(mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_ouro_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_ouro_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    let mut args: ModelArgs = serde_json::from_reader(file)?;
    args.normalize();
    Ok(args)
}

/// Ouro uses the SmolLM/ChatML vocabulary: turns open with `<|im_start|>` and
/// close with `<|im_end|>`. The `-Thinking` variant additionally emits
/// `<think>…</think>` reasoning blocks (the chat template prefills the opening
/// `<think>` when `enable_thinking`). The base variant never emits `<think>`,
/// so carrying the marker for both is harmless.
impl crate::engine::chat_template_openai::ChatTemplateModel for Model {
    fn markers(&self) -> crate::engine::stream_parser::MarkerSet {
        crate::engine::stream_parser::MarkerSet {
            think: Some(("<think>".into(), "</think>".into())),
            tool_call: None,
            channel: None,
            quote: None,
            tool_call_format: crate::engine::stream_parser::ToolCallFormat::QwenJsonOrXml,
        }
    }

    fn resolve_special_tokens(
        &self,
        _template: &str,
        _tokenizer: &crate::engine::mlx_lm_utils::tokenizer::Tokenizer,
    ) -> crate::engine::chat_template_openai::SpecialTokens {
        crate::engine::chat_template_openai::SpecialTokens::empty()
    }

    fn stop_token_ids(
        &self,
        tokenizer: &crate::engine::mlx_lm_utils::tokenizer::Tokenizer,
    ) -> Vec<u32> {
        // ChatML terminators resolved by name against this checkpoint's vocab,
        // plus the config-declared EOS as a backstop.
        let mut ids = crate::engine::chat_template_openai::resolve_token_ids(
            tokenizer,
            &["<|im_end|>", "<|endoftext|>"],
        );
        if !ids.contains(&self.args.eos_token_id) {
            ids.push(self.args.eos_token_id);
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed `mlx-community/Ouro-2.6B-4bit/config.json`. Verifies the
    /// looped-LM fields parse and the cache-layer product is 4 × 48 = 192.
    #[test]
    fn parses_ouro_config_and_cache_layers() {
        let raw = r#"{
            "model_type": "ouro",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 5632,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-06,
            "vocab_size": 49152,
            "num_key_value_heads": 16,
            "max_position_embeddings": 65536,
            "rope_theta": 1000000.0,
            "head_dim": 128,
            "tie_word_embeddings": false,
            "rope_scaling": null,
            "total_ut_steps": 4,
            "eos_token_id": 2
        }"#;
        let mut args: ModelArgs = serde_json::from_str(raw).expect("parse ouro config");
        args.normalize();
        assert_eq!(args.total_ut_steps, 4);
        assert_eq!(args.num_hidden_layers, 48);
        assert_eq!(args.head_dim, 128);
        assert_eq!(args.total_cache_layers(), 192);
    }

    /// `total_ut_steps` defaults to 4 and `head_dim` is derived when absent.
    #[test]
    fn defaults_total_ut_steps_and_derives_head_dim() {
        let raw = r#"{
            "model_type": "ouro",
            "hidden_size": 2048,
            "num_hidden_layers": 12,
            "intermediate_size": 5632,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-06,
            "vocab_size": 49152,
            "num_key_value_heads": 16,
            "max_position_embeddings": 65536,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
            "rope_scaling": null
        }"#;
        let mut args: ModelArgs = serde_json::from_str(raw).expect("parse ouro config");
        args.normalize();
        assert_eq!(args.total_ut_steps, 4, "total_ut_steps default");
        assert_eq!(args.head_dim, 128, "derived 2048/16");
        assert_eq!(args.eos_token_id, 2, "eos default");
        assert_eq!(args.total_cache_layers(), 48);
    }
}
