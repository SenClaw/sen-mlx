//! Llama-family inference (Llama-3.x, Llama-3.2, and `mlx-community/nesso-*`).
//!
//! Architecturally identical to Qwen3 minus the per-head Q/K RMSNorm: Llama
//! attention writes Q/K directly through RoPE without intermediate
//! normalisation. Param naming matches HF safetensors keys exactly:
//!
//! - `model.embed_tokens.weight`
//! - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
//! - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
//! - `model.layers.{i}.input_layernorm.weight`
//! - `model.layers.{i}.post_attention_layernorm.weight`
//! - `model.norm.weight`
//! - `lm_head.weight` (only when `tie_word_embeddings == false`)
//!
//! ## Sharing with Qwen3
//!
//! `ModelInput`, `AttentionInput`, and `sample` are reused from
//! [`super::qwen3`] — they're data-only types operating on a generic
//! `C: KeyValueCache`, so the engine (`mlx_native.rs`) can drive both
//! architectures through the same call sites.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::zeros_dtype,
    quantization::MaybeQuantized,
    Array, Dtype,
};
use serde::Deserialize;
use serde_json::Value;
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
// Reuse the input types from qwen3 verbatim — they are data carriers, not
// architecture-specific.
pub use super::qwen3::{sample, AttentionInput, ModelInput};

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
    /// Optional in some Llama configs (derived from `hidden_size / num_attention_heads`).
    #[serde(default)]
    pub head_dim: i32,
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    /// EOS id from `config.json`. Falls back to `2` (Llama-2 default) if missing.
    /// Variants like Nesso use a custom value (e.g. `128256`).
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
}

fn default_eos() -> u32 {
    2
}

impl ModelArgs {
    pub fn is_qwen2(&self) -> bool {
        self.model_type.to_lowercase().starts_with("qwen2")
    }

    pub fn normalize(&mut self) {
        if self.head_dim <= 0 && self.num_attention_heads > 0 {
            self.head_dim = self.hidden_size / self.num_attention_heads;
        }
        if self.is_qwen2() {
            // Qwen2/Qwen2.5 checkpoints have Q/K/V projection bias but no O-proj
            // bias. `Attention::new` handles that split explicitly.
            self.attention_bias = true;
        }
    }
}

fn zero_embedding_bf16(embedding_count: i32, dimensions: i32) -> Result<nn::Embedding, Exception> {
    let weight = zeros_dtype(&[embedding_count, dimensions], Dtype::Bfloat16)?;
    Ok(nn::Embedding {
        weight: Param::new(weight),
    })
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
            .bias(args.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(args.attention_bias && !args.is_qwen2())
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

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
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
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let self_attn = Attention::new(args)?;
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size, args.mlp_bias)?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
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

        let self_attn_input = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
            rope_offset,
        };
        let r = self.self_attn.forward(self_attn_input)?;
        let h = x.add(r)?;
        let r = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct LlamaModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl LlamaModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());
        let embed_tokens = if args.is_qwen2() {
            zero_embedding_bf16(1, args.hidden_size)?
        } else {
            nn::Embedding::new(args.vocab_size, args.hidden_size)?
        };
        let layers = (0..args.num_hidden_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for LlamaModel
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

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| None).collect();
        }
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
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

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: LlamaModel,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = LlamaModel::new(&args)?;
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
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out),
                MaybeQuantized::Quantized(q) => q.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <LlamaModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_llama_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_llama_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    let mut args: ModelArgs = serde_json::from_reader(file)?;
    args.normalize();
    Ok(args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_llama_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_llama_model_args(model_dir)?;
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
        shard_files.push(model_dir.join("model.safetensors"));
    }

    for weight_file in shard_files {
        model.load_safetensors(weight_file)?;
    }
    Ok(model)
}

/// Llama-3 and Qwen2 chat templates keep their header/sentinel tokens inline.
/// Neither needs runtime `bos_token` / `eos_token` injection.
impl crate::engine::chat_template_openai::ChatTemplateModel for Model {
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
        // Config EOS (custom values like Nesso 128256 and Qwen2 `<|im_end|>`
        // live here) plus well-known chat terminators resolved by name.
        let mut ids = vec![self.args.eos_token_id];
        for id in crate::engine::chat_template_openai::resolve_token_ids(
            tokenizer,
            &["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
        ) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        for id in [128_009u32, 128_001] {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }
}
