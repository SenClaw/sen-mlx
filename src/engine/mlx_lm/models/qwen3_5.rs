//! Qwen3.5 hybrid model (GatedDeltaNet + full attention), including OptiQ quants.
//!
//! Ported from `mlx-lm` `qwen3_5.py` / `qwen3_next.py`. Weight keys use the
//! `language_model.model.*` prefix (mlx-community layout).

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{concatenate_axis, indexing::IndexOp, ones_dtype, sigmoid},
    quantization::{MaybeQuantized, Quantizable},
    transforms::{eval, eval_params},
    Array, Dtype,
};
use serde::Deserialize;
use serde_json::Value;

use super::super::{
    cache::{KvCache, KvFetchResult, Qwen35LinearCache},
    error::Error,
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        scaled_dot_product_attention, AttentionMask,
    },
};
use super::gated_delta::gated_delta_update;
use super::qwen3::{AttentionInput, Mlp};

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TextModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: i32,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

fn default_full_attn_interval() -> i32 {
    4
}
fn default_partial_rotary() -> f32 {
    0.25
}
fn default_rope_theta() -> f32 {
    100_000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub text_config: TextModelArgs,
}

impl ModelArgs {
    pub fn from_config_json(cfg: &Value) -> Result<Self, Error> {
        if let Some(text) = cfg.get("text_config") {
            let mut text_config: TextModelArgs = serde_json::from_value(text.clone())?;
            if let Some(rp) = text.get("rope_parameters").and_then(|v| v.as_object()) {
                if let Some(f) = rp.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                    text_config.partial_rotary_factor = f as f32;
                }
                if let Some(t) = rp.get("rope_theta").and_then(|v| v.as_f64()) {
                    text_config.rope_theta = t as f32;
                }
            }
            Ok(Self {
                model_type: cfg
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("qwen3_5")
                    .to_string(),
                text_config,
            })
        } else {
            let text_config: TextModelArgs = serde_json::from_value(cfg.clone())?;
            Ok(Self {
                model_type: text_config.model_type.clone(),
                text_config,
            })
        }
    }
}

// -----------------------------------------------------------------------------
// Gated RMSNorm + GatedDeltaNet
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct RmsNormGated {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl RmsNormGated {
    pub fn new(dim: i32, eps: f32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(ones_dtype(&[dim], Dtype::Float32)?),
            eps,
        })
    }

    pub fn forward(&self, hidden: &Array, gate: &Array) -> Result<Array, Exception> {
        // mlx-lm Qwen3NextRMSNormGated: rms_norm(hidden); silu(gate) * normed
        let normed = nn::RmsNorm {
            weight: self.weight.clone(),
            eps: self.eps,
        }
        .forward(hidden)?;
        let gate_f = nn::silu(&gate.as_dtype(Dtype::Float32)?)?;
        let out = gate_f.multiply(&normed.as_dtype(Dtype::Float32)?)?;
        out.as_dtype(hidden.dtype())
    }
}

fn stateless_rms_norm(x: &Array, eps: f32) -> Result<Array, Exception> {
    let dim = *x.shape().last().unwrap_or(&1);
    let ones = ones_dtype(&[dim], x.dtype())?;
    mlx_rs::fast::rms_norm(x, &ones, eps)
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
#[allow(non_snake_case)]
pub struct GatedDeltaNet {
    pub hidden_size: i32,
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub conv_dim: i32,
    pub d_conv: i32,

    #[quantizable]
    #[param]
    pub in_proj_qkv: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub in_proj_z: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub in_proj_b: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub in_proj_a: MaybeQuantized<nn::Linear>,
    #[param]
    pub conv1d: nn::Conv1d,
    #[param]
    pub dt_bias: Param<Array>,
    #[param]
    #[allow(non_snake_case)]
    pub A_log: Param<Array>,
    #[param]
    pub norm: RmsNormGated,
    #[quantizable]
    #[param]
    pub out_proj: MaybeQuantized<nn::Linear>,
}

impl GatedDeltaNet {
    pub fn new(args: &TextModelArgs) -> Result<Self, Exception> {
        let num_v = args.linear_num_value_heads;
        let num_k = args.linear_num_key_heads;
        let head_k = args.linear_key_head_dim;
        let head_v = args.linear_value_head_dim;
        let key_dim = head_k * num_k;
        let value_dim = head_v * num_v;
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = args.linear_conv_kernel_dim;

        let in_proj_qkv = nn::LinearBuilder::new(args.hidden_size, key_dim * 2 + value_dim)
            .bias(false)
            .build()?;
        let in_proj_z = nn::LinearBuilder::new(args.hidden_size, value_dim)
            .bias(false)
            .build()?;
        let in_proj_b = nn::LinearBuilder::new(args.hidden_size, num_v)
            .bias(false)
            .build()?;
        let in_proj_a = nn::LinearBuilder::new(args.hidden_size, num_v)
            .bias(false)
            .build()?;
        let out_proj = nn::LinearBuilder::new(value_dim, args.hidden_size)
            .bias(false)
            .build()?;
        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, d_conv)
            .groups(conv_dim)
            .bias(false)
            .padding(0)
            .build()?;

        Ok(Self {
            hidden_size: args.hidden_size,
            num_v_heads: num_v,
            num_k_heads: num_k,
            head_k_dim: head_k,
            head_v_dim: head_v,
            key_dim,
            value_dim,
            conv_dim,
            d_conv,
            in_proj_qkv: MaybeQuantized::Original(in_proj_qkv),
            in_proj_z: MaybeQuantized::Original(in_proj_z),
            in_proj_b: MaybeQuantized::Original(in_proj_b),
            in_proj_a: MaybeQuantized::Original(in_proj_a),
            conv1d,
            dt_bias: Param::new(ones_dtype(&[num_v], Dtype::Float32)?),
            A_log: Param::new(mlx_rs::ops::zeros_dtype(&[num_v], Dtype::Float32)?),
            norm: RmsNormGated::new(head_v, args.rms_norm_eps)?,
            out_proj: MaybeQuantized::Original(out_proj),
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        cache: &mut Qwen35LinearCache,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let b_size = shape[0];
        let seq_len = shape[1];
        let dtype = x.dtype();

        let x_bc = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?.reshape(&[
            b_size,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ])?;
        let b = self.in_proj_b.forward(x)?;
        let a = self.in_proj_a.forward(x)?;

        let prev = cache.conv_state_or_init(b_size, dtype)?.clone();
        let xbc_aug = concatenate_axis(&[prev, x_bc.clone()], 1)?;
        let mut x_bc_conv = self.conv1d.forward(&xbc_aug)?;
        x_bc_conv = nn::silu(&x_bc_conv)?;
        let total_len = xbc_aug.shape()[1];
        let new_conv = xbc_aug.index((.., (total_len - (self.d_conv - 1))..total_len, ..));
        cache.set_conv_state(new_conv);

        let split_a = self.key_dim;
        let split_b = self.key_dim + self.num_k_heads * self.head_k_dim;
        let q = x_bc_conv.index((.., .., 0..split_a)).reshape(&[
            b_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ])?;
        let k = x_bc_conv.index((.., .., split_a..split_b)).reshape(&[
            b_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ])?;
        let v = x_bc_conv
            .index((.., .., split_b..self.conv_dim))
            .reshape(&[b_size, seq_len, self.num_v_heads, self.head_v_dim])?;

        // RMS-norm THEN scale (mlx-lm `q = inv_scale**2 * rms_norm(q)`,
        // `k = inv_scale * rms_norm(k)`). The previous code multiplied by the
        // scale *before* rms_norm, which is scale-invariant and cancels — that
        // left q,k at L2 = sqrt(head_dim) and the delta-rule state diverged to
        // inf/NaN. Applying the scale after gives k unit-L2 and q at
        // 1/sqrt(head_dim), exactly as the reference.
        let inv_scale = (self.head_k_dim as f32).sqrt().recip();
        let q = stateless_rms_norm(&q, 1e-6)?.multiply(&array!(inv_scale * inv_scale))?;
        let k = stateless_rms_norm(&k, 1e-6)?.multiply(&array!(inv_scale))?;

        let state_in = cache.ssm_state_or_init(b_size)?.clone();
        let (y, state_out) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b,
            self.A_log.as_ref(),
            self.dt_bias.as_ref(),
            Some(&state_in),
            mask,
        )?;
        cache.set_ssm_state(state_out);
        cache.advance(seq_len);

        let y_out = self.norm.forward(&y, &z)?;
        self.out_proj
            .forward(&y_out.reshape(&[b_size, seq_len, self.value_dim])?)
    }
}

// -----------------------------------------------------------------------------
// Full attention (Qwen3-Next style output gate)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct FullAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope_dims: i32,

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
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl FullAttention {
    pub fn new(args: &TextModelArgs) -> Result<Self, Exception> {
        let n_heads = args.num_attention_heads;
        let n_kv = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let rope_dims = (head_dim as f32 * args.partial_rotary_factor).round() as i32;
        let q_proj = nn::LinearBuilder::new(args.hidden_size, n_heads * head_dim * 2)
            .bias(args.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(args.hidden_size, n_kv * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(args.hidden_size, n_kv * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, args.hidden_size)
            .bias(false)
            .build()?;
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let rope = initialize_rope(
            rope_dims,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;
        Ok(Self {
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            rope_dims,
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

impl<C> Module<AttentionInput<'_, C>> for FullAttention
where
    C: super::super::cache::KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            mut cache,
            rope_offset,
        } = input;
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let rope_off = i32::try_from(rope_offset)
            .map_err(|_| Exception::custom("rope_offset exceeds i32::MAX"))?;

        let q_out = self.q_proj.forward(x)?;
        let reshaped = q_out.reshape(&[b, l, self.n_heads, self.head_dim * 2])?;
        let mid = self.head_dim;
        let queries = reshaped.index((.., .., .., 0..mid));
        let gate = reshaped.index((.., .., .., mid..));
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Norm on [B, H, L, D] — keep that layout for RoPE, KV cache, and SDPA (mlx-lm qwen3_next).
        let mut queries = self
            .q_norm
            .forward(&queries.transpose_axes(&[0, 2, 1, 3])?)?;
        let mut keys = self.k_norm.forward(
            &keys
                .reshape(&[b, l, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut values = values
            .reshape(&[b, l, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let fetch = if let Some(cache) = cache.as_mut() {
            let q_in = nn::RopeInputBuilder::new(&queries)
                .offset(rope_off)
                .build()?;
            queries = self.rope.forward(q_in)?;
            let k_in = nn::RopeInputBuilder::new(&keys).offset(rope_off).build()?;
            keys = self.rope.forward(k_in)?;
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
                return Err(Exception::custom("TurboQuant not supported for Qwen3.5"));
            }
        }
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, l, -1])?;

        let gated = output.multiply(&sigmoid(&gate.reshape(&[b, l, -1])?)?)?;
        self.o_proj.forward(&gated)
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
// Decoder layer + backbone
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    pub is_linear: bool,
    #[quantizable]
    #[param]
    pub linear_attn: Option<GatedDeltaNet>,
    #[quantizable]
    #[param]
    pub self_attn: Option<FullAttention>,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
}

impl DecoderLayer {
    pub fn new(args: &TextModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let is_linear = (layer_idx + 1) % args.full_attention_interval != 0;
        let (linear_attn, self_attn) = if is_linear {
            (Some(GatedDeltaNet::new(args)?), None)
        } else {
            (None, Some(FullAttention::new(args)?))
        };
        Ok(Self {
            is_linear,
            linear_attn,
            self_attn,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TextBackbone {
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
    pub full_attention_interval: i32,
}

impl TextBackbone {
    pub fn new(args: &TextModelArgs) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i))
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
            full_attention_interval: args.full_attention_interval,
        })
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut [Option<KvCache>],
        rope_offset: usize,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(inputs)?;

        let fa_mask = create_attention_mask(&h, caches, rope_offset, None)?;
        let mask_for_fa = match fa_mask.as_ref() {
            Some(AttentionMask::Array(a)) => Some(a),
            _ => None,
        };

        for (layer, slot) in self.layers.iter_mut().zip(caches.iter_mut()) {
            let normed = layer.input_layernorm.forward(&h)?;
            let r = if layer.is_linear {
                let cache = slot
                    .as_mut()
                    .and_then(KvCache::as_qwen35_linear_mut)
                    .ok_or_else(|| {
                        Exception::custom("Qwen3.5 linear layer needs Qwen35Linear cache slot")
                    })?;
                layer
                    .linear_attn
                    .as_mut()
                    .expect("linear_attn")
                    .forward(&normed, cache, None)?
            } else {
                let attn_input = AttentionInput {
                    x: &normed,
                    mask: mask_for_fa,
                    cache: slot.as_mut(),
                    rope_offset,
                };
                layer
                    .self_attn
                    .as_mut()
                    .expect("self_attn")
                    .forward(attn_input)?
            };
            h = h.add(&r)?;
            let mlp_out = layer
                .mlp
                .forward(&layer.post_attention_layernorm.forward(&h)?)?;
            h = h.add(&mlp_out)?;
            eval(&[h.clone()])?;
        }
        self.norm.forward(&h)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct LanguageModel {
    #[quantizable]
    #[param]
    pub model: TextBackbone,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub language_model: LanguageModel,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let text = &args.text_config;
        let lm_head = if text.tie_word_embeddings {
            None
        } else {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(text.hidden_size, text.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        };
        Ok(Self {
            args: args.clone(),
            language_model: LanguageModel {
                model: TextBackbone::new(text)?,
                lm_head,
            },
        })
    }

    pub fn make_cache(&self) -> Vec<Option<KvCache>> {
        let args = &self.args.text_config;
        let key_dim = args.linear_key_head_dim * args.linear_num_key_heads;
        let value_dim = args.linear_value_head_dim * args.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        (0..args.num_hidden_layers)
            .map(|i| {
                if (i + 1) % args.full_attention_interval != 0 {
                    Some(KvCache::qwen35_linear(
                        conv_dim,
                        args.linear_conv_kernel_dim,
                        args.linear_num_value_heads,
                        args.linear_value_head_dim,
                        args.linear_key_head_dim,
                    ))
                } else {
                    Some(KvCache::fp16_with_max(args.max_position_embeddings))
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
        let h = self
            .language_model
            .model
            .forward(inputs, caches, rope_offset)?;
        match &mut self.language_model.lm_head {
            Some(lm) => lm.forward(&h),
            None => match &mut self.language_model.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&h),
                MaybeQuantized::Quantized(e) => e.as_linear(&h),
            },
        }
    }

    /// Force all weights to be materialized now. Called at load time so the
    /// weights are resident after `warm_up` (rather than lazily realized on the
    /// first forward — which otherwise makes RSS jump by the full weight size
    /// on request #1 and stalls that request). `eval(&[])` was a no-op.
    pub fn eval(&self) -> Result<(), Exception> {
        eval_params(self.parameters())
    }
}

// -----------------------------------------------------------------------------
// Load + OptiQ mixed quant
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub weight_map: HashMap<String, String>,
}

pub fn get_qwen35_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let raw = std::fs::read_to_string(model_dir.as_ref().join("config.json"))?;
    let cfg: Value = serde_json::from_str(&raw)?;
    ModelArgs::from_config_json(&cfg)
}

fn optiq_bits_map(cfg: &Value) -> HashMap<String, (i32, i32)> {
    let mut map = HashMap::new();
    let q = cfg
        .get("quantization")
        .or_else(|| cfg.get("quantization_config"));
    let Some(q) = q.and_then(|v| v.as_object()) else {
        return map;
    };
    let default_gs = q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32;
    let default_bits = q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32;
    for (k, v) in q {
        if matches!(k.as_str(), "group_size" | "bits" | "mode") {
            continue;
        }
        let Some(obj) = v.as_object() else { continue };
        let bits = obj
            .get("bits")
            .and_then(|x| x.as_i64())
            .unwrap_or(default_bits as i64) as i32;
        let gs = obj
            .get("group_size")
            .and_then(|x| x.as_i64())
            .unwrap_or(default_gs as i64) as i32;
        map.insert(k.clone(), (gs, bits));
    }
    map
}

fn optiq_lookup(path: &str, map: &HashMap<String, (i32, i32)>, default: (i32, i32)) -> (i32, i32) {
    map.get(path).copied().unwrap_or(default)
}

fn quantize_maybe_linear(
    slot: &mut MaybeQuantized<nn::Linear>,
    path: &str,
    map: &HashMap<String, (i32, i32)>,
    default: (i32, i32),
) -> Result<(), Exception> {
    if slot.is_quantized() {
        return Ok(());
    }
    let (gs, bits) = optiq_lookup(path, map, default);
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
    map: &HashMap<String, (i32, i32)>,
    default: (i32, i32),
) -> Result<(), Exception> {
    if slot.is_quantized() {
        return Ok(());
    }
    let (gs, bits) = optiq_lookup(path, map, default);
    let placeholder = nn::Embedding::new(1, 1)?;
    match std::mem::replace(slot, MaybeQuantized::Original(placeholder)) {
        MaybeQuantized::Original(embed) => {
            *slot = MaybeQuantized::Quantized(embed.try_into_quantized(gs, bits)?);
        }
        MaybeQuantized::Quantized(q) => *slot = MaybeQuantized::Quantized(q),
    }
    Ok(())
}

fn quantize_mlp(
    mlp: &mut Mlp,
    prefix: &str,
    map: &HashMap<String, (i32, i32)>,
    default: (i32, i32),
) -> Result<(), Exception> {
    quantize_maybe_linear(
        &mut mlp.gate_proj,
        &format!("{prefix}.gate_proj"),
        map,
        default,
    )?;
    quantize_maybe_linear(
        &mut mlp.down_proj,
        &format!("{prefix}.down_proj"),
        map,
        default,
    )?;
    quantize_maybe_linear(&mut mlp.up_proj, &format!("{prefix}.up_proj"), map, default)?;
    Ok(())
}

fn quantize_gated_delta(
    net: &mut GatedDeltaNet,
    prefix: &str,
    map: &HashMap<String, (i32, i32)>,
    default: (i32, i32),
) -> Result<(), Exception> {
    quantize_maybe_linear(
        &mut net.in_proj_qkv,
        &format!("{prefix}.in_proj_qkv"),
        map,
        default,
    )?;
    quantize_maybe_linear(
        &mut net.in_proj_z,
        &format!("{prefix}.in_proj_z"),
        map,
        default,
    )?;
    quantize_maybe_linear(
        &mut net.in_proj_b,
        &format!("{prefix}.in_proj_b"),
        map,
        default,
    )?;
    quantize_maybe_linear(
        &mut net.in_proj_a,
        &format!("{prefix}.in_proj_a"),
        map,
        default,
    )?;
    quantize_maybe_linear(
        &mut net.out_proj,
        &format!("{prefix}.out_proj"),
        map,
        default,
    )?;
    Ok(())
}

fn quantize_full_attention(
    attn: &mut FullAttention,
    prefix: &str,
    map: &HashMap<String, (i32, i32)>,
    default: (i32, i32),
) -> Result<(), Exception> {
    quantize_maybe_linear(&mut attn.q_proj, &format!("{prefix}.q_proj"), map, default)?;
    quantize_maybe_linear(&mut attn.k_proj, &format!("{prefix}.k_proj"), map, default)?;
    quantize_maybe_linear(&mut attn.v_proj, &format!("{prefix}.v_proj"), map, default)?;
    quantize_maybe_linear(&mut attn.o_proj, &format!("{prefix}.o_proj"), map, default)?;
    Ok(())
}

fn quantize_decoder_layer(
    layer: &mut DecoderLayer,
    layer_idx: i32,
    map: &HashMap<String, (i32, i32)>,
    default: (i32, i32),
) -> Result<(), Exception> {
    let prefix = format!("language_model.model.layers.{layer_idx}");
    if layer.is_linear {
        if let Some(linear_attn) = layer.linear_attn.as_mut() {
            quantize_gated_delta(linear_attn, &format!("{prefix}.linear_attn"), map, default)?;
        }
    } else if let Some(self_attn) = layer.self_attn.as_mut() {
        quantize_full_attention(self_attn, &format!("{prefix}.self_attn"), map, default)?;
    }
    quantize_mlp(&mut layer.mlp, &format!("{prefix}.mlp"), map, default)?;
    Ok(())
}

/// OptiQ checkpoints declare per-path `bits` / `group_size` (mixed 4- and 8-bit).
fn apply_optiq_quantization(mut model: Model, cfg: &Value) -> Result<Model, Exception> {
    let q = cfg
        .get("quantization")
        .or_else(|| cfg.get("quantization_config"));
    let default = (
        q.and_then(|v| v.get("group_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(64) as i32,
        q.and_then(|v| v.get("bits"))
            .and_then(|v| v.as_i64())
            .unwrap_or(4) as i32,
    );
    let map = optiq_bits_map(cfg);

    quantize_maybe_embedding(
        &mut model.language_model.model.embed_tokens,
        "language_model.model.embed_tokens",
        &map,
        default,
    )?;

    for (i, layer) in model.language_model.model.layers.iter_mut().enumerate() {
        quantize_decoder_layer(layer, i as i32, &map, default)?;
    }

    if let Some(lm_head) = model.language_model.lm_head.as_mut() {
        quantize_maybe_linear(lm_head, "language_model.lm_head", &map, default)?;
    }

    Ok(model)
}

fn sanitize_weights(weights: &mut HashMap<String, Array>) {
    // Match mlx-lm `TextModel.sanitize`: only shift HF-style layernorm weights when
    // conv1d still needs layout fix. Do NOT touch `.linear_attn.norm.weight`.
    let has_unsanitized_conv1d = weights
        .iter()
        .any(|(k, v)| k.contains("conv1d.weight") && v.shape().last().copied() != Some(1));
    let should_shift_norm = has_unsanitized_conv1d;
    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    ];
    weights.retain(|k, _| !k.contains("mtp."));
    for (k, v) in weights.iter_mut() {
        if k.contains("conv1d.weight") && v.shape().last().copied() != Some(1) {
            if let Ok(t) = v.transpose_axes(&[0, 2, 1]) {
                *v = t;
            }
        }
        if should_shift_norm && norm_suffixes.iter().any(|s| k.ends_with(s)) && v.shape().len() == 1
        {
            if let Ok(shifted) = v.add(&array!(1.0_f32)) {
                *v = shifted;
            }
        }
    }
}

pub fn load_qwen35_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let raw_cfg = std::fs::read_to_string(model_dir.join("config.json"))?;
    let cfg: Value = serde_json::from_str(&raw_cfg)?;
    let args = ModelArgs::from_config_json(&cfg)?;
    let model = Model::new(args)?;
    let mut model =
        apply_optiq_quantization(model, &cfg).map_err(|e| Error::Other(e.to_string().into()))?;

    let index_path = model_dir.join("model.safetensors.index.json");
    let shard_files: Vec<std::path::PathBuf> = if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let map: WeightMap = serde_json::from_str(&json)?;
        map.weight_map
            .values()
            .map(|f| model_dir.join(f))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    } else {
        vec![model_dir.join("model.safetensors")]
    };

    let mut all_weights: HashMap<String, Array> = HashMap::new();
    for shard in &shard_files {
        let loaded = mlx_rs::Array::load_safetensors(shard)
            .map_err(|e| Error::Other(format!("load shard: {e:?}").into()))?;
        for (k, v) in loaded {
            all_weights.insert(k.to_string(), v);
        }
    }
    sanitize_weights(&mut all_weights);

    let mut embed_w = None;
    let mut embed_s = None;
    let mut embed_b = None;
    let mut params = model.parameters_mut().flatten();
    let mut loaded = 0usize;
    let mut missed = 0usize;
    for (key, value) in all_weights {
        let key = if let Some(rest) = key.strip_prefix("model.language_model.") {
            format!("language_model.model.{rest}")
        } else if key.starts_with("language_model.") {
            key
        } else {
            format!("language_model.{key}")
        };
        match key.as_str() {
            "language_model.model.embed_tokens.weight" => {
                embed_w = Some(value);
                loaded += 1;
                continue;
            }
            "language_model.model.embed_tokens.scales" => {
                embed_s = Some(value);
                loaded += 1;
                continue;
            }
            "language_model.model.embed_tokens.biases" => {
                embed_b = Some(value);
                loaded += 1;
                continue;
            }
            _ => {}
        }
        if let Some(slot) = params.get_mut(key.as_str()) {
            **slot = value;
            loaded += 1;
        } else if let Some(stripped) = key.strip_suffix(".weight") {
            let remapped = format!("{stripped}.inner.weight");
            if let Some(slot) = params.get_mut(remapped.as_str()) {
                **slot = value;
                loaded += 1;
                continue;
            }
            missed += 1;
        } else if let Some(stripped) = key.strip_suffix(".bias") {
            let remapped = format!("{stripped}.inner.bias");
            if let Some(slot) = params.get_mut(remapped.as_str()) {
                **slot = value;
                loaded += 1;
                continue;
            }
            missed += 1;
        } else {
            missed += 1;
        }
    }
    if let Some(w) = embed_w {
        match &mut model.language_model.model.embed_tokens {
            MaybeQuantized::Quantized(q) => {
                q.inner.weight.value = w;
                if let Some(s) = embed_s {
                    q.scales.value = s;
                }
                if let Some(b) = embed_b {
                    q.biases.value = b;
                }
            }
            MaybeQuantized::Original(e) => {
                e.weight.value = w;
            }
        }
    }
    tracing::info!("[qwen3_5] safetensor load: {loaded} matched, {missed} unmatched");
    model
        .eval()
        .map_err(|e| Error::Other(format!("eval: {e:?}").into()))?;
    Ok(model)
}

/// Qwen3.5 reuses ChatML markers (`<|im_start|>` / `<|im_end|>`); no
/// `bos_token` / `eos_token` injection needed.
impl crate::engine::chat_template_openai::ChatTemplateModel for Model {
    /// Qwen 3.5 (GatedDeltaNet hybrid): same `<think>` / `<tool_call>` outer
    /// envelope as Qwen 3, but tool-call bodies are typically the Hermes-style
    /// `<function=NAME><parameter=K>v</parameter>…</function>` XML rather than
    /// the JSON form Qwen 3 prefers. The parser's `QwenJsonOrXml` mode handles
    /// both, so we share the Qwen preset — but this method is declared
    /// separately so Qwen 3.5 can diverge (e.g. add an arch-specific marker)
    /// without touching Qwen 3.
    fn markers(&self) -> crate::engine::stream_parser::MarkerSet {
        crate::engine::stream_parser::MarkerSet::qwen()
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
        // CRITICAL: Qwen3.5's chat terminator `<|im_end|>` is token 248046, but
        // `text_config.eos_token_id` is 248044 (`<|endoftext|>`). Resolving by
        // name picks the right id for this 248320-token vocab; we also add the
        // config eos. Stopping only on the config eos lets the model run past
        // `<|im_end|>` into fake `<|im_start|>user…` turns (runaway loop).
        let mut ids = crate::engine::chat_template_openai::resolve_token_ids(
            tokenizer,
            &["<|im_end|>", "<|endoftext|>"],
        );
        if let Some(eos) = self.args.text_config.eos_token_id {
            if !ids.contains(&eos) {
                ids.push(eos);
            }
        }
        if ids.is_empty() {
            ids = vec![248_046, 248_044];
        }
        ids
    }
}
