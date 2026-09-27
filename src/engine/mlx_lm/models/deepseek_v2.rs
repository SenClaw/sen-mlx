//! DeepSeek-V2 inference — `model_type = deepseek_v2`.
//!
//! Supports `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` (16B MoE,
//! 2.4B active). This is the FIRST model in this crate to combine two
//! previously-unported subsystems:
//!
//! - **Multi-head Latent Attention (MLA)** — instead of standard per-head K/V
//!   projections, DeepSeek compresses KV into a low-rank latent
//!   (`kv_lora_rank=512`) plus a small decoupled RoPE key (`qk_rope_head_dim=64`)
//!   that is shared across heads. We implement the *naive* form (decompress to
//!   full per-head K/V, cache the expanded tensors) which is mathematically
//!   exact and fits the existing asymmetric FP16 KV cache (K head-dim 192, V
//!   head-dim 128) with no cache.rs changes. The "absorbed" latent-only form is
//!   a later memory optimization.
//! - **DeepSeekMoE** — fine-grained experts (`n_routed_experts=64`,
//!   `num_experts_per_tok=6`) plus always-on shared experts
//!   (`n_shared_experts=2`). The routed experts are a single grouped quantized
//!   matmul ([`crate::engine::mlx_lm::utils::moe::gather_qmm`]) over weights
//!   stacked as `switch_mlp.{gate,up,down}_proj` of shape `[E, out, in]`.
//!
//! Config specifics for Coder-V2-Lite (verified from the repo `config.json`):
//! `q_lora_rank=null` (single dense `q_proj`, no q low-rank), `scoring_func`
//! softmax, `topk_method` "greedy" (no group masking, `n_group=1`),
//! `norm_topk_prob=false`, `routed_scaling_factor=1.0`, `first_k_dense_replace=1`
//! (layer 0 dense, layers 1..26 MoE). RoPE is YaRN (`factor=40`, `mscale=0.707`)
//! with the mscale folded **squared** into the attention softmax scale.

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
        argpartition_axis, broadcast_to, concatenate_axis, indexing::IndexOp, softmax_axis,
        zeros_dtype,
    },
    quantization::MaybeQuantized,
    Array, Dtype,
};
use serde::Deserialize;
use serde_json::Value;

use super::super::{
    cache::{KeyValueCache, KvCache, KvFetchResult},
    error::Error,
    utils::{
        create_causal_mask, scaled_dot_product_attention,
        yarn::{apply_yarn_rope, compute_yarn_freqs, yarn_get_mscale},
    },
};
use crate::engine::mlx_lm::utils::moe::gather_qmm;

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScaling {
    #[serde(default)]
    pub factor: f32,
    #[serde(default)]
    pub mscale: f32,
    #[serde(default)]
    pub mscale_all_dim: f32,
    #[serde(default)]
    pub beta_fast: f32,
    #[serde(default)]
    pub beta_slow: f32,
    #[serde(default)]
    pub original_max_position_embeddings: i32,
    #[serde(default, rename = "type")]
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub moe_intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,

    // MLA
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    pub kv_lora_rank: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,

    // MoE
    pub n_routed_experts: i32,
    pub num_experts_per_tok: i32,
    pub n_shared_experts: i32,
    #[serde(default = "default_first_k_dense")]
    pub first_k_dense_replace: i32,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: i32,
    #[serde(default = "default_routed_scaling")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub scoring_func: Option<String>,
    #[serde(default)]
    pub topk_method: Option<String>,

    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    /// Folded from the outer config scope by [`get_deepseek_v2_model_args`].
    #[serde(skip)]
    pub eos_token_ids: Vec<u32>,
    #[serde(skip)]
    pub bos_token_id: Option<u32>,
}

fn default_first_k_dense() -> i32 {
    0
}
fn default_moe_layer_freq() -> i32 {
    1
}
fn default_routed_scaling() -> f32 {
    1.0
}
fn default_tie() -> bool {
    false
}

impl ModelArgs {
    /// Total per-head query/key dim before V (nope + decoupled rope).
    pub fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Whether layer `idx` is a MoE layer (vs a dense MLP layer).
    pub fn is_moe_layer(&self, idx: i32) -> bool {
        self.n_routed_experts > 0
            && idx >= self.first_k_dense_replace
            && (self.moe_layer_freq <= 0 || idx % self.moe_layer_freq == 0)
    }

    /// Attention softmax scale: `q_head_dim^-0.5`, with the YaRN `mscale`
    /// folded in **squared** when rope scaling is active (DeepSeek reference).
    pub fn softmax_scale(&self) -> f32 {
        let base = (self.q_head_dim() as f32).powf(-0.5);
        match &self.rope_scaling {
            Some(rs) if rs.factor > 1.0 && rs.mscale_all_dim != 0.0 => {
                let m = yarn_get_mscale(rs.factor, rs.mscale_all_dim);
                base * m * m
            }
            _ => base,
        }
    }
}

// -----------------------------------------------------------------------------
// Multi-head Latent Attention (naive form)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mla {
    pub n_heads: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub q_head_dim: i32,
    pub v_head_dim: i32,
    pub kv_lora_rank: i32,
    pub scale: f32,
    pub rope_theta: f32,
    pub mscale: f32,
    pub sliding_window: i32,
    /// Precomputed YaRN frequencies for the decoupled rope dims.
    pub yarn_freqs: Option<Array>,

    /// q_lora_rank is null for Lite → a single dense q projection.
    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub kv_a_proj_with_mqa: MaybeQuantized<nn::Linear>,
    #[param]
    pub kv_a_layernorm: nn::RmsNorm,
    #[quantizable]
    #[param]
    pub kv_b_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
}

impl Mla {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let q_head_dim = args.q_head_dim();
        let mk = |i: i32, o: i32| nn::LinearBuilder::new(i, o).bias(false).build();

        // Lite: q_lora_rank == None → dense q_proj. (The q_a/q_b low-rank path
        // for the larger DeepSeek-V2 checkpoints is not needed here.)
        let q_proj = mk(dim, n_heads * q_head_dim)?;
        let kv_a_proj_with_mqa = mk(dim, args.kv_lora_rank + args.qk_rope_head_dim)?;
        let kv_a_layernorm = nn::RmsNormBuilder::new(args.kv_lora_rank)
            .eps(args.rms_norm_eps)
            .build()?;
        let kv_b_proj = mk(
            args.kv_lora_rank,
            n_heads * (args.qk_nope_head_dim + args.v_head_dim),
        )?;
        let o_proj = mk(n_heads * args.v_head_dim, dim)?;

        let yarn_freqs = args.rope_scaling.as_ref().map(|rs| {
            compute_yarn_freqs(
                args.qk_rope_head_dim,
                args.rope_theta,
                rs.factor,
                rs.original_max_position_embeddings,
                rs.beta_fast,
                rs.beta_slow,
            )
        });

        Ok(Self {
            n_heads,
            qk_nope_head_dim: args.qk_nope_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim,
            q_head_dim,
            v_head_dim: args.v_head_dim,
            kv_lora_rank: args.kv_lora_rank,
            scale: args.softmax_scale(),
            rope_theta: args.rope_theta,
            // Rope cos/sin mscale: ratio of mscale to mscale_all_dim. For
            // Coder-V2-Lite both are 0.707 → ratio 1.0 (no pre-scaling).
            mscale: 1.0,
            sliding_window: 0,
            yarn_freqs,
            q_proj: MaybeQuantized::Original(q_proj),
            kv_a_proj_with_mqa: MaybeQuantized::Original(kv_a_proj_with_mqa),
            kv_a_layernorm,
            kv_b_proj: MaybeQuantized::Original(kv_b_proj),
            o_proj: MaybeQuantized::Original(o_proj),
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // Queries: [B, L, H, q_head_dim] -> [B, H, L, q_head_dim].
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, self.q_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q_nope = q.index((.., .., .., 0..self.qk_nope_head_dim));
        let q_pe = q.index((.., .., .., self.qk_nope_head_dim..self.q_head_dim));

        // Compressed KV + decoupled rope key.
        let c = self.kv_a_proj_with_mqa.forward(x)?;
        let compressed_kv = c.index((.., .., 0..self.kv_lora_rank));
        let k_pe = c
            .index((
                ..,
                ..,
                self.kv_lora_rank..(self.kv_lora_rank + self.qk_rope_head_dim),
            ))
            .reshape(&[B, L, 1, self.qk_rope_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B, 1, L, rope]

        // Decompress to per-head k_nope + values.
        let kv = self
            .kv_b_proj
            .forward(&self.kv_a_layernorm.forward(&compressed_kv)?)?
            .reshape(&[B, L, self.n_heads, self.qk_nope_head_dim + self.v_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B, H, L, nope+v]
        let k_nope = kv.index((.., .., .., 0..self.qk_nope_head_dim));
        let values = kv.index((
            ..,
            ..,
            ..,
            self.qk_nope_head_dim..(self.qk_nope_head_dim + self.v_head_dim),
        ));

        // YaRN RoPE on the decoupled dims only.
        let q_pe = apply_yarn_rope(
            &q_pe,
            self.qk_rope_head_dim,
            self.rope_theta,
            self.yarn_freqs.as_ref(),
            self.mscale,
            rope_offset,
            true,
        )?;
        let k_pe = apply_yarn_rope(
            &k_pe,
            self.qk_rope_head_dim,
            self.rope_theta,
            self.yarn_freqs.as_ref(),
            self.mscale,
            rope_offset,
            true,
        )?;

        // Assemble full per-head queries/keys; k_pe is shared across heads.
        let k_pe_b = broadcast_to(&k_pe, &[B, self.n_heads, L, self.qk_rope_head_dim])?;
        let queries = concatenate_axis(&[q_nope, q_pe], -1)?; // [B, H, L, q_head_dim]
        let keys = concatenate_axis(&[k_nope, k_pe_b], -1)?; // [B, H, L, q_head_dim]

        // Cache the expanded K (head-dim q_head_dim) and V (head-dim v_head_dim);
        // SteppingKeyValueCache supports the asymmetry.
        let (k_full, v_full) = if let Some(cache) = cache {
            match cache.update_and_fetch(keys, values)? {
                KvFetchResult::Fp16(k, v) => (k, v),
                KvFetchResult::TurboQuant => {
                    return Err(Exception::custom(
                        "DeepSeek-V2 KV cache must be FP16 (TurboQuant not wired)",
                    ));
                }
            }
        } else {
            (keys, values)
        };

        let out = scaled_dot_product_attention(
            queries,
            k_full,
            v_full,
            None::<&mut KvCache>,
            self.scale,
            mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, self.n_heads * self.v_head_dim])?;

        self.o_proj.forward(&out)
    }
}

// -----------------------------------------------------------------------------
// Dense MLP (SwiGLU) — dense layers + the shared-expert path
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
        let gated = nn::silu(self.gate_proj.forward(x)?)?.multiply(self.up_proj.forward(x)?)?;
        self.down_proj.forward(&gated)
    }
}

// -----------------------------------------------------------------------------
// Stacked quantized experts (SwitchGLU via gather_qmm)
// -----------------------------------------------------------------------------

/// One stacked quantized expert projection: `weight [E, out, in_packed]`,
/// `scales`/`biases [E, out, in/group_size]`. Loaded verbatim from the
/// `switch_mlp.*` tensors (already quantized + stacked on disk) — NOT quantized
/// at runtime, so these are plain `#[param]` arrays, not `MaybeQuantized`.
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
        // Real shapes are filled by the weight loader (`**slot = value`); start
        // with 1-element placeholders so the param keys exist for matching.
        let z = || zeros_dtype(&[1], Dtype::Float32);
        Ok(Self {
            group_size,
            bits,
            weight: Param::new(z()?),
            scales: Param::new(z()?),
            biases: Param::new(z()?),
        })
    }

    /// `x [..., 1, in]` × selected experts `indices [..., K]` → `[..., K, 1, out]`.
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
        // [B, L, 1, 1, hidden] so batch dims broadcast against indices [B, L, K].
        let x_e = x.reshape(&[B, L, 1, 1, H])?;
        let gate = nn::silu(self.gate_proj.forward(&x_e, indices)?)?;
        let up = self.up_proj.forward(&x_e, indices)?;
        let h = gate.multiply(&up)?; // [B, L, K, 1, moe_inter]
        let out = self.down_proj.forward(&h, indices)?; // [B, L, K, 1, hidden]
        let os = out.shape();
        out.reshape(&[os[0], os[1], os[2], os[4]]) // squeeze the M=1 dim
    }
}

// -----------------------------------------------------------------------------
// DeepSeekMoE block
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DeepseekMoe {
    pub num_experts_per_tok: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,

    /// Router — UNQUANTIZED in the checkpoint (`mlp.gate.weight` only), so a
    /// plain `nn::Linear`, never wrapped in `MaybeQuantized`.
    #[param]
    pub gate: nn::Linear,
    #[param]
    pub switch_mlp: SwitchGlu,
    #[quantizable]
    #[param]
    pub shared_experts: Mlp,
}

impl DeepseekMoe {
    pub fn new(args: &ModelArgs, group_size: i32, bits: i32) -> Result<Self, Exception> {
        let gate = nn::LinearBuilder::new(args.hidden_size, args.n_routed_experts)
            .bias(false)
            .build()?;
        let switch_mlp = SwitchGlu::new(group_size, bits)?;
        let shared_inter = args.moe_intermediate_size * args.n_shared_experts;
        let shared_experts = Mlp::new(args.hidden_size, shared_inter)?;
        Ok(Self {
            num_experts_per_tok: args.num_experts_per_tok,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
            gate,
            switch_mlp,
            shared_experts,
        })
    }

    #[allow(non_snake_case)]
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let s = x.shape();
        let (B, L) = (s[0], s[1]);
        let k = self.num_experts_per_tok;

        // Router: softmax over experts, then (greedy) top-k. Coder-V2-Lite uses
        // scoring_func=softmax, topk_method=greedy, n_group=1 → no group mask.
        let logits = self.gate.forward(x)?; // [B, L, E]
        let scores = softmax_axis(&logits, -1, Some(true))?;

        // Top-k experts via argpartition on -scores (order within the k is
        // irrelevant — the combine is a weighted sum).
        let neg = scores.multiply(&array!(-1.0_f32))?;
        let part = argpartition_axis(&neg, k, -1)?;
        let inds = part.index((.., .., 0..k)); // [B, L, K]
        let mut weights = scores.take_along_axis(&inds, -1)?; // [B, L, K]
        if self.norm_topk_prob {
            let denom = weights.sum_axes(&[-1], true)?.add(&array!(1e-20_f32))?;
            weights = weights.divide(&denom)?;
        }
        if (self.routed_scaling_factor - 1.0).abs() > f32::EPSILON {
            weights = weights.multiply(&array!(self.routed_scaling_factor))?;
        }

        // Routed experts: [B, L, K, hidden] * weights[..., None] summed over K.
        let expert_out = self.switch_mlp.forward(x, &inds)?;
        let weights = weights.reshape(&[B, L, k, 1])?;
        let routed = expert_out.multiply(&weights)?.sum_axes(&[-2], false)?;

        // Always-on shared experts.
        let shared = self.shared_experts.forward(x)?;
        routed.add(&shared)
    }
}

// -----------------------------------------------------------------------------
// Decoder layer
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    #[quantizable]
    #[param]
    pub self_attn: Mla,
    /// Exactly one of `mlp` / `moe` is `Some`, chosen by `first_k_dense_replace`.
    #[quantizable]
    #[param]
    pub mlp: Option<Mlp>,
    #[quantizable]
    #[param]
    pub moe: Option<DeepseekMoe>,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl DecoderLayer {
    pub fn new(args: &ModelArgs, idx: i32, group_size: i32, bits: i32) -> Result<Self, Exception> {
        let self_attn = Mla::new(args)?;
        let (mlp, moe) = if args.is_moe_layer(idx) {
            (None, Some(DeepseekMoe::new(args, group_size, bits)?))
        } else {
            (
                Some(Mlp::new(args.hidden_size, args.intermediate_size)?),
                None,
            )
        };
        let mk_norm = || {
            nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()
        };
        Ok(Self {
            self_attn,
            mlp,
            moe,
            input_layernorm: mk_norm()?,
            post_attention_layernorm: mk_norm()?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KvCache>,
        rope_offset: i32,
    ) -> Result<Array, Exception> {
        let r =
            self.self_attn
                .forward(&self.input_layernorm.forward(x)?, mask, cache, rope_offset)?;
        let h = x.add(&r)?;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let r = match (self.moe.as_mut(), self.mlp.as_mut()) {
            (Some(moe), _) => moe.forward(&normed)?,
            (None, Some(mlp)) => mlp.forward(&normed)?,
            (None, None) => return Err(Exception::custom("layer has neither mlp nor moe")),
        };
        h.add(&r)
    }
}

// -----------------------------------------------------------------------------
// Backbone + top-level model
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DeepseekV2Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl DeepseekV2Model {
    pub fn new(args: &ModelArgs, group_size: i32, bits: i32) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i, group_size, bits))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            args: args.clone(),
            embed_tokens: MaybeQuantized::Original(embed_tokens),
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

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut [Option<KvCache>],
        rope_offset: usize,
    ) -> Result<Array, Exception> {
        let rope_off = i32::try_from(rope_offset)
            .map_err(|_| Exception::custom("rope_offset exceeds i32::MAX"))?;
        let mut h = self.embed(inputs)?;

        let seq = h.dim(1);
        let mask = if seq <= 1 {
            None
        } else {
            Some(create_causal_mask(seq, Some(rope_off), None, None)?)
        };

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let cache = caches.get_mut(idx).and_then(|c| c.as_mut());
            h = layer.forward(&h, mask.as_ref(), cache, rope_off)?;
        }
        self.norm.forward(&h)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: DeepseekV2Model,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs, group_size: i32, bits: i32) -> Result<Self, Exception> {
        let model = DeepseekV2Model::new(&args, group_size, bits)?;
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
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// One FP16 KV cache per layer (MLA caches expanded K/V; asymmetric dims are
    /// handled by `SteppingKeyValueCache`). TurboQuant is not wired for MLA.
    pub fn make_caches(&self, max_kv_tokens: i32) -> Vec<Option<KvCache>> {
        let cap = max_kv_tokens.max(1);
        (0..self.args.num_hidden_layers)
            .map(|_| Some(KvCache::fp16_with_max(cap)))
            .collect()
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut [Option<KvCache>],
        rope_offset: usize,
    ) -> Result<Array, Exception> {
        let out = self.model.forward(inputs, caches, rope_offset)?;
        match self.lm_head.as_mut() {
            Some(lm) => lm.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&out),
                MaybeQuantized::Quantized(q) => q.as_linear(&out),
            },
        }
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

pub fn get_deepseek_v2_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let path = model_dir.as_ref().join("config.json");
    let raw = std::fs::read_to_string(&path)?;
    let root: Value = serde_json::from_str(&raw)?;
    let mut args: ModelArgs = serde_json::from_value(root.clone())?;

    args.eos_token_ids = match root.get("eos_token_id") {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => vec![100_001],
    };
    args.bos_token_id = root
        .get("bos_token_id")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .or(Some(100_000));
    Ok(args)
}

// -----------------------------------------------------------------------------
// Chat template integration
// -----------------------------------------------------------------------------

impl crate::engine::chat_template_openai::ChatTemplateModel for Model {
    /// DeepSeek-Coder-V2 emits plain text (no special reasoning/tool markers in
    /// the Lite instruct template) — use the empty marker set.
    fn markers(&self) -> crate::engine::stream_parser::MarkerSet {
        crate::engine::stream_parser::MarkerSet::default()
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
        _tokenizer: &crate::engine::mlx_lm_utils::tokenizer::Tokenizer,
    ) -> Vec<u32> {
        if self.args.eos_token_ids.is_empty() {
            vec![100_001]
        } else {
            self.args.eos_token_ids.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_lite_config(dir: &std::path::Path) {
        let cfg = serde_json::json!({
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_hidden_layers": 27,
            "intermediate_size": 10944,
            "moe_intermediate_size": 1408,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "vocab_size": 102400,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "q_lora_rank": null,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": false,
            "scoring_func": "softmax",
            "topk_method": "greedy",
            "tie_word_embeddings": false,
            "eos_token_id": 100001,
            "bos_token_id": 100000,
            "rope_scaling": {
                "beta_fast": 32, "beta_slow": 1, "factor": 40,
                "mscale": 0.707, "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 4096, "type": "yarn"
            }
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
    }

    /// MLA caches K with head-dim `q_head_dim` (192 = nope 128 + rope 64) and V
    /// with head-dim `v_head_dim` (128). The whole naive-MLA design hinges on
    /// the fast SDPA accepting `qk_dim != v_dim` (output takes the V dim). This
    /// validates that primitive on-device without needing the full model.
    #[test]
    fn sdpa_accepts_asymmetric_kv_head_dims() {
        let (b, h, lq, lk, qk, vd) = (1i32, 2i32, 3i32, 5i32, 192i32, 128i32);
        let q = Array::full::<f32>(&[b, h, lq, qk], array!(0.01_f32)).unwrap();
        let k = Array::full::<f32>(&[b, h, lk, qk], array!(0.01_f32)).unwrap();
        let v = Array::full::<f32>(&[b, h, lk, vd], array!(1.0_f32)).unwrap();
        let out =
            scaled_dot_product_attention(q, k, v, None::<&mut KvCache>, 0.1147, None).unwrap();
        assert_eq!(
            out.shape(),
            &[b, h, lq, vd],
            "MLA SDPA must accept qk_dim != v_dim and output the V head-dim"
        );
    }

    #[test]
    fn parses_config_and_derives() {
        let tmp = std::env::temp_dir().join("deepseek_v2_args_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        write_lite_config(&tmp);

        let a = get_deepseek_v2_model_args(&tmp).unwrap();
        assert_eq!(a.q_head_dim(), 192);
        assert_eq!(a.q_lora_rank, None);
        assert_eq!(a.eos_token_ids, vec![100_001]);
        assert_eq!(a.bos_token_id, Some(100_000));
        // Layer 0 dense, layers 1.. MoE (first_k_dense_replace = 1).
        assert!(!a.is_moe_layer(0));
        assert!(a.is_moe_layer(1));
        assert!(a.is_moe_layer(26));

        // softmax scale = q_head_dim^-0.5 * mscale^2, mscale = yarn(40, 0.707).
        let m = yarn_get_mscale(40.0, 0.707);
        let expect = (192.0_f32).powf(-0.5) * m * m;
        assert!((a.softmax_scale() - expect).abs() < 1e-6);
        // Sanity: ~0.1147 per the DeepSeek reference.
        assert!(
            (a.softmax_scale() - 0.1147).abs() < 1e-3,
            "scale {}",
            a.softmax_scale()
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
