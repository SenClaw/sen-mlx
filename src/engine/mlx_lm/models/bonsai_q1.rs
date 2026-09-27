//! Bonsai-Q1 target-capable engine: packed 1.25-bpw weight storage.
//!
//! Unlike `DiffusionEngine::load_q1` which dequantizes to fp32 at load (32 GB
//! residency on 8B), this engine holds MLX's `Q1_0_g128` affine encoding
//! verbatim: `w[row, col] = scales[row, col/128] * bit(col) + biases[row,
//! col/128]`. Dequant happens inline inside the MLX quantized matmul kernel
//! once upstream MLX provides bits=1 affine support.
//!
//! Residency: ~1.25 GB for Bonsai-8B-mlx-1bit, ~260 MB for Bonsai-1.7B-mlx-1bit.
//!
//! Scope: Rust-side loader and engine implementation. Runtime routing is held
//! back in `higgs-engine` until the upstream MLX dependency supports bits=1
//! affine quantization.

#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    // Quantization math uses small bounded dims (head_dim, GROUP_SIZE=128, vocab) and
    // bit-packed u32→f32 conversions where precision/sign loss is intentional.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    // Dequant kernel + safetensors loader index into manually-bounds-checked slices.
    clippy::indexing_slicing,
    // Decode loop reuses names (q, k, v, t0) across rope/sdpa/o_proj stages by design.
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same,
    // Loader unwraps on safetensors slices after explicit shape validation; load failure
    // paths return ShapeMismatch via map_err elsewhere.
    clippy::unwrap_used,
    clippy::map_unwrap_or,
    // YarnRoPE / Q1 / KV abbreviations are domain terms, not items to backtick.
    clippy::doc_markdown,
    clippy::doc_lazy_continuation,
    clippy::missing_const_for_fn,
    clippy::manual_flatten,
    clippy::if_then_some_else_none,
    clippy::suboptimal_flops,
)]

use half::f16;
use std::path::Path;

use mlx_rs::{error::Exception, fast, ops, Array, Dtype};
use safetensors::SafeTensors;

use super::super::cache::{KeyValueCache, KvCache, KvFetchResult, SteppingKeyValueCache};
use super::super::error::Error;
use super::super::utils::{
    create_attention_mask, scaled_dot_product_attention,
    yarn::{apply_yarn_rope, compute_yarn_freqs, yarn_get_mscale},
    AttentionMask,
};

/// Loads packed Q1 checkpoint and uploads to MLX device memory.
pub fn load_bonsai_q1_model<P: AsRef<Path>>(model_dir: P) -> Result<BonsaiQ1Gpu, Error> {
    let engine = BonsaiQ1Engine::load(model_dir)
        .map_err(|s| Error::Other(format!("bonsai-q1: {s}").into()))?;
    engine.to_gpu().map_err(Into::into)
}

/// Inference handle + special tokens for MLX native routing.
pub struct LoadedBonsaiQ1 {
    pub gpu: BonsaiQ1Gpu,
    pub eos_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
}

impl std::fmt::Debug for LoadedBonsaiQ1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedBonsaiQ1")
            .field("layers", &self.gpu.layers.len())
            .field("eos_token_ids", &self.eos_token_ids)
            .field("bos_token_id", &self.bos_token_id)
            .finish_non_exhaustive()
    }
}

pub fn load_bonsai_q1_bundle(model_dir: &Path) -> Result<LoadedBonsaiQ1, Error> {
    let gpu = load_bonsai_q1_model(model_dir)?;
    let eos = eos_ids_from_hf_dir(model_dir)?;
    let bos = bos_token_id_from_hf_dir(model_dir);
    Ok(LoadedBonsaiQ1 {
        gpu,
        eos_token_ids: eos,
        bos_token_id: bos,
    })
}

fn bos_token_id_from_hf_dir(model_dir: &Path) -> Option<u32> {
    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    v.get("bos_token_id")
        .and_then(|x| x.as_u64())
        .map(|x| x as u32)
}

fn eos_ids_from_hf_dir(model_dir: &Path) -> Result<Vec<u32>, Error> {
    let cfg_path = model_dir.join("config.json");
    let tok_path = model_dir.join("tokenizer_config.json");
    let mut out = Vec::new();
    if let Ok(raw) = std::fs::read_to_string(&cfg_path) {
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        match v.get("eos_token_id").cloned() {
            Some(serde_json::Value::Number(n)) => {
                if let Some(id) = n.as_u64() {
                    out.push(id as u32);
                }
            }
            Some(serde_json::Value::Array(arr)) => {
                out.extend(arr.iter().filter_map(|x| x.as_u64().map(|x| x as u32)));
            }
            _ => {}
        }
    }
    if out.is_empty() {
        if let Ok(raw) = std::fs::read_to_string(&tok_path) {
            let v: serde_json::Value = serde_json::from_str(&raw)?;
            if let Some(id) = v.get("eos_token_id").and_then(|x| x.as_u64()) {
                out.push(id as u32);
            }
        }
    }
    Ok(out)
}

pub const GROUP_SIZE: usize = 128;
const BITS: i32 = 1;
const GROUP_SIZE_I32: i32 = GROUP_SIZE as i32;

/// Packed 1-bit linear layer with affine per-group dequant.
///
/// Layout (matches MLX 1-bit `QuantizedLinear`, `PrismML` fork):
///   - `w_packed`: `[out_features, in_features/32]` u32, bit `col%32` of word
///     `col/32` is the raw 1-bit weight for column `col`.
///   - `scales`, `biases`: `[out_features, in_features/128]` f16, one per group
///     of 128 input columns.
///
/// Effective: 1 bit/weight + 32 bits/group / 128 weights = **1.25 bpw**.
pub struct PackedQ1Linear {
    pub w_packed: Vec<u32>,
    pub scales: Vec<f16>,
    pub biases: Vec<f16>,
    pub out_features: usize,
    pub in_features: usize,
}

impl PackedQ1Linear {
    pub const fn resident_bytes(&self) -> usize {
        self.w_packed.len() * 4 + self.scales.len() * 2 + self.biases.len() * 2
    }

    /// Dequantize a single row to fp32 (reference path for correctness tests).
    ///
    /// Not used on the hot path — P2 replaces this with a Metal kernel that
    /// fuses dequant into the matmul.
    pub fn dequant_row_to_fp32(&self, row: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.in_features);
        let n_groups = self.in_features / GROUP_SIZE;
        let packed_cols = self.in_features / 32;
        let w_row = &self.w_packed[row * packed_cols..(row + 1) * packed_cols];
        let s_row = &self.scales[row * n_groups..(row + 1) * n_groups];
        let b_row = &self.biases[row * n_groups..(row + 1) * n_groups];
        for col in 0..self.in_features {
            let word = w_row[col / 32];
            let bit = ((word >> (col % 32)) & 1) as f32;
            let group = col / GROUP_SIZE;
            out[col] = s_row[group].to_f32().mul_add(bit, b_row[group].to_f32());
        }
    }
}

pub struct BonsaiQ1LayerWeights {
    pub q_proj: PackedQ1Linear,
    pub k_proj: PackedQ1Linear,
    pub v_proj: PackedQ1Linear,
    pub o_proj: PackedQ1Linear,
    pub gate_proj: PackedQ1Linear,
    pub up_proj: PackedQ1Linear,
    pub down_proj: PackedQ1Linear,
    pub q_norm: Vec<f16>,
    pub k_norm: Vec<f16>,
    pub input_norm: Vec<f16>,
    pub post_attn_norm: Vec<f16>,
}

impl BonsaiQ1LayerWeights {
    pub fn resident_bytes(&self) -> usize {
        self.q_proj.resident_bytes()
            + self.k_proj.resident_bytes()
            + self.v_proj.resident_bytes()
            + self.o_proj.resident_bytes()
            + self.gate_proj.resident_bytes()
            + self.up_proj.resident_bytes()
            + self.down_proj.resident_bytes()
            + (self.q_norm.len()
                + self.k_norm.len()
                + self.input_norm.len()
                + self.post_attn_norm.len())
                * 2
    }
}

#[derive(Debug, Clone)]
pub struct BonsaiQ1Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// YARN scaling factor if present (Bonsai-8B uses `factor=4.0, original=16384`).
    pub rope_yarn_factor: Option<f64>,
    pub rope_original_max_seq: Option<usize>,
    pub tie_word_embeddings: bool,
}

pub struct BonsaiQ1Engine {
    pub config: BonsaiQ1Config,
    pub layers: Vec<BonsaiQ1LayerWeights>,
    /// Token embedding stored packed (dequants inline at embed lookup time).
    pub embed: PackedQ1Linear,
    /// Untied LM head for 8B (`tie_word_embeddings: false`). None for 1.7B.
    pub lm_head: Option<PackedQ1Linear>,
    pub final_norm: Vec<f16>,
}

impl BonsaiQ1Engine {
    pub const fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn resident_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(BonsaiQ1LayerWeights::resident_bytes)
            .sum();
        let lm_head_bytes = self
            .lm_head
            .as_ref()
            .map_or(0, PackedQ1Linear::resident_bytes);
        layer_bytes + self.embed.resident_bytes() + lm_head_bytes + self.final_norm.len() * 2
    }

    /// Load from a `HuggingFace` directory containing `config.json` +
    /// `model.safetensors` in MLX 1-bit affine-quant format.
    #[allow(clippy::too_many_lines)]
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let dir = model_dir.as_ref();

        let cfg_txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&cfg_txt).map_err(|e| format!("config.json parse: {e}"))?;

        let u64_of = |k: &str| -> Result<u64, String> {
            cfg[k]
                .as_u64()
                .ok_or_else(|| format!("config.json missing u64 '{k}'"))
        };
        let hidden = u64_of("hidden_size")? as usize;
        let heads = u64_of("num_attention_heads")? as usize;
        let kv_heads = u64_of("num_key_value_heads")? as usize;
        let head_dim = cfg["head_dim"].as_u64().map_or(128, |v| v as usize);
        let inter = u64_of("intermediate_size")? as usize;
        let layers_n = u64_of("num_hidden_layers")? as usize;
        let vocab = u64_of("vocab_size")? as usize;

        let rms_norm_eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap_or(1_000_000.0);
        let tie_word_embeddings = cfg["tie_word_embeddings"].as_bool().unwrap_or(false);

        let (rope_yarn_factor, rope_original_max_seq) = cfg
            .get("rope_scaling")
            .and_then(|rs| {
                (rs.get("rope_type").and_then(|v| v.as_str()) == Some("yarn")).then(|| {
                    let f = rs.get("factor").and_then(serde_json::Value::as_f64);
                    let o = rs
                        .get("original_max_position_embeddings")
                        .and_then(serde_json::Value::as_u64)
                        .map(|v| v as usize);
                    (f, o)
                })
            })
            .unwrap_or((None, None));

        let quant = cfg
            .get("quantization")
            .ok_or("missing quantization block")?;
        let q_bits = quant.get("bits").and_then(serde_json::Value::as_u64);
        let q_group = quant.get("group_size").and_then(serde_json::Value::as_u64);
        if q_bits != Some(1) || q_group != Some(GROUP_SIZE as u64) {
            return Err(format!(
                "expected quantization {{bits:1, group_size:{GROUP_SIZE}}}, got bits={q_bits:?} \
                 group_size={q_group:?}"
            ));
        }

        let st_path = dir.join("model.safetensors");
        let st_data = std::fs::read(&st_path).map_err(|e| format!("read safetensors: {e}"))?;
        let tensors = SafeTensors::deserialize(&st_data)
            .map_err(|e| format!("deserialize safetensors: {e}"))?;

        let config = BonsaiQ1Config {
            hidden,
            layers: layers_n,
            heads,
            kv_heads,
            head_dim,
            inter,
            vocab,
            rms_norm_eps,
            rope_theta,
            rope_yarn_factor,
            rope_original_max_seq,
            tie_word_embeddings,
        };

        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let embed = load_packed(
            &tensors,
            "model.embed_tokens",
            vocab,
            hidden,
            "embed_tokens",
        )?;
        let lm_head = if tie_word_embeddings {
            None
        } else {
            Some(load_packed(&tensors, "lm_head", vocab, hidden, "lm_head")?)
        };
        let final_norm = load_f16(&tensors, "model.norm.weight")?;
        if final_norm.len() != hidden {
            return Err(format!(
                "final_norm len {} != hidden {hidden}",
                final_norm.len()
            ));
        }

        let mut layers = Vec::with_capacity(layers_n);
        for i in 0..layers_n {
            let p = format!("model.layers.{i}");
            let attn = format!("{p}.self_attn");
            let mlp = format!("{p}.mlp");

            let layer = BonsaiQ1LayerWeights {
                q_proj: load_packed(&tensors, &format!("{attn}.q_proj"), q_dim, hidden, "q_proj")?,
                k_proj: load_packed(
                    &tensors,
                    &format!("{attn}.k_proj"),
                    kv_dim,
                    hidden,
                    "k_proj",
                )?,
                v_proj: load_packed(
                    &tensors,
                    &format!("{attn}.v_proj"),
                    kv_dim,
                    hidden,
                    "v_proj",
                )?,
                o_proj: load_packed(&tensors, &format!("{attn}.o_proj"), hidden, q_dim, "o_proj")?,
                gate_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.gate_proj"),
                    inter,
                    hidden,
                    "gate_proj",
                )?,
                up_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.up_proj"),
                    inter,
                    hidden,
                    "up_proj",
                )?,
                down_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.down_proj"),
                    hidden,
                    inter,
                    "down_proj",
                )?,
                q_norm: load_f16(&tensors, &format!("{attn}.q_norm.weight"))?,
                k_norm: load_f16(&tensors, &format!("{attn}.k_norm.weight"))?,
                input_norm: load_f16(&tensors, &format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: load_f16(
                    &tensors,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
            };
            layers.push(layer);
        }

        let engine = Self {
            config,
            layers,
            embed,
            lm_head,
            final_norm,
        };
        let resident_mb = engine.resident_bytes() as f64 / (1024.0 * 1024.0);
        tracing::info!(
            layers = engine.config.layers,
            hidden = engine.config.hidden,
            heads = engine.config.heads,
            kv_heads = engine.config.kv_heads,
            head_dim = engine.config.head_dim,
            inter = engine.config.inter,
            vocab = engine.config.vocab,
            tied_embed = engine.config.tie_word_embeddings,
            packed_resident_mb = format!("{resident_mb:.1}"),
            "BonsaiQ1Engine::load",
        );
        Ok(engine)
    }
}

// ---------------------------------------------------------------------------
// GPU-ready mirror — built once from the packed engine.
// ---------------------------------------------------------------------------

/// MLX-resident 1-bit linear: weight as uint32 packed, scales/biases as f16,
/// same shape as `PackedQ1Linear` but ready for `ops::quantized_matmul`.
pub struct BonsaiQ1GpuLinear {
    pub w: Array,
    pub scales: Array,
    pub biases: Array,
    pub out_features: i32,
    pub in_features: i32,
}

impl BonsaiQ1GpuLinear {
    fn from_packed(p: &PackedQ1Linear) -> Result<Self, Exception> {
        let out = i32::try_from(p.out_features)
            .map_err(|_| Exception::custom("out_features overflows i32"))?;
        let inf = i32::try_from(p.in_features)
            .map_err(|_| Exception::custom("in_features overflows i32"))?;
        let packed_cols = inf / 32;
        let n_groups = inf / GROUP_SIZE_I32;

        let w = Array::from_slice(&p.w_packed, &[out, packed_cols]);
        let scales_f32: Vec<f32> = p.scales.iter().map(|h| h.to_f32()).collect();
        let biases_f32: Vec<f32> = p.biases.iter().map(|h| h.to_f32()).collect();
        let scales = Array::from_slice(&scales_f32, &[out, n_groups]).as_dtype(Dtype::Float16)?;
        let biases = Array::from_slice(&biases_f32, &[out, n_groups]).as_dtype(Dtype::Float16)?;

        Ok(Self {
            w,
            scales,
            biases,
            out_features: out,
            in_features: inf,
        })
    }

    /// `y = x @ dequant(w, scales, biases).T` via fused bits=1 qmm.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        ops::quantized_matmul(
            x,
            &self.w,
            &self.scales,
            &self.biases,
            true,
            GROUP_SIZE_I32,
            BITS,
        )
    }
}

pub struct BonsaiQ1GpuLayer {
    pub q_proj: BonsaiQ1GpuLinear,
    pub k_proj: BonsaiQ1GpuLinear,
    pub v_proj: BonsaiQ1GpuLinear,
    pub o_proj: BonsaiQ1GpuLinear,
    pub gate_proj: BonsaiQ1GpuLinear,
    pub up_proj: BonsaiQ1GpuLinear,
    pub down_proj: BonsaiQ1GpuLinear,
    pub q_norm: Array,
    pub k_norm: Array,
    pub input_norm: Array,
    pub post_attn_norm: Array,
}

pub struct BonsaiQ1Gpu {
    pub config: BonsaiQ1Config,
    pub layers: Vec<BonsaiQ1GpuLayer>,
    pub embed: BonsaiQ1GpuLinear,
    pub lm_head: Option<BonsaiQ1GpuLinear>,
    pub final_norm: Array,
    /// YARN-scaled `RoPE` frequencies (per `head_dim/2`). None if no YARN.
    pub yarn_freqs: Option<Array>,
    pub yarn_mscale: f32,
    pub attention_scale: f32,
}

fn f16_vec_to_array(weights: &[f16]) -> Result<Array, Exception> {
    let f32s: Vec<f32> = weights.iter().map(|h| h.to_f32()).collect();
    let len =
        i32::try_from(weights.len()).map_err(|_| Exception::custom("norm len overflows i32"))?;
    Array::from_slice(&f32s, &[len]).as_dtype(Dtype::Float16)
}

impl BonsaiQ1Engine {
    /// Consume the packed engine and materialize MLX arrays.
    ///
    /// Frees the `Vec<u32>` / `Vec<f16>` residency once copied to MLX.
    pub fn to_gpu(self) -> Result<BonsaiQ1Gpu, Exception> {
        let mut gpu_layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            gpu_layers.push(BonsaiQ1GpuLayer {
                q_proj: BonsaiQ1GpuLinear::from_packed(&layer.q_proj)?,
                k_proj: BonsaiQ1GpuLinear::from_packed(&layer.k_proj)?,
                v_proj: BonsaiQ1GpuLinear::from_packed(&layer.v_proj)?,
                o_proj: BonsaiQ1GpuLinear::from_packed(&layer.o_proj)?,
                gate_proj: BonsaiQ1GpuLinear::from_packed(&layer.gate_proj)?,
                up_proj: BonsaiQ1GpuLinear::from_packed(&layer.up_proj)?,
                down_proj: BonsaiQ1GpuLinear::from_packed(&layer.down_proj)?,
                q_norm: f16_vec_to_array(&layer.q_norm)?,
                k_norm: f16_vec_to_array(&layer.k_norm)?,
                input_norm: f16_vec_to_array(&layer.input_norm)?,
                post_attn_norm: f16_vec_to_array(&layer.post_attn_norm)?,
            });
        }

        let embed = BonsaiQ1GpuLinear::from_packed(&self.embed)?;
        let lm_head = self
            .lm_head
            .as_ref()
            .map(BonsaiQ1GpuLinear::from_packed)
            .transpose()?;
        let final_norm = f16_vec_to_array(&self.final_norm)?;

        // YARN precompute.
        let head_dim_i = i32::try_from(self.config.head_dim)
            .map_err(|_| Exception::custom("head_dim overflows i32"))?;
        let base = self.config.rope_theta as f32;
        let (yarn_freqs, yarn_mscale) = match self.config.rope_yarn_factor {
            Some(factor) if factor > 1.0 => {
                let orig_seq = self.config.rope_original_max_seq.ok_or_else(|| {
                    Exception::custom(
                        "rope_yarn_factor > 1.0 requires \
                         rope_scaling.original_max_position_embeddings",
                    )
                })?;
                let orig = i32::try_from(orig_seq)
                    .map_err(|_| Exception::custom("orig_max_seq overflows i32"))?;
                let factor_f = factor as f32;
                let freqs = compute_yarn_freqs(head_dim_i, base, factor_f, orig, 32.0, 1.0);
                (Some(freqs), yarn_get_mscale(factor_f, 1.0))
            }
            _ => (None, 1.0),
        };

        let head_dim_f = head_dim_i as f32;
        let attention_scale = head_dim_f.sqrt().recip();

        Ok(BonsaiQ1Gpu {
            config: self.config,
            layers: gpu_layers,
            embed,
            lm_head,
            final_norm,
            yarn_freqs,
            yarn_mscale,
            attention_scale,
        })
    }
}

#[inline]
fn kv_fp16_mut<'a>(
    slot: Option<&'a mut KvCache>,
) -> Result<&'a mut SteppingKeyValueCache, Exception> {
    match slot {
        Some(KvCache::Fp16(kv)) => Ok(kv),
        Some(_) => Err(Exception::custom(
            "bonsai-q1: expected FP16 KV cache — disable TurboQuant (kv_cache_bits) for this model",
        )),
        None => Err(Exception::custom("bonsai-q1: missing KV slot")),
    }
}

impl BonsaiQ1Gpu {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Gather embedding rows for a token-ID tensor.
    ///
    /// Uses MLX dequantize after gathering the selected packed rows. This path
    /// requires bits=1 affine support in the active MLX runtime.
    fn embed_rows(&self, ids: &Array) -> Result<Array, Exception> {
        let shape = ids.shape().to_vec();
        let flat = ids.flatten(None, None)?;
        let w = self.embed.w.take_axis(&flat, 0)?;
        let s = self.embed.scales.take_axis(&flat, 0)?;
        let b = self.embed.biases.take_axis(&flat, 0)?;
        let out = ops::dequantize(&w, &s, &b, GROUP_SIZE_I32, BITS)?;
        let mut ret_shape: Vec<i32> = shape;
        ret_shape.push(-1);
        out.reshape(&ret_shape)
    }

    fn apply_rope(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let head_dim = i32::try_from(self.config.head_dim)
            .map_err(|_| Exception::custom("head_dim overflows i32"))?;
        apply_yarn_rope(
            x,
            head_dim,
            self.config.rope_theta as f32,
            self.yarn_freqs.as_ref(),
            self.yarn_mscale,
            offset,
            false, // Qwen3 layout
        )
    }

    fn forward_trunk(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<KvCache>>,
        rope_off: usize,
    ) -> Result<Array, Exception> {
        forward_trunk_free(self, cache, inputs, rope_off)
    }

    pub fn forward_all_logits_native(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<KvCache>>,
        rope_off: usize,
    ) -> Result<Array, Exception> {
        let h = self.forward_trunk(inputs, cache, rope_off)?;
        self.project_logits(&h)
    }

    fn project_logits(&self, h: &Array) -> Result<Array, Exception> {
        let logits = match &self.lm_head {
            Some(head) => head.forward(h)?,
            None => self.embed.forward(h)?,
        };
        logits.as_dtype(Dtype::Float32)
    }
}

/// Decoder trunk `[B,T,hidden]` (final RMSNorm).
#[allow(non_snake_case)]
pub fn forward_trunk_free(
    gpu: &BonsaiQ1Gpu,
    cache: &mut Vec<Option<KvCache>>,
    inputs: &Array,
    rope_off_usize: usize,
) -> Result<Array, Exception> {
    let rope_i = i32::try_from(rope_off_usize)
        .map_err(|_| Exception::custom("bonsai-q1 rope_offset exceeds i32::MAX"))?;

    let shape = inputs.shape();
    let B = *shape
        .first()
        .ok_or_else(|| Exception::custom("bonsai-q1 inputs must have >= 2 dims"))?;
    let T = *shape
        .get(1)
        .ok_or_else(|| Exception::custom("bonsai-q1 inputs must have >= 2 dims"))?;

    if cache.len() != gpu.layers.len() {
        return Err(Exception::custom(format!(
            "bonsai-q1 cache len {} != num_layers {}",
            cache.len(),
            gpu.layers.len()
        )));
    }

    let mut h = gpu.embed_rows(inputs)?;

    let mask = create_attention_mask(&h, cache, rope_off_usize, Some(true))?;

    let heads = i32::try_from(gpu.config.heads)
        .map_err(|_| Exception::custom("bonsai-q1 heads overflows i32"))?;
    let kv_heads = i32::try_from(gpu.config.kv_heads)
        .map_err(|_| Exception::custom("bonsai-q1 kv_heads overflows i32"))?;
    let rms_eps = gpu.config.rms_norm_eps;

    for (layer, layer_slot) in gpu.layers.iter().zip(cache.iter_mut()) {
        let normed = fast::rms_norm(&h, &layer.input_norm, rms_eps)?;

        let q = layer.q_proj.forward(&normed)?;
        let k = layer.k_proj.forward(&normed)?;
        let v = layer.v_proj.forward(&normed)?;

        let q = q
            .reshape(&[B, T, heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[B, T, kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[B, T, kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = fast::rms_norm(&q, &layer.q_norm, rms_eps)?;
        let k = fast::rms_norm(&k, &layer.k_norm, rms_eps)?;

        let q = gpu.apply_rope(&q, rope_i)?;
        let k = gpu.apply_rope(&k, rope_i)?;

        let mask_arr_opt: Option<&Array> = match &mask {
            Some(AttentionMask::Array(a)) => Some(a),
            Some(AttentionMask::Causal) => {
                return Err(Exception::custom(
                    "bonsai-q1: expected dense causal mask (return_array)",
                ));
            }
            None => None,
        };

        let kv_inner = kv_fp16_mut(layer_slot.as_mut())?;
        let fetch = kv_inner.update_and_fetch(k, v)?;
        let attn_out = match fetch {
            KvFetchResult::Fp16(keys, values) => scaled_dot_product_attention(
                q,
                keys,
                values,
                Some(kv_inner),
                gpu.attention_scale,
                mask_arr_opt,
            )?,
            KvFetchResult::TurboQuant => {
                return Err(Exception::custom("bonsai-q1 TurboQuant KV is not wired"));
            }
        };

        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, T, -1])?;
        let attn_out = layer.o_proj.forward(&attn_out)?;
        let h_post_attn = h.add(&attn_out)?;

        let normed_post = fast::rms_norm(&h_post_attn, &layer.post_attn_norm, rms_eps)?;
        let gate = layer.gate_proj.forward(&normed_post)?;
        let up = layer.up_proj.forward(&normed_post)?;
        let mlp_hidden = mlx_rs::nn::silu(&gate)?.multiply(&up)?;
        let mlp_out = layer.down_proj.forward(&mlp_hidden)?;

        h = h_post_attn.add(&mlp_out)?;
    }

    fast::rms_norm(&h, &gpu.final_norm, rms_eps)
}

fn load_packed(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    who: &str,
) -> Result<PackedQ1Linear, String> {
    if in_features % GROUP_SIZE != 0 {
        return Err(format!(
            "{who}: in_features {in_features} not divisible by group_size {GROUP_SIZE}"
        ));
    }
    let packed_cols = in_features / 32;
    let n_groups = in_features / GROUP_SIZE;

    let w_view = tensors
        .tensor(&format!("{prefix}.weight"))
        .map_err(|e| format!("{who}: {prefix}.weight: {e}"))?;
    let s_view = tensors
        .tensor(&format!("{prefix}.scales"))
        .map_err(|e| format!("{who}: {prefix}.scales: {e}"))?;
    let b_view = tensors
        .tensor(&format!("{prefix}.biases"))
        .map_err(|e| format!("{who}: {prefix}.biases: {e}"))?;

    let w_bytes = w_view.data();
    let s_bytes = s_view.data();
    let b_bytes = b_view.data();

    let expected_w_bytes = out_features * packed_cols * 4;
    if w_bytes.len() != expected_w_bytes {
        return Err(format!(
            "{who}: weight byte-size mismatch: got {} expected {}",
            w_bytes.len(),
            expected_w_bytes,
        ));
    }
    let expected_sb_bytes = out_features * n_groups * 2;
    if s_bytes.len() != expected_sb_bytes {
        return Err(format!(
            "{who}: scales byte-size mismatch: got {} expected {}",
            s_bytes.len(),
            expected_sb_bytes,
        ));
    }
    if b_bytes.len() != expected_sb_bytes {
        return Err(format!(
            "{who}: biases byte-size mismatch: got {} expected {}",
            b_bytes.len(),
            expected_sb_bytes,
        ));
    }

    let w_packed: Vec<u32> = w_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let scales = bytes_to_f16_vec(s_bytes);
    let biases = bytes_to_f16_vec(b_bytes);

    Ok(PackedQ1Linear {
        w_packed,
        scales,
        biases,
        out_features,
        in_features,
    })
}

fn load_f16(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f16>, String> {
    let view = tensors.tensor(name).map_err(|e| format!("{name}: {e}"))?;
    Ok(bytes_to_f16_vec(view.data()))
}

fn bytes_to_f16_vec(b: &[u8]) -> Vec<f16> {
    b.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Bonsai‑Q1 chat templates inject `{{ bos_token }}` similarly to Gemma‑2 —
/// decode `bos_token_id` / `eos_token_ids[0]` recovered from `config.json`.
impl crate::engine::chat_template_openai::ChatTemplateModel for LoadedBonsaiQ1 {
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
            self.bos_token_id
                .and_then(|id| tokenizer.decode(std::slice::from_ref(&id), false).ok())
        } else {
            None
        };
        if need_bos {
            match &bos {
                Some(s) if !s.is_empty() => tracing::debug!(
                    "[local-mlx-native] chat_template bos_token injected for Bonsai-Q1 (decoded len={})",
                    s.len()
                ),
                _ => tracing::warn!(
                    "[local-mlx-native] chat_template mentions bos_token — could not decode from bos_token_id={:?}",
                    self.bos_token_id
                ),
            }
        }
        let eos = if need_eos {
            self.eos_token_ids
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
        self.eos_token_ids.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
