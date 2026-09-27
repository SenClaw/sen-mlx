//! Mamba-1 inference for **Falcon-Mamba** checkpoints
//! (e.g. `mlx-community/falcon-mamba-7b-4bit`).
//!
//! Falcon-Mamba is a Mamba-1 (selective SSM) backbone — *not* the head-grouped
//! Mamba-2 in [`super::mamba2`]. The only Falcon-specific tweak is the
//! `use_bcdt_rms` flag: a single-row RMSNorm (γ = ones, ε = 1e-6) is applied
//! to each of `delta`, `B`, `C` after `x_proj` but *before* `softplus(dt_proj)`.
//!
//! Layout follows the official `mlx-lm` reference at
//! [`ml-explore/mlx-lm:mlx_lm/models/mamba.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py).
//! Safetensors keys match the `mlx-community/falcon-mamba-7b-*` checkpoints:
//! `backbone.embeddings`, `backbone.layers.<i>.{mixer,norm}`,
//! `backbone.norm_f`, `lm_head`.
//!
//! ## Algorithmic core
//!
//! For each timestep `t` and each channel `c ∈ [0, d_inner)`:
//!
//! ```text
//! deltaBC = x_proj(x[t])          // [B, dt_rank + 2 * d_state]
//! delta, B, C = split(deltaBC)
//! (delta, B, C) = mixer_rms(.)    // when use_bcdt_rms (Falcon-Mamba)
//! delta = softplus(dt_proj(delta))// [B, d_inner]
//! new_state[c] = (delta[c] * x[t,c]) ⊗ B[t]
//!               + state[c] * exp(delta[c] * A[c])      // A: [d_inner, d_state]
//! y[t,c] = new_state[c] @ C[t]^T  +  D[c] * x[t,c]
//! ```
//!
//! The reference implementation walks `T` sequentially; we keep the same
//! O(L) loop for now (Mamba-1's selective SSM has no published chunked /
//! Metal-kernel form in this build). Prefill and decode share the same path.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    fast,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        concatenate_axis,
        indexing::{IndexOp, NewAxis},
        ones_dtype,
    },
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::super::{
    cache::{KvCache, Mamba1Cache},
    error::Error,
};

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// HuggingFace `config.json` schema for Mamba-1 / Falcon-Mamba.
///
/// Notes vs. the canonical `mlx-lm` Python `ModelArgs`:
/// - `rms_norm_eps` is aliased from `layer_norm_epsilon` (HF's field name; the
///   final norm weights use the same eps as the residual norms).
/// - `time_step_rank` may be the literal string `"auto"` in some checkpoints,
///   which means `ceil(hidden_size / 16)`. Falcon-Mamba ships a concrete int
///   (`256` for the 7B), but we accept either.
/// - `use_bcdt_rms` defaults to `true` when `model_type == "falcon_mamba"` and
///   `false` otherwise (matches `__post_init__` in the reference).
/// - Unrecognised fields (`time_step_*`, `rescale_prenorm_residual`,
///   `quantization*`, etc.) are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub state_size: i32,
    pub num_hidden_layers: i32,
    pub conv_kernel: i32,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_true")]
    pub use_conv_bias: bool,
    /// Either an integer or the string `"auto"`; [`ModelArgs::normalize`] resolves it.
    pub time_step_rank: TimeStepRank,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_bcdt_rms: Option<bool>,
    #[serde(default = "default_mixer_eps", alias = "layer_norm_epsilon")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_mixer_eps")]
    pub mixer_rms_eps: f32,
}

fn default_true() -> bool {
    true
}
fn default_mixer_eps() -> f32 {
    1e-6
}

/// Resolved-at-load `time_step_rank` (`int` in JSON, or `"auto"`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TimeStepRank {
    Int(i32),
    Str(String),
}

impl ModelArgs {
    /// Resolve fields that HF leaves implicit. Call once after deserialisation.
    pub fn normalize(&mut self) {
        // `time_step_rank = "auto"` -> ceil(hidden_size / 16)
        if let TimeStepRank::Str(s) = &self.time_step_rank {
            if s == "auto" {
                let auto = (self.hidden_size + 15) / 16;
                self.time_step_rank = TimeStepRank::Int(auto);
            }
        }
        // Falcon-Mamba: BCdt mixer RMSNorm on by default.
        if self.use_bcdt_rms.is_none() {
            self.use_bcdt_rms = Some(self.model_type == "falcon_mamba");
        }
    }

    pub fn time_step_rank_i32(&self) -> i32 {
        match &self.time_step_rank {
            TimeStepRank::Int(i) => *i,
            // `normalize()` should have replaced "auto"; fall back conservatively.
            TimeStepRank::Str(_) => (self.hidden_size + 15) / 16,
        }
    }

    pub fn use_bcdt_rms_resolved(&self) -> bool {
        self.use_bcdt_rms
            .unwrap_or(self.model_type == "falcon_mamba")
    }
}

// -----------------------------------------------------------------------------
// Mamba-1 mixer block
// -----------------------------------------------------------------------------

/// One Mamba-1 mixer. Mirrors `MambaBlock` in the reference Python.
///
/// Param naming matches HF safetensors keys for `mlx-community/falcon-mamba-*`
/// and `state-spaces/mamba-*`: `in_proj`, `conv1d`, `x_proj`, `dt_proj`,
/// `A_log`, `D`, `out_proj`. Linear projections are wrapped in
/// [`MaybeQuantized`] so 4/8-bit checkpoints round-trip through
/// `mlx_rs::nn::quantize`.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
#[allow(non_snake_case)]
pub struct MambaBlock {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub d_state: i32,
    pub d_conv: i32,
    pub time_step_rank: i32,
    pub use_bcdt_rms: bool,
    pub mixer_rms_eps: f32,

    /// `hidden_size -> 2 * intermediate_size` (split into `x`, `z`).
    #[quantizable]
    #[param]
    pub in_proj: MaybeQuantized<nn::Linear>,

    /// Depthwise short conv over `intermediate_size` channels (causal pad applied
    /// manually so this builder uses `padding = 0`).
    #[param]
    pub conv1d: nn::Conv1d,

    /// `intermediate_size -> time_step_rank + 2 * d_state` (split into `delta`, `B`, `C`).
    #[quantizable]
    #[param]
    pub x_proj: MaybeQuantized<nn::Linear>,

    /// `time_step_rank -> intermediate_size`, bias=true (selective discretisation).
    #[quantizable]
    #[param]
    pub dt_proj: MaybeQuantized<nn::Linear>,

    /// `[intermediate_size, d_state]` — log-parameterised diagonal of `A`. The
    /// block consumes `A = -exp(A_log)`.
    #[param]
    #[allow(non_snake_case)]
    pub A_log: Param<Array>,

    /// `[intermediate_size]` — skip-connection scaling.
    #[param]
    #[allow(non_snake_case)]
    pub D: Param<Array>,

    #[quantizable]
    #[param]
    pub out_proj: MaybeQuantized<nn::Linear>,
}

impl MambaBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dt_rank = args.time_step_rank_i32();

        // Use bf16-zero placeholders — see [`super::mamba2::zero_linear_bf16`]
        // for the RAM-peak rationale. Safetensors load overwrites every slot.
        let in_proj = super::mamba2::zero_linear_bf16(
            args.hidden_size,
            args.intermediate_size * 2,
            args.use_bias,
        )?;
        let conv1d = nn::Conv1dBuilder::new(
            args.intermediate_size,
            args.intermediate_size,
            args.conv_kernel,
        )
        .groups(args.intermediate_size)
        .bias(args.use_conv_bias)
        .padding(0)
        .build()?;
        let x_proj = super::mamba2::zero_linear_bf16(
            args.intermediate_size,
            dt_rank + 2 * args.state_size,
            false,
        )?;
        let dt_proj = super::mamba2::zero_linear_bf16(dt_rank, args.intermediate_size, true)?;
        let out_proj = super::mamba2::zero_linear_bf16(
            args.intermediate_size,
            args.hidden_size,
            args.use_bias,
        )?;

        // Placeholders — concrete weights are populated by safetensors load.
        let a_log = mlx_rs::ops::zeros::<f32>(&[args.intermediate_size, args.state_size])?;
        let d_param = mlx_rs::ops::zeros::<f32>(&[args.intermediate_size])?;

        Ok(Self {
            hidden_size: args.hidden_size,
            intermediate_size: args.intermediate_size,
            d_state: args.state_size,
            d_conv: args.conv_kernel,
            time_step_rank: dt_rank,
            use_bcdt_rms: args.use_bcdt_rms_resolved(),
            mixer_rms_eps: args.mixer_rms_eps,
            in_proj: MaybeQuantized::Original(in_proj),
            conv1d,
            x_proj: MaybeQuantized::Original(x_proj),
            dt_proj: MaybeQuantized::Original(dt_proj),
            A_log: Param::new(a_log),
            D: Param::new(d_param),
            out_proj: MaybeQuantized::Original(out_proj),
        })
    }

    /// Single-row RMSNorm with γ = ones and the configured ε. Applied to each of
    /// `delta`, `B`, `C` after `x_proj` for Falcon-Mamba.
    fn mixer_rms(&self, t: &Array) -> Result<Array, Exception> {
        let last = *t.shape().last().expect("non-empty trailing dim");
        let weight = ones_dtype(&[last], t.dtype())?;
        fast::rms_norm(t, &weight, self.mixer_rms_eps)
    }

    /// Forward one chunk of tokens through the mixer.
    ///
    /// `x` is `[B, L, hidden_size]`. The cache is mandatory: prefill passes a
    /// freshly allocated [`Mamba1Cache`] (state lazily zeroed on first use),
    /// decode passes the same cache back unchanged.
    pub fn forward(&mut self, x: &Array, cache: &mut Mamba1Cache) -> Result<Array, Exception> {
        let shape = x.shape();
        let b_size = shape[0];
        let seq_len = shape[1];
        let dtype = x.dtype();
        let d_inner = self.intermediate_size;

        // ---- 1. in_proj -> split into (x, z) -------------------------------
        let xz = self.in_proj.forward(x)?;
        // Slice the last axis into the first half (x) and second half (z).
        // Equivalent to `mx.split(xz, 2, axis=-1)` in the Python reference.
        let mut x_inner = xz.index((.., .., 0..d_inner));
        let z = xz.index((.., .., d_inner..(2 * d_inner)));

        // ---- 2. depthwise causal conv --------------------------------------
        // Concatenate the rolling conv state (NLC) in front of x_inner; conv1d
        // with padding=0 and kernel=K collapses the length back to seq_len.
        let prev = cache.conv_state_or_init(b_size, dtype)?.clone();
        let x_aug = concatenate_axis(&[prev, x_inner.clone()], 1)?;
        let conv_out = self.conv1d.forward(&x_aug)?;
        x_inner = nn::silu(&conv_out)?;

        // Save trailing (d_conv - 1) tokens for the next step.
        let total_len = x_aug.shape()[1];
        let new_conv_state = x_aug.index((.., (total_len - (self.d_conv - 1))..total_len, ..));
        cache.set_conv_state(new_conv_state);

        // ---- 3. SSM scan ---------------------------------------------------
        // A = -exp(A_log); shape [d_inner, d_state]
        let a = self.A_log.as_ref().exp()?.multiply(&array!(-1.0_f32))?;

        let dt_rank = self.time_step_rank;
        let d_state = self.d_state;

        let mut state: Option<Array> = cache.ssm_state().cloned();
        let mut outs: Vec<Array> = Vec::with_capacity(seq_len as usize);

        for t in 0..seq_len {
            // x_t: [B, d_inner]
            let x_t = x_inner.index((.., t, ..));

            // deltaBC = x_proj(x_t) -> [B, dt_rank + 2 * d_state]
            let delta_bc = self.x_proj.forward(&x_t)?;
            let mut delta = delta_bc.index((.., 0..dt_rank));
            let mut b_t = delta_bc.index((.., dt_rank..(dt_rank + d_state)));
            let mut c_t = delta_bc.index((.., (dt_rank + d_state)..(dt_rank + 2 * d_state)));

            if self.use_bcdt_rms {
                delta = self.mixer_rms(&delta)?;
                b_t = self.mixer_rms(&b_t)?;
                c_t = self.mixer_rms(&c_t)?;
            }

            // delta = softplus(dt_proj(delta)); shape [B, d_inner]
            let delta = nn::softplus(&self.dt_proj.forward(&delta)?)?;

            // new_state = (delta * x_t)[:, :, None] * B_t[:, None, :]
            //   shape [B, d_inner, d_state]
            let dt_x = delta.multiply(&x_t)?;
            let mut new_state =
                dt_x.index((.., .., NewAxis))
                    .multiply(&b_t.index((.., NewAxis, ..)))?;

            if let Some(prev_state) = state.as_ref() {
                // state * exp(delta[:, :, None] * A[None, :, :])
                let decay = delta
                    .index((.., .., NewAxis))
                    .multiply(&a.index((NewAxis, .., ..)))?
                    .exp()?;
                new_state = new_state.add(&prev_state.multiply(&decay)?)?;
            }
            state = Some(new_state.clone());

            // y = new_state @ C_t[:, :, None] -> [B, d_inner, 1] -> [B, d_inner]
            let y_t = new_state
                .matmul(&c_t.index((.., .., NewAxis)))?
                .squeeze_axes(&[2])?;
            // y += D * x_t
            let y_t = y_t.add(&self.D.as_ref().index((NewAxis, ..)).multiply(&x_t)?)?;
            outs.push(y_t.index((.., NewAxis, ..)));

            // Bound the lazy graph depth — see Mamba-2's `SCAN_EVAL_CHUNK`.
            // Eval *every* accumulated `y_t` (not just the last) so their
            // state-snapshot dependencies can be GC'd.
            const SCAN_EVAL_CHUNK: i32 = 128;
            if (t + 1) % SCAN_EVAL_CHUNK == 0 && (t + 1) < seq_len {
                let mut batch: Vec<Array> = outs.iter().cloned().collect();
                if let Some(s) = state.as_ref() {
                    batch.push(s.clone());
                }
                mlx_rs::transforms::eval(&batch)?;
            }
        }

        let y = if outs.len() == 1 {
            outs.into_iter().next().expect("len==1")
        } else {
            let refs: Vec<&Array> = outs.iter().collect();
            concatenate_axis(&refs, 1)?
        };

        if let Some(s) = state {
            cache.set_ssm_state(s);
        }
        cache.advance(seq_len);

        // ---- 4. SwiGLU gate + out_proj -------------------------------------
        // swiglu(z, y) = silu(z) * y  (reference: `out_proj(swiglu(z, y))`)
        let gated = nn::silu(&z)?.multiply(&y)?;
        self.out_proj.forward(&gated)
    }
}

// -----------------------------------------------------------------------------
// Decoder block + backbone + model
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct ResidualBlock {
    #[quantizable]
    #[param]
    pub mixer: MambaBlock,

    #[param]
    pub norm: nn::RmsNorm,
}

impl ResidualBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            mixer: MambaBlock::new(args)?,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }

    pub fn forward(&mut self, x: &Array, cache: &mut Mamba1Cache) -> Result<Array, Exception> {
        let normed = self.norm.forward(x)?;
        let h = self.mixer.forward(&normed, cache)?;
        x.add(&h)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MambaBackbone {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embeddings: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<ResidualBlock>,

    #[param]
    pub norm_f: nn::RmsNorm,
}

impl MambaBackbone {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embeddings = super::mamba2::zero_embedding_bf16(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|_| ResidualBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm_f = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embeddings: MaybeQuantized::Original(embeddings),
            layers,
            norm_f,
        })
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut Vec<Option<KvCache>>,
    ) -> Result<Array, Exception> {
        let mut h = self.embeddings.forward(inputs)?;

        if caches.is_empty() {
            *caches = (0..self.layers.len()).map(|_| None).collect();
        }
        for (layer, slot) in self.layers.iter_mut().zip(caches.iter_mut()) {
            let cache = slot
                .as_mut()
                .and_then(KvCache::as_mamba1_mut)
                .ok_or_else(|| {
                    Exception::custom(
                        "Mamba1 layer requires a KvCache::Mamba1 slot; \
                         allocate with KvCache::mamba1(...)",
                    )
                })?;
            h = layer.forward(&h, cache)?;
        }
        self.norm_f.forward(&h)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub backbone: MambaBackbone,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let backbone = MambaBackbone::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(super::mamba2::zero_linear_bf16(
                args.hidden_size,
                args.vocab_size,
                false,
            )?))
        } else {
            None
        };
        Ok(Self {
            args,
            backbone,
            lm_head,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Allocate a fresh per-layer Mamba-1 cache vector sized to this model.
    pub fn make_cache(&self) -> Vec<Option<KvCache>> {
        (0..self.args.num_hidden_layers)
            .map(|_| {
                Some(KvCache::mamba1(
                    self.args.intermediate_size,
                    self.args.conv_kernel,
                    self.args.state_size,
                ))
            })
            .collect()
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut Vec<Option<KvCache>>,
    ) -> Result<Array, Exception> {
        let h = self.backbone.forward(inputs, caches)?;
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&h),
            None => match &mut self.backbone.embeddings {
                MaybeQuantized::Original(e) => e.as_linear(&h),
                MaybeQuantized::Quantized(e) => e.as_linear(&h),
            },
        }
    }
}

// -----------------------------------------------------------------------------
// Loaders
// -----------------------------------------------------------------------------

pub fn load_falcon_mamba_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_falcon_mamba_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let raw = std::fs::read_to_string(model_dir.as_ref().join("config.json"))?;
    // Same `Infinity` workaround as Mamba-2 — some HF Mamba configs ship
    // `time_step_limit: [0.0, Infinity]`, which `serde_json` rejects. See
    // [`super::mamba2::sanitize_non_finite_json`] for the rationale.
    let sanitized = super::mamba2::sanitize_non_finite_json(&raw);
    let mut args: ModelArgs = serde_json::from_str(&sanitized)?;
    args.normalize();
    Ok(args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

/// Load weights for [`Model`] from a directory of safetensors shards.
///
/// Handles both plain bf16/f16 checkpoints and `nn::quantize`d ones (4-bit /
/// 8-bit `mlx-community/falcon-mamba-7b-*bit`):
///
/// 1. Parse `config.json` and (optionally) the `quantization` block.
/// 2. Build the param tree (`Model::new`); if quantised, run `nn::quantize`
///    on it so [`MaybeQuantized`] slots flip to `Quantized` and expose
///    `.inner.weight` + `.scales` + `.biases` keys.
/// 3. Stream every safetensor shard, mapping HF keys onto our slots:
///    - `*.conv1d.weight` with PyTorch layout `[C, 1, K]` is transposed to
///      `[C, K, 1]` (mlx-rs NLC).
///    - For quantised models, `proj.weight` → `proj.inner.weight`,
///      `proj.bias` → `proj.inner.bias`.
///    - `backbone.embeddings.{weight,scales,biases}` are captured separately
///      and applied directly into the `QuantizedEmbedding` struct (the
///      mlx-rs 0.25.x `QuantizedEmbedding` is missing `#[param]` annotations
///      on its inner fields, so it doesn't appear in `parameters_mut()`).
pub fn load_falcon_mamba_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    use mlx_rs::module::{ModuleParameters, ModuleParametersExt};

    let model_dir = model_dir.as_ref();
    let mut args = get_falcon_mamba_model_args(model_dir)?;

    // Resolve `time_step_rank = "auto"` etc. before building the param tree.
    args.normalize();
    let model = Model::new(args)?;

    // Optional `quantization: { group_size, bits }` (mlx-community 4-bit / 8-bit).
    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path).map_err(Error::from)?;
    let cfg: serde_json::Value = serde_json::from_str(&raw).map_err(Error::from)?;
    let quant = cfg
        .get("quantization")
        .or_else(|| cfg.get("quantization_config"))
        .and_then(|q| {
            let g = q.get("group_size")?.as_i64()? as i32;
            let b = q.get("bits")?.as_i64()? as i32;
            Some((g, b))
        });

    let mut model = if let Some((group_size, bits)) = quant {
        tracing::info!("[falcon_mamba] quantizing layers: group_size={group_size}, bits={bits}");
        // Skip `m.eval()` — see [`super::mamba2::load_mamba2_model`] for why
        // eagerly evaluating the lazy `quantize(bf16_zeros)` ops would blow
        // out RAM only to have the safetensors load overwrite everything.
        mlx_rs::nn::quantize(model, Some(group_size), Some(bits))
            .map_err(|e| Exception::custom(format!("nn::quantize failed: {e:?}")))?
    } else {
        model
    };

    // Collect shard paths (sharded or single-file).
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
            return Err(Error::from(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "no model.safetensors.index.json or model.safetensors in {}",
                    model_dir.display()
                ),
            )));
        }
        shard_files.push(single);
    }

    let is_quant = quant.is_some();
    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    let mut unfilled: HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };

    // QuantizedEmbedding workaround: mlx-rs 0.25.3 leaves the `inner`/`scales`/
    // `biases` fields of `QuantizedEmbedding` without `#[param]` annotations,
    // so they never show up in `parameters_mut()`. Capture the safetensors
    // entries by raw key and write them in via direct field mutation below.
    let mut embed_weight: Option<Array> = None;
    let mut embed_scales: Option<Array> = None;
    let mut embed_biases: Option<Array> = None;

    for shard in &shard_files {
        let loaded = Array::load_safetensors(shard)?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, mut value) in loaded {
            let key = raw_key.as_str();

            // Reshape depthwise conv weights from PyTorch [C, 1, K] -> mlx-rs [C, K, 1].
            if key.ends_with(".conv1d.weight") {
                let shape = value.shape().to_vec();
                if shape.len() == 3 && shape[1] == 1 {
                    value = value.transpose_axes(&[0, 2, 1])?;
                }
            }

            // Embedding triple (handled separately, see comment above).
            match key {
                "backbone.embeddings.weight" => {
                    embed_weight = Some(value);
                    total_loaded += 1;
                    continue;
                }
                "backbone.embeddings.scales" => {
                    embed_scales = Some(value);
                    total_loaded += 1;
                    continue;
                }
                "backbone.embeddings.biases" => {
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
                if let Some(stripped) = key.strip_suffix(".weight") {
                    let remapped = format!("{stripped}.inner.weight");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled.remove(&remapped);
                        continue;
                    }
                }
                if let Some(stripped) = key.strip_suffix(".bias") {
                    let remapped = format!("{stripped}.inner.bias");
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
        match &mut model.backbone.embeddings {
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
                tracing::info!(
                    "[falcon_mamba] embeddings (QuantizedEmbedding) populated via direct mutation"
                );
            }
            MaybeQuantized::Original(e) => {
                if let Some(w) = embed_weight {
                    e.weight.value = w;
                }
                tracing::info!(
                    "[falcon_mamba] embeddings (Embedding) populated via direct mutation"
                );
            }
        }
    }

    tracing::info!(
        "[falcon_mamba] safetensor load: {total_loaded} matched, {total_missed} unmatched (quantized={is_quant})"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[falcon_mamba] sample unmatched keys: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled.is_empty() {
        let mut samples: Vec<&String> = unfilled.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[falcon_mamba] {} parameter slot(s) NOT populated — first few: {}",
            unfilled.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        return Err(Exception::custom(
            "no safetensor keys matched the Mamba-1 / Falcon-Mamba parameter tree",
        )
        .into());
    }

    model.eval()?;
    // Same RAM-cleanup as Mamba-2: drop MLX's transient buffer pool after
    // load+eval. See [`super::mamba2::load_mamba2_model`].
    unsafe {
        mlx_sys::mlx_clear_cache();
    }
    Ok(model)
}

/// Falcon‑Mamba shares Mistral-style `<s>` / `</s>` markers — look them up by
/// literal token name in the tokenizer vocabulary.
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
            tokenizer
                .token_to_id("<s>")
                .and_then(|id| tokenizer.decode(std::slice::from_ref(&id), false).ok())
        } else {
            None
        };
        let eos = if need_eos {
            tokenizer
                .token_to_id("</s>")
                .and_then(|id| tokenizer.decode(std::slice::from_ref(&id), false).ok())
        } else {
            None
        };
        SpecialTokens { bos, eos }
    }

    fn stop_token_ids(
        &self,
        tokenizer: &crate::engine::mlx_lm_utils::tokenizer::Tokenizer,
    ) -> Vec<u32> {
        // Falcon-Mamba: terminators resolved by name; default to id 2 (`</s>`).
        let mut ids = crate::engine::chat_template_openai::resolve_token_ids(
            tokenizer,
            &["</s>", "<|endoftext|>", "[/INST]"],
        );
        if ids.is_empty() {
            ids.push(2);
        }
        ids
    }
}
