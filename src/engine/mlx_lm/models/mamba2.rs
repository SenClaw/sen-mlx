//! Mamba-2 (SSD) inference for `mlx-community/mamba2-*` checkpoints.
//!
//! Layout follows the official HuggingFace Mamba-2 architecture and is compatible
//! with the Python reference in `mlx-examples/llms/mlx_lm/models/mamba2.py`.
//!
//! ## Algorithmic notes
//!
//! Mamba-2 collapses the per-channel SSM of Mamba-1 into a *head-wise* recurrence
//! (`State Space Duality`, "SSD"). For each head `h`:
//!
//! ```text
//! state[t] = state[t-1] * exp(dt[t] * A[h])
//!          + dt[t] * x[t] (outer) B[group(h), t]
//! y[t]     = state[t] @ C[group(h), t]^T + D[h] * x[t]
//! ```
//!
//! where `B`/`C` are shared across heads in the same `n_groups` group.
//!
//! ### Scan backend
//!
//! The crate-public [`SsmScanBackend`] trait abstracts over implementations of the
//! core recurrence. [`SequentialScan`] (the default) is a correct but O(L) per-token
//! loop using `mlx_rs::ops`. A future chunked SSD or Metal-kernel backend can be
//! dropped in by implementing the trait — the block forward path itself is unchanged.

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
        broadcast_to, concatenate_axis,
        indexing::{IndexOp, NewAxis},
        zeros_dtype,
    },
    quantization::MaybeQuantized,
    Array, Dtype,
};

/// Build an `nn::Linear` with **bf16 zero** placeholder weight/bias instead of
/// the f32 random uniform that [`nn::LinearBuilder::build`] would emit.
///
/// We never train these models — every slot is overwritten by safetensors at
/// load time — so init *values* are irrelevant. The dtype, however, dominates
/// peak RAM for big checkpoints:
///
/// `LinearBuilder::build()` calls `random::uniform::<_, f32>` which, when
/// `nn::quantize` forces evaluation on a 7B checkpoint (64 layers ×
/// 4096↔16384 `in_proj` projections), materialises ~28 GB of f32 before each
/// int4 replacement is ready. Falling back to bf16 zeros (a) halves that to
/// ~14 GB and (b) lets MLX skip the random-state side effect entirely.
pub(crate) fn zero_linear_bf16(
    in_dim: i32,
    out_dim: i32,
    with_bias: bool,
) -> Result<nn::Linear, Exception> {
    let weight = zeros_dtype(&[out_dim, in_dim], Dtype::Bfloat16)?;
    let bias = if with_bias {
        Some(zeros_dtype(&[out_dim], Dtype::Bfloat16)?)
    } else {
        None
    };
    Ok(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

/// Build an `nn::Embedding` with bf16 zero placeholder. Same rationale as
/// [`zero_linear_bf16`] — `nn::Embedding::new` calls `random::normal::<f32>`
/// which dominates RAM for large vocabularies (e.g. 32k × 4096 × 4 bytes
/// ≈ 512 MiB) the moment `nn::quantize` evaluates it.
pub(crate) fn zero_embedding_bf16(
    embedding_count: i32,
    dimensions: i32,
) -> Result<nn::Embedding, Exception> {
    let weight = zeros_dtype(&[embedding_count, dimensions], Dtype::Bfloat16)?;
    Ok(nn::Embedding {
        weight: Param::new(weight),
    })
}
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::super::{
    cache::{KvCache, Mamba2Cache},
    error::Error,
};

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// HuggingFace `config.json` schema for Mamba-2.
///
/// Adapts to the real `mlx-community/mamba2-*` layout:
/// - `rms_norm_eps` is aliased from `layer_norm_epsilon` (HF's field name).
/// - `intermediate_size` is optional in the JSON and derived as
///   `expand * hidden_size` when absent.
/// - `time_step_*` and other unused fields are not declared so serde silently
///   ignores them (avoids the `Infinity` value HF writes into
///   `time_step_limit`, which is not valid JSON).
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub vocab_size: i32,
    #[serde(default = "default_eps", alias = "layer_norm_epsilon")]
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    /// Optional in HF JSON — derived as `expand * hidden_size` if zero/missing.
    #[serde(default)]
    pub intermediate_size: i32,
    #[serde(default = "default_expand")]
    pub expand: i32,
    pub state_size: i32,
    pub conv_kernel: i32,
    pub n_groups: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: i32,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "default_true")]
    pub use_conv_bias: bool,
    /// HF `eos_token_id` (Mamba-Codestral sets `0`; real sequence end is usually `</s>` = 2).
    #[serde(default)]
    pub eos_token_id: i32,
    /// HF `bos_token_id` (Codestral: `0`; chat templates use literal `<s>` = 1).
    #[serde(default)]
    pub bos_token_id: i32,
}

/// Mistral V1 instruct Jinja used by `mistralai/Mamba-Codestral-*` and other `[INST]` models.
/// System text is prepended to the **last** user turn (Mistral convention).
pub const MISTRAL_V1_INSTRUCT_CHAT_TEMPLATE: &str = r#"{%- if messages[0]["role"] == "system" %}
 {%- set system_message = messages[0]["content"] %}
 {%- set loop_messages = messages[1:] %}
{%- else %}
 {%- set loop_messages = messages %}
{%- endif %}
{{- bos_token }}
{%- for message in loop_messages %}
 {%- if message["role"] == "user" %}
 {%- if loop.last and system_message is defined %}
 {{- "[INST] " + system_message + "\n\n" + message["content"] + "[/INST]" }}
 {%- else %}
 {{- "[INST] " + message["content"] + "[/INST]" }}
 {%- endif %}
 {%- elif message["role"] == "assistant" %}
 {{- " " + message["content"] + eos_token }}
 {%- else %}
 {{- raise_exception("Only user and assistant roles are supported, with optional system message!") }}
 {%- endif %}
{%- endfor %}"#;

fn default_eps() -> f32 {
    1e-5
}
fn default_expand() -> i32 {
    2
}
fn default_chunk_size() -> i32 {
    256
}
fn default_true() -> bool {
    true
}

impl ModelArgs {
    /// Resolve fields that HF leaves implicit. Call once after deserialisation.
    pub fn normalize(&mut self) {
        if self.intermediate_size <= 0 {
            self.intermediate_size = self.expand.max(1) * self.hidden_size;
        }
    }

    pub fn conv_dim(&self) -> i32 {
        self.intermediate_size + 2 * self.n_groups * self.state_size
    }

    /// Token ids that should end generation for this checkpoint.
    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        // HF often sets `eos_token_id: 0` (padding); still honour non-zero config ids.
        if self.eos_token_id > 0 {
            ids.push(self.eos_token_id as u32);
        }
        ids
    }
}

// -----------------------------------------------------------------------------
// SSM scan backend
// -----------------------------------------------------------------------------

/// Pluggable backend for the SSD recurrence. The default implementation,
/// [`SequentialScan`], walks the time axis with `mlx_rs::ops`. Future chunked /
/// Metal-kernel backends implement the same trait to swap transparently.
pub trait SsmScanBackend {
    /// Run the SSM recurrence over `seq_len` timesteps.
    ///
    /// Tensor shapes (channels-first SSM convention):
    /// - `x`:   `[B, L, n_heads, head_dim]`
    /// - `dt`:  `[B, L, n_heads]` (already after `softplus(. + dt_bias)`)
    /// - `A`:   `[n_heads]`
    /// - `b`:   `[B, L, n_groups, d_state]`
    /// - `c`:   `[B, L, n_groups, d_state]`
    /// - `d`:   `[n_heads]`
    /// - `state_in`: `[B, n_heads, head_dim, d_state]`
    ///
    /// Returns `(y, state_out)` with `y: [B, L, n_heads, head_dim]` and
    /// `state_out` matching `state_in`'s shape.
    #[allow(clippy::too_many_arguments)]
    fn scan(
        &self,
        x: &Array,
        dt: &Array,
        a: &Array,
        b: &Array,
        c: &Array,
        d: &Array,
        state_in: &Array,
    ) -> Result<(Array, Array), Exception>;
}

/// Per-token sequential scan. Correct for both prefill and decode; O(L).
///
/// Implementation strategy: hold the live state as an [`Array`] and reduce over
/// the time dimension by indexing each token. This is purposefully simple — it
/// uses only ops that exist in `mlx-rs` v0.25.x and gives a reference against
/// which a future chunked SSD implementation can be validated.
#[derive(Debug, Default, Clone, Copy)]
pub struct SequentialScan;

/// How many timesteps to walk before forcing an `eval` to bound the lazy
/// graph depth (and free per-step intermediates back to MLX's pool).
///
/// Without this, prefilling a long prompt accumulates a chained
/// multiply-add graph of length `seq_len` per layer; for a 64-layer / 2k-token
/// prefill the MLX active set has been measured at ~9 GB and the buffer pool
/// at another ~10 GB before the final `eval_all_caches` collapses everything.
/// Materialising every `SCAN_EVAL_CHUNK` steps keeps the active set at the
/// chunk's working-set size instead of the full prompt's.
const SCAN_EVAL_CHUNK: i32 = 128;

impl SsmScanBackend for SequentialScan {
    fn scan(
        &self,
        x: &Array,
        dt: &Array,
        a: &Array,
        b: &Array,
        c: &Array,
        d: &Array,
        state_in: &Array,
    ) -> Result<(Array, Array), Exception> {
        use mlx_rs::transforms::eval;

        let x_shape = x.shape();
        let b_size = x_shape[0];
        let seq_len = x_shape[1];
        let n_heads = x_shape[2];
        let head_dim = x_shape[3];
        let n_groups = b.shape()[2];
        let d_state = b.shape()[3];
        let heads_per_group = n_heads / n_groups.max(1);

        let mut state = state_in.clone();
        let mut outs: Vec<Array> = Vec::with_capacity(seq_len as usize);

        for t in 0..seq_len {
            // [B, n_heads]
            let dt_t = dt.index((.., t, ..));
            // [B, n_heads, 1, 1] for broadcasting against state[B, H, head_dim, d_state]
            let dt_t_b = dt_t.index((.., .., NewAxis, NewAxis));
            // dA[h] = exp(dt * A[h]); [B, n_heads, 1, 1]
            let a_b = a.index((NewAxis, .., NewAxis, NewAxis));
            let d_a = (dt_t_b.clone().multiply(&a_b)?).exp()?;

            // x_t: [B, n_heads, head_dim], B_t: [B, n_groups, d_state]
            let x_t = x.index((.., t, .., ..));
            let b_t = b.index((.., t, .., ..));
            let c_t = c.index((.., t, .., ..));

            // Expand B/C from per-group to per-head: tile heads_per_group along axis 1.
            let b_t = repeat_groups_to_heads(&b_t, heads_per_group, n_groups, d_state, b_size)?;
            let c_t = repeat_groups_to_heads(&c_t, heads_per_group, n_groups, d_state, b_size)?;

            // dBx = dt * x[..,None] * B[:,:,None,:]  -> [B, H, head_dim, d_state]
            let dt_x = dt_t_b.multiply(&x_t.index((.., .., .., NewAxis)))?;
            let d_b_x = dt_x.multiply(&b_t.index((.., .., NewAxis, ..)))?;

            // state <- state * dA + dBx
            state = state.multiply(&d_a)?.add(&d_b_x)?;

            // y = sum_state ⋅ C  -> [B, H, head_dim]
            //   = (state * C[:,:,None,:]).sum(-1)
            let y_t = state
                .multiply(&c_t.index((.., .., NewAxis, ..)))?
                .sum_axes(&[-1], false)?;

            // Skip connection: y += D[h] * x_t
            let d_b = d.index((NewAxis, .., NewAxis));
            let y_t = y_t.add(&d_b.multiply(&x_t)?)?;
            outs.push(y_t.index((.., NewAxis, .., ..)));

            // Force materialisation every `SCAN_EVAL_CHUNK` steps so the lazy
            // graph doesn't grow to seq_len-deep chains of state updates.
            //
            // Crucially we must eval *every* `y_t` accumulated in `outs` so
            // far — not just the last one — because each one references the
            // state snapshot at its timestep. Leaving the earlier outs lazy
            // would force MLX to keep state[0..t] alive through them.
            if SCAN_EVAL_CHUNK > 0 && (t + 1) % SCAN_EVAL_CHUNK == 0 && (t + 1) < seq_len {
                let mut batch: Vec<Array> = outs.iter().cloned().collect();
                batch.push(state.clone());
                eval(&batch)?;
            }
        }

        let y = if outs.len() == 1 {
            outs.into_iter().next().expect("len==1")
        } else {
            let refs: Vec<&Array> = outs.iter().collect();
            concatenate_axis(&refs, 1)?
        };
        // y shape: [B, L, n_heads, head_dim]
        let _ = (b_size, head_dim); // retained for shape-check intent
        Ok((y, state))
    }
}

fn repeat_groups_to_heads(
    t: &Array,
    repeats: i32,
    n_groups: i32,
    d_state: i32,
    batch: i32,
) -> Result<Array, Exception> {
    // t: [B, n_groups, d_state] -> [B, n_groups, repeats, d_state] -> [B, n_groups*repeats, d_state]
    if repeats == 1 {
        return Ok(t.clone());
    }
    let expanded = broadcast_to(
        &t.index((.., .., NewAxis, ..)),
        &[batch, n_groups, repeats, d_state],
    )?;
    expanded.reshape(&[batch, n_groups * repeats, d_state])
}

// -----------------------------------------------------------------------------
// Mamba-2 mixer block
// -----------------------------------------------------------------------------

/// One Mamba-2 mixer. Mirrors `MambaMixer` from the reference Python.
///
/// Param naming matches HF safetensors keys for `mlx-community/mamba2-*`:
/// `in_proj`, `conv1d`, `dt_bias`, `A_log`, `D`, `norm`, `out_proj`.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
#[allow(non_snake_case)]
pub struct Mamba2Mixer {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub n_heads: i32,
    pub head_dim: i32,
    pub n_groups: i32,
    pub d_state: i32,
    pub d_conv: i32,
    pub conv_dim: i32,

    #[quantizable]
    #[param]
    pub in_proj: MaybeQuantized<nn::Linear>,

    /// Depthwise short conv over `conv_dim` channels.
    #[param]
    pub conv1d: nn::Conv1d,

    /// `[n_heads]` — softplus bias for the discretisation step.
    #[param]
    pub dt_bias: Param<Array>,

    /// `[n_heads]` — log-parameterised diagonal of `A`. The block consumes
    /// `A = -exp(A_log)`.
    #[param]
    #[allow(non_snake_case)]
    pub A_log: Param<Array>,

    /// `[n_heads]` — skip-connection scaling.
    #[param]
    #[allow(non_snake_case)]
    pub D: Param<Array>,

    /// RMSGroupNorm over `intermediate_size`, partitioned across `n_groups`.
    /// `mlx_rs::nn::RmsNorm` is used directly: Mamba-2's reference applies the
    /// norm head-wise (reshape to `[B, L, n_groups, group_dim]`, normalise the
    /// last axis, then reshape back) — we keep that inline in `forward`.
    #[param]
    pub norm: nn::RmsNorm,

    #[quantizable]
    #[param]
    pub out_proj: MaybeQuantized<nn::Linear>,
}

impl Mamba2Mixer {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let conv_dim = args.conv_dim();
        // in_proj output = [z, xBC, dt] = d_inner + (d_inner + 2*n_groups*d_state) + n_heads
        let proj_out = args.intermediate_size + conv_dim + args.num_heads;
        // Skip `LinearBuilder::build()` — see [`zero_linear_bf16`] for why we
        // can't afford the f32 random init on 7B Mamba checkpoints.
        let in_proj = zero_linear_bf16(args.hidden_size, proj_out, args.use_bias)?;
        let out_proj = zero_linear_bf16(args.intermediate_size, args.hidden_size, args.use_bias)?;

        // Depthwise short conv: groups == channels, kernel = d_conv, causal pad
        // is applied manually so we keep `padding = 0` here.
        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, args.conv_kernel)
            .groups(conv_dim)
            .bias(args.use_conv_bias)
            .padding(0)
            .build()?;

        let norm = nn::RmsNormBuilder::new(args.intermediate_size)
            .eps(args.rms_norm_eps)
            .build()?;

        // Placeholders — concrete weights are populated by safetensors load.
        let dt_bias = mlx_rs::ops::zeros::<f32>(&[args.num_heads])?;
        let a_log = mlx_rs::ops::zeros::<f32>(&[args.num_heads])?;
        let d_param = mlx_rs::ops::zeros::<f32>(&[args.num_heads])?;

        Ok(Self {
            hidden_size: args.hidden_size,
            intermediate_size: args.intermediate_size,
            n_heads: args.num_heads,
            head_dim: args.head_dim,
            n_groups: args.n_groups,
            d_state: args.state_size,
            d_conv: args.conv_kernel,
            conv_dim,
            in_proj: MaybeQuantized::Original(in_proj),
            conv1d,
            dt_bias: Param::new(dt_bias),
            A_log: Param::new(a_log),
            D: Param::new(d_param),
            norm,
            out_proj: MaybeQuantized::Original(out_proj),
        })
    }

    /// Forward one chunk of tokens through the mixer.
    ///
    /// `x` is `[B, L, hidden_size]`. The cache is mandatory: prefill passes a
    /// freshly allocated [`Mamba2Cache`] (state lazily zeroed on first use),
    /// decode passes the same cache back unchanged.
    pub fn forward(
        &mut self,
        x: &Array,
        cache: &mut Mamba2Cache,
        scan: &dyn SsmScanBackend,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let b_size = shape[0];
        let seq_len = shape[1];
        let dtype = x.dtype();

        // ---- 1. in_proj + split ---------------------------------------------
        // proj: [B, L, d_inner + conv_dim + n_heads]
        let proj = self.in_proj.forward(x)?;
        let d_inner = self.intermediate_size;
        let d_conv_dim = self.conv_dim;
        let n_h = self.n_heads;

        let z = proj.index((.., .., 0..d_inner));
        let x_bc = proj.index((.., .., d_inner..(d_inner + d_conv_dim)));
        let dt = proj.index((.., .., (d_inner + d_conv_dim)..(d_inner + d_conv_dim + n_h)));

        // ---- 2. depthwise causal conv ---------------------------------------
        // Concatenate the rolling conv state (NLC) in front of x_bc, run conv1d,
        // then update the cache window. mlx-rs Conv1d expects NLC input.
        let prev = cache.conv_state_or_init(b_size, dtype)?.clone();
        let xbc_aug = concatenate_axis(&[prev, x_bc.clone()], 1)?;
        // After conv with kernel=d_conv and no padding, length collapses to
        // (prev_len + seq_len) - (d_conv - 1) == seq_len.
        let mut x_bc_conv = self.conv1d.forward(&xbc_aug)?;
        x_bc_conv = nn::silu(&x_bc_conv)?;

        // Save the trailing (d_conv - 1) tokens for the next step.
        let total_len = xbc_aug.shape()[1];
        let new_conv_state = xbc_aug.index((.., (total_len - (self.d_conv - 1))..total_len, ..));
        cache.set_conv_state(new_conv_state);

        // Split the conv output into x, B, C along channel axis.
        let split_a = d_inner;
        let split_b = d_inner + self.n_groups * self.d_state;
        let x_ssm = x_bc_conv.index((.., .., 0..split_a));
        let b_ssm = x_bc_conv.index((.., .., split_a..split_b));
        let c_ssm = x_bc_conv.index((.., .., split_b..d_conv_dim));

        // Reshape into SSM-friendly axes.
        // x: [B, L, n_heads, head_dim]
        let x_ssm = x_ssm.reshape(&[b_size, seq_len, self.n_heads, self.head_dim])?;
        // B/C: [B, L, n_groups, d_state]
        let b_ssm = b_ssm.reshape(&[b_size, seq_len, self.n_groups, self.d_state])?;
        let c_ssm = c_ssm.reshape(&[b_size, seq_len, self.n_groups, self.d_state])?;

        // ---- 3. dt + A discretisation ---------------------------------------
        // dt = softplus(dt_raw + dt_bias); broadcast bias across [B, L, n_heads]
        let dt_biased = dt.add(&self.dt_bias.as_ref().index((NewAxis, NewAxis, ..)))?;
        let dt = nn::softplus(&dt_biased)?;
        // A = -exp(A_log); shape [n_heads]
        let a = self.A_log.as_ref().exp()?.multiply(&array!(-1.0_f32))?;

        // ---- 4. SSM scan -----------------------------------------------------
        let state_in = cache.ssm_state_or_init(b_size, dtype)?.clone();
        let (y, state_out) =
            scan.scan(&x_ssm, &dt, &a, &b_ssm, &c_ssm, self.D.as_ref(), &state_in)?;
        cache.set_ssm_state(state_out);
        cache.advance(seq_len);

        // y: [B, L, n_heads, head_dim] -> [B, L, d_inner]
        let y = y.reshape(&[b_size, seq_len, d_inner])?;

        // ---- 5. gated RMSNorm + out_proj ------------------------------------
        // `MambaRMSNormGated`: gate is multiplied IN before the norm, then the
        // norm's `weight` (γ) scales the result. The reference (HF transformers
        // and mlx-examples) both apply silu(gate) -> mul -> rms-norm -> γ, NOT
        // norm -> mul. Doing the gate after the norm would skip the γ scale on
        // the gated branch and inflates variance.
        let gated_pre = y.multiply(&nn::silu(&z)?)?;
        let y_out = self.norm.forward(&gated_pre)?;
        self.out_proj.forward(&y_out)
    }
}

// -----------------------------------------------------------------------------
// Decoder block + model
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mamba2Block {
    #[quantizable]
    #[param]
    pub mixer: Mamba2Mixer,

    #[param]
    pub norm: nn::RmsNorm,
}

impl Mamba2Block {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            mixer: Mamba2Mixer::new(args)?,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        cache: &mut Mamba2Cache,
        scan: &dyn SsmScanBackend,
    ) -> Result<Array, Exception> {
        let normed = self.norm.forward(x)?;
        let h = self.mixer.forward(&normed, cache, scan)?;
        x.add(&h)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mamba2Backbone {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embeddings: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<Mamba2Block>,

    #[param]
    pub norm_f: nn::RmsNorm,
}

impl Mamba2Backbone {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embeddings = zero_embedding_bf16(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|_| Mamba2Block::new(args))
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
        scan: &dyn SsmScanBackend,
    ) -> Result<Array, Exception> {
        use mlx_rs::transforms::eval;

        let mut h = self.embeddings.forward(inputs)?;

        if caches.is_empty() {
            *caches = (0..self.layers.len()).map(|_| None).collect();
        }
        // For long prefills (e.g. 2k tokens) the per-layer activations are
        // large (~hundreds of MB each in bf16). Without an `eval` between
        // layers, MLX builds one giant lazy graph spanning all 64 layers and
        // keeps every intermediate live until the final `eval_all_caches`
        // call — measured 9+ GB active for Mamba-Codestral. Force-materialise
        // `h` (and the cache state we just wrote) after each layer so the
        // working set stays at one layer's transient instead of 64 layers'.
        for (layer, slot) in self.layers.iter_mut().zip(caches.iter_mut()) {
            let cache = slot
                .as_mut()
                .and_then(KvCache::as_mamba2_mut)
                .ok_or_else(|| {
                    Exception::custom(
                        "Mamba2 layer requires a KvCache::Mamba2 slot; \
                         allocate with KvCache::mamba2(...)",
                    )
                })?;
            h = layer.forward(&h, cache, scan)?;
            // Cheap on decode (h is 1-token wide); critical on prefill.
            eval(&[h.clone()])?;
        }
        self.norm_f.forward(&h)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub backbone: Mamba2Backbone,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let backbone = Mamba2Backbone::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(zero_linear_bf16(
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

    /// Allocate a fresh per-layer SSM cache vector sized to this model.
    pub fn make_cache(&self) -> Vec<Option<KvCache>> {
        (0..self.args.num_hidden_layers)
            .map(|_| {
                Some(KvCache::mamba2(
                    self.args.conv_dim(),
                    self.args.conv_kernel,
                    self.args.num_heads,
                    self.args.head_dim,
                    self.args.state_size,
                ))
            })
            .collect()
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut Vec<Option<KvCache>>,
        scan: &dyn SsmScanBackend,
    ) -> Result<Array, Exception> {
        let h = self.backbone.forward(inputs, caches, scan)?;
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

pub fn load_mamba2_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_mamba2_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let raw = std::fs::read_to_string(model_dir.as_ref().join("config.json"))?;
    // HF Mamba-2 configs embed `time_step_limit: [0.0, Infinity]`; `Infinity`
    // is not a valid JSON literal so `serde_json` rejects the file even when
    // the field is `#[serde(default)]`-skipped. Replace the bare literals with
    // `null` before parsing — the fields are ignored downstream.
    let sanitized = sanitize_non_finite_json(&raw);
    let mut args: ModelArgs = serde_json::from_str(&sanitized)?;
    args.normalize();
    Ok(args)
}

/// Replace bare `Infinity` / `-Infinity` / `NaN` JSON literals with `null`.
///
/// HF Mamba checkpoints write `time_step_limit: [0.0, Infinity]`. These are not
/// valid JSON values per RFC 8259, so `serde_json` refuses them outright (the
/// tokenizer fails before the struct visitor gets a chance to skip the field).
/// We do a coarse `str::replace` because:
///   1. These literals never appear inside any HF config *string* value.
///   2. The fields that carry them (`time_step_*`) are ignored by our
///      `ModelArgs` anyway, so turning them into `null` is lossless.
pub(crate) fn sanitize_non_finite_json(raw: &str) -> String {
    raw.replace("-Infinity", "null")
        .replace("Infinity", "null")
        .replace("NaN", "null")
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, String>,
}

// Note: numerical tests for `SequentialScan` live in `tests/mlx_mamba2_scan.rs`
// (integration test) because the lib test binary currently fails to link due to
// unrelated pre-existing breakage in `src/agent/agent_pool/tests.rs`.
#[cfg(any())]
mod tests {
    //! Numerical tests for [`SequentialScan`].
    //!
    //! Each test builds a tiny, fully-determined SSD configuration and
    //! cross-checks `SequentialScan::scan` against a hand-rolled f32 reference
    //! loop encoding the canonical recurrence
    //!
    //! ```text
    //! state[t] = state[t-1] * exp(dt[t] * A)
    //!          + dt[t] * x[t] (outer) B[group(h), t]
    //! y[t]     = state[t] @ C[group(h), t]^T + D * x[t]
    //! ```
    //!
    //! Where shapes mirror `SsmScanBackend::scan`:
    //!   x:  [B, L, H, P]   dt: [B, L, H]   A,D: [H]
    //!   B,C: [B, L, G, N]  state: [B, H, P, N]

    use super::*;
    use mlx_rs::transforms::eval;

    /// Hand-rolled f32 reference scan. Mirrors the docstring formula 1:1.
    #[allow(clippy::too_many_arguments)]
    fn reference_scan(
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        d: &[f32],
        state_in: &[f32],
        b_size: usize,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        n_groups: usize,
        d_state: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let heads_per_group = n_heads / n_groups;
        // state: [B, H, P, N]
        let mut state = state_in.to_vec();
        let mut y = vec![0.0_f32; b_size * seq_len * n_heads * head_dim];

        for bi in 0..b_size {
            for t in 0..seq_len {
                for h in 0..n_heads {
                    let g = h / heads_per_group;
                    let dt_v = dt[(bi * seq_len + t) * n_heads + h];
                    let a_v = a[h];
                    let d_a = (dt_v * a_v).exp();

                    // state[bi, h] in R^{P x N}
                    let s_base = ((bi * n_heads + h) * head_dim) * d_state;
                    // x_t: [P], B_t/C_t: [N]
                    let x_base = ((bi * seq_len + t) * n_heads + h) * head_dim;
                    let b_base = ((bi * seq_len + t) * n_groups + g) * d_state;
                    let c_base = b_base;

                    // Update state and accumulate y.
                    for p in 0..head_dim {
                        let x_v = x[x_base + p];
                        for n in 0..d_state {
                            let dbx = dt_v * x_v * b[b_base + n];
                            let idx = s_base + p * d_state + n;
                            state[idx] = state[idx] * d_a + dbx;
                        }
                    }
                    // y[bi, t, h] = sum_n state[bi,h,p,n] * C[n] + D[h] * x[p]
                    for p in 0..head_dim {
                        let mut acc = 0.0_f32;
                        for n in 0..d_state {
                            acc += state[s_base + p * d_state + n] * c[c_base + n];
                        }
                        y[x_base + p] = acc + d[h] * x[x_base + p];
                    }
                }
            }
        }
        (y, state)
    }

    fn mlx_scan_collect(
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        d: &[f32],
        state_in: &[f32],
        b_size: i32,
        seq_len: i32,
        n_heads: i32,
        head_dim: i32,
        n_groups: i32,
        d_state: i32,
    ) -> (Vec<f32>, Vec<f32>) {
        let x_a = Array::from_slice(x, &[b_size, seq_len, n_heads, head_dim]);
        let dt_a = Array::from_slice(dt, &[b_size, seq_len, n_heads]);
        let a_a = Array::from_slice(a, &[n_heads]);
        let b_a = Array::from_slice(b, &[b_size, seq_len, n_groups, d_state]);
        let c_a = Array::from_slice(c, &[b_size, seq_len, n_groups, d_state]);
        let d_a = Array::from_slice(d, &[n_heads]);
        let s_a = Array::from_slice(state_in, &[b_size, n_heads, head_dim, d_state]);

        let scan = SequentialScan;
        let (y, s_out) = scan
            .scan(&x_a, &dt_a, &a_a, &b_a, &c_a, &d_a, &s_a)
            .expect("scan");
        eval(&[y.clone(), s_out.clone()]).expect("eval");
        (
            y.as_slice::<f32>().to_vec(),
            s_out.as_slice::<f32>().to_vec(),
        )
    }

    fn approx_eq(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let allow = tol * (1.0 + e.abs());
            assert!(diff <= allow, "elem {i}: |{a} - {e}| = {diff} > {allow}");
        }
    }

    /// Skip-connection only: A=−∞ (state decays to 0), B=0, C=0, D=1, state_in=0.
    /// Expected: y == x.
    #[test]
    fn scan_skip_connection_only() {
        let (b, l, h, p, g, n) = (1, 4, 2, 3, 1, 2);
        let x: Vec<f32> = (0..(b * l * h * p)).map(|i| i as f32 * 0.1).collect();
        let dt = vec![1.0; b * l * h];
        // A_h hugely negative so exp(dt*A) ≈ 0 in f32 — kills any state contribution.
        let a = vec![-1.0e6; h];
        let b_vec = vec![0.0; b * l * g * n];
        let c_vec = vec![0.0; b * l * g * n];
        let d = vec![1.0; h];
        let state_in = vec![0.0; b * h * p * n];

        let (y, _) = mlx_scan_collect(
            &x, &dt, &a, &b_vec, &c_vec, &d, &state_in, b as i32, l as i32, h as i32, p as i32,
            g as i32, n as i32,
        );
        approx_eq(&y, &x, 1e-5);
    }

    /// Pure state decay: x=0, B=0, D=0, state_in=1, C=1, A=ln(0.5)/dt.
    /// Expected: state and y decay by 0.5 each step.
    #[test]
    fn scan_pure_state_decay() {
        let (b, l, h, p, g, n) = (1, 3, 1, 2, 1, 2);
        let dt = vec![1.0; b * l * h];
        let half = 0.5_f32.ln();
        let a = vec![half; h];
        let x = vec![0.0; b * l * h * p];
        let b_vec = vec![0.0; b * l * g * n];
        let c_vec = vec![1.0; b * l * g * n];
        let d = vec![0.0; h];
        let state_in = vec![1.0; b * h * p * n];

        let (y, s_out) = mlx_scan_collect(
            &x, &dt, &a, &b_vec, &c_vec, &d, &state_in, b as i32, l as i32, h as i32, p as i32,
            g as i32, n as i32,
        );

        // After L=3 steps, each state element ≈ 1/8.
        for v in &s_out {
            assert!((v - 0.125).abs() < 1e-5, "state decay: {v}");
        }
        // y[t,p] = sum_n state[t,p,n] * 1; at t=2 each y elem should equal 2 * 0.125 = 0.25.
        // (n=2, sum over both d_state lanes.)
        let last_y_start = (b * (l - 1) * h * p) as usize;
        for v in &y[last_y_start..last_y_start + (h * p) as usize] {
            assert!((v - 0.25).abs() < 1e-5, "last step y: {v}");
        }
    }

    /// Cross-check the full recurrence against the hand-rolled reference loop
    /// on a small but non-trivial case with grouped B/C (heads_per_group = 2).
    #[test]
    fn scan_matches_reference_grouped() {
        let (b, l, h, p, g, n) = (1, 5, 4, 3, 2, 2);
        // Deterministic linspace-style fills so failures point at a specific index.
        let x: Vec<f32> = (0..(b * l * h * p))
            .map(|i| (i as f32 * 0.013) - 0.2)
            .collect();
        let dt: Vec<f32> = (0..(b * l * h)).map(|i| 0.1 + i as f32 * 0.07).collect();
        let a: Vec<f32> = (0..h).map(|i| -0.3 - 0.1 * i as f32).collect();
        let b_vec: Vec<f32> = (0..(b * l * g * n))
            .map(|i| 0.05 + (i as f32 * 0.011))
            .collect();
        let c_vec: Vec<f32> = (0..(b * l * g * n))
            .map(|i| -0.04 + (i as f32 * 0.009))
            .collect();
        let d: Vec<f32> = (0..h).map(|i| 0.2 + 0.1 * i as f32).collect();
        let state_in: Vec<f32> = (0..(b * h * p * n))
            .map(|i| 0.02 + i as f32 * 0.003)
            .collect();

        let (y_ref, s_ref) =
            reference_scan(&x, &dt, &a, &b_vec, &c_vec, &d, &state_in, b, l, h, p, g, n);
        let (y_mlx, s_mlx) = mlx_scan_collect(
            &x, &dt, &a, &b_vec, &c_vec, &d, &state_in, b as i32, l as i32, h as i32, p as i32,
            g as i32, n as i32,
        );
        approx_eq(&y_mlx, &y_ref, 5e-5);
        approx_eq(&s_mlx, &s_ref, 5e-5);
    }

    /// Sanity: a one-step scan with seq_len=1 must produce one timestep and
    /// leave state advanced by exactly one update.
    #[test]
    fn scan_single_step_shape() {
        let (b, l, h, p, g, n) = (2, 1, 2, 2, 1, 2);
        let x: Vec<f32> = (0..(b * l * h * p)).map(|i| i as f32 * 0.01).collect();
        let dt = vec![0.5; b * l * h];
        let a = vec![-0.2; h];
        let b_vec: Vec<f32> = (0..(b * l * g * n))
            .map(|i| 0.1 + 0.01 * i as f32)
            .collect();
        let c_vec: Vec<f32> = (0..(b * l * g * n))
            .map(|i| 0.2 + 0.01 * i as f32)
            .collect();
        let d = vec![0.0; h];
        let state_in = vec![0.0; b * h * p * n];

        let (y_ref, s_ref) =
            reference_scan(&x, &dt, &a, &b_vec, &c_vec, &d, &state_in, b, l, h, p, g, n);
        let (y_mlx, s_mlx) = mlx_scan_collect(
            &x, &dt, &a, &b_vec, &c_vec, &d, &state_in, b as i32, l as i32, h as i32, p as i32,
            g as i32, n as i32,
        );
        assert_eq!(y_mlx.len(), b * l * h * p);
        assert_eq!(s_mlx.len(), b * h * p * n);
        approx_eq(&y_mlx, &y_ref, 5e-5);
        approx_eq(&s_mlx, &s_ref, 5e-5);
    }
}

/// Load weights for [`Model`] from a directory of safetensors shards.
///
/// Handles the **conv1d weight layout** difference between PyTorch and mlx-rs:
/// HF saves depthwise Conv1d weight as `[C, 1, K]` (PyTorch NCL convention),
/// whereas `mlx_rs::nn::Conv1d` expects `[C_out, K, C_in/groups]` = `[C, K, 1]`
/// for the same depthwise op. We transpose axes `(1, 2)` on every conv1d weight
/// we see so checkpoints from `mlx-community/mamba2-*` (and any HF→safetensors
/// pipeline that preserved PT layout) load directly.
///
/// Unmatched keys are tracked and logged at info/warn so missing-weight bugs
/// surface immediately instead of producing silently zero-initialised tensors.
pub fn load_mamba2_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
    use mlx_rs::quantization::MaybeQuantized;

    let model_dir = model_dir.as_ref();
    let args = get_mamba2_model_args(model_dir)?;
    let model = Model::new(args)?;

    // Optional `quantization: { group_size, bits }` (mlx-community 4-bit / 8-bit,
    // e.g. `Mamba-Codestral-7B-v0.1-4bit`). When present, run `nn::quantize`
    // first so `MaybeQuantized` slots flip to `Quantized` and expose
    // `.inner.weight` + `.scales` + `.biases`.
    let cfg_raw = std::fs::read_to_string(model_dir.join("config.json"))?;
    let cfg_sanitized = sanitize_non_finite_json(&cfg_raw);
    let cfg: serde_json::Value = serde_json::from_str(&cfg_sanitized)?;
    let quant = cfg
        .get("quantization")
        .or_else(|| cfg.get("quantization_config"))
        .and_then(|q| {
            let g = q.get("group_size")?.as_i64()? as i32;
            let b = q.get("bits")?.as_i64()? as i32;
            Some((g, b))
        });

    let mut model = if let Some((group_size, bits)) = quant {
        tracing::info!("[mamba2] quantizing layers: group_size={group_size}, bits={bits}");
        // NB: deliberately *no* `m.eval()` here. `nn::quantize` produces
        // lazy `quantize(bf16_zeros)` ops for every `MaybeQuantized<Linear>`.
        // Evaluating them eagerly would materialise all 64 layers' bf16
        // placeholders (~14 GB peak) *only to overwrite them seconds later*
        // from the safetensors load below. Skipping the eval lets MLX GC
        // the lazy graph the moment we assign safetensors arrays into the
        // params — no materialisation ever happens.
        mlx_rs::nn::quantize(model, Some(group_size), Some(bits))
            .map_err(|e| Exception::custom(format!("nn::quantize failed: {e:?}")))?
    } else {
        model
    };
    let is_quant = quant.is_some();

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

    let mut unfilled: std::collections::HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };
    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    // QuantizedEmbedding workaround — see [`super::falcon_mamba`] for the
    // mlx-rs 0.25.x missing-`#[param]` rationale.
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

            // Embedding triple is captured separately and applied via direct
            // mutation below (see `embed_*` declarations).
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
                    "[mamba2] embeddings (QuantizedEmbedding) populated via direct mutation"
                );
            }
            MaybeQuantized::Original(e) => {
                if let Some(w) = embed_weight {
                    e.weight.value = w;
                }
                tracing::info!("[mamba2] embeddings (Embedding) populated via direct mutation");
            }
        }
    }

    tracing::info!(
        "[mamba2] safetensor load: {total_loaded} matched, {total_missed} unmatched (quantized={is_quant})"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[mamba2] sample unmatched keys: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled.is_empty() {
        let mut samples: Vec<&String> = unfilled.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[mamba2] {} parameter slot(s) NOT populated — first few: {}",
            unfilled.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        return Err(
            Exception::custom("no safetensor keys matched the Mamba-2 parameter tree").into(),
        );
    }

    model.eval()?;
    // Release the MLX device-pool buffers used by the quantize / safetensors
    // load. Without this, MLX retains a few GB of transient buffers in its
    // cache pool that look like a leak in `rss` to the user. Cheap & idempotent.
    unsafe {
        mlx_sys::mlx_clear_cache();
    }

    Ok(model)
}

/// True when the tokenizer defines Mistral-style `[INST]` / `[/INST]` special
/// tokens — the marker that `chat_template_fallback` should prefer the Mistral
/// V1 [INST] template (Mamba-Codestral and friends ship no `chat_template`).
fn tokenizer_has_mistral_inst_tokens(tokenizer_json: &Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(tokenizer_json) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    let Some(added) = v.get("added_tokens").and_then(|a| a.as_array()) else {
        return false;
    };
    let mut has_inst = false;
    let mut has_end_inst = false;
    for entry in added {
        let Some(content) = entry.get("content").and_then(|c| c.as_str()) else {
            continue;
        };
        if content == "[INST]" {
            has_inst = true;
        }
        if content == "[/INST]" {
            has_end_inst = true;
        }
    }
    has_inst && has_end_inst
}

/// Mamba-class fallback for `chat_template_openai::load_chat_template_from_dir`.
/// Returns the Mistral V1 `[INST]` template for checkpoints that ship the
/// `[INST]` / `[/INST]` tokens (Mamba-Codestral *Instruct), otherwise `None`
/// (caller falls back to a plain `User: … / Assistant:` transcript).
pub fn chat_template_fallback(model_dir: &Path) -> std::io::Result<Option<String>> {
    let tok = model_dir.join("tokenizer.json");
    if tokenizer_has_mistral_inst_tokens(&tok) {
        tracing::info!(
            "[local-mlx-native] no chat template in model_dir; using Mistral V1 [INST] fallback"
        );
        return Ok(Some(MISTRAL_V1_INSTRUCT_CHAT_TEMPLATE.to_string()));
    }
    tracing::warn!(
        "[local-mlx-native] no chat template in model_dir; using plain User/Assistant transcript"
    );
    Ok(None)
}

/// Mamba‑2 `[INST]` template references `{{ bos_token }}` / `{{ eos_token }}`.
/// Prefer the tokenizer's literal `<s>` / `</s>` strings, falling back to
/// `args.bos_token_id` / `args.stop_token_ids()[0]` when those special tokens
/// aren't named in the tokenizer vocabulary.
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
                .or(if self.args.bos_token_id > 0 {
                    Some(self.args.bos_token_id as u32)
                } else {
                    None
                })
                .and_then(|id| tokenizer.decode(std::slice::from_ref(&id), false).ok())
        } else {
            None
        };
        let eos = if need_eos {
            tokenizer
                .token_to_id("</s>")
                .or_else(|| self.args.stop_token_ids().first().copied())
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
        // Config stop ids (`eos_token_id` when >0) plus Mistral/Mamba chat
        // terminators resolved by name; default to id 2 (`</s>`) if none found.
        let mut ids = self.args.stop_token_ids();
        for id in crate::engine::chat_template_openai::resolve_token_ids(
            tokenizer,
            &["</s>", "<|endoftext|>", "[/INST]"],
        ) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        if ids.is_empty() {
            ids.push(2);
        }
        ids
    }
}

#[cfg(test)]
mod mistral_template_tests {
    use super::*;

    #[test]
    fn codestral_stop_ids_skip_zero_eos() {
        let args = ModelArgs {
            model_type: "mamba2".into(),
            hidden_size: 4096,
            num_hidden_layers: 64,
            vocab_size: 32768,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            intermediate_size: 8192,
            expand: 2,
            state_size: 128,
            conv_kernel: 4,
            n_groups: 8,
            num_heads: 128,
            head_dim: 64,
            chunk_size: 256,
            use_bias: false,
            use_conv_bias: true,
            eos_token_id: 0,
            bos_token_id: 0,
        };
        assert!(args.stop_token_ids().is_empty());
    }

    #[test]
    fn mistral_chat_template_contains_inst_markers() {
        assert!(MISTRAL_V1_INSTRUCT_CHAT_TEMPLATE.contains("[INST]"));
        assert!(MISTRAL_V1_INSTRUCT_CHAT_TEMPLATE.contains("bos_token"));
    }
}
