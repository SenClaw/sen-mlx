use std::sync::atomic::{AtomicU64, Ordering};

use mlx_rs::{
    error::Exception,
    ops::indexing::{IndexOp, TryIndexMutOp},
    ops::{concatenate_axis, zeros_dtype},
    transforms::eval,
    Array, Dtype,
};
use turboquant::{attention::QuantizedKVCache, packed::TurboQuantConfig};

/// Default TurboQuant activation threshold (tokens before quantizing KV storage).
///
/// Once activated, every new KV pair routes through the `turboquant-rs` CPU
/// path (f16→f32 cast + heap-allocated quant batch + 4-bit pack). For typical
/// tool-calling prompts (~120 tokens / MCP tool × ~100 tools ≈ 12 K tokens),
/// activating at 2 048 forces the entire post-2 048 chunk through the slow
/// CPU path during **prefill**, easily blowing past the LLM turn timeout.
///
/// Setting this to 16 384 means TQ only kicks in for genuinely long sessions
/// (multi-turn accumulating past 16 K) where the RAM saving is meaningful.
/// Users who want earlier quantization can override `tq_activate_at` in
/// `settings.json`.
pub const DEFAULT_TQ_ACTIVATE_AT: i32 = 16_384;

/// Growth chunk for pre-allocated FP16 KV buffers (mlx_lm / Higgs stepping).
const KV_CACHE_STEP: i32 = 256;

static TQ_SEED: AtomicU64 = AtomicU64::new(0x5eed_c0de);

fn next_tq_seed() -> u64 {
    TQ_SEED.fetch_add(1, Ordering::Relaxed)
}

/// Per-layer cache: FP16 attention KV, TurboQuant attention KV, Mamba-1 SSM state,
/// or Mamba-2 SSM state.
///
/// SSM layers (Mamba-1 / Mamba-2) do not store KV pairs; they carry a fixed-size
/// recurrent state (conv_state + ssm_state). When wrapped in this enum,
/// [`KeyValueCache`] methods for SSM layers behave as a no-op (length reported as
/// tokens-seen, attention ops bypassed). Mamba blocks access the cache directly
/// via [`KvCache::as_mamba1_mut`] / [`KvCache::as_mamba2_mut`].
/// Cloning a cache copies the **MLX Array handles** (which are Arc-internal).
/// Because every `update_*` method *replaces* the Option<Array> slot with a
/// freshly-built Array (`slice_axis2`, `slice_update_axis2`, …) instead of
/// mutating in place, snapshots stay independent of subsequent generation
/// steps — this is what makes the [`PrefixCache`](super::prefix_cache) safe.
///
/// `TurboQuantKeyValueCache` cannot be cloned (it owns a `QuantizedKVCache`
/// from `turboquant-rs` which is non-`Clone`); prefix caching is skipped for
/// turns that activated TurboQuant.
#[derive(Debug)]
pub enum KvCache {
    Fp16(SteppingKeyValueCache),
    TurboQuant(TurboQuantKeyValueCache),
    Mamba1(Mamba1Cache),
    Mamba2(Mamba2Cache),
    /// Qwen3.5 GatedDeltaNet: conv rolling window + SSM state.
    Qwen35Linear(Qwen35LinearCache),
}

impl KvCache {
    /// Snapshot-clone for prefix caching. Returns `None` if the variant is
    /// not safe to clone (TurboQuant active state, etc.) — caller treats
    /// that as "skip prefix cache for this turn".
    pub fn try_snapshot(&self) -> Option<Self> {
        match self {
            Self::Fp16(c) => Some(Self::Fp16(c.snapshot_clone())),
            Self::TurboQuant(c) if !c.is_turbo_active() => Some(Self::Fp16(c.staging_clone())),
            Self::TurboQuant(_) => None,
            Self::Mamba1(c) => Some(Self::Mamba1(c.clone())),
            Self::Mamba2(c) => Some(Self::Mamba2(c.clone())),
            Self::Qwen35Linear(c) => Some(Self::Qwen35Linear(c.clone())),
        }
    }

    /// Drop the last `n` cached tokens. Used by the prefix cache to strip
    /// the per-turn assistant generation suffix (`<|im_start|>assistant\n`)
    /// from a snapshot so subsequent turns can hit the shared prefix.
    ///
    /// Only meaningful for attention KV caches (`Fp16`); SSM / linear caches
    /// have fixed-size state with no notion of "last N tokens", so it's a
    /// no-op there (and prefix caching is disabled for them at call site).
    pub fn trim_by(&mut self, n: usize) {
        match self {
            Self::Fp16(c) => c.trim_by(n),
            Self::TurboQuant(_) | Self::Mamba1(_) | Self::Mamba2(_) | Self::Qwen35Linear(_) => {
                // No-op: TurboQuant active path is excluded from prefix cache
                // upstream; SSM/linear caches are recurrent and have no
                // per-token slice to trim.
            }
        }
    }

    /// Force the cache buffers to materialize into compact storage sized to
    /// the current `stored_len`. Used right after a snapshot is taken so the
    /// snapshot doesn't pin the live cache's full preallocated buffer
    /// (~4.4 GB for Qwen3-4B + 32 K KV). See
    /// [`SteppingKeyValueCache::compact_to_stored_len`].
    pub fn compact_to_stored_len(&mut self) {
        if let Self::Fp16(c) = self {
            c.compact_to_stored_len();
        }
    }
}

impl KvCache {
    pub fn fp16_with_max(max_seq_len: i32) -> Self {
        Self::Fp16(SteppingKeyValueCache::with_max(max_seq_len))
    }

    /// FP16 KV with a decode-time sliding window (see
    /// [`SteppingKeyValueCache::with_decode_window`]). Prefill grows to
    /// `max_seq_len`; decode is bounded to the last `decode_window` keys.
    pub fn fp16_decode_windowed(max_seq_len: i32, decode_window: i32) -> Self {
        Self::Fp16(SteppingKeyValueCache::with_decode_window(
            max_seq_len,
            decode_window,
        ))
    }

    pub fn turboquant_with_max(
        bits: u8,
        head_dim: i32,
        n_kv_heads: i32,
        max_seq_len: i32,
        activate_at: i32,
    ) -> Self {
        Self::TurboQuant(TurboQuantKeyValueCache::with_max(
            bits,
            head_dim,
            n_kv_heads,
            max_seq_len,
            activate_at,
        ))
    }

    /// Allocate a Mamba-2 SSM state cache for a single layer.
    pub fn mamba2(conv_dim: i32, d_conv: i32, n_heads: i32, head_dim: i32, d_state: i32) -> Self {
        Self::Mamba2(Mamba2Cache::new(
            conv_dim, d_conv, n_heads, head_dim, d_state,
        ))
    }

    pub fn qwen35_linear(conv_dim: i32, d_conv: i32, n_v_heads: i32, d_v: i32, d_k: i32) -> Self {
        Self::Qwen35Linear(Qwen35LinearCache::new(
            conv_dim, d_conv, n_v_heads, d_v, d_k,
        ))
    }

    /// Allocate a Mamba-1 SSM state cache for a single layer.
    ///
    /// Mamba-1's recurrence is per-channel (no head grouping), so we only need
    /// `d_inner` (= `intermediate_size`) and `d_state` to size the state.
    pub fn mamba1(d_inner: i32, d_conv: i32, d_state: i32) -> Self {
        Self::Mamba1(Mamba1Cache::new(d_inner, d_conv, d_state))
    }

    pub fn as_mamba2_mut(&mut self) -> Option<&mut Mamba2Cache> {
        match self {
            Self::Mamba2(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_qwen35_linear_mut(&mut self) -> Option<&mut Qwen35LinearCache> {
        match self {
            Self::Qwen35Linear(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_mamba1_mut(&mut self) -> Option<&mut Mamba1Cache> {
        match self {
            Self::Mamba1(c) => Some(c),
            _ => None,
        }
    }

    pub fn is_mamba2(&self) -> bool {
        matches!(self, Self::Mamba2(_))
    }

    pub fn is_mamba1(&self) -> bool {
        matches!(self, Self::Mamba1(_))
    }

    pub(crate) fn eval_targets(&self) -> Vec<Array> {
        match self {
            Self::Fp16(c) => c.eval_targets(),
            Self::TurboQuant(c) => c.eval_targets(),
            Self::Mamba1(c) => c.eval_targets(),
            Self::Mamba2(c) => c.eval_targets(),
            Self::Qwen35Linear(c) => c.eval_targets(),
        }
    }

    /// One-line description of cache kind for log lines (e.g. `"fp16"`,
    /// `"tq4"`, `"mamba2"`). Avoids dumping `Debug` impl (verbose, contains
    /// internal `Array` handles).
    pub fn kind_label(&self) -> &'static str {
        match self {
            Self::Fp16(_) => "fp16",
            Self::TurboQuant(c) => match c.bits {
                3 => "tq3",
                4 => "tq4",
                _ => "tq",
            },
            Self::Mamba1(_) => "mamba1",
            Self::Mamba2(_) => "mamba2",
            Self::Qwen35Linear(_) => "qwen35-linear",
        }
    }

    /// Approximate **per-layer** cache memory (bytes). Sums K + V buffers
    /// (FP16 → 2 B/elt) for attention caches; SSM caches return their state
    /// tensor footprint. Used for `[mem] kv cache: …` log lines so the
    /// operator can see RAM usage growing turn-by-turn.
    pub fn approx_bytes(&self) -> usize {
        match self {
            Self::Fp16(c) => c.approx_bytes(),
            Self::TurboQuant(c) => c.approx_bytes(),
            Self::Mamba1(c) => c.approx_bytes(),
            Self::Mamba2(c) => c.approx_bytes(),
            Self::Qwen35Linear(c) => c.approx_bytes(),
        }
    }
}

/// Sum [`KvCache::approx_bytes`] across all layers + a one-line summary of
/// (cache kind, tokens-stored). Cheap (just shape inspection) so it can run
/// at every prefill/decode milestone.
pub fn summarize_caches(caches: &[Option<KvCache>]) -> (usize, i32, &'static str) {
    let mut total_bytes = 0usize;
    let mut max_stored = 0i32;
    let mut kind: &'static str = "empty";
    for c in caches.iter().flatten() {
        total_bytes += c.approx_bytes();
        max_stored = max_stored.max(c.stored_len());
        kind = c.kind_label();
    }
    (total_bytes, max_stored, kind)
}

impl KeyValueCache for KvCache {
    fn is_quantized(&self) -> bool {
        match self {
            Self::Fp16(c) => c.is_quantized(),
            Self::TurboQuant(c) => c.is_quantized(),
            Self::Mamba1(_) | Self::Mamba2(_) | Self::Qwen35Linear(_) => false,
        }
    }

    fn group_size(&self) -> Option<i32> {
        match self {
            Self::Fp16(c) => c.group_size(),
            Self::TurboQuant(c) => c.group_size(),
            Self::Mamba1(_) | Self::Mamba2(_) | Self::Qwen35Linear(_) => None,
        }
    }

    fn bits(&self) -> Option<i32> {
        match self {
            Self::Fp16(c) => c.bits(),
            Self::TurboQuant(c) => c.bits(),
            Self::Mamba1(_) | Self::Mamba2(_) | Self::Qwen35Linear(_) => None,
        }
    }

    fn stored_len(&self) -> i32 {
        match self {
            Self::Fp16(c) => c.stored_len(),
            Self::TurboQuant(c) => c.stored_len(),
            Self::Mamba1(c) => c.tokens_seen(),
            Self::Mamba2(c) => c.tokens_seen(),
            Self::Qwen35Linear(c) => c.tokens_seen(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Self::Fp16(c) => c.max_size(),
            Self::TurboQuant(c) => c.max_size(),
            Self::Mamba1(_) | Self::Mamba2(_) | Self::Qwen35Linear(_) => None,
        }
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<KvFetchResult, Exception> {
        match self {
            Self::Fp16(c) => c.update_and_fetch(keys, values),
            Self::TurboQuant(c) => c.update_and_fetch(keys, values),
            Self::Mamba1(_) => Err(Exception::custom(
                "Mamba1 cache does not support KV update_and_fetch; \
                 use KvCache::as_mamba1_mut from the Mamba block",
            )),
            Self::Mamba2(_) => Err(Exception::custom(
                "Mamba2 cache does not support KV update_and_fetch; \
                 use KvCache::as_mamba2_mut from the Mamba block",
            )),
            Self::Qwen35Linear(_) => Err(Exception::custom(
                "Qwen3.5 linear cache does not support KV update_and_fetch",
            )),
        }
    }

    fn turboquant_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        n_heads: i32,
        n_kv_heads: i32,
    ) -> Result<Option<Array>, Exception> {
        match self {
            Self::Fp16(c) => c.turboquant_attention(queries, scale, mask, n_heads, n_kv_heads),
            Self::TurboQuant(c) => {
                c.turboquant_attention(queries, scale, mask, n_heads, n_kv_heads)
            }
            Self::Mamba1(_) | Self::Mamba2(_) | Self::Qwen35Linear(_) => Ok(None),
        }
    }
}

fn materialize_pair(k: Array, v: Array) -> Result<(Array, Array), Exception> {
    eval(&[k.clone(), v.clone()])?;
    Ok((k, v))
}

pub fn eval_all_caches(caches: &mut [Option<KvCache>]) -> Result<(), Exception> {
    let mut batch = Vec::new();
    for cache in caches.iter_mut().flatten() {
        batch.extend(cache.eval_targets());
    }
    if !batch.is_empty() {
        eval(&batch)?;
    }
    Ok(())
}

/// Normalize settings `kv_cache_bits` (2 → TQ3).
pub fn normalize_turboquant_bits(bits: u8) -> u8 {
    match bits {
        4 => 4,
        2 | 3 => 3,
        _ => 3,
    }
}

pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    fn group_size(&self) -> Option<i32> {
        None
    }

    fn bits(&self) -> Option<i32> {
        None
    }

    /// Tokens currently in cache (for attention mask width), not RoPE position.
    fn stored_len(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<KvFetchResult, Exception>;

    /// When TurboQuant storage is active, run approximate GQA attention on CPU.
    fn turboquant_attention(
        &mut self,
        _queries: &Array,
        _scale: f32,
        _mask: Option<&Array>,
        _n_heads: i32,
        _n_kv_heads: i32,
    ) -> Result<Option<Array>, Exception> {
        Ok(None)
    }
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn stored_len(&self) -> i32 {
        T::stored_len(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<KvFetchResult, Exception> {
        T::update_and_fetch(self, keys, values)
    }

    fn turboquant_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        n_heads: i32,
        n_kv_heads: i32,
    ) -> Result<Option<Array>, Exception> {
        T::turboquant_attention(self, queries, scale, mask, n_heads, n_kv_heads)
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedKeys {
    pub keys: Array,
    pub scales: Array,
    pub biases: Array,
}

#[derive(Debug, Clone)]
pub struct QuantizedValues {
    pub values: Array,
    pub scales: Array,
    pub biases: Array,
}

#[derive(Debug)]
pub enum KvFetchResult {
    Fp16(Array, Array),
    /// Attention uses [`super::utils::turboquant_attn::turboquant_gqa_attention`].
    TurboQuant,
}

/// FP16 KV: `slice_update` writes + grow-by-256 (unbounded) or single alloc of `max` (bounded).
///
/// RoPE positions come from the **caller** (`ModelInput::rope_offset`), not this struct.
#[derive(Debug, Clone)]
pub struct SteppingKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    stored_len: i32,
    max_seq_len: Option<i32>,
    /// Sliding-window cap that applies **only to single-token (decode) writes**.
    ///
    /// During multi-token prefill the buffer grows to `max_seq_len` so every
    /// query keeps its full attention window (a single forward pass over an
    /// `L`-token chunk needs all `L` keys — a query at position `p` attends
    /// `(p-window, p]`, and collectively those windows span the whole chunk, so
    /// the buffer cannot be bounded below the chunk length without dropping keys
    /// that later queries in the same pass still need).
    ///
    /// Once decode starts (1 token / step) eviction is exact: keep the last
    /// `decode_window` keys, which is precisely the sliding window for the new
    /// query. `None` = no sliding behavior (full-attention layers / other
    /// models) — identical to the previous behavior.
    decode_window: Option<i32>,
    /// Ring write head — `Some(h)` means the K/V buffers are **exactly**
    /// `decode_window` rows, every row is a live in-window key, and they are
    /// stored **rotated**: index `h` holds the oldest key and is the slot the
    /// next single-token write overwrites in place.
    ///
    /// Why rotate at all: the pre-ring eviction path dropped the oldest key
    /// with `slice_axis2(k, 1, stored_len)`, which left the buffer one row
    /// short of `required_slots` and so re-entered the grow branch — a trim,
    /// a `zeros` pad and a `concatenate` — **every decode step past the
    /// window**. That is two full copies of the whole K buffer and two of V,
    /// per sliding layer, per token. A ring overwrites one row and copies
    /// nothing, which makes decode past the window exactly as cheap as decode
    /// below it.
    ///
    /// Rotation is safe without any change at the model: attention is
    /// permutation-invariant along the key axis (softmax over keys, then a
    /// weighted sum of the matching values — permuting `(k, v)` pairs
    /// consistently leaves the result unchanged), keys carry their RoPE phase
    /// from when they were written, and RoPE offsets come from the caller
    /// rather than this struct. The one requirement is that the consumer pass
    /// **no mask** for these layers on decode, since every stored row is a
    /// valid key — which is what `gemma4::Gemma4TextModel::forward` already
    /// does at `seq <= 1`.
    ///
    /// The reordered key axis does change floating-point accumulation order in
    /// SDPA, so this is a *token-parity* optimization, not a bit-identity one.
    ring_head: Option<i32>,
    step: i32,
}

pub type ConcatKeyValueCache = SteppingKeyValueCache;

impl Default for SteppingKeyValueCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SteppingKeyValueCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            stored_len: 0,
            max_seq_len: None,
            decode_window: None,
            ring_head: None,
            step: KV_CACHE_STEP,
        }
    }

    /// Pre-sized KV cache. The current `update_dense` path allocates a
    /// **dense** buffer of `max_seq_len` slots on the first write, which
    /// wastes RAM when the actual prompt is much smaller than `max_seq_len`
    /// (e.g. `max_kv_tokens=32000` allocates ~4.4 GB even for a 14 K-token
    /// prompt).
    ///
    /// We accept the over-allocation as a deliberate trade-off: switching
    /// to incremental growth (`KV_CACHE_STEP=256` chunks) would multiply
    /// the per-layer `concatenate_axis` + `eval` work by `seq_len/step`,
    /// adding ~250 GPU syncs for a 14 K prefill. On M-series unified memory
    /// the RAM is cheap; CPU↔GPU dispatch overhead is not. Callers that
    /// want lower RAM should set `max_kv_tokens` closer to their actual
    /// `prompt + max_new` budget via `settings.json`.
    pub fn with_max(max_seq_len: i32) -> Self {
        Self {
            keys: None,
            values: None,
            stored_len: 0,
            max_seq_len: Some(max_seq_len.max(1)),
            decode_window: None,
            ring_head: None,
            step: KV_CACHE_STEP,
        }
    }

    /// Pre-sized KV cache with a **decode-time sliding window**. Identical to
    /// [`with_max`](Self::with_max) for multi-token prefill writes (the buffer
    /// grows up to `max_seq_len`), but single-token decode writes evict to keep
    /// only the last `decode_window` keys. Used by sliding-window attention
    /// layers (e.g. Gemma-4) so the decode KV — and the per-step SDPA work — is
    /// bounded by the window instead of the full sequence length. See
    /// [`decode_window`](Self::decode_window).
    pub fn with_decode_window(max_seq_len: i32, decode_window: i32) -> Self {
        Self {
            keys: None,
            values: None,
            stored_len: 0,
            max_seq_len: Some(max_seq_len.max(1)),
            decode_window: Some(decode_window.max(1)),
            ring_head: None,
            step: KV_CACHE_STEP,
        }
    }

    pub fn seq_len(&self) -> i32 {
        self.stored_len
    }

    pub(crate) fn eval_targets(&self) -> Vec<Array> {
        let mut out = Vec::with_capacity(2);
        if let Some(k) = &self.keys {
            out.push(k.clone());
        }
        if let Some(v) = &self.values {
            out.push(v.clone());
        }
        out
    }

    /// Bytes used by the K + V buffers for this layer. Assumes the FP16 KV
    /// stepping layout (`numel × 2 bytes`) — actual element size could be
    /// BF16/FP16 which are both 2 B, so the figure is correct in practice.
    pub(crate) fn approx_bytes(&self) -> usize {
        fn array_bytes(a: &Array) -> usize {
            let n: usize = a.shape().iter().map(|&d| d.max(0) as usize).product();
            n * 2 // f16 / bf16 = 2 B per element
        }
        let mut total = 0usize;
        if let Some(k) = &self.keys {
            total += array_bytes(k);
        }
        if let Some(v) = &self.values {
            total += array_bytes(v);
        }
        total
    }

    pub fn trim_by(&mut self, n: usize) {
        let trim = i32::try_from(n).unwrap_or(i32::MAX);
        if trim <= 0 {
            return;
        }
        // "Last n tokens" is a positional claim, so a rotated buffer has to be
        // linearized before the tail slice means anything.
        if self.ring_head.is_some() {
            let _ = self.unrotate();
        }
        let new_len = self.stored_len.saturating_sub(trim);
        if new_len < self.stored_len {
            if let (Some(k), Some(v)) = (&self.keys, &self.values) {
                if let (Ok(k2), Ok(v2)) = (slice_axis2(k, 0, new_len), slice_axis2(v, 0, new_len)) {
                    self.keys = Some(k2);
                    self.values = Some(v2);
                }
            }
            self.stored_len = new_len;
        }
    }

    /// Force the K/V buffers to materialize into **independent storage**
    /// sized exactly to `stored_len`. Pre-fix the K/V might be a view into
    /// the live cache's much larger preallocated buffer (e.g. shape
    /// `[1, 8, 32000, 128]` even though only the first `stored_len` slots
    /// are used) — that pins ~4.4 GB per snapshot for Qwen3-4B + 32 K KV.
    ///
    /// `slice_update_axis2(zeros, view, 0, stored_len)` allocates a fresh
    /// `[..., stored_len, ...]` buffer and copies the view's data in. The
    /// snapshot then owns its own compact buffer; the live cache's larger
    /// buffer is no longer pinned by the snapshot's Arc ref.
    pub fn compact_to_stored_len(&mut self) {
        let Some(k) = self.keys.as_ref() else {
            return;
        };
        let Some(v) = self.values.as_ref() else {
            return;
        };
        let k_shape = k.shape();
        let v_shape = v.shape();
        if k_shape.len() != 4 || v_shape.len() != 4 {
            return;
        }
        let cur_t = k_shape[2];
        if cur_t == self.stored_len {
            // Already compact (no padding to drop).
            return;
        }
        let target_t = self.stored_len.max(0);
        let mut new_k_shape = k_shape.to_vec();
        new_k_shape[2] = target_t;
        let mut new_v_shape = v_shape.to_vec();
        new_v_shape[2] = target_t;
        let Ok(empty_k) = zeros_dtype(&new_k_shape, k.dtype()) else {
            return;
        };
        let Ok(empty_v) = zeros_dtype(&new_v_shape, v.dtype()) else {
            return;
        };
        let Ok(stored_k) = slice_axis2(k, 0, target_t) else {
            return;
        };
        let Ok(stored_v) = slice_axis2(v, 0, target_t) else {
            return;
        };
        if let Ok(compact_k) = slice_update_axis2(&empty_k, &stored_k, 0, target_t) {
            if let Ok(compact_v) = slice_update_axis2(&empty_v, &stored_v, 0, target_t) {
                // Force materialization so the resulting Array owns its
                // storage instead of staying lazy (which would defer the
                // copy and the live buffer would stay pinned).
                let _ = eval(&[compact_k.clone(), compact_v.clone()]);
                self.keys = Some(compact_k);
                self.values = Some(compact_v);
            }
        }
    }

    fn dim(shape: &[i32], i: usize, label: &'static str) -> Result<i32, Exception> {
        shape
            .get(i)
            .copied()
            .ok_or_else(|| Exception::custom(format!("KV cache: missing dim {i} ({label})")))
    }

    /// Rewrite a rotated ring buffer back into chronological order and leave
    /// ring mode. Costs one copy, and only runs on the rare paths that need
    /// positional meaning back: a multi-token write landing on a cache that
    /// already decoded, `trim_by`, and taking a prefix-cache snapshot.
    fn unrotate(&mut self) -> Result<(), Exception> {
        let Some(head) = self.ring_head.take() else {
            return Ok(());
        };
        let len = self.stored_len;
        if head <= 0 || head >= len {
            // Head at slot 0 means the buffer is already [oldest … newest].
            return Ok(());
        }
        let (Some(k), Some(v)) = (self.keys.as_ref(), self.values.as_ref()) else {
            return Ok(());
        };
        let k2 = concatenate_axis(&[slice_axis2(k, head, len)?, slice_axis2(k, 0, head)?], 2)?;
        let v2 = concatenate_axis(&[slice_axis2(v, head, len)?, slice_axis2(v, 0, head)?], 2)?;
        self.keys = Some(k2);
        self.values = Some(v2);
        Ok(())
    }

    /// A clone that is safe to hand to the prefix cache: rotated storage is
    /// linearized first, since a snapshot is replayed as a *positional* prefix.
    pub(crate) fn snapshot_clone(&self) -> Self {
        let mut c = self.clone();
        let _ = c.unrotate();
        c
    }

    /// First evicting decode write: build the `w`-row ring out of the last
    /// `w - 1` cached keys plus this one, and park the head on slot 0 (the
    /// oldest key, hence the next slot to be overwritten). One copy, once.
    fn ring_enter(
        &mut self,
        keys: &Array,
        values: &Array,
        w: i32,
    ) -> Result<(Array, Array), Exception> {
        let keep = w - 1;
        let (ring_k, ring_v) = if keep <= 0 {
            (keys.clone(), values.clone())
        } else {
            let k = self
                .keys
                .as_ref()
                .ok_or_else(|| Exception::custom("ring_enter: keys missing"))?;
            let v = self
                .values
                .as_ref()
                .ok_or_else(|| Exception::custom("ring_enter: values missing"))?;
            let start = self.stored_len - keep;
            (
                concatenate_axis(&[slice_axis2(k, start, self.stored_len)?, keys.clone()], 2)?,
                concatenate_axis(
                    &[slice_axis2(v, start, self.stored_len)?, values.clone()],
                    2,
                )?,
            )
        };
        self.keys = Some(ring_k.clone());
        self.values = Some(ring_v.clone());
        self.stored_len = w;
        self.ring_head = Some(0);
        Ok((ring_k, ring_v))
    }

    /// Steady-state decode write: overwrite the oldest slot in place, advance
    /// the head. No eviction slice, no grow, no `concatenate` — the whole point
    /// of the ring.
    fn ring_write(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array), Exception> {
        let w = self.stored_len.max(1);
        let head = self.ring_head.unwrap_or(0).rem_euclid(w);
        let k_buf = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("ring_write: keys missing"))?;
        let v_buf = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("ring_write: values missing"))?;
        let k = slice_update_axis2(k_buf, keys, head, 1)?;
        let v = slice_update_axis2(v_buf, values, head, 1)?;
        self.keys = Some(k.clone());
        self.values = Some(v.clone());
        self.ring_head = Some((head + 1) % w);
        Ok((k, v))
    }

    fn update_dense(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array), Exception> {
        let k_shape = keys.shape();
        let v_shape = values.shape();
        let new_tokens = Self::dim(k_shape, 2, "keys T")?;

        // ── Sliding-window ring ──────────────────────────────────────────
        // Only single-token (decode) writes rotate. A multi-token write still
        // means prefill, which needs chronological storage, so a rotated buffer
        // is linearized first and then falls through to the append path below.
        if let Some(w) = self.decode_window {
            if new_tokens == 1 && self.keys.is_some() {
                if self.ring_head.is_some() {
                    return self.ring_write(keys, values);
                }
                if self.stored_len >= w {
                    return self.ring_enter(keys, values, w);
                }
            } else if self.ring_head.is_some() {
                self.unrotate()?;
            }
        }

        // Decode-time sliding window: a single-token (decode) write evicts down
        // to `decode_window`; multi-token prefill writes use the normal
        // `max_seq_len` cap so every query keeps its full window. See the field
        // doc on `decode_window`.
        let max_cap = match self.decode_window {
            Some(w) if new_tokens == 1 => Some(w),
            _ => self.max_seq_len,
        };
        let target_stored = match max_cap {
            Some(m) => (self.stored_len + new_tokens).min(m),
            None => self.stored_len + new_tokens,
        };

        let drop = (self.stored_len + new_tokens - target_stored).max(0);
        if drop > 0 {
            if let (Some(k), Some(v)) = (&self.keys, &self.values) {
                self.keys = Some(slice_axis2(k, drop, self.stored_len)?);
                self.values = Some(slice_axis2(v, drop, self.stored_len)?);
            }
            self.stored_len -= drop;
        }

        let write_pos = self.stored_len;
        let required_slots = write_pos + new_tokens;

        // Capacity of the existing K/V buffer along the time axis. Differs
        // from `stored_len` when the buffer was pre-allocated past current
        // tokens (max_cap path, first-alloc preallocation) OR when the buffer
        // was sliced smaller (prefix-cache restore via `trim_by` sets
        // shape[2] == stored_len). We always check the *real* shape so a
        // restored snapshot can still grow into more writes.
        //
        // Pre-fix: `need_grow = need_alloc` for max_cap mode → restored
        // snapshot's smaller buffer was never grown, `slice_update_axis2`
        // wrote past the end and mlx returned shape `(...,0,...)` causing
        // `broadcast_shapes` error mid-prefill.
        let cap_now = match self.keys.as_ref() {
            Some(k) => Self::dim(k.shape(), 2, "cached keys T")?,
            None => 0,
        };
        let need_grow = cap_now < required_slots;

        if need_grow {
            let b = Self::dim(k_shape, 0, "keys B")?;
            let n_kv_heads = Self::dim(k_shape, 1, "keys H")?;
            let k_head_dim = Self::dim(k_shape, 3, "keys D")?;
            let v_head_dim = Self::dim(v_shape, 3, "values D")?;

            // Total post-grow buffer size along the time axis.
            //
            // **Hybrid grow** (max_cap mode) — pure doubling overshoots near
            // `max_cap`: e.g. cap=16 384, required=16 896 → double to
            // 32 768 → capped at m=32 000 (the full sliding window), wasting
            // 15 K slots ≈ 1.8 GB when the prompt only needs ~17 K. So:
            //   - **Below 8 K** doubling (rapid amortized grow during early
            //     prefill chunks; ~6 grows to reach 8 K).
            //   - **At or above 8 K** linear `+2 K` per grow (tight cap so
            //     the final buffer is close to actual stored_len, not 2×).
            //
            // For a 14 K prefill: cap progression 512→1024→2048→4096→8192→
            // 10240→12288→14336→16384 (9 grows total, final ≈ 16 K instead
            // of 32 K). Saves ~50 % live KV RAM when prompt ≪ max_cap.
            // Pre-existing 2× overshoot kept for small caps where the
            // absolute waste is negligible (a few hundred MB).
            const DOUBLE_THRESHOLD: i32 = 8192;
            const LINEAR_GROW_STEP: i32 = 2048;
            let new_cap = match max_cap {
                Some(m) => {
                    let target = if cap_now < DOUBLE_THRESHOLD {
                        cap_now.saturating_mul(2).max(required_slots).max(self.step)
                    } else {
                        cap_now.saturating_add(LINEAR_GROW_STEP).max(required_slots)
                    };
                    target.min(m)
                }
                None => {
                    let n_steps = (self.step + new_tokens - 1) / self.step;
                    let grow = n_steps * self.step;
                    (cap_now + grow).max(required_slots)
                }
            };

            let (grown_k, grown_v) = match (self.keys.take(), self.values.take()) {
                (Some(old_k), Some(old_v)) => {
                    // Trim existing buffer to stored_len (strips zero padding
                    // past the live data; for restored snapshots cap_now ==
                    // stored_len, so this is a no-op identity slice).
                    let trimmed_k = if cap_now > self.stored_len {
                        slice_axis2(&old_k, 0, self.stored_len)?
                    } else {
                        old_k
                    };
                    let trimmed_v = if cap_now > self.stored_len {
                        slice_axis2(&old_v, 0, self.stored_len)?
                    } else {
                        old_v
                    };
                    // Pad up to new_cap from stored_len (NOT from cap_now —
                    // we already stripped padding above).
                    let pad_slots = (new_cap - self.stored_len).max(0);
                    let pad_k = zeros_dtype(&[b, n_kv_heads, pad_slots, k_head_dim], keys.dtype())?;
                    let pad_v =
                        zeros_dtype(&[b, n_kv_heads, pad_slots, v_head_dim], values.dtype())?;
                    (
                        concatenate_axis(&[trimmed_k, pad_k], 2)?,
                        concatenate_axis(&[trimmed_v, pad_v], 2)?,
                    )
                }
                _ => {
                    // First-alloc — no existing buffer to keep.
                    let fresh_k = zeros_dtype(&[b, n_kv_heads, new_cap, k_head_dim], keys.dtype())?;
                    let fresh_v =
                        zeros_dtype(&[b, n_kv_heads, new_cap, v_head_dim], values.dtype())?;
                    (fresh_k, fresh_v)
                }
            };
            self.keys = Some(grown_k);
            self.values = Some(grown_v);
        }

        let k_buf = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("Keys cannot be None after grow"))?;
        let v_buf = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("Values cannot be None after grow"))?;

        self.keys = Some(slice_update_axis2(k_buf, keys, write_pos, new_tokens)?);
        self.values = Some(slice_update_axis2(v_buf, values, write_pos, new_tokens)?);

        self.stored_len += new_tokens;

        let result_k = slice_axis2(
            self.keys
                .as_ref()
                .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?,
            0,
            self.stored_len,
        )?;
        let result_v = slice_axis2(
            self.values
                .as_ref()
                .ok_or_else(|| Exception::custom("Values cannot be None after update"))?,
            0,
            self.stored_len,
        )?;

        Ok((result_k, result_v))
    }
}

impl KeyValueCache for SteppingKeyValueCache {
    fn stored_len(&self) -> i32 {
        self.stored_len
    }

    fn max_size(&self) -> Option<i32> {
        self.max_seq_len
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<KvFetchResult, Exception> {
        let (k, v) = self.update_dense(&keys, &values)?;
        let (k, v) = materialize_pair(k, v)?;
        Ok(KvFetchResult::Fp16(k, v))
    }
}

/// TurboQuant KV per layer (one `QuantizedKVCache` with `num_layers = 1`).
///
/// Before `activate_at` tokens (sum of `stored_len` updates), uses FP16 [`SteppingKeyValueCache`].
/// After activation, pushes packed TQ blocks and serves attention via turboquant-rs.
pub struct TurboQuantKeyValueCache {
    staging: SteppingKeyValueCache,
    tq: QuantizedKVCache,
    active: bool,
    activate_at: i32,
    bits: u8,
    head_dim: i32,
    n_kv_heads: i32,
}

impl std::fmt::Debug for TurboQuantKeyValueCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurboQuantKeyValueCache")
            .field("active", &self.active)
            .field("activate_at", &self.activate_at)
            .field("bits", &self.bits)
            .field("stored_len", &self.stored_len())
            .finish()
    }
}

impl TurboQuantKeyValueCache {
    pub fn with_max(
        bits: u8,
        head_dim: i32,
        n_kv_heads: i32,
        max_seq_len: i32,
        activate_at: i32,
    ) -> Self {
        let bits = normalize_turboquant_bits(bits);
        let config = TurboQuantConfig::new(bits, head_dim as usize)
            .expect("TurboQuantConfig::new validated at runtime");
        Self {
            staging: SteppingKeyValueCache::with_max(max_seq_len),
            tq: QuantizedKVCache::new(config, 1, next_tq_seed()),
            active: false,
            activate_at: activate_at.max(0),
            bits,
            head_dim,
            n_kv_heads,
        }
    }

    pub fn tq(&self) -> &QuantizedKVCache {
        &self.tq
    }

    pub fn head_dim(&self) -> i32 {
        self.head_dim
    }

    pub fn is_turbo_active(&self) -> bool {
        self.active
    }

    /// Clone just the FP16 staging half — used by [`KvCache::try_snapshot`]
    /// before TurboQuant activates so prefix caching still works for the
    /// `prompt_len < tq_activate_at` path (the common case after my default
    /// bump to 16384). Once TQ activates, snapshots are skipped because
    /// `QuantizedKVCache` is not `Clone`.
    pub(crate) fn staging_clone(&self) -> SteppingKeyValueCache {
        self.staging.clone()
    }

    fn tokens_in_tq(&self) -> i32 {
        if !self.active {
            return 0;
        }
        let entries = self.tq.entry_count(0);
        (entries / self.n_kv_heads as usize) as i32
    }

    fn maybe_activate(&mut self) -> Result<(), Exception> {
        if self.active {
            return Ok(());
        }
        if self.staging.stored_len < self.activate_at {
            return Ok(());
        }
        if let (Some(k), Some(v)) = (self.staging.keys.as_ref(), self.staging.values.as_ref()) {
            let k = slice_axis2(k, 0, self.staging.stored_len)?;
            let v = slice_axis2(v, 0, self.staging.stored_len)?;
            push_kv_arrays(&mut self.tq, 0, &k, &v, self.n_kv_heads)?;
        }
        self.staging.keys = None;
        self.staging.values = None;
        self.staging.stored_len = 0;
        self.active = true;
        Ok(())
    }

    fn trim_tq_if_needed(&mut self) -> Result<(), Exception> {
        let Some(max) = self.staging.max_seq_len else {
            return Ok(());
        };
        let max = max as usize;
        let n_h = self.n_kv_heads as usize;
        let n = self.tq.entry_count(0);
        let tokens = n / n_h;
        if tokens <= max {
            return Ok(());
        }
        let drop_tokens = tokens - max;
        let drop_entries = drop_tokens * n_h;
        let keys = self
            .tq
            .dequantize_keys_range(0, drop_entries, n)
            .map_err(|e| Exception::custom(format!("tq trim keys: {e}")))?;
        let vals = self
            .tq
            .dequantize_values_range(0, drop_entries, n)
            .map_err(|e| Exception::custom(format!("tq trim values: {e}")))?;
        let config = TurboQuantConfig::new(self.bits, self.head_dim as usize)
            .map_err(|e| Exception::custom(format!("tq trim TurboQuantConfig::new: {e}")))?;
        let seed = self.tq.qjl_seed();
        let mut fresh = QuantizedKVCache::new(config, 1, seed);
        let key_refs: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let val_refs: Vec<&[f32]> = vals.iter().map(|v| v.as_slice()).collect();
        fresh
            .push_batch(0, &key_refs, &val_refs)
            .map_err(|e| Exception::custom(format!("tq trim re-push: {e}")))?;
        self.tq = fresh;
        Ok(())
    }

    pub(crate) fn eval_targets(&self) -> Vec<Array> {
        if self.active {
            Vec::new()
        } else {
            self.staging.eval_targets()
        }
    }

    /// Per-layer bytes. While staging (pre-activation) the storage matches
    /// FP16 KV; once active the storage is TQ packed at `bits` bits per
    /// element (plus small header / scale overhead, ignored here — rough
    /// estimate only). Used for `[mem] kv cache:` log lines.
    pub(crate) fn approx_bytes(&self) -> usize {
        if !self.active {
            return self.staging.approx_bytes();
        }
        let entries = self.tq.entry_count(0);
        // bits per element, K + V = 2× the per-element cost
        let bytes_per_elem = self.head_dim as usize * self.bits as usize / 8;
        // entries are (tokens × heads); each entry holds head_dim packed elems
        // for one of K or V (entry_count already counts K and V separately).
        entries * bytes_per_elem
    }
}

impl KeyValueCache for TurboQuantKeyValueCache {
    fn is_quantized(&self) -> bool {
        self.active
    }

    fn bits(&self) -> Option<i32> {
        self.active.then_some(self.bits as i32)
    }

    fn stored_len(&self) -> i32 {
        if self.active {
            self.tokens_in_tq()
        } else {
            self.staging.stored_len()
        }
    }

    fn max_size(&self) -> Option<i32> {
        self.staging.max_seq_len
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<KvFetchResult, Exception> {
        if !self.active {
            let out = self
                .staging
                .update_and_fetch(keys.clone(), values.clone())?;
            let KvFetchResult::Fp16(k, v) = out else {
                return Err(Exception::custom("staging must return FP16"));
            };
            self.maybe_activate()?;
            if self.active {
                push_kv_arrays(&mut self.tq, 0, &k, &v, self.n_kv_heads)?;
                self.trim_tq_if_needed()?;
                return Ok(KvFetchResult::TurboQuant);
            }
            return Ok(KvFetchResult::Fp16(k, v));
        }
        push_kv_arrays(&mut self.tq, 0, &keys, &values, self.n_kv_heads)?;
        self.trim_tq_if_needed()?;
        Ok(KvFetchResult::TurboQuant)
    }

    fn turboquant_attention(
        &mut self,
        queries: &Array,
        scale: f32,
        mask: Option<&Array>,
        n_heads: i32,
        n_kv_heads: i32,
    ) -> Result<Option<Array>, Exception> {
        if !self.active {
            return Ok(None);
        }
        super::utils::turboquant_attn::turboquant_gqa_attention(
            queries, self, scale, mask, n_heads, n_kv_heads,
        )
        .map(Some)
    }
}

fn push_kv_arrays(
    tq: &mut QuantizedKVCache,
    layer: usize,
    keys: &Array,
    values: &Array,
    n_kv_heads: i32,
) -> Result<(), Exception> {
    eval(&[keys.clone(), values.clone()])?;
    let k = keys.as_dtype(Dtype::Float32)?;
    let v = values.as_dtype(Dtype::Float32)?;
    let sh = k.shape();
    if sh.len() != 4 {
        return Err(Exception::custom(
            "push_kv_arrays: keys must be 4D [B,H,T,D]",
        ));
    }
    let t = sh[2] as usize;
    let h = n_kv_heads as usize;
    let d = sh[3] as usize;
    let k_flat = k.as_slice::<f32>();
    let v_flat = v.as_slice::<f32>();
    let mut key_bufs = Vec::with_capacity(t * h);
    let mut val_bufs = Vec::with_capacity(t * h);
    for ti in 0..t {
        for hi in 0..h {
            let start = (hi * t + ti) * d;
            key_bufs.push(k_flat[start..start + d].to_vec());
            val_bufs.push(v_flat[start..start + d].to_vec());
        }
    }
    let key_refs: Vec<&[f32]> = key_bufs.iter().map(|s| s.as_slice()).collect();
    let val_refs: Vec<&[f32]> = val_bufs.iter().map(|s| s.as_slice()).collect();
    tq.push_batch(layer, &key_refs, &val_refs)
        .map_err(|e| Exception::custom(format!("turboquant push_batch: {e}")))
}

fn slice_axis2(arr: &Array, start: i32, end: i32) -> Result<Array, Exception> {
    Ok(arr.index((.., .., start..end, ..)))
}

fn slice_update_axis2(
    target: &Array,
    update: &Array,
    start: i32,
    n: i32,
) -> Result<Array, Exception> {
    let mut out = target.clone();
    out.try_index_mut((.., .., start..start + n, ..), update.clone())?;
    Ok(out)
}

/// Mamba-2 per-layer recurrent state.
///
/// Two pieces of state are carried across timesteps (mirroring `mlx-lm`'s
/// `mamba2.py` reference):
///
/// - **conv_state**: rolling window into the depthwise short conv, shape
///   `[B, d_conv - 1, conv_dim]` where `conv_dim = d_inner + 2 * n_groups * d_state`.
///   Stored channels-last to match `mlx_rs::nn::Conv1d` NLC layout, so the block
///   can `concatenate_axis(&[conv_state, xBC_token], 1)` directly.
/// - **ssm_state**: per-head SSM hidden state, shape `[B, n_heads, head_dim, d_state]`.
///   On the very first prefill call the cache is zero-initialised lazily so the
///   block can support arbitrary batch sizes without re-allocating up front.
///
/// `tokens_seen` tracks total absolute position so callers can advance positional
/// state if needed (mirrors `RopeInput::offset` semantics for SSM layers).
#[derive(Debug, Clone)]
pub struct Mamba2Cache {
    pub conv_dim: i32,
    pub d_conv: i32,
    pub n_heads: i32,
    pub head_dim: i32,
    pub d_state: i32,
    conv_state: Option<Array>,
    ssm_state: Option<Array>,
    tokens_seen: i32,
}

impl Mamba2Cache {
    pub fn new(conv_dim: i32, d_conv: i32, n_heads: i32, head_dim: i32, d_state: i32) -> Self {
        Self {
            conv_dim,
            d_conv,
            n_heads,
            head_dim,
            d_state,
            conv_state: None,
            ssm_state: None,
            tokens_seen: 0,
        }
    }

    pub fn tokens_seen(&self) -> i32 {
        self.tokens_seen
    }

    pub fn advance(&mut self, n: i32) {
        self.tokens_seen = self.tokens_seen.saturating_add(n);
    }

    /// Channels-last conv state of length `d_conv - 1`, lazily zero-initialised.
    pub fn conv_state_or_init(&mut self, batch: i32, dtype: Dtype) -> Result<&Array, Exception> {
        if self.conv_state.is_none() {
            let pad = (self.d_conv - 1).max(0);
            self.conv_state = Some(zeros_dtype(&[batch, pad, self.conv_dim], dtype)?);
        }
        Ok(self
            .conv_state
            .as_ref()
            .expect("conv_state initialised above"))
    }

    pub fn set_conv_state(&mut self, state: Array) {
        self.conv_state = Some(state);
    }

    /// Per-head SSM hidden state `[B, n_heads, head_dim, d_state]`, lazily init.
    pub fn ssm_state_or_init(&mut self, batch: i32, dtype: Dtype) -> Result<&Array, Exception> {
        if self.ssm_state.is_none() {
            self.ssm_state = Some(zeros_dtype(
                &[batch, self.n_heads, self.head_dim, self.d_state],
                dtype,
            )?);
        }
        Ok(self
            .ssm_state
            .as_ref()
            .expect("ssm_state initialised above"))
    }

    pub fn set_ssm_state(&mut self, state: Array) {
        self.ssm_state = Some(state);
    }

    pub(crate) fn eval_targets(&self) -> Vec<Array> {
        let mut out = Vec::with_capacity(2);
        if let Some(c) = &self.conv_state {
            out.push(c.clone());
        }
        if let Some(s) = &self.ssm_state {
            out.push(s.clone());
        }
        out
    }

    pub(crate) fn approx_bytes(&self) -> usize {
        fn array_bytes(a: &Array) -> usize {
            let n: usize = a.shape().iter().map(|&d| d.max(0) as usize).product();
            n * 2
        }
        let mut total = 0;
        if let Some(c) = &self.conv_state {
            total += array_bytes(c);
        }
        if let Some(s) = &self.ssm_state {
            total += array_bytes(s);
        }
        total
    }
}

/// Mamba-1 per-layer recurrent state.
///
/// Two pieces of state are carried across timesteps, matching the reference
/// `mlx-lm` `mamba.py` (`ArraysCache(size=2)` slots):
///
/// - **conv_state**: rolling window into the depthwise short conv,
///   shape `[B, d_conv - 1, d_inner]`. Stored channels-last to match the
///   `mlx_rs::nn::Conv1d` NLC layout so the block can
///   `concatenate_axis(&[conv_state, x_inner_token], 1)` directly.
/// - **ssm_state**: per-channel SSM hidden state, shape `[B, d_inner, d_state]`.
///   Unlike Mamba-2, Mamba-1 has no head/group structure — every channel of
///   `d_inner` carries its own `d_state`-wide hidden state.
///
/// Both slots are lazily zero-initialised on the first prefill call so the
/// block can support arbitrary batch sizes without re-allocating up front.
/// `tokens_seen` tracks total absolute position (mirrors [`Mamba2Cache`]).
#[derive(Debug, Clone)]
pub struct Mamba1Cache {
    pub d_inner: i32,
    pub d_conv: i32,
    pub d_state: i32,
    conv_state: Option<Array>,
    ssm_state: Option<Array>,
    tokens_seen: i32,
}

impl Mamba1Cache {
    pub fn new(d_inner: i32, d_conv: i32, d_state: i32) -> Self {
        Self {
            d_inner,
            d_conv,
            d_state,
            conv_state: None,
            ssm_state: None,
            tokens_seen: 0,
        }
    }

    pub fn tokens_seen(&self) -> i32 {
        self.tokens_seen
    }

    pub fn advance(&mut self, n: i32) {
        self.tokens_seen = self.tokens_seen.saturating_add(n);
    }

    /// Channels-last conv state of length `d_conv - 1`, lazily zero-initialised.
    pub fn conv_state_or_init(&mut self, batch: i32, dtype: Dtype) -> Result<&Array, Exception> {
        if self.conv_state.is_none() {
            let pad = (self.d_conv - 1).max(0);
            self.conv_state = Some(zeros_dtype(&[batch, pad, self.d_inner], dtype)?);
        }
        Ok(self
            .conv_state
            .as_ref()
            .expect("conv_state initialised above"))
    }

    pub fn set_conv_state(&mut self, state: Array) {
        self.conv_state = Some(state);
    }

    /// Has the SSM hidden state been populated yet (i.e. seen any tokens)?
    pub fn has_ssm_state(&self) -> bool {
        self.ssm_state.is_some()
    }

    /// Per-channel SSM hidden state `[B, d_inner, d_state]`. Returns `None` until
    /// the first token has been processed — Mamba-1's recurrence skips the
    /// `state * exp(dt * A)` decay term on the very first step (matching
    /// `state is not None` in the Python reference).
    pub fn ssm_state(&self) -> Option<&Array> {
        self.ssm_state.as_ref()
    }

    pub fn set_ssm_state(&mut self, state: Array) {
        self.ssm_state = Some(state);
    }

    pub(crate) fn eval_targets(&self) -> Vec<Array> {
        let mut out = Vec::with_capacity(2);
        if let Some(c) = &self.conv_state {
            out.push(c.clone());
        }
        if let Some(s) = &self.ssm_state {
            out.push(s.clone());
        }
        out
    }

    pub(crate) fn approx_bytes(&self) -> usize {
        fn array_bytes(a: &Array) -> usize {
            let n: usize = a.shape().iter().map(|&d| d.max(0) as usize).product();
            n * 2
        }
        let mut total = 0;
        if let Some(c) = &self.conv_state {
            total += array_bytes(c);
        }
        if let Some(s) = &self.ssm_state {
            total += array_bytes(s);
        }
        total
    }
}

/// Qwen3.5 GatedDeltaNet recurrent state (conv window + linear SSM).
#[derive(Debug, Clone)]
pub struct Qwen35LinearCache {
    pub conv_dim: i32,
    pub d_conv: i32,
    pub n_v_heads: i32,
    pub d_v: i32,
    pub d_k: i32,
    conv_state: Option<Array>,
    ssm_state: Option<Array>,
    tokens_seen: i32,
}

impl Qwen35LinearCache {
    pub fn new(conv_dim: i32, d_conv: i32, n_v_heads: i32, d_v: i32, d_k: i32) -> Self {
        Self {
            conv_dim,
            d_conv,
            n_v_heads,
            d_v,
            d_k,
            conv_state: None,
            ssm_state: None,
            tokens_seen: 0,
        }
    }

    pub fn tokens_seen(&self) -> i32 {
        self.tokens_seen
    }

    pub fn advance(&mut self, n: i32) {
        self.tokens_seen = self.tokens_seen.saturating_add(n);
    }

    pub fn conv_state_or_init(&mut self, batch: i32, dtype: Dtype) -> Result<&Array, Exception> {
        if self.conv_state.is_none() {
            let pad = (self.d_conv - 1).max(0);
            self.conv_state = Some(zeros_dtype(&[batch, pad, self.conv_dim], dtype)?);
        }
        Ok(self.conv_state.as_ref().expect("conv_state"))
    }

    pub fn set_conv_state(&mut self, state: Array) {
        self.conv_state = Some(state);
    }

    pub fn ssm_state_or_init(&mut self, batch: i32) -> Result<&Array, Exception> {
        if self.ssm_state.is_none() {
            self.ssm_state = Some(zeros_dtype(
                &[batch, self.n_v_heads, self.d_v, self.d_k],
                Dtype::Float32,
            )?);
        }
        Ok(self.ssm_state.as_ref().expect("ssm_state"))
    }

    pub fn set_ssm_state(&mut self, state: Array) {
        self.ssm_state = Some(state);
    }

    pub(crate) fn eval_targets(&self) -> Vec<Array> {
        let mut out = Vec::with_capacity(2);
        if let Some(c) = &self.conv_state {
            out.push(c.clone());
        }
        if let Some(s) = &self.ssm_state {
            out.push(s.clone());
        }
        out
    }

    pub(crate) fn approx_bytes(&self) -> usize {
        fn array_bytes(a: &Array) -> usize {
            let n: usize = a.shape().iter().map(|&d| d.max(0) as usize).product();
            // ssm_state for Qwen3.5 is FP32 (4 B); conv is FP16. Approximate as 2 B avg.
            n * 2
        }
        let mut total = 0;
        if let Some(c) = &self.conv_state {
            total += array_bytes(c);
        }
        if let Some(s) = &self.ssm_state {
            total += array_bytes(s);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::{array, Array};

    fn make_kv(seq_len: i32, fill: f32) -> (Array, Array) {
        let t = Array::full::<f32>(&[1, 1, seq_len, 1], array!(fill)).unwrap();
        (t.clone(), t)
    }

    /// The live keys along the time axis, in **storage** order (rotated once
    /// the ring engages).
    fn time_axis(cache: &SteppingKeyValueCache) -> Vec<f32> {
        let keys = cache.keys.as_ref().expect("keys");
        (0..cache.stored_len())
            .map(|i| keys.index((.., .., i, ..)).item::<f32>())
            .collect()
    }

    fn sorted(mut v: Vec<f32>) -> Vec<f32> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    #[test]
    fn stepping_window_evicts() {
        let mut cache = SteppingKeyValueCache::with_max(2);
        for step in 0..4 {
            let (k, v) = make_kv(1, step as f32);
            let _ = cache.update_and_fetch(k, v).unwrap();
        }
        assert_eq!(cache.stored_len(), 2);
        let keys = cache.keys.as_ref().unwrap();
        assert_eq!(keys.index((.., .., 0, ..)).item::<f32>(), 2.0);
        assert_eq!(keys.index((.., .., 1, ..)).item::<f32>(), 3.0);
    }

    /// Decode-windowed cache (sliding-window attention layers): a **multi-token
    /// prefill** write is retained in full — every prefill query keeps its
    /// window — while **single-token decode** writes evict down to the window.
    /// This is the invariant that lets Gemma-4 bound decode-phase sliding KV to
    /// `sliding_window` without changing the (already windowed) attention math.
    #[test]
    fn decode_window_evicts_only_on_single_token_writes() {
        let window = 4;
        let mut cache = SteppingKeyValueCache::with_decode_window(64_000, window);

        // Prefill: one 10-token write, values 0..10 along the time axis. A
        // multi-token write ignores the decode window, so all 10 are kept.
        let prefill: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let pk = Array::from_slice(&prefill, &[1, 1, 10, 1]);
        let _ = cache.update_and_fetch(pk.clone(), pk).unwrap();
        assert_eq!(
            cache.stored_len(),
            10,
            "multi-token prefill write must not be windowed"
        );

        // Decode: single-token writes 10, 11, 12 — each evicts to the window.
        for v in 10..13 {
            let (k, vv) = make_kv(1, v as f32);
            let _ = cache.update_and_fetch(k, vv).unwrap();
        }
        assert_eq!(
            cache.stored_len(),
            window,
            "single-token decode writes cap at the window"
        );
        // Retained = most recent `window` keys: 9 (last prefill token), 10, 11,
        // 12 — as a *set*. Storage order is rotated once the ring engages (see
        // `ring_rotates_in_place_and_holds_the_window`), and attention is
        // permutation-invariant along the key axis, so order is not asserted
        // here. `snapshot_clone` is what restores chronological order.
        assert_eq!(sorted(time_axis(&cache)), vec![9.0, 10.0, 11.0, 12.0]);
    }

    /// Steady-state decode on a sliding-window layer must overwrite one ring
    /// slot per token: the physical buffer stays exactly `window` rows, the
    /// head advances modulo the window, and the retained *set* is always the
    /// last `window` keys — even though their storage order is rotated.
    ///
    /// The rotation is the optimization. The pre-ring path evicted with a tail
    /// slice, which left the buffer one row short of the required slots and so
    /// re-entered the grow branch (trim + `zeros` + `concatenate`) on *every*
    /// token past the window.
    #[test]
    fn ring_rotates_in_place_and_holds_the_window() {
        let window = 4;
        let mut cache = SteppingKeyValueCache::with_decode_window(64_000, window);

        let prefill: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let pk = Array::from_slice(&prefill, &[1, 1, 10, 1]);
        let _ = cache.update_and_fetch(pk.clone(), pk).unwrap();
        assert!(cache.ring_head.is_none(), "prefill must not rotate");

        for v in 10..30 {
            let (k, vv) = make_kv(1, v as f32);
            let _ = cache.update_and_fetch(k, vv).unwrap();

            assert_eq!(cache.stored_len(), window);
            assert_eq!(
                cache.keys.as_ref().unwrap().shape()[2],
                window,
                "ring must never grow the physical buffer"
            );
            assert!(cache.ring_head.is_some(), "decode past the window rotates");

            let expected: Vec<f32> = ((v - window + 1)..=v).map(|i| i as f32).collect();
            assert_eq!(
                sorted(time_axis(&cache)),
                expected,
                "ring set after writing {v}"
            );
        }

        // `ring_enter` fired on the first decode token (10) and parked the head
        // at 0; tokens 11..=29 are 19 in-place writes, so the head has wrapped
        // to 19 % 4.
        assert_eq!(cache.ring_head, Some(3));
    }

    /// A snapshot is replayed as a *positional* prefix, so the prefix cache
    /// must never see a rotated buffer. `snapshot_clone` linearizes; the live
    /// cache keeps its rotation.
    #[test]
    fn snapshot_clone_unrotates_the_ring() {
        let window = 4;
        let mut cache = SteppingKeyValueCache::with_decode_window(64_000, window);
        let prefill: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let pk = Array::from_slice(&prefill, &[1, 1, 6, 1]);
        let _ = cache.update_and_fetch(pk.clone(), pk).unwrap();
        for v in 6..9 {
            let (k, vv) = make_kv(1, v as f32);
            let _ = cache.update_and_fetch(k, vv).unwrap();
        }
        assert!(cache.ring_head.is_some());

        let snap = cache.snapshot_clone();
        assert!(snap.ring_head.is_none(), "snapshot leaves ring mode");
        assert_eq!(
            sorted(time_axis(&snap)),
            vec![5.0, 6.0, 7.0, 8.0],
            "same keys"
        );
        assert_eq!(
            time_axis(&snap),
            vec![5.0, 6.0, 7.0, 8.0],
            "in chronological order"
        );
        assert!(
            cache.ring_head.is_some(),
            "taking a snapshot must not disturb the live cache"
        );
    }

    /// A multi-token write means prefill, which needs chronological storage.
    /// Landing one on a cache that already decoded must linearize first, then
    /// append — not interleave new keys into ring slots.
    #[test]
    fn multi_token_write_after_ring_linearizes_first() {
        let window = 4;
        let mut cache = SteppingKeyValueCache::with_decode_window(64_000, window);
        let prefill: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let pk = Array::from_slice(&prefill, &[1, 1, 6, 1]);
        let _ = cache.update_and_fetch(pk.clone(), pk).unwrap();
        for v in 6..9 {
            let (k, vv) = make_kv(1, v as f32);
            let _ = cache.update_and_fetch(k, vv).unwrap();
        }
        assert!(cache.ring_head.is_some());

        let more = Array::from_slice(&[9.0_f32, 10.0], &[1, 1, 2, 1]);
        let _ = cache.update_and_fetch(more.clone(), more).unwrap();

        assert!(
            cache.ring_head.is_none(),
            "multi-token write leaves the ring"
        );
        assert_eq!(cache.stored_len(), 6, "4 windowed keys + 2 appended");
        assert_eq!(
            time_axis(&cache),
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "chronological after linearize + append"
        );
    }

    /// `trim_by` drops the *newest* n tokens, which is a positional claim, so a
    /// rotated buffer has to be linearized before the tail slice means anything.
    #[test]
    fn trim_by_after_ring_drops_the_newest_keys() {
        let window = 4;
        let mut cache = SteppingKeyValueCache::with_decode_window(64_000, window);
        let prefill: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let pk = Array::from_slice(&prefill, &[1, 1, 6, 1]);
        let _ = cache.update_and_fetch(pk.clone(), pk).unwrap();
        for v in 6..9 {
            let (k, vv) = make_kv(1, v as f32);
            let _ = cache.update_and_fetch(k, vv).unwrap();
        }
        // Ring holds {5,6,7,8} rotated.
        cache.trim_by(2);
        assert!(cache.ring_head.is_none());
        assert_eq!(cache.stored_len(), 2);
        assert_eq!(time_axis(&cache), vec![5.0, 6.0]);
    }

    /// Degenerate window of 1: the ring is a single slot that every decode
    /// token overwrites.
    #[test]
    fn ring_window_of_one_keeps_only_the_newest_key() {
        let mut cache = SteppingKeyValueCache::with_decode_window(64_000, 1);
        let prefill: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let pk = Array::from_slice(&prefill, &[1, 1, 4, 1]);
        let _ = cache.update_and_fetch(pk.clone(), pk).unwrap();
        for v in 4..8 {
            let (k, vv) = make_kv(1, v as f32);
            let _ = cache.update_and_fetch(k, vv).unwrap();
            assert_eq!(cache.stored_len(), 1);
            assert_eq!(time_axis(&cache), vec![v as f32]);
        }
    }

    /// Buffer capacity should double per grow event, capped at `max_cap`.
    /// Verifies the new lazy-allocation behaviour vs. the pre-fix
    /// "preallocate full max" pattern.
    #[test]
    fn doubling_grow_bounded_by_max_cap() {
        let mut cache = SteppingKeyValueCache::with_max(10000);
        // Write 512 tokens — cap should be 512 (one step worth), NOT 10000.
        let (k, v) = make_kv(512, 1.0);
        let _ = cache.update_and_fetch(k, v).unwrap();
        let keys = cache.keys.as_ref().unwrap();
        let cap = keys.shape()[2];
        assert!(
            cap < 10000,
            "expected lazy growth, got cap={cap} (should be << max_cap=10000)"
        );
        assert!(cap >= 512, "cap must be at least required (512), got {cap}");
        // Write another 512 — cap should at most double or grow to required.
        let prev_cap = cap;
        let (k, v) = make_kv(512, 2.0);
        let _ = cache.update_and_fetch(k, v).unwrap();
        let new_cap = cache.keys.as_ref().unwrap().shape()[2];
        assert!(
            new_cap <= prev_cap * 2,
            "doubling means new cap ≤ 2× prev: {new_cap} vs {prev_cap}×2"
        );
        assert!(new_cap >= 1024, "must fit required 1024, got {new_cap}");
    }

    /// Once cap is ≥ DOUBLE_THRESHOLD (8K), growth switches to linear +2K
    /// per event — avoids 2× overshoot near max_cap that would pin nearly
    /// the entire sliding-window buffer.
    #[test]
    fn hybrid_grow_uses_linear_above_threshold() {
        let mut cache = SteppingKeyValueCache::with_max(64000);
        // Fill cache up past the doubling threshold with one write.
        let (k, v) = make_kv(10_000, 1.0);
        let _ = cache.update_and_fetch(k, v).unwrap();
        let cap_at_10k = cache.keys.as_ref().unwrap().shape()[2];
        assert!(
            cap_at_10k >= 10_000 && cap_at_10k <= 16_384,
            "first big write may double up to ~16K, got {cap_at_10k}"
        );
        // Now add 100 more tokens. cap should grow by ≤ LINEAR_GROW_STEP
        // (2 048), NOT double to ~32 K.
        let (k, v) = make_kv(100, 2.0);
        let _ = cache.update_and_fetch(k, v).unwrap();
        let cap_after = cache.keys.as_ref().unwrap().shape()[2];
        // If grow happened, it should be linear (+2048), not doubled.
        if cap_after > cap_at_10k {
            assert!(
                cap_after - cap_at_10k <= 2_048,
                "expected linear +≤2K grow above 8K threshold, got cap_at_10k={cap_at_10k} → cap_after={cap_after} (delta={})",
                cap_after - cap_at_10k,
            );
        }
    }
}
