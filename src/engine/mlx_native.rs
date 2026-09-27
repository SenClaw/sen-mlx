//! Native MLX inference engine.
//!
//! Apple Silicon only. Uses **`mlx-rs`** plus weights/templates vendored in-tree (see [`super::mlx_lm`]).
//!
//! Supported architectures (autodetected from `config.json::model_type`):
//! - **Qwen3** — `qwen3*` (transformer + GQA, with optional MLX quantization & TurboQuant KV).
//! - **Qwen3.5** — `qwen3_5*` (hybrid GatedDeltaNet + full attention, OptiQ quants).
//! - **Llama/Qwen2** — `llama` / `qwen2` (Llama-3.x, Llama-3.2, Nesso, Qwen2.x; GQA, no Q/K norm, custom EOS from config).
//! - **Gemma-2** — `gemma2` (alternating sliding-window/global attention, Gemma RMSNorm +1, attn/final soft-capping).
//! - **Gemma-3** — `gemma3` / `gemma3_text` (hybrid sliding+full attention, Gemma RMSNorm, 4 norms/block, dual-RoPE).
//! - **Mamba-2** — `mamba2` (SSD recurrence, fixed-size SSM cache; ignores `kv_cache_bits`).
//! - **Bonsai-Q1** — MLX 1-bit affine checkpoints (`bonsai*` in `model_type`; FP16 KV only).
//!
//! KV cache: **`SteppingKeyValueCache` (FP16)** — `slice_update` + grow-by-256, sliding window.
//! Optional **`TurboQuantKeyValueCache`** when `kv_cache_bits` is set (`turboquant-rs` storage).
//! RoPE uses **`ModelInput::rope_offset`** (caller `usize`), not cache-internal position.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::mpsc;

use super::runtime::{LocalModelRuntime, RuntimeEndpoint, RuntimeHealth, RuntimeStatus};

/// Architecture-tagged model handle. The transformer (Qwen3) and SSM (Mamba-2)
/// paths share a single engine but have wildly different forward signatures,
/// cache shapes, and stop-token sets — so we keep one variant per supported arch.
enum ModelKind {
    Qwen3(super::mlx_lm::models::qwen3::Model),
    Qwen35(super::mlx_lm::models::qwen3_5::Model),
    Llama(super::mlx_lm::models::llama::Model),
    Gemma2(super::mlx_lm::models::gemma2::Gemma2CausalLM),
    Gemma3(super::mlx_lm::models::gemma3::Model),
    Gemma4(super::mlx_lm::models::gemma4::Model),
    DeepSeekV2(super::mlx_lm::models::deepseek_v2::Model),
    Mamba2(super::mlx_lm::models::mamba2::Model),
    FalconMamba(super::mlx_lm::models::falcon_mamba::Model),
    BonsaiQ1(super::mlx_lm::models::bonsai_q1::LoadedBonsaiQ1),
    Ouro(super::mlx_lm::models::ouro::Model),
}

impl ModelKind {
    fn arch_name(&self) -> &'static str {
        match self {
            Self::Qwen3(_) => "qwen3",
            Self::Qwen35(_) => "qwen3_5",
            Self::Llama(_) => "llama",
            Self::Gemma2(_) => "gemma2",
            Self::Gemma3(_) => "gemma3",
            Self::Gemma4(_) => "gemma4",
            Self::DeepSeekV2(_) => "deepseek_v2",
            Self::Mamba2(_) => "mamba2",
            Self::FalconMamba(_) => "falcon_mamba",
            Self::BonsaiQ1(_) => "bonsai_q1",
            Self::Ouro(_) => "ouro",
        }
    }

    /// Per-arch dispatch into [`ChatTemplateModel`]. Each variant delegates to
    /// the model's own impl so `bos_token` / `eos_token` decoding lives next
    /// to the `args` struct that defines them.
    fn resolve_special_tokens(
        &self,
        template: &str,
        tokenizer: &super::mlx_lm_utils::tokenizer::Tokenizer,
    ) -> super::chat_template_openai::SpecialTokens {
        use super::chat_template_openai::ChatTemplateModel;
        match self {
            Self::Qwen3(m) => m.resolve_special_tokens(template, tokenizer),
            Self::Qwen35(m) => m.resolve_special_tokens(template, tokenizer),
            Self::Llama(m) => m.resolve_special_tokens(template, tokenizer),
            Self::Gemma2(m) => m.resolve_special_tokens(template, tokenizer),
            Self::Gemma3(m) => m.resolve_special_tokens(template, tokenizer),
            Self::Gemma4(m) => m.resolve_special_tokens(template, tokenizer),
            Self::DeepSeekV2(m) => m.resolve_special_tokens(template, tokenizer),
            Self::Mamba2(m) => m.resolve_special_tokens(template, tokenizer),
            Self::FalconMamba(m) => m.resolve_special_tokens(template, tokenizer),
            Self::BonsaiQ1(b) => b.resolve_special_tokens(template, tokenizer),
            Self::Ouro(o) => o.resolve_special_tokens(template, tokenizer),
        }
    }

    /// Per-arch dispatch into [`ChatTemplateModel::markers`] — what dialect
    /// the loaded model emits. Each MLX model in `mlx_lm/models/<arch>` owns
    /// its own `MarkerSet` (Gemma-4 harmony, Qwen 3 think/tool_call, etc.);
    /// `ModelKind` just routes the call. Default (`MarkerSet::empty()`) for
    /// archs that haven't declared a structured dialect — plain-text output.
    fn markers(&self) -> super::stream_parser::MarkerSet {
        use super::chat_template_openai::ChatTemplateModel;
        match self {
            Self::Qwen3(m) => m.markers(),
            Self::Qwen35(m) => m.markers(),
            Self::Llama(m) => m.markers(),
            Self::Gemma2(m) => m.markers(),
            Self::Gemma3(m) => m.markers(),
            Self::Gemma4(m) => m.markers(),
            Self::DeepSeekV2(m) => m.markers(),
            Self::Mamba2(m) => m.markers(),
            Self::FalconMamba(m) => m.markers(),
            Self::BonsaiQ1(b) => b.markers(),
            Self::Ouro(o) => o.markers(),
        }
    }

    /// Per-arch dispatch into [`ChatTemplateModel::stop_token_ids`]. The
    /// terminator set is config-specific and lives next to each model's `args`
    /// (see the model files in `mlx_lm/models/`); this just routes to it.
    fn stop_token_ids(&self, tokenizer: &super::mlx_lm_utils::tokenizer::Tokenizer) -> Vec<u32> {
        use super::chat_template_openai::ChatTemplateModel;
        match self {
            Self::Qwen3(m) => m.stop_token_ids(tokenizer),
            Self::Qwen35(m) => m.stop_token_ids(tokenizer),
            Self::Llama(m) => m.stop_token_ids(tokenizer),
            Self::Gemma2(m) => m.stop_token_ids(tokenizer),
            Self::Gemma3(m) => m.stop_token_ids(tokenizer),
            Self::Gemma4(m) => m.stop_token_ids(tokenizer),
            Self::DeepSeekV2(m) => m.stop_token_ids(tokenizer),
            Self::Mamba2(m) => m.stop_token_ids(tokenizer),
            Self::FalconMamba(m) => m.stop_token_ids(tokenizer),
            Self::BonsaiQ1(b) => b.stop_token_ids(tokenizer),
            Self::Ouro(o) => o.stop_token_ids(tokenizer),
        }
    }

    /// KV-cache slot multiplier for looped LMs: how many independent KV slots a
    /// looped model keeps *per logical layer* (one per recurrent sweep). Ouro
    /// keeps `total_ut_steps` (4) → its KV footprint is 4× a dense model at the
    /// same window. `1` for every non-looped architecture. Used to scale the KV
    /// window down so a looped model's total KV memory matches a dense model.
    fn kv_recurrence_multiplier(&self) -> i32 {
        match self {
            Self::Ouro(o) => o.total_ut_steps().max(1),
            _ => 1,
        }
    }

    /// Apply a per-request recurrent-depth override (looped LMs only). `Some(n)`
    /// runs `min(n, trained_steps)` sweeps; `None`/`Some(0)` restores the trained
    /// depth. No-op for every non-looped architecture. Returns the effective
    /// depth that will run for Ouro (for logging), `None` otherwise.
    fn set_runtime_recurrence(&mut self, steps: Option<u32>) -> Option<i32> {
        match self {
            Self::Ouro(o) => {
                let req = steps.and_then(|n| i32::try_from(n).ok());
                o.set_active_ut_steps(req);
                // Recompute the clamped effective depth for the caller's log.
                let trained = o.total_ut_steps();
                Some(match req {
                    Some(n) if n >= 1 => n.min(trained),
                    _ => trained,
                })
            }
            _ => None,
        }
    }
}

/// Heavyweight cached state: model weights + tokenizer + (optional) rendered
/// chat template. Populated by `load()` and dropped by `unload()`, so RAM can
/// be freed on demand without restarting the daemon.
struct Loaded {
    model: ModelKind,
    tokenizer: super::mlx_lm_utils::tokenizer::Tokenizer,
    /// `None` for base models that ship no `chat_template` (e.g. Mamba-2 base).
    /// Generation falls back to plain concatenation in that case.
    chat_template: Option<String>,
    /// Per-model parser config loaded from `tokenizer_config.json` — markers
    /// (channel / tool_call / think / quote), bos/eos, etc. The OUTPUT-stream
    /// parser in `query_llm.rs` reads this verbatim so the canonical
    /// `(visible, reasoning, tool_calls)` shape doesn't depend on hardcoded
    /// dialects; new local models just ship their `tokenizer_config.json`.
    parser_config: super::stream_parser::ParserConfig,
    n_layers: usize,
    /// Attention-only dims; `0` for SSM-only models (Mamba-2 leaves them unused).
    head_dim: i32,
    n_kv_heads: i32,
    /// In-memory prefix KV cache. Multi-turn agent chats reuse the
    /// `[system + tools + previous turns]` prefix — caching the post-prefill
    /// KV state cuts turn-2+ prefill from 30–50 s down to ~3-5 s. Dropped
    /// with the rest of `Loaded` on idle unload (cache is rebuilt on the
    /// first turn after reload).
    prefix_cache: super::mlx_lm::prefix_cache::PrefixCache,
    /// Sampling knobs the checkpoint itself ships in `generation_config.json`.
    /// Used only where the user left the corresponding setting unset.
    gen_defaults: GenerationDefaults,
}

/// The sampling defaults a checkpoint publishes in `generation_config.json`.
///
/// These are the values the model was tuned and evaluated with — Gemma 4, for
/// instance, ships `{"temperature": 1.0, "top_k": 64, "top_p": 0.95}` — so
/// reading them beats a per-architecture table in this file: a new checkpoint
/// gets its own recommended sampling with no code change, and a checkpoint
/// that ships nothing simply leaves every field `None` and falls through to
/// the existing hardcoded fallbacks.
///
/// Missing file, unparseable JSON, and wrong-typed fields are all "no
/// opinion", never an error: sampling defaults must not be able to fail a load.
#[derive(Debug, Clone, Copy, Default)]
struct GenerationDefaults {
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
}

impl GenerationDefaults {
    fn from_model_dir(model_dir: &Path) -> Self {
        let Ok(raw) = std::fs::read_to_string(model_dir.join("generation_config.json")) else {
            return Self::default();
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
            return Self::default();
        };
        // `do_sample: false` means the checkpoint asks for greedy decoding, in
        // which case its temperature/top_k/top_p are decoration and must not be
        // promoted into a sampling config.
        if v.get("do_sample").and_then(|d| d.as_bool()) == Some(false) {
            return Self {
                temperature: Some(0.0),
                ..Self::default()
            };
        }
        Self {
            temperature: v
                .get("temperature")
                .and_then(|t| t.as_f64())
                .map(|t| t as f32)
                .filter(|t| *t >= 0.0),
            top_k: v
                .get("top_k")
                .and_then(|k| k.as_i64())
                .map(|k| k as i32)
                .filter(|k| *k > 0),
            top_p: v
                .get("top_p")
                .and_then(|p| p.as_f64())
                .map(|p| p as f32)
                .filter(|p| *p > 0.0 && *p <= 1.0),
        }
    }
}

/// **Process-wide serialization for ALL heavy MLX/Metal work.**
///
/// MLX uses a shared global Metal device + default stream that is NOT safe for
/// concurrent model loads / generations / cache frees across threads. The main
/// agent and the background cognitive-reflection pipeline each drive their own
/// cached `MlxNativeEngine` (often *different* models, e.g. gemma-4 vs
/// Qwen2.5-0.5B), so their per-instance `loaded` mutexes don't serialize each
/// other — running both at once corrupts Metal state and **SIGSEGVs**. Every
/// load / generate / unload / KV-free acquires this single lock so MLX work is
/// strictly serialized across the whole process.
static MLX_SERIAL: Mutex<()> = Mutex::new(());

/// Acquire [`MLX_SERIAL`], recovering from a poisoned lock (a prior MLX panic
/// must not wedge all future inference). Blocks until it's this caller's turn —
/// use only from blocking/`spawn_blocking` contexts (load + generate).
/// `pub(crate)` so other MLX subsystems (e.g. the TTS backends) can serialize
/// against LLM inference instead of corrupting Metal state.
pub(crate) fn mlx_serial_lock() -> std::sync::MutexGuard<'static, ()> {
    MLX_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Non-blocking [`MLX_SERIAL`] acquire. Returns `None` when MLX is busy (a
/// load/generate is in flight). Used by the async-invoked free paths
/// (`unload` / `release_kv_cache`) so they (a) never block a tokio worker
/// waiting on a long generation and (b) never free Metal buffers concurrently
/// with that generation — the caller simply skips and retries later.
/// `pub(crate)` so other MLX subsystems' reapers (e.g. the TTS idle-model
/// eviction) can use the same skip-when-busy discipline.
pub(crate) fn mlx_serial_try_lock() -> Option<std::sync::MutexGuard<'static, ()>> {
    match MLX_SERIAL.try_lock() {
        Ok(g) => Some(g),
        Err(std::sync::TryLockError::WouldBlock) => None,
        Err(std::sync::TryLockError::Poisoned(g)) => Some(g.into_inner()),
    }
}

/// In-process MLX inference engine. Caches the loaded model so subsequent
/// chats reuse weights instead of re-reading safetensors every call.
pub struct MlxNativeEngine {
    model_dir: PathBuf,
    model_id: String,
    /// `settings.json` `kv_cache_bits` (`3` / `4` → TurboQuant storage; `None` → FP16 stepping).
    kv_cache_bits: Option<u8>,
    /// Arc-wrapped so we can clone the handle into the blocking worker that
    /// runs generation. MLX state is non-Send through naked references, but
    /// `Arc<Mutex<...>>` is.
    loaded: Arc<Mutex<Option<Loaded>>>,
    status: Mutex<RuntimeStatus>,
    /// `(prompt_tokens, generated_tokens)` of the most recent generation on
    /// this engine — the numbers that previously only reached the throughput
    /// log. Read by `query_llm` right after a turn to build real usage for
    /// token accounting (generations are serialized per engine, so
    /// "most recent" is the caller's own turn).
    last_usage: Arc<Mutex<Option<(usize, usize)>>>,
}

impl MlxNativeEngine {
    pub fn new(model_dir: &Path, model_id: &str, kv_cache_bits: Option<u8>) -> Self {
        Self {
            model_dir: model_dir.to_path_buf(),
            model_id: model_id.to_owned(),
            kv_cache_bits,
            loaded: Arc::new(Mutex::new(None)),
            status: Mutex::new(RuntimeStatus::NotInstalled),
            last_usage: Arc::new(Mutex::new(None)),
        }
    }

    /// True when weights are currently in memory.
    pub fn is_loaded(&self) -> bool {
        self.loaded.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// `(prompt_tokens, generated_tokens)` of the most recent generation.
    pub fn last_usage(&self) -> Option<(usize, usize)> {
        self.last_usage.lock().ok().and_then(|g| *g)
    }

    /// Drop the cached model + tokenizer + prefix cache, freeing the bulk
    /// of RAM used by this engine. Safe to call when nothing is loaded
    /// (no-op). Explicitly clears MLX's buffer pool so reclaimed memory
    /// returns to the OS immediately instead of waiting for the next pool
    /// cycle.
    pub fn unload(&self) {
        // Freeing model weights + `mlx_clear_cache` must not run while another
        // engine is mid-generate on the shared Metal device. If MLX is busy,
        // skip — the idle sweep / next model switch retries.
        let Some(_mlx_serial) = mlx_serial_try_lock() else {
            tracing::debug!(
                "[local-mlx-native] unload skipped — MLX busy (generation in progress)"
            );
            return;
        };
        if let Ok(mut g) = self.loaded.lock() {
            // Dropping the `Loaded` releases model weights, tokenizer, AND
            // every PrefixCacheEntry's snapshot Arrays. Whatever MLX
            // buffers those Arc-pinned end up free-listed once their refs
            // hit zero — the `mlx_clear_cache` call below then evicts the
            // free list back to the OS.
            let prev = g.take();
            if let Some(loaded) = prev.as_ref() {
                tracing::info!(
                    "[local-mlx-native] unload: dropping model + prefix cache ({} entries, {})",
                    loaded.prefix_cache.len(),
                    fmt_bytes(loaded.prefix_cache.total_bytes()),
                );
            }
            drop(prev);
        }
        // Force MLX pool release so the freed-up KV / model buffers return
        // to the OS immediately. Without this, the OS RSS often stays high
        // until the next allocation cycle pressures MLX to reclaim.
        unsafe {
            mlx_sys::mlx_clear_cache();
        }
        // Also clear the compile cache so the next reload doesn't ship
        // with stale Metal kernels keyed on the old weights.
        mlx_rs::transforms::compile::clear_cache();
        self.set_status(RuntimeStatus::Stopped);
        let (active, cached, _) = mlx_mem_mib();
        tracing::info!(
            "[local-mlx-native] unload complete — mlx active={:.0} cache={:.0} MiB",
            active,
            cached,
        );
    }

    /// Force a load now (used by the UI "Load" button). Equivalent to the
    /// lazy load that happens on first generate_stream.
    pub fn warm_up(&self) -> anyhow::Result<()> {
        ensure_loaded_blocking(self)
    }

    /// Release the prefix-cache KV (and push MLX's freed pool back to the OS)
    /// **without** unloading the model weights. Cheap to call when the host is
    /// under memory pressure (OS memory-pressure notification, another model
    /// loading, etc.) — the next turn just pays a full prefill instead of a
    /// cache hit. No-op when nothing is loaded. Contrast [`Self::unload`], which
    /// drops the weights too.
    pub fn release_kv_cache(&self) {
        // Same Metal-safety constraint as `unload`: skip if MLX is busy.
        let Some(_mlx_serial) = mlx_serial_try_lock() else {
            return;
        };
        if let Ok(mut g) = self.loaded.lock() {
            if let Some(loaded) = g.as_mut() {
                let freed = loaded.prefix_cache.clear();
                if freed > 0 {
                    tracing::info!(
                        "[local-mlx-native] release_kv_cache: dropped prefix-cache KV ({})",
                        fmt_bytes(freed),
                    );
                }
            }
        }
        unsafe {
            mlx_sys::mlx_clear_cache();
        }
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Settings fingerprint for registry reload (see `local_models/settings.json` `kv_cache_bits`).
    /// Get the parser config for the currently-loaded model. Loads weights if
    /// not yet loaded (parser config is built during the same one-shot load
    /// that populates `Loaded`). Used by `query_llm::query_local_mlx` so the
    /// output-stream parser uses the model's own `tokenizer_config.json`
    /// markers — not a model-id-derived guess.
    pub fn parser_config(&self) -> anyhow::Result<super::stream_parser::ParserConfig> {
        self.warm_up()?;
        let guard = self.loaded.lock().unwrap();
        let state = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        Ok(state.parser_config.clone())
    }

    pub fn kv_cache_bits(&self) -> Option<u8> {
        self.kv_cache_bits
    }

    fn set_status(&self, s: RuntimeStatus) {
        if let Ok(mut g) = self.status.lock() {
            *g = s;
        }
    }
}

#[async_trait]
impl LocalModelRuntime for MlxNativeEngine {
    async fn ensure_installed(&self) -> anyhow::Result<()> {
        if !self.model_dir.exists() {
            anyhow::bail!(
                "model directory not found: {} — download weights first \
                 (e.g. `huggingface-cli download {}`)",
                self.model_dir.display(),
                self.model_id
            );
        }
        for required in ["tokenizer.json", "tokenizer_config.json"] {
            let p = self.model_dir.join(required);
            if !p.exists() {
                anyhow::bail!("missing {} in {}", required, self.model_dir.display());
            }
        }
        Ok(())
    }

    async fn start(&self, _model: &str) -> anyhow::Result<RuntimeEndpoint> {
        self.ensure_installed().await?;
        self.set_status(RuntimeStatus::Ready);
        Ok(RuntimeEndpoint {
            base_url: None,
            model_name: self.model_id.clone(),
            adapt: "local-mlx-native".to_owned(),
            api_key: None,
        })
    }

    async fn stop(&self) -> anyhow::Result<()> {
        self.set_status(RuntimeStatus::Stopped);
        Ok(())
    }

    async fn health(&self) -> anyhow::Result<RuntimeHealth> {
        let status = self
            .status
            .lock()
            .map(|g| *g)
            .unwrap_or(RuntimeStatus::Error);
        Ok(RuntimeHealth {
            status,
            message: None,
        })
    }

    fn supports_native_stream(&self) -> bool {
        true
    }

    async fn generate_stream(
        &self,
        messages: Vec<serde_json::Value>,
        tools: Vec<serde_json::Value>,
        tx: mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        self.stream_openai_to_channel(&messages, &tools, tx).await
    }
}

impl MlxNativeEngine {
    /// Canonical event-stream entry point. The engine generates as usual, but
    /// the raw token stream is piped through the model's own
    /// [`stream_parser::LocalStreamParser`] (built from the loaded
    /// [`stream_parser::ParserConfig`]) before reaching `event_tx` — so
    /// callers receive [`stream_parser::ParserEvent::{Visible, Reasoning,
    /// ToolCall}`] regardless of which model produced the output (Qwen
    /// `<think>`/`<tool_call>`, Gemma-4 `<|channel>`/`<|tool_call>`, …).
    ///
    /// This is the recommended interface for new consumers — adding a new
    /// local-model arch only requires its `tokenizer_config.json` (parser
    /// auto-discovers markers); no consumer-side changes.
    ///
    /// Backward compat: [`stream_openai_to_channel`] still streams raw text
    /// for callers that want to do their own parsing.
    pub async fn stream_events_to_channel(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        event_tx: mpsc::Sender<super::stream_parser::ParserEvent>,
    ) -> anyhow::Result<()> {
        let config = self.parser_config()?;
        let (raw_tx, raw_rx) = mpsc::channel::<String>(64);
        let parser = super::stream_parser::LocalStreamParser::new(config.markers.clone());

        let gen_fut = self.stream_openai_to_channel(messages, tools, raw_tx);
        let pipe_fut = super::stream_parser::pipe_text_stream_to_events(raw_rx, parser, event_tx);

        let ((), gen_res) = tokio::join!(pipe_fut, gen_fut);
        gen_res
    }

    /// Public native-stream entry point used by `query_llm.rs` (OpenAI-shaped messages + tools).
    pub async fn stream_openai_to_channel(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        tx: mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        let loaded = Arc::clone(&self.loaded);
        let model_dir = self.model_dir.clone();
        let model_id = self.model_id.clone();
        let messages = messages.to_vec();
        let tools = tools.to_vec();
        let kv_bits = self.kv_cache_bits;
        let last_usage = Arc::clone(&self.last_usage);

        let join = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            generate_with_cache(
                &loaded,
                &model_dir,
                &model_id,
                &messages,
                &tools,
                kv_bits,
                tx,
                &last_usage,
            )
        });

        join.await??;
        Ok(())
    }
}

/// Synchronous, blocking-thread version of `MlxNativeEngine::warm_up`.
/// Populates `loaded` (if currently None) with Model + Tokenizer + template.
fn ensure_loaded_blocking(engine: &MlxNativeEngine) -> anyhow::Result<()> {
    // Fast path: already loaded — no MLX work, so skip the global serialization.
    if engine.loaded.lock().map(|g| g.is_some()).unwrap_or(false) {
        return Ok(());
    }
    // Serialize the (heavy, Metal-touching) weight load against any concurrent
    // generate/load on another engine — see `MLX_SERIAL`.
    let _mlx_serial = mlx_serial_lock();
    let mut guard = engine
        .loaded
        .lock()
        .map_err(|_| anyhow::anyhow!("loaded mutex poisoned"))?;
    if guard.is_some() {
        return Ok(());
    }
    let state = load_state(&engine.model_dir, &engine.model_id)?;
    *guard = Some(state);
    engine.set_status(RuntimeStatus::Ready);
    Ok(())
}

/// Pin MLX's global default device to the Apple-Silicon GPU (Metal) and log it
/// once, so "is this actually running on the GPU?" is verifiable from the daemon
/// log rather than assumed. MLX already defaults to `gpu()` with the `metal`
/// feature; this makes the guarantee explicit and visible (no silent CPU
/// fallback) and is a no-op on every call after the first.
fn ensure_mlx_gpu_default() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        use mlx_rs::Device;
        Device::set_default(&Device::gpu());
        match Device::try_default() {
            Ok(dev) => match dev.get_type() {
                Ok(mlx_rs::DeviceType::Gpu) => {
                    tracing::info!("[local-mlx-native] compute device: {dev} (Metal GPU) ✓")
                }
                Ok(mlx_rs::DeviceType::Cpu) => tracing::warn!(
                    "[local-mlx-native] compute device fell back to CPU ({dev}) — decode will be slow"
                ),
                Err(e) => tracing::warn!("[local-mlx-native] device type query failed: {e:?}"),
            },
            Err(e) => tracing::warn!("[local-mlx-native] default device query failed: {e:?}"),
        }
    });
}

fn load_state(model_dir: &Path, model_id: &str) -> anyhow::Result<Loaded> {
    use super::mlx_lm_utils::tokenizer::Tokenizer;

    ensure_mlx_gpu_default();
    let arch = detect_architecture(model_id, model_dir)?;
    let tokenizer_file = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e:?}"))?;

    // Single source of truth for both input rendering and output parsing:
    // ParserConfig loads `tokenizer_config.json` + the chat template (probing
    // `tokenizer_config.json::chat_template` → `chat_template.jinja` →
    // `chat_template.json` in that order). The engine then uses the same
    // template string for chat rendering (via `apply_chat_template_*`) and the
    // same MarkerSet for output-stream normalisation — no double load, no
    // chance of input and output drifting apart for new models.
    let mut parser_config = super::stream_parser::ParserConfig::from_model_dir(model_dir, model_id)
        .map_err(|e| anyhow::anyhow!("ParserConfig::from_model_dir failed: {e}"))?;
    let chat_template = parser_config.chat_template.clone();

    let (model, n_layers, head_dim, n_kv_heads) = match arch {
        Arch::Qwen3 => {
            if chat_template.is_none() {
                anyhow::bail!("Qwen3 chat template missing in tokenizer_config.json");
            }
            let m = load_qwen3_any(model_dir)
                .map_err(|e| anyhow::anyhow!("load_qwen3 failed: {e:?}"))?;
            let n = m.args.num_hidden_layers as usize;
            let hd = m.args.head_dim;
            let kvh = m.args.num_key_value_heads;
            (ModelKind::Qwen3(m), n, hd, kvh)
        }
        Arch::Qwen35 => {
            if chat_template.is_none() {
                anyhow::bail!("Qwen3.5 chat template missing in tokenizer_config.json");
            }
            let m = super::mlx_lm::models::qwen3_5::load_qwen35_model(model_dir)
                .map_err(|e| anyhow::anyhow!("load_qwen35 failed: {e:?}"))?;
            let tc = &m.args.text_config;
            let n = tc.num_hidden_layers as usize;
            let hd = tc.head_dim;
            let kvh = tc.num_key_value_heads;
            (ModelKind::Qwen35(m), n, hd, kvh)
        }
        Arch::Llama => {
            let m = load_llama_any(model_dir)
                .map_err(|e| anyhow::anyhow!("load_llama failed: {e:?}"))?;
            let n = m.args.num_hidden_layers as usize;
            let hd = m.args.head_dim;
            let kvh = m.args.num_key_value_heads;
            (ModelKind::Llama(m), n, hd, kvh)
        }
        Arch::Gemma2 => {
            if chat_template.is_none() {
                anyhow::bail!("Gemma 2 chat template missing in tokenizer_config.json");
            }
            let m = super::mlx_lm::models::gemma2::load_gemma2_model(model_dir)
                .map_err(|e| anyhow::anyhow!("load_gemma2 failed: {e:?}"))?;
            let n = m.args.num_hidden_layers as usize;
            let hd = m.args.head_dim;
            let kvh = m.args.num_key_value_heads;
            (ModelKind::Gemma2(m), n, hd, kvh)
        }
        Arch::Gemma3 => {
            let m = load_gemma3_any(model_dir)
                .map_err(|e| anyhow::anyhow!("load_gemma3 failed: {e:?}"))?;
            let n = m.args.num_hidden_layers as usize;
            let hd = m.args.head_dim;
            let kvh = m.args.num_key_value_heads;
            (ModelKind::Gemma3(m), n, hd, kvh)
        }
        Arch::Gemma4 => {
            let m = load_gemma4_any(model_dir)
                .map_err(|e| anyhow::anyhow!("load_gemma4 failed: {e:?}"))?;
            let n = m.args.num_hidden_layers as usize;
            let hd = m.args.head_dim;
            let kvh = m.args.num_key_value_heads;
            (ModelKind::Gemma4(m), n, hd, kvh)
        }
        Arch::DeepSeekV2 => {
            let m = load_deepseek_v2_any(model_dir)
                .map_err(|e| anyhow::anyhow!("load_deepseek_v2 failed: {e:?}"))?;
            let n = m.args.num_hidden_layers as usize;
            // Nominal head_dim/kv_heads — only used to size TurboQuant, which
            // MLA does not use (it caches expanded K/V via fp16_with_max).
            let hd = m.args.q_head_dim();
            let kvh = m.args.num_key_value_heads;
            (ModelKind::DeepSeekV2(m), n, hd, kvh)
        }
        Arch::Mamba2 => {
            let m = load_mamba2_any(model_dir)
                .map_err(|e| anyhow::anyhow!("load_mamba2 failed: {e:?}"))?;
            let n = m.args.num_hidden_layers as usize;
            (ModelKind::Mamba2(m), n, 0, 0)
        }
        Arch::FalconMamba => {
            let m = load_falcon_mamba_any(model_dir)
                .map_err(|e| anyhow::anyhow!("load_falcon_mamba failed: {e:?}"))?;
            let n = m.args.num_hidden_layers as usize;
            (ModelKind::FalconMamba(m), n, 0, 0)
        }
        Arch::BonsaiQ1 => {
            if chat_template.is_none() {
                anyhow::bail!("Bonsai-Q1 chat template missing in tokenizer_config.json");
            }
            let b = super::mlx_lm::models::bonsai_q1::load_bonsai_q1_bundle(model_dir)
                .map_err(|e| anyhow::anyhow!("load_bonsai_q1 failed: {e:?}"))?;
            let n = b.gpu.config.layers;
            let hd = i32::try_from(b.gpu.config.head_dim)
                .map_err(|_| anyhow::anyhow!("bonsai-q1 head_dim does not fit i32"))?;
            let kvh = i32::try_from(b.gpu.config.kv_heads)
                .map_err(|_| anyhow::anyhow!("bonsai-q1 kv_heads does not fit i32"))?;
            (ModelKind::BonsaiQ1(b), n, hd, kvh)
        }
        Arch::Ouro => {
            if chat_template.is_none() {
                anyhow::bail!("Ouro chat template missing in tokenizer_config.json");
            }
            let m =
                load_ouro_any(model_dir).map_err(|e| anyhow::anyhow!("load_ouro failed: {e:?}"))?;
            // Ouro reuses `num_hidden_layers` blocks across `total_ut_steps`
            // recurrent sweeps, with a distinct KV slot per (sweep, layer). Report
            // the product so the engine sizes the cache vector to all 192 slots
            // (`load_ouro_any` keeps them indexed `ut * num_hidden_layers + layer`).
            let n = m.args.total_cache_layers();
            let hd = m.args.head_dim;
            let kvh = m.args.num_key_value_heads;
            (ModelKind::Ouro(m), n, hd, kvh)
        }
    };
    tracing::info!(
        "[local-mlx-native] cached state ready for {model_id} (arch={}, {n_layers} layers, head_dim={head_dim}, kv_heads={n_kv_heads})",
        model.arch_name(),
    );
    // The loaded model is the authoritative source for its own dialect — each
    // `mlx_lm::models::<arch>` declares its `MarkerSet` via `ChatTemplateModel`.
    // If the model returns non-empty markers, prefer them over file-based
    // detection (they're the code-level source of truth; the file path is a
    // best-guess fallback for archs that haven't implemented `markers()` yet).
    let model_markers = model.markers();
    if !model_markers.is_empty() {
        if model_markers != parser_config.markers {
            tracing::info!(
                "[local-mlx-native] using {} model's own markers (overriding file-detected set)",
                model.arch_name()
            );
        }
        parser_config.markers = model_markers;
    }

    tracing::info!(
        "[local-mlx-native] parser_config for {model_id}: arch={} channel={} tool_call={} think={} quote={} format={:?}",
        model.arch_name(),
        parser_config.markers.channel.is_some(),
        parser_config.markers.tool_call.is_some(),
        parser_config.markers.think.is_some(),
        parser_config.markers.quote.is_some(),
        parser_config.markers.tool_call_format,
    );

    // Looped LMs (Ouro) snapshot 192 KV slots per prefix entry — 4× a dense
    // model. Cap the prefix cache to a single, tighter-budgeted entry so it
    // can't stack multiple oversized snapshots on top of the live KV (the main
    // driver of the daemon's RAM ballooning to ~20 GB).
    let kv_mult = model.kv_recurrence_multiplier();
    let prefix_cache = if kv_mult > 1 {
        use super::mlx_lm::prefix_cache::{PrefixCache, MAX_TOTAL_BYTES};
        PrefixCache::with_limits(1, MAX_TOTAL_BYTES / kv_mult as usize)
    } else {
        super::mlx_lm::prefix_cache::PrefixCache::new()
    };

    Ok(Loaded {
        model,
        tokenizer,
        chat_template,
        parser_config,
        n_layers,
        head_dim,
        n_kv_heads,
        prefix_cache,
        gen_defaults: GenerationDefaults::from_model_dir(model_dir),
    })
}

/// Current process RSS (MiB) via macOS Mach `task_info`. Returns **0** on failure or non-macOS.
#[allow(deprecated)] // `libc::mach_task_self` deprecated in favor of `mach2`; keep single deps for MLX path.
fn rss_mib() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: libc::time_value_t,
            system_time: libc::time_value_t,
            policy: i32,
            suspend_count: i32,
        }
        const MACH_TASK_BASIC_INFO: libc::c_int = 20;
        const MACH_TASK_BASIC_INFO_COUNT: u32 = 12;

        let mut info: MachTaskBasicInfo = std::mem::zeroed();
        let mut count: u32 = MACH_TASK_BASIC_INFO_COUNT;
        let kr = libc::task_info(
            libc::mach_task_self(),
            MACH_TASK_BASIC_INFO as _,
            &mut info as *mut _ as _,
            &mut count,
        );
        if kr == 0 {
            info.resident_size as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        0.0
    }
}

/// MLX Metal allocator stats (**active**, **cache**, **peak**) in MiB (via `mlx-sys`, same MLX build as `mlx-rs`).
///
/// **active**: bytes referenced by live arrays. **cache**: freed buffers held in MLX’s pool.
/// **peak**: high-water since start (or since last [`mlx_sys::mlx_reset_peak_memory`]).
#[inline]
fn mlx_mem_mib() -> (f64, f64, f64) {
    unsafe {
        let mut active: usize = 0;
        let mut cache: usize = 0;
        let mut peak: usize = 0;
        mlx_sys::mlx_get_active_memory(&mut active);
        mlx_sys::mlx_get_cache_memory(&mut cache);
        mlx_sys::mlx_get_peak_memory(&mut peak);
        const M: f64 = 1024.0 * 1024.0;
        (active as f64 / M, cache as f64 / M, peak as f64 / M)
    }
}

fn mlx_log_generate_done(label: &'static str, rss_start: f64, generated_count: usize) {
    let (ma, mc, mp) = mlx_mem_mib();
    tracing::info!(
        "[local-mlx-native][mem] {} — rss={:.0}(Δ{:.0}) | mlx active={:.0} cache={:.0} peak={:.0} MiB | new_tokens={}",
        label,
        rss_mib(),
        rss_mib() - rss_start,
        ma,
        mc,
        mp,
        generated_count,
    );
}

/// Human-readable byte count for log lines (`"3.42 GB"`, `"512 MB"`, …).
fn fmt_bytes(n: usize) -> String {
    const KB: f64 = 1024.0;
    let n = n as f64;
    if n < KB {
        return format!("{n:.0} B");
    }
    if n < KB * KB {
        return format!("{:.0} KB", n / KB);
    }
    if n < KB * KB * KB {
        return format!("{:.1} MB", n / (KB * KB));
    }
    format!("{:.2} GB", n / (KB * KB * KB))
}

/// One-line summary of cumulative KV-cache state across all layers. Cheap
/// (shape inspection only); intended for prefill/decode milestone logs so
/// the operator can see how RAM grows turn-by-turn and which cache variant
/// is active (FP16 vs TQ4 vs SSM).
fn mlx_log_kv_cache(label: &str, caches: &[Option<super::mlx_lm::cache::KvCache>]) {
    let (bytes, max_stored, kind) = super::mlx_lm::cache::summarize_caches(caches);
    let n_layers = caches.iter().filter(|c| c.is_some()).count();
    tracing::info!(
        "[local-mlx-native][mem] {} — kv cache: {} ({}) across {} layer(s), longest stored={}",
        label,
        fmt_bytes(bytes),
        kind,
        n_layers,
        max_stored,
    );
}

/// Throughput summary for a finished turn: prefill tok/s + decode tok/s,
/// computed from caller-tracked `Instant` markers. Lets the operator compare
/// runs at a glance (e.g. tweaking `kv_cache_bits` should change decode
/// tok/s; chunked prefill should change prefill tok/s; tools-heavy prompts
/// blow up the prompt-tokens column).
fn mlx_log_throughput(
    prompt_tokens: usize,
    prefill_elapsed: std::time::Duration,
    decode_tokens: usize,
    decode_elapsed: std::time::Duration,
) {
    let prefill_ms = prefill_elapsed.as_millis().max(1) as f64;
    let decode_ms = decode_elapsed.as_millis().max(1) as f64;
    let prefill_tps = (prompt_tokens as f64) * 1000.0 / prefill_ms;
    let decode_tps = (decode_tokens as f64) * 1000.0 / decode_ms;
    tracing::info!(
        "[local-mlx-native][perf] turn done — prefill {} tok / {:.2} s ({:.1} tok/s) | decode {} tok / {:.2} s ({:.1} tok/s)",
        prompt_tokens,
        prefill_ms / 1000.0,
        prefill_tps,
        decode_tokens,
        decode_ms / 1000.0,
        decode_tps,
    );
}

/// Drop MLX’s pooled device buffers (`mlx_clear_cache`) and the lazy compile cache before the next turn.
fn mlx_release_after_turn() {
    unsafe {
        mlx_sys::mlx_clear_cache();
    }
    let (a, c, _) = mlx_mem_mib();
    tracing::info!(
        "[local-mlx-native][mem] MLX buffer pool cleared — active={:.0} cache={:.0} MiB",
        a,
        c,
    );
    mlx_rs::transforms::compile::clear_cache();
}

/// Load the model's Jinja chat template. Probes `tokenizer_config.json` →
/// `chat_template.jinja` → arch-specific fallback (Mistral `[INST]` when the
/// detected arch is Mamba-class and the tokenizer ships `[INST]` tokens).
fn load_mlx_chat_template(model_dir: &Path, model_id: &str) -> anyhow::Result<Option<String>> {
    let detected = detect_architecture(model_id, model_dir)?;
    super::chat_template_openai::load_chat_template_from_dir(model_dir, model_id, |dir| {
        match detected {
            Arch::Mamba2 | Arch::FalconMamba => {
                super::mlx_lm::models::mamba2::chat_template_fallback(dir)
            }
            _ => Ok(None),
        }
    })
    .map_err(|e| anyhow::anyhow!("load chat template: {e}"))
}

/// Decoded token ids counted back from the current step (HF repetition penalty window).
const MLX_REPEN_DECODE_WINDOW: usize = 128;

fn mlx_recent_decode_push(buf: &mut VecDeque<u32>, tok: u32, window: usize) {
    if window == 0 {
        return;
    }
    buf.push_back(tok);
    // O(1) amortized eviction (vs. `Vec::drain(..)` memmove of the whole
    // window every token). The window is only ever overgrown by one push.
    while buf.len() > window {
        buf.pop_front();
    }
}

/// Truncation applied before the draw: `top_k` / `top_p`, resolved once per
/// turn from the user's settings falling back to the checkpoint's own
/// `generation_config.json`. `None` on both fields = the untruncated
/// full-vocabulary draw this engine used to do unconditionally.
#[derive(Debug, Clone, Copy, Default)]
struct SamplingTruncation {
    top_k: Option<i32>,
    top_p: Option<f32>,
}

fn sample_decode_token_id(
    last_logits: &mlx_rs::Array,
    temperature: f32,
    repetition_penalty: f32,
    truncation: SamplingTruncation,
    recent_decode_ids: &VecDeque<u32>,
    forbidden_ids: &[u32],
) -> anyhow::Result<mlx_rs::Array> {
    use mlx_rs::ops::which;
    use mlx_rs::{Array, Dtype};

    let mlx_sample = |logits: &Array| {
        super::mlx_lm::sampling::sample_with(
            logits,
            temperature,
            truncation.top_k,
            truncation.top_p,
        )
    };

    let apply_penalty = repetition_penalty > 1.0 && !recent_decode_ids.is_empty();
    // Fast path: nothing to adjust — sample the raw logits directly.
    if forbidden_ids.is_empty() && !apply_penalty {
        return mlx_sample(last_logits).map_err(|e| anyhow::anyhow!("mlx sample: {e:?}"));
    }
    if !forbidden_ids.is_empty() {
        tracing::debug!(
            "[local-mlx-native] sample adjust path with forbidden_ids={:?} (mask to -inf on GPU)",
            forbidden_ids
        );
    }

    // GPU-side logit adjustment. Everything below stays **lazy** so it folds into
    // the single `eval` the decode loop already forces (via `.item()` / `eval`)
    // — no host round-trip, no full-vocab copy. The previous CPU path cast to
    // f32, pulled the whole ~152K-entry row to a `Vec`, mutated it, and re-eval'd
    // twice per token. `last_logits` is `[1, vocab]`.
    //
    // Cast to f32 once: quantized lm_heads emit BF16/FP16, mixing those with f32
    // scalar penalties is dtype-fragile, and f32 reproduces the old CPU path
    // bit-for-bit on greedy argmax.
    let mut logits = if last_logits.dtype() == Dtype::Float32 {
        last_logits.clone()
    } else {
        last_logits
            .as_dtype(Dtype::Float32)
            .map_err(|e| anyhow::anyhow!("logits cast to f32: {e:?}"))?
    };

    // (a) HF repetition penalty: gather the logits of recently-emitted tokens,
    // scale (>0 → /penalty, ≤0 → *penalty), scatter back. Gather-then-scatter is
    // idempotent for duplicate ids (the same source value is written back), so
    // no dedup is needed — matches the "apply once per unique id" convention.
    if apply_penalty {
        let ids: Vec<i32> = recent_decode_ids.iter().map(|&t| t as i32).collect();
        let idx = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let vals = logits
            .take_along_axis(&idx, -1)
            .map_err(|e| anyhow::anyhow!("penalty gather: {e:?}"))?;
        let p = Array::from_f32(repetition_penalty);
        let positive = vals
            .gt(Array::from_f32(0.0))
            .map_err(|e| anyhow::anyhow!("penalty cmp: {e:?}"))?;
        let scaled = which(
            &positive,
            &vals
                .divide(&p)
                .map_err(|e| anyhow::anyhow!("penalty div: {e:?}"))?,
            &vals
                .multiply(&p)
                .map_err(|e| anyhow::anyhow!("penalty mul: {e:?}"))?,
        )
        .map_err(|e| anyhow::anyhow!("penalty select: {e:?}"))?;
        logits = logits
            .put_along_axis(&idx, &scaled, -1)
            .map_err(|e| anyhow::anyhow!("penalty scatter: {e:?}"))?;
    }

    // (b) Forbidden-token mask: scatter -inf so those tokens carry zero
    // probability mass. Keeps stop tokens (`<|im_end|>`, `<|endoftext|>`) out of
    // `<tool_call>…</tool_call>` bodies even when 4-bit precision biases toward
    // an early stop mid-args.
    if !forbidden_ids.is_empty() {
        let ids: Vec<i32> = forbidden_ids.iter().map(|&t| t as i32).collect();
        let idx = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let neg = Array::from_slice(&vec![f32::NEG_INFINITY; ids.len()], &[1, ids.len() as i32]);
        logits = logits
            .put_along_axis(&idx, &neg, -1)
            .map_err(|e| anyhow::anyhow!("forbidden scatter: {e:?}"))?;
    }

    mlx_sample(&logits).map_err(|e| anyhow::anyhow!("mlx sample: {e:?}"))
}

/// Synchronous generation entry. Runs on a blocking worker, holding the
/// engine's `loaded` mutex for the duration of one inference call.
fn preprocess_openai_messages_for_mlx(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    use super::thinking_parse::strip_thinking_blocks;
    messages
        .iter()
        .map(|msg| {
            if msg.get("role").and_then(|v| v.as_str()) != Some("assistant") {
                return msg.clone();
            }
            let mut out = msg.clone();
            let Some(obj) = out.as_object_mut() else {
                return out;
            };
            // Strip any prior `<think>…</think>` from the assistant text.
            if let Some(content) = obj.get("content").and_then(|c| c.as_str()) {
                let stripped = strip_thinking_blocks(content);
                obj.insert("content".into(), serde_json::Value::String(stripped));
            }
            // Normalize tool-call arguments from the OpenAI JSON-*string* form into
            // a parsed object. Chat templates that re-render prior tool calls by
            // iterating `tool_call.arguments|items` (Qwen3.5 / Hermes) fail with
            // "cannot convert value into pairs" when arguments is a string.
            if let Some(tcs) = obj.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                for tc in tcs.iter_mut() {
                    let Some(func) = tc.get_mut("function").and_then(|f| f.as_object_mut()) else {
                        continue;
                    };
                    if let Some(args_str) = func.get("arguments").and_then(|a| a.as_str()) {
                        let parsed = serde_json::from_str::<serde_json::Value>(args_str)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        func.insert("arguments".into(), parsed);
                    }
                }
            }
            out
        })
        .collect()
}

fn openai_messages_to_plain_transcript(messages: &[serde_json::Value]) -> String {
    let mut text = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = match msg.get("content") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| {
                    p.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if content.is_empty() {
            continue;
        }
        text.push_str(&format!("{role}: {content}\n"));
    }
    text.push_str("Assistant:");
    text
}

#[allow(clippy::too_many_arguments)]
fn generate_with_cache(
    loaded: &Arc<Mutex<Option<Loaded>>>,
    model_dir: &Path,
    model_id: &str,
    messages: &[serde_json::Value],
    tools: &[serde_json::Value],
    _kv_cache_bits: Option<u8>,
    tx: mpsc::Sender<String>,
    last_usage: &Mutex<Option<(usize, usize)>>,
) -> anyhow::Result<()> {
    use super::mlx_lm::cache::{eval_all_caches, KvCache, DEFAULT_TQ_ACTIVATE_AT};
    use mlx_rs::ops::indexing::{IndexOp, NewAxis};
    use mlx_rs::transforms::{async_eval, eval};
    use mlx_rs::Array;

    // Serialize all MLX/Metal work process-wide (load + prefill + decode) so the
    // main-agent model and the background cognitive model can't run on the
    // shared Metal device concurrently and SIGSEGV. Held for the whole call.
    let _mlx_serial = mlx_serial_lock();

    let mut guard = loaded
        .lock()
        .map_err(|_| anyhow::anyhow!("loaded mutex poisoned"))?;
    if guard.is_none() {
        *guard = Some(load_state(model_dir, model_id)?);
    }
    // Safe: we just ensured Some.
    let state = guard.as_mut().unwrap();
    let tokenizer = &mut state.tokenizer;
    let template_opt = state.chat_template.clone();
    let n_layers = state.n_layers;

    let settings_dir = model_dir.parent().unwrap_or(model_dir);
    let gen_opt = crate::settings::load(settings_dir);

    let messages = preprocess_openai_messages_for_mlx(messages);
    let enable_thinking = gen_opt.enable_thinking;

    // Looped-LM recurrent-depth override (Ouro / LoopLM). Applied before prefill
    // so both prefill and decode run at the selected depth; a no-op for every
    // other architecture. The effective (clamped) depth is logged per turn.
    if let Some(eff) = state.model.set_runtime_recurrence(gen_opt.recurrence_steps) {
        tracing::info!("[local-mlx-native] Ouro recurrent sweeps this turn: {eff}");
    }

    let max_prompt_tokens = gen_opt
        .max_prompt_tokens
        .unwrap_or(crate::settings::DEFAULT_MAX_PROMPT_TOKENS)
        .clamp(512, 262_144) as usize;
    let max_new_tokens = gen_opt
        .max_new_tokens
        .unwrap_or(crate::settings::DEFAULT_MAX_NEW_TOKENS)
        .clamp(1, 8192) as usize;
    let max_kv_tokens = gen_opt
        .max_kv_tokens
        .unwrap_or(crate::settings::DEFAULT_KV_WINDOW_TOKENS)
        .clamp(128, 262_144) as i32;
    // Looped LMs (Ouro) keep one KV slot per (recurrent sweep, layer) → their KV
    // footprint is `total_ut_steps`× a dense model at the same window. Honouring
    // the 20 480-token default verbatim made Ouro-2.6B's KV reach ~30 GB
    // (192 slots × 20 480 × 16 × 128 × 2 × 2 B). On the DEFAULT window, size the
    // window to a fixed ~2 GB KV budget instead (window = budget / per-token-bytes
    // over all 192 slots). An EXPLICIT `max_kv_tokens` is honoured verbatim — the
    // user has opted into the RAM cost for more context.
    let kv_mult = state.model.kv_recurrence_multiplier();
    let max_kv_tokens = if gen_opt.max_kv_tokens.is_none() {
        // Bound live-KV RAM to a fixed byte budget when the window isn't set
        // explicitly. This is a **no-op for low-per-token dense models** (the
        // budget window exceeds `DEFAULT_KV_WINDOW_TOKENS`, so the `.clamp` keeps
        // the default), but it caps the archs whose KV would otherwise balloon on
        // the 20 480-token default:
        //   - **Looped LMs (Ouro)** — `kv_mult`× slots (`n_layers` already
        //     ×total_ut_steps); the default reached ~20–30 GB.
        //   - **MLA (DeepSeek)** — caches the *expanded* K-192 + V-128 per head
        //     × 16 heads × 27 layers ≈ 264 KB/token, so 20 480 tokens is ~5.4 GB
        //     KV → >15 GB total with the 8.3 GB of 4-bit weights.
        // An EXPLICIT `max_kv_tokens` bypasses this — the user opted into the RAM
        // cost for a larger context window.
        const KV_BUDGET_BYTES: i64 = 4 * 1024 * 1024 * 1024;
        // Per token over all KV slots: n_layers × n_kv_heads × head_dim × 2 (K+V)
        // × 2 B (fp16). Uses `head_dim` for both K and V (slightly conservative
        // for MLA, whose V dim is smaller than K).
        let per_token = (n_layers as i64)
            * (state.head_dim.max(1) as i64)
            * (state.n_kv_heads.max(1) as i64)
            * 4;
        let budget_window =
            (KV_BUDGET_BYTES / per_token.max(1)).clamp(1024, max_kv_tokens as i64) as i32;
        if budget_window < max_kv_tokens {
            let budget_gib = KV_BUDGET_BYTES / (1024 * 1024 * 1024);
            let why = if kv_mult > 1 {
                "recurrent slots"
            } else {
                "high per-token MLA KV"
            };
            tracing::info!(
                "[local-mlx-native] KV window {max_kv_tokens}→{budget_window} tokens \
                 (≈{budget_gib} GB KV budget, {why}); set max_kv_tokens explicitly for more context"
            );
        }
        budget_window
    } else {
        max_kv_tokens
    };
    let max_kv_usize = max_kv_tokens as usize;
    // KV is a *sliding* window: every decode token evicts the oldest cached
    // token once the cache is full. If the prompt already fills the cache,
    // the first decode step pushes the system prompt / tool schemas out and
    // the model loses its grounding within a few tokens — classic symptoms
    // are mid-answer repetition spirals (`7 7 7 …`).
    //
    // Decode reserve = how many decode tokens can run before sliding starts
    // evicting the system block. Capped at 2 048 (most tool-calling turns
    // are well under that) so very large KV windows don't waste headroom
    // that could hold more tool schemas. Hard ceiling at KV/4 protects
    // small-KV configs from over-reserving.
    const DECODE_RESERVE_CAP: usize = 2048;
    let decode_reserve = max_new_tokens
        .min(DECODE_RESERVE_CAP)
        .min(max_kv_usize / 4)
        .max(256);
    // Single budget = tighter of (RAM cap, KV cap minus decode reserve).
    // Render is iterative — trimming inputs and re-rendering is the only
    // way to keep the prompt structurally valid; raw token-level truncation
    // destroys the chat template's role headers and tool-schema framing.
    let budget = max_prompt_tokens
        .min(max_kv_usize.saturating_sub(decode_reserve))
        .max(512);
    // Each attempt drops one conversation turn OR shrinks the tool roster, then
    // re-renders. Long agentic histories can hold many turns, so this must be
    // high enough to drain the whole history AND still reach the tool-dropping
    // phase — otherwise the loop exits over-budget and falls back to the
    // structure-preserving splice. Renders are cheap (~tens of ms) and the loop
    // exits as soon as it fits, so this only matters for pathological prompts.
    const MAX_FIT_ATTEMPTS: usize = 128;

    let mut prompt: Vec<u32> = if let Some(template) = template_opt {
        let special = state
            .model
            .resolve_special_tokens(template.as_str(), tokenizer);
        let initial_msg_count = messages.len();
        let initial_tool_count = tools.len();
        let mut work_messages: Vec<serde_json::Value> = messages.clone();
        let mut work_tools: Vec<serde_json::Value> = tools.to_vec();

        // Tool-definition shape differs by template. The MLX path feeds
        // HF-flattened tools (`{name, description, parameters}`) which the
        // Qwen/Llama templates index directly (`tool.name`). Gemma-4's template
        // indexes the OpenAI-nested form (`tool['function']['name']`) and raises
        // an UndefinedError on a flattened tool — so wrap each flattened tool
        // back into `{type:"function", function:{…}}` for Gemma-4. Idempotent:
        // tools that already carry a `function` key are left untouched.
        if matches!(state.model, ModelKind::Gemma4(_)) {
            for t in work_tools.iter_mut() {
                if t.get("function").is_none() {
                    let inner = t.clone();
                    *t = serde_json::json!({ "type": "function", "function": inner });
                }
            }
        }

        // Fit-to-budget loop:
        // 1. Render with current (messages, tools).
        // 2. If under budget — done.
        // 3. Else drop the oldest non-system / non-last-user message.
        // 4. Else (no more middle messages) halve the tools list from the end.
        // 5. Stop after MAX_FIT_ATTEMPTS or when nothing else can be trimmed.
        // Fit-to-budget: re-render after each trim. Trim order preserves the
        // most valuable context — sacrifice the oldest conversation turn first
        // (keeps the full tool roster = capability), and only once no middle
        // turns remain start dropping tools from the end. Crucially, message
        // drops and tool drops draw from the SAME attempt budget below, so a
        // long history can no longer starve the tool-dropping phase (the old
        // bug: 24 attempts fully consumed dropping messages, leaving all tools
        // in place and forcing a destructive head-truncation). The cap is just
        // a non-convergence guard — `drop_oldest_openai_middle_message` always
        // keeps the leading system block + the last user turn, so the system
        // prompt and the current question survive every step.
        let mut attempts = 0usize;
        let mut tokens: Vec<u32> = loop {
            let encs = tokenizer
                .apply_chat_template_json_and_encode(
                    template.clone(),
                    model_id,
                    &work_messages,
                    &work_tools,
                    Some(true),
                    enable_thinking,
                    special.bos.as_deref(),
                    special.eos.as_deref(),
                )
                .map_err(|e| anyhow::anyhow!("chat template apply failed: {e:?}"))?;
            let tokens: Vec<u32> = encs.iter().flat_map(|e| e.get_ids()).copied().collect();
            if tokens.len() <= budget || attempts >= MAX_FIT_ATTEMPTS {
                break tokens;
            }
            attempts += 1;
            // 1. Sacrifice the oldest conversation turn (system + last user kept).
            if super::mlx_prompt::drop_oldest_openai_middle_message(&mut work_messages) {
                continue;
            }
            // 2. No droppable turns left — shrink the tool roster from the end.
            if !work_tools.is_empty() {
                // Linear-scale estimate: tokens ≈ fixed_cost + per_tool × count.
                // Project the new tool count that should land just under
                // budget — much better than blind halving (which dropped 28→14
                // even when 28 only overshot by ~2 tokens). Subtract one more
                // for safety so we converge inside the budget on the next
                // render. Fall back to `len-1` if the estimate is degenerate.
                let scaled = (work_tools.len() as u128 * budget as u128) / tokens.len() as u128;
                let mut new_len = (scaled as usize).saturating_sub(1);
                if new_len >= work_tools.len() {
                    new_len = work_tools.len().saturating_sub(1);
                }
                let dropped = work_tools.len() - new_len;
                tracing::warn!(
                    "[local-mlx-native] prompt {} tokens > budget {}; estimated per-tool cost → dropped {} tool(s) (kept {})",
                    tokens.len(),
                    budget,
                    dropped,
                    new_len,
                );
                work_tools.truncate(new_len);
                continue;
            }
            // 3. Only the system block + last user turn remain and it still
            //    overflows — stop; the head+tail splice below preserves both
            //    the system prompt and the generation tail.
            break tokens;
        };

        tracing::info!(
            "[local-mlx-native] chat template: {} → {} message(s), {} → {} tool(s), {} tokens (budget {})",
            initial_msg_count,
            work_messages.len(),
            initial_tool_count,
            work_tools.len(),
            tokens.len(),
            budget,
        );

        if tokens.len() > budget {
            // Last resort: even the system block + last user turn (no droppable
            // middle, possibly no tools) overflow the window. Splice out the
            // middle rather than draining the head — preserves the system prompt
            // and the generation tail (see `splice_head_tail_to_budget`).
            let before = tokens.len();
            let overflow = before - budget;
            tokens = super::mlx_prompt::splice_head_tail_to_budget(tokens, budget);
            tracing::warn!(
                "[local-mlx-native] prompt {} > budget {} after {} trim attempt(s); spliced out {} middle token(s) → {} (system prompt + generation tail preserved)",
                before,
                budget,
                attempts,
                overflow,
                tokens.len(),
            );
        }
        tokens
    } else {
        // No chat template (Mamba-2 base etc.) — fall back to a plain transcript.
        // No tool / chat structure to preserve, so token-level head-truncation
        // is acceptable here.
        let text = openai_messages_to_plain_transcript(&messages);
        let enc = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("plain encode failed: {e:?}"))?;
        let mut tokens = enc.get_ids().to_vec();
        if tokens.len() > budget {
            let drop = tokens.len() - budget;
            tracing::warn!(
                "[local-mlx-native] truncating plain-transcript prompt {} → {} tokens",
                tokens.len(),
                budget,
            );
            tokens.drain(..drop);
        }
        tokens
    };

    tracing::info!(
        "[local-mlx-native] prompt {} tokens, max_kv_window {}, head={:?} tail={:?}",
        prompt.len(),
        max_kv_tokens,
        &prompt[..prompt.len().min(8)],
        &prompt[prompt.len().saturating_sub(8)..]
    );

    // Does the rendered prompt end INSIDE a `<think>` block? Some chat templates
    // (Qwen3.5 with `enable_thinking=true`) prefill `<think>\n` as the assistant
    // prefix, so the model generates the reasoning + (eventually) `</think>` but
    // never the opening tag. We re-emit a synthetic `<think>\n` as the first
    // stream chunk so the downstream reasoning/answer split sees a well-formed
    // block — including the common case where a long chain-of-thought never
    // closes within the token budget (otherwise the whole reasoning would leak
    // into the visible answer). Template-agnostic: detected from the prompt tail.
    let thinking_prefilled = {
        let n = prompt.len();
        let tail = tokenizer
            .decode(&prompt[n.saturating_sub(16)..], false)
            .unwrap_or_default();
        match tail.rfind("<think>") {
            Some(o) => tail.rfind("</think>").map_or(true, |c| c < o),
            None => false,
        }
    };

    let prompt_tokens = Array::from(&prompt[..]).index(NewAxis);

    let configured_tq_bits = _kv_cache_bits.or(gen_opt.kv_cache_bits).filter(|b| *b > 0);
    let tq_activate_at = gen_opt
        .tq_activate_at
        .map(|v| v as i32)
        .unwrap_or(DEFAULT_TQ_ACTIVATE_AT);

    // TurboQuant auto-disable guard.
    //
    // TQ runs its per-token KV quantization on CPU (`turboquant-rs`). Once
    // activated, every subsequent decode token routes through:
    //   `Array.eval → cast f16→f32 → heap Vec<Vec<f32>> → 4-bit pack`
    // …which costs ~500 ms – 1 s per token on M-series vs ~30 ms on FP16 GPU
    // (20–30× slower). Worse, *activation itself* dumps the full staging
    // buffer (16 K tokens × 36 layers × 8 heads = 4.7 M quant ops) through
    // the same path in one shot — easily 30–60 s of stall.
    //
    // Symptom: decode logs the first 20 tokens then goes silent for minutes
    // until the LLM_TIMEOUT fires.
    //
    // Auto-disable the configured `kv_cache_bits` when **prompt + decode
    // headroom > tq_activate_at**, i.e. when TQ would activate during the
    // current turn. Falls back to FP16 KV — the user's effective RAM is
    // similar (prompt already fills most of the FP16 buffer) and decode
    // stays on the fast GPU path.
    let tq_bits = if let Some(bits) = configured_tq_bits {
        let projected_max = prompt.len() as i32 + max_new_tokens as i32;
        if projected_max > tq_activate_at {
            tracing::warn!(
                "[local-mlx-native] TurboQuant TQ{} auto-disabled for this turn: \
                 prompt({}) + max_new({}) = {} > tq_activate_at({}). \
                 TQ activation mid-decode is ~20× slower and routinely hits LLM_TIMEOUT. \
                 Using FP16 KV for this turn. Raise `tq_activate_at` ≥ {} in settings.json \
                 to keep TurboQuant enabled.",
                bits,
                prompt.len(),
                max_new_tokens,
                projected_max,
                tq_activate_at,
                projected_max,
            );
            None
        } else {
            tracing::info!(
                "[local-mlx-native] TurboQuant TQ{} enabled (activate_at={}, prompt={}, projected_max={})",
                bits,
                tq_activate_at,
                prompt.len(),
                projected_max,
            );
            Some(bits)
        }
    } else {
        None
    };
    let head_dim = state.head_dim;
    let n_kv_heads = state.n_kv_heads;

    // Generation terminators — each model derives its own set from its config
    // (`eos_token_id(s)` and/or tokenizer-resolved named specials like
    // `<|im_end|>`). See `ChatTemplateModel::stop_token_ids` in the per-model
    // files under `mlx_lm/models/`. Empty = run to max_new_tokens.
    let stop_ids: Vec<u32> = state.model.stop_token_ids(tokenizer);
    tracing::info!(
        "[local-mlx-native] stop token ids ({}): {:?}",
        state.model.arch_name(),
        stop_ids
    );
    let is_stop = move |t: u32| stop_ids.contains(&t);

    // Per-arch KV / SSM cache allocation (TurboQuant only where wired).
    let mut cache: Vec<Option<KvCache>> = match &state.model {
        ModelKind::Qwen3(_) => {
            if let Some(bits) = tq_bits {
                tracing::info!(
                    "[local-mlx-native] TurboQuant KV: TQ{bits} (activate_at={tq_activate_at}, max_kv_window {max_kv_tokens})"
                );
            }
            (0..n_layers)
                .map(|_| {
                    Some(if let Some(bits) = tq_bits {
                        KvCache::turboquant_with_max(
                            bits,
                            head_dim,
                            n_kv_heads,
                            max_kv_tokens,
                            tq_activate_at,
                        )
                    } else {
                        KvCache::fp16_with_max(max_kv_tokens)
                    })
                })
                .collect()
        }
        ModelKind::Qwen35(m) => {
            if tq_bits.is_some() {
                tracing::warn!(
                    "[local-mlx-native] kv_cache_bits ignored for Qwen3.5 (TurboQuant KV not wired for full-attn layers)"
                );
            }
            m.make_cache()
        }
        ModelKind::Llama(_) => {
            if let Some(bits) = tq_bits {
                tracing::info!(
                    "[local-mlx-native] TurboQuant KV: TQ{bits} (activate_at={tq_activate_at}, max_kv_window {max_kv_tokens})"
                );
            }
            (0..n_layers)
                .map(|_| {
                    Some(if let Some(bits) = tq_bits {
                        KvCache::turboquant_with_max(
                            bits,
                            head_dim,
                            n_kv_heads,
                            max_kv_tokens,
                            tq_activate_at,
                        )
                    } else {
                        KvCache::fp16_with_max(max_kv_tokens)
                    })
                })
                .collect()
        }
        ModelKind::Gemma2(_) => {
            if tq_bits.is_some() {
                tracing::warn!(
                    "[local-mlx-native] kv_cache_bits ignored for Gemma-2 (TurboQuant KV path not wired for this architecture)"
                );
            }
            (0..n_layers)
                .map(|_| Some(KvCache::fp16_with_max(max_kv_tokens)))
                .collect()
        }
        ModelKind::Gemma3(m) => {
            if tq_bits.is_some() {
                tracing::warn!(
                    "[local-mlx-native] kv_cache_bits ignored for Gemma-3 (TurboQuant path \
                     not yet wired); hybrid layers use unified FP16 KV cap, sliding via mask only"
                );
            }
            m.make_caches(max_kv_tokens)
        }
        ModelKind::Gemma4(m) => {
            if tq_bits.is_some() {
                tracing::warn!(
                    "[local-mlx-native] kv_cache_bits ignored for Gemma-4 (TurboQuant path \
                     not wired); non-shared layers use FP16 KV, KV-shared layers reuse them"
                );
            }
            m.make_caches(max_kv_tokens)
        }
        ModelKind::DeepSeekV2(m) => {
            if tq_bits.is_some() {
                tracing::warn!(
                    "[local-mlx-native] kv_cache_bits ignored for DeepSeek-V2 (MLA caches \
                     expanded K/V via FP16; TurboQuant not wired)"
                );
            }
            m.make_caches(max_kv_tokens)
        }
        ModelKind::Mamba2(m) => {
            if tq_bits.is_some() {
                tracing::warn!(
                    "[local-mlx-native] kv_cache_bits ignored for Mamba-2 (fixed-size SSM state)"
                );
            }
            m.make_cache()
        }
        ModelKind::FalconMamba(m) => {
            if tq_bits.is_some() {
                tracing::warn!(
                    "[local-mlx-native] kv_cache_bits ignored for Falcon-Mamba (fixed-size SSM state)"
                );
            }
            m.make_cache()
        }
        ModelKind::BonsaiQ1(_) => {
            if tq_bits.is_some() {
                tracing::warn!(
                    "[local-mlx-native] kv_cache_bits ignored for Bonsai-Q1 (TurboQuant KV path not wired for this architecture)"
                );
            }
            (0..n_layers)
                .map(|_| Some(KvCache::fp16_with_max(max_kv_tokens)))
                .collect()
        }
        ModelKind::Ouro(_) => {
            // `n_layers` here is `total_ut_steps * num_hidden_layers` — one slot
            // per (recurrent sweep, layer). Per-slot caches are ordinary KV
            // caches, so TurboQuant applies the same as for Llama/Qwen3.
            if let Some(bits) = tq_bits {
                tracing::info!(
                    "[local-mlx-native] TurboQuant KV: TQ{bits} (activate_at={tq_activate_at}, max_kv_window {max_kv_tokens})"
                );
            }
            (0..n_layers)
                .map(|_| {
                    Some(if let Some(bits) = tq_bits {
                        KvCache::turboquant_with_max(
                            bits,
                            head_dim,
                            n_kv_heads,
                            max_kv_tokens,
                            tq_activate_at,
                        )
                    } else {
                        KvCache::fp16_with_max(max_kv_tokens)
                    })
                })
                .collect()
        }
    };

    let mut rope_offset = 0usize;

    // ── Prefix cache restore ──────────────────────────────────────────────
    // Multi-turn agent chats repeatedly send `[system + tools + prior turns]`
    // prompts where 80–90 % of the tokens are unchanged from the previous
    // turn. Restoring the post-prefill KV snapshot for the longest matching
    // prefix lets prefill skip directly to the new suffix.
    //
    // Constraints:
    // - Only for KvCache variants that support clone (`Fp16`; TQ-active
    //   skipped). Filtered by `tq_bits.is_none()` here — TQ allocation
    //   forces snapshot to `None` for active layers anyway.
    // - Only for Qwen-family arches whose chat template ends with the
    //   3-token assistant generation prompt (`<|im_start|>assistant\n`).
    //   Other arches use different suffix lengths — would need per-arch
    //   `gen_suffix_len` to extend.
    const QWEN3_GEN_SUFFIX_LEN: usize = 3;
    // Qwen3.5 (`Qwen35LinearCache`) uses a **strict-extension** prefix-cache
    // path, NOT the trimmed one used by pure-attention arches. Its linear-attn
    // layers carry a recurrent SSM/conv state with no per-token slice, so the
    // trimmed path (store key = prompt[..len-gen_suffix], `trim_by` no-op on
    // recurrent state) leaves the state 3 tokens AHEAD of the key → a HIT
    // restores a misaligned state → non-deterministic output (the bug this
    // path used to hit, hence the old exclusion).
    //
    // The fix: store Qwen3.5 with the FULL prompt as key and NO trim, so the
    // recurrent state aligns exactly at `prompt.len()`. Within one agentic
    // request the Qwen3.5 chat template keeps each assistant turn's `<think>`
    // reasoning (`loop.index0 > ns.last_query_index`), so every tool turn's
    // prompt is a strict extension of the previous → the full-length key is a
    // prefix of next turn's prompt and HITs, skipping re-prefill of the shared
    // body (the snapshot boundary is computed below in `recurrent_snap_at`).
    //
    // Determinism (verified by the strict-extension A/B in `examples/mlx_bench`,
    // MLX_BENCH_EXT_DETERMINISM=1): Qwen3.5-**0.8B** restore is byte-identical
    // to a from-scratch prefill. Qwen3.5-**4B**-OptiQ stays coherent and shares
    // a long (~60-token) prefix but is NOT bit-identical — a tiny floating-point
    // difference in the recurrent/mixed-bit-quant path eventually flips one close
    // greedy argmax. This is inherent to KV/state reuse (no production engine
    // guarantees bit-identity across a cache hit vs recompute); we accept it for
    // the ~90%-of-prefill saving on multi-turn agentic Qwen3.5 (per user opt-in).
    let recurrent_strict = matches!(&state.model, ModelKind::Qwen35(_));
    // Gemma-4's assistant generation prompt `<|turn>model\n` is also 3 tokens
    // (105, "model", "\n"), the same length as Qwen3's `<|im_start|>assistant\n`,
    // so the same `QWEN3_GEN_SUFFIX_LEN` snapshot-trim path applies. Pure-
    // attention arch (no recurrent state), so the trimmed-replay path is sound.
    let prefix_cache_eligible = tq_bits.is_none()
        && matches!(
            &state.model,
            ModelKind::Qwen3(_) | ModelKind::Llama(_) | ModelKind::Qwen35(_) | ModelKind::Gemma4(_)
        );
    let mut prefill_start = 0usize;
    if prefix_cache_eligible {
        // Reclaim stale snapshots (default TTL 5 min). Run on every turn —
        // cheap O(entries) check + actual free when stale. Without this,
        // a forgotten conversation pins ~2.5 GB of KV until the model
        // itself unloads (which can be 5–60 min depending on
        // `idle_unload_secs`).
        state.prefix_cache.evict_idle();
        // Memory-pressure guard: if this process is already over the configured
        // RSS budget, drop the prefix-cache KV now (weights stay loaded) and
        // return MLX's pool to the OS. Trades this turn's potential cache hit
        // for a bounded footprint. Disabled (None / 0) by default.
        if let Some(limit) = gen_opt.kv_release_rss_mib.filter(|&m| m > 0) {
            let rss = rss_mib();
            if rss > limit as f64 {
                let freed = state.prefix_cache.clear();
                unsafe {
                    mlx_sys::mlx_clear_cache();
                }
                tracing::info!(
                    "[local-mlx-native] memory pressure: RSS {:.0} MiB > {} MiB budget → released prefix-cache KV ({})",
                    rss,
                    limit,
                    fmt_bytes(freed),
                );
            }
        }
        if let Some(hit) = state.prefix_cache.find_longest_match(&prompt) {
            // Need at least 1 new token to prefill for logits sampling. Full
            // match would require a 1-token re-forward edge case we don't
            // bother with — fall back to full prefill (rare in practice
            // since the assistant generation suffix changes each turn).
            if hit.tokens.len() < prompt.len() {
                for (slot, snap_opt) in cache.iter_mut().zip(hit.caches.iter()) {
                    *slot = snap_opt.as_ref().and_then(|s| s.try_snapshot());
                }
                rope_offset = hit.rope_offset;
                prefill_start = hit.tokens.len();
                tracing::info!(
                    "[local-mlx-native] prefix cache HIT — restored {} KV tokens \
                     (skip {:.1}% of prefill), prefill suffix = {} tokens",
                    prefill_start,
                    100.0 * (prefill_start as f64) / (prompt.len() as f64),
                    prompt.len() - prefill_start,
                );
            } else {
                tracing::info!(
                    "[local-mlx-native] prefix cache full match ({} tokens) — skip restore, full prefill",
                    hit.tokens.len(),
                );
            }
        } else {
            tracing::info!(
                "[local-mlx-native] prefix cache miss (entries={}) — full prefill of {} tokens",
                state.prefix_cache.len(),
                prompt.len(),
            );
        }
    }

    use super::mlx_lm::models::qwen3::ModelInput;
    use mlx_rs::module::Module;

    // Per Qwen3 model card: temp=0.6 in thinking mode, 0.7 in non-thinking mode
    // (https://huggingface.co/Qwen/Qwen3-4B#best-practices).
    //
    // Pure greedy (temp=0) is a trap for Qwen3-4B-4bit: a single biased token
    // (a 4-bit precision artifact — e.g. preferring `!` to start a markdown
    // image preview for search results) becomes a deterministic dead end the
    // model can never escape. Empirically Qwen3-4B-4bit + greedy + 14 K-token
    // tools-heavy prompts emits `![](...)` as turn 1 every single time.
    // Sampling at the recommended temp gives the correct `<tool_call>` start
    // a fair chance against the artifact.
    let decode_temperature = gen_opt
        .temperature
        .map(|t| t.clamp(0.0_f32, 4.0_f32))
        .unwrap_or_else(|| match &state.model {
            ModelKind::Gemma2(_)
            | ModelKind::Gemma3(_)
            | ModelKind::Gemma4(_)
            | ModelKind::BonsaiQ1(_) => 0.65_f32,
            ModelKind::Qwen3(_) | ModelKind::Qwen35(_) | ModelKind::Ouro(_) => {
                if gen_opt.enable_thinking.unwrap_or(false) {
                    0.6_f32
                } else {
                    0.7_f32
                }
            }
            // DeepSeek recommends ~0.6; greedy (0.0) invites loops on the 4-bit
            // MoE since we leave it off the prefix cache for v1.
            ModelKind::DeepSeekV2(_) => 0.6_f32,
            _ => 0.0_f32,
        });
    // Pure greedy (temp=0) plus rep_penalty=1.0 is a recipe for loops once
    // the KV window slides and the model loses early context. Default to a
    // mild penalty for Qwen / Llama transformer paths so greedy decoding
    // still breaks out of `7 7 7 …` cycles; Gemma / Bonsai already use
    // sampling + a stronger penalty.
    let decode_repetition_penalty = gen_opt
        .repetition_penalty
        .map(|p| p.clamp(1.0_f32, 2.0_f32))
        .unwrap_or_else(|| match &state.model {
            ModelKind::Gemma2(_)
            | ModelKind::Gemma3(_)
            | ModelKind::Gemma4(_)
            | ModelKind::BonsaiQ1(_) => 1.15_f32,
            ModelKind::Qwen3(_)
            | ModelKind::Qwen35(_)
            | ModelKind::Llama(_)
            | ModelKind::Ouro(_)
                if decode_temperature == 0.0_f32 =>
            {
                1.05_f32
            }
            _ => 1.0_f32,
        });
    // Top-k / top-p truncation. Precedence is user setting → checkpoint's own
    // `generation_config.json` → off. "Off" is the historical behaviour (an
    // untruncated draw over the whole vocabulary), so a checkpoint that
    // publishes nothing and a user who sets nothing get exactly what they got
    // before. Gemma 4 publishes `top_k: 64, top_p: 0.95`, so it now samples
    // the way its own checkpoint asks to be sampled.
    //
    // A setting of `0` (top_k) or `1.0` (top_p) is an explicit "disable" and
    // must beat the checkpoint default, which is why the `unwrap_or_else` sits
    // outside the validity filter — `sample_with` treats those values as off.
    let decode_truncation = SamplingTruncation {
        top_k: gen_opt
            .top_k
            .map(|k| k as i32)
            .or(state.gen_defaults.top_k),
        top_p: gen_opt.top_p.or(state.gen_defaults.top_p),
    };
    let mut recent_decode_ids: VecDeque<u32> = VecDeque::new();

    // Structural tool-call enforcement state — Qwen3 family only. When we see
    // `<tool_call>` open (token 151657), we forbid stop tokens until the
    // matching `</tool_call>` close (token 151658). Without this guard a
    // 4-bit quantized Qwen3 routinely emits `<|im_end|>` mid-args and the
    // parser receives unclosed JSON.
    const QWEN3_TOOL_CALL_OPEN: u32 = 151_657;
    const QWEN3_TOOL_CALL_CLOSE: u32 = 151_658;
    const QWEN3_STOP_TOKENS_TO_MASK: &[u32] = &[
        151_645, /* <|im_end|> */
        151_643, /* <|endoftext|> */
    ];
    let qwen3_family = matches!(&state.model, ModelKind::Qwen3(_) | ModelKind::Qwen35(_));
    let mut inside_tool_call = false;

    let max_tokens: usize = max_new_tokens;
    let mut buffer: Vec<Array> = Vec::new();
    let mut hit_stop = false;
    let mut generated_count = 0usize;

    // Throughput markers — used by `mlx_log_throughput` at end-of-turn so the
    // operator can see prefill tok/s and decode tok/s without doing log math.
    let turn_start = std::time::Instant::now();
    let mut prefill_done_at: Option<std::time::Instant> = None;
    let prompt_token_count = prompt.len();
    let rss_start = rss_mib();
    let (mlx_a0, mlx_c0, _) = mlx_mem_mib();
    tracing::info!(
        "[local-mlx-native][mem] generate start — rss={:.0} MiB | mlx active={:.0} cache={:.0} MiB | prompt={} max_new={}, temperature={}, repetition_penalty={}",
        rss_start,
        mlx_a0,
        mlx_c0,
        prompt.len(),
        max_new_tokens,
        decode_temperature,
        decode_repetition_penalty
    );

    let scan = super::mlx_lm::models::mamba2::SequentialScan;

    // Chunked prefill — split long prompts into bounded chunks so the lazy
    // graph (transformer working set) doesn't balloon to 10+ GB on Metal.
    // Each chunk's KV updates are materialized before the next chunk runs.
    // Only the final chunk's logits feed sampling — lm_head over earlier
    // chunks remains unevaluated (lazy graph pruning).
    //
    // 512 is a sweet spot on M-series: small enough that activations fit in
    // GPU caches, large enough that per-chunk dispatch overhead amortizes.
    //
    // Overridable via `prefill_chunk_tokens` in settings.json so the value can
    // be swept without a rebuild. It is worth sweeping: chunk size is a whole
    // scheduling regime, not a constant factor, and the sweet spot moves with
    // the machine, the model's activation width, and the prompt length. Clamped
    // to a sane band because both ends fail badly — tiny chunks pay per-chunk
    // dispatch on every one of them, huge chunks materialize an activation
    // graph the GPU has to hold at once.
    const PREFILL_CHUNK_DEFAULT: usize = 512;
    let prefill_chunk = gen_opt
        .prefill_chunk_tokens
        .map(|c| (c as usize).clamp(32, 4096))
        .unwrap_or(PREFILL_CHUNK_DEFAULT);
    // SSM-only architectures consume the full sequence at once via the
    // sequential scan; chunking would force re-doing the scan and break
    // their convolution windows. Skip for Mamba / Falcon-Mamba.
    let chunked_supported = !matches!(
        &state.model,
        // DeepSeek-V2 v1 has no forward_hidden/forward_last_token yet → run
        // single-shot prefill only (excluded here so it never reaches the
        // chunked path's unreachable!() catch-all).
        ModelKind::Mamba2(_) | ModelKind::FalconMamba(_) | ModelKind::DeepSeekV2(_)
    );

    // Recurrent arches (Qwen3.5): the SSM/conv state can't be trimmed, so the
    // prefix cache must snapshot it at a CLEAN token boundary that the next turn
    // reproduces byte-for-byte. The generation prompt
    // (`<|im_start|>assistant\n<think>\n`) ends in a regular `\n` that BPE-merges
    // with the reasoning text generated next turn, so a key ending there never
    // re-matches (verified: it always missed). Instead we snapshot just before
    // the FINAL `<|im_start|>` (the gen-prompt start): the preceding `\n` (after
    // the last `<|im_end|>`) is always followed by the special `<|im_start|>`
    // token, which never merges → the key is stable across turns. Within one
    // agentic request each turn appends `[assistant tool-call][tool result]`, so
    // this boundary is a strict prefix of the next turn's prompt → HIT, and only
    // the newest turn's tokens are re-prefilled.
    let recurrent_snap_at: Option<usize> = if recurrent_strict {
        // `tokenizer` (the `&mut state.tokenizer` bound earlier for detokenize)
        // is reborrowed immutably here — token_to_id takes `&self`.
        tokenizer
            .token_to_id("<|im_start|>")
            .and_then(|im| prompt.iter().rposition(|&t| t == im))
            .filter(|&b| {
                b > prefill_start
                    && b >= super::mlx_lm::prefix_cache::MIN_PREFIX_LEN
                    && b < prompt.len()
            })
    } else {
        None
    };
    let mut recurrent_snapshot: Option<Vec<Option<super::mlx_lm::cache::KvCache>>> = None;

    // Force chunked path when we restored from prefix cache — single-shot
    // would feed the entire `prompt_tokens` array (length N) into forward,
    // which would re-walk positions [0..prefill_start] that are already
    // populated by the restored KV state. Chunked path slices the prompt
    // from `prefill_start` onwards, only doing work for the new suffix.
    let logits = if (!chunked_supported || prompt.len() <= prefill_chunk) && prefill_start == 0 {
        // Small prompt or SSM arch — single forward pass.
        match &mut state.model {
            ModelKind::Qwen3(m) => {
                let input = ModelInput {
                    inputs: &prompt_tokens,
                    mask: None,
                    cache: &mut cache,
                    rope_offset,
                };
                m.forward(input)
                    .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?
            }
            ModelKind::Qwen35(m) => m
                .forward(&prompt_tokens, &mut cache, rope_offset)
                .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?,
            ModelKind::Llama(m) => {
                let input = ModelInput {
                    inputs: &prompt_tokens,
                    mask: None,
                    cache: &mut cache,
                    rope_offset,
                };
                m.forward(input)
                    .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?
            }
            ModelKind::Gemma2(m) => {
                let input = ModelInput {
                    inputs: &prompt_tokens,
                    mask: None,
                    cache: &mut cache,
                    rope_offset,
                };
                m.forward(input)
                    .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?
            }
            ModelKind::Gemma3(m) => {
                let input = ModelInput {
                    inputs: &prompt_tokens,
                    mask: None,
                    cache: &mut cache,
                    rope_offset,
                };
                m.forward(input)
                    .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?
            }
            ModelKind::Gemma4(m) => m
                .forward(&prompt_tokens, &mut cache, rope_offset)
                .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?,
            ModelKind::DeepSeekV2(m) => m
                .forward(&prompt_tokens, &mut cache, rope_offset)
                .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?,
            ModelKind::Mamba2(m) => m
                .forward(&prompt_tokens, &mut cache, &scan)
                .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?,
            ModelKind::FalconMamba(m) => m
                .forward(&prompt_tokens, &mut cache)
                .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?,
            ModelKind::BonsaiQ1(b) => b
                .gpu
                .forward_all_logits_native(&prompt_tokens, &mut cache, rope_offset)
                .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?,
            ModelKind::Ouro(m) => {
                let input = ModelInput {
                    inputs: &prompt_tokens,
                    mask: None,
                    cache: &mut cache,
                    rope_offset,
                };
                m.forward(input)
                    .map_err(|e| anyhow::anyhow!("prefill forward failed: {e:?}"))?
            }
        }
    } else {
        let to_prefill = prompt.len() - prefill_start;
        let chunk_count = to_prefill.div_ceil(prefill_chunk).max(1);
        tracing::info!(
            "[local-mlx-native] chunked prefill: {} tokens (skipping {} restored) in {} chunks of ≤{}",
            to_prefill,
            prefill_start,
            chunk_count,
            prefill_chunk,
        );
        let mut chunk_logits: Option<mlx_rs::Array> = None;
        let mut cursor = prefill_start;
        while cursor < prompt.len() {
            let mut end = (cursor + prefill_chunk).min(prompt.len());
            // Break the chunk exactly at the recurrent snapshot boundary so the
            // SSM/conv state can be captured aligned to the cache key.
            if let Some(b) = recurrent_snap_at {
                if cursor < b && b < end {
                    end = b;
                }
            }
            let chunk_arr = Array::from(&prompt[cursor..end]).index(NewAxis);
            let is_last_chunk = end == prompt.len();
            // Final chunk: project LM head only on the last position →
            // 1 × vocab values. Intermediate chunks: skip LM head entirely
            // via `forward_hidden` — KV writes are the only side-effect we
            // need, and `(L × vocab)` of wasted lm_head compute per
            // intermediate chunk dominated prefill time for tools-heavy
            // prompts (Qwen3 vocab = 151 936).
            //
            // Saving for 14 K-token prompt, chunk 512: ~28 chunks × 511 ×
            // 152 K = ~2.2 G MAC ops avoided on the LM head alone.
            let out = match &mut state.model {
                ModelKind::Qwen3(m) => {
                    let input = ModelInput {
                        inputs: &chunk_arr,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    if is_last_chunk {
                        m.forward_last_token(input)
                            .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?
                    } else {
                        m.forward_hidden(input)
                            .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?
                    }
                }
                ModelKind::Qwen35(m) => m
                    .forward(&chunk_arr, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?,
                ModelKind::Llama(m) => {
                    let input = ModelInput {
                        inputs: &chunk_arr,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    m.forward(input)
                        .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?
                }
                ModelKind::Gemma2(m) => {
                    let input = ModelInput {
                        inputs: &chunk_arr,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    m.forward(input)
                        .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?
                }
                ModelKind::Gemma3(m) => {
                    let input = ModelInput {
                        inputs: &chunk_arr,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    m.forward(input)
                        .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?
                }
                ModelKind::Gemma4(m) => m
                    .forward(&chunk_arr, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?,
                ModelKind::BonsaiQ1(b) => b
                    .gpu
                    .forward_all_logits_native(&chunk_arr, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?,
                ModelKind::Ouro(m) => {
                    let input = ModelInput {
                        inputs: &chunk_arr,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    if is_last_chunk {
                        m.forward_last_token(input)
                            .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?
                    } else {
                        m.forward_hidden(input)
                            .map_err(|e| anyhow::anyhow!("prefill chunk forward failed: {e:?}"))?
                    }
                }
                // SSM paths are routed to single-shot above.
                _ => unreachable!("chunked_supported guard excludes Mamba variants"),
            };
            rope_offset += end - cursor;
            // Materialize this chunk's KV cache writes before next chunk
            // attends to them.
            eval_all_caches(&mut cache)
                .map_err(|e| anyhow::anyhow!("prefill chunk cache eval failed: {e:?}"))?;
            // Recurrent prefix-cache snapshot: capture the cache state aligned to
            // the clean boundary (just before the gen prompt) once we reach it.
            if let Some(b) = recurrent_snap_at {
                if end == b && recurrent_snapshot.is_none() {
                    recurrent_snapshot =
                        super::mlx_lm::prefix_cache::PrefixCache::snapshot_layers_trimmed(
                            &cache, 0,
                        );
                }
            }
            if is_last_chunk {
                chunk_logits = Some(out);
            } else {
                // **Force eval of the intermediate chunk's hidden output**
                // so MLX materializes (then can release) the chunk's lazy
                // graph nodes — without this, MLX's buffer pool grows to
                // ~10 GB during a 14 K-token prefill (28 chunks of ~430 MB
                // activations each pinned as lazy graph holds refs). After
                // this eval `out` goes out of scope and its underlying
                // buffer becomes pool-reclaimable on the next clear.
                let _ = eval(&[out]);
                // Periodic pool flush every 8 chunks (~4 K tokens of
                // intermediates) to keep transient peak memory bounded.
                // Each call is a sync barrier (~2-5 ms) but reclaims
                // hundreds of MB → net win on memory-constrained machines.
                if cursor > 0 && (cursor / prefill_chunk) % 8 == 7 {
                    unsafe {
                        mlx_sys::mlx_clear_cache();
                    }
                }
            }
            cursor = end;
        }
        chunk_logits.expect("at least one chunk processed")
    };
    if (!chunked_supported || prompt.len() <= prefill_chunk) && prefill_start == 0 {
        rope_offset += prompt.len();
        eval_all_caches(&mut cache)
            .map_err(|e| anyhow::anyhow!("prefill cache eval failed: {e:?}"))?;
    }
    // Sample first decode token. Skip the redundant eval calls — `.item()`
    // below forces the whole graph chain (logits → sample → token) in one
    // CPU↔GPU sync instead of four.
    let last_logits = logits.index((.., -1, ..));
    let forbidden_now: &[u32] = if qwen3_family && inside_tool_call {
        QWEN3_STOP_TOKENS_TO_MASK
    } else {
        &[]
    };
    let mut next_token = sample_decode_token_id(
        &last_logits,
        decode_temperature,
        decode_repetition_penalty,
        decode_truncation,
        &recent_decode_ids,
        forbidden_now,
    )?;
    let first_id = next_token.item::<u32>();
    mlx_recent_decode_push(&mut recent_decode_ids, first_id, MLX_REPEN_DECODE_WINDOW);
    // Track tool-call open/close so subsequent decode steps can mask stops.
    if qwen3_family {
        if first_id == QWEN3_TOOL_CALL_OPEN {
            inside_tool_call = true;
        } else if first_id == QWEN3_TOOL_CALL_CLOSE {
            inside_tool_call = false;
        }
    }
    if is_stop(first_id) {
        hit_stop = true;
    } else {
        buffer.push(next_token.clone());
        generated_count += 1;
    }

    // Stream the synthetic opening `<think>` first (template prefilled it into
    // the prompt, so the model never emits it) — keeps the reasoning/answer
    // split well-formed downstream. Sent before any generated text is flushed.
    if thinking_prefilled && tx.blocking_send("<think>\n".to_string()).is_err() {
        mlx_release_after_turn();
        return Ok(());
    }

    {
        let (ma, mc, mp) = mlx_mem_mib();
        tracing::info!(
            "[local-mlx-native][mem] after prefill — rss={:.0}(Δ{:.0}) | mlx active={:.0} cache={:.0} peak={:.0} MiB",
            rss_mib(),
            rss_mib() - rss_start,
            ma,
            mc,
            mp
        );
        mlx_log_kv_cache("after prefill", &cache);
    }
    prefill_done_at = Some(std::time::Instant::now());

    // Store post-prefill KV state under `prompt[..prompt.len() - gen_suffix]`
    // so the NEXT turn's prompt (which has different assistant content where
    // turn-N's gen suffix used to be) can still hit the cache for the shared
    // body. Skip when caching disabled (TQ active, non-Qwen arch) or when
    // the trimmed key would be below the minimum useful length.
    if prefix_cache_eligible {
        if recurrent_strict {
            // Recurrent arches (Qwen3.5): store the SSM/conv snapshot captured
            // mid-prefill at the clean boundary (just before the gen prompt).
            // Key = prompt up to that boundary; the state aligns exactly there
            // (no trim possible for recurrent state — that misalignment was the
            // old non-determinism bug).
            if let (Some(snap), Some(at)) = (recurrent_snapshot.take(), recurrent_snap_at) {
                let key: Vec<u32> = prompt[..at].to_vec();
                state.prefix_cache.store(key, snap, at);
                tracing::info!(
                    "[local-mlx-native] prefix cache stored ({} tokens, recurrent boundary, entries now={})",
                    at,
                    state.prefix_cache.len(),
                );
            }
        } else {
            let snapshot_key_len = prompt.len().saturating_sub(QWEN3_GEN_SUFFIX_LEN);
            if snapshot_key_len >= super::mlx_lm::prefix_cache::MIN_PREFIX_LEN {
                // Snapshot now (post-suffix prefill) and trim by gen_suffix so
                // the cached KV state aligns with the trimmed key length.
                let snap = super::mlx_lm::prefix_cache::PrefixCache::snapshot_layers_trimmed(
                    &cache,
                    QWEN3_GEN_SUFFIX_LEN,
                );
                if let Some(snap) = snap {
                    let key: Vec<u32> = prompt[..snapshot_key_len].to_vec();
                    state.prefix_cache.store(key, snap, snapshot_key_len);
                    tracing::info!(
                        "[local-mlx-native] prefix cache stored ({} tokens, entries now={})",
                        snapshot_key_len,
                        state.prefix_cache.len(),
                    );
                }
            }
        }
    }

    // Decode operates on single tokens — the prefill's working-set pool
    // (often >10 GB after a 2k-token prompt) is dead weight from here on.
    // Drop MLX's cached buffers back to the OS so RSS doesn't carry them
    // through every decode step.
    unsafe {
        mlx_sys::mlx_clear_cache();
    }

    // One decode forward over a `[1, 1]` token array → logits. Shared by the
    // async-lookahead path below; keeps the 8-arch dispatch in one place.
    macro_rules! decode_forward {
        ($inputs:expr) => {
            match &mut state.model {
                ModelKind::Qwen3(m) => m
                    .forward(ModelInput {
                        inputs: $inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    })
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Qwen35(m) => m
                    .forward($inputs, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Llama(m) => m
                    .forward(ModelInput {
                        inputs: $inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    })
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Gemma2(m) => m
                    .forward(ModelInput {
                        inputs: $inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    })
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Gemma3(m) => m
                    .forward(ModelInput {
                        inputs: $inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    })
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Gemma4(m) => m
                    .forward($inputs, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::DeepSeekV2(m) => m
                    .forward($inputs, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Mamba2(m) => m
                    .forward($inputs, &mut cache, &scan)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::FalconMamba(m) => m
                    .forward($inputs, &mut cache)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::BonsaiQ1(b) => b
                    .gpu
                    .forward_all_logits_native($inputs, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Ouro(m) => m
                    .forward(ModelInput {
                        inputs: $inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    })
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
            }
        };
    }

    // Gemma-4 emits harmony-style structural tokens — `<|channel>…<channel|>`
    // (thinking) and `<|tool_call>call:NAME{…}<tool_call|>` (tool calls, with
    // `<|"|>`-wrapped string args). The downstream parser needs those markers,
    // so DON'T strip specials when detokenizing Gemma-4. Every other arch
    // strips them (`<think>`/`<tool_call>` are plain tokens there).
    let decode_skip_special = !matches!(state.model, ModelKind::Gemma4(_));

    // Drain the 20-token stream buffer: detokenize and push to the channel.
    // `return`s the whole turn early if the receiver hung up. Shared verbatim
    // between the async and synchronous decode paths.
    macro_rules! flush_buffer {
        () => {
            if buffer.len() >= 20 {
                eval(&buffer).map_err(|e| anyhow::anyhow!("eval failed: {e:?}"))?;
                if generated_count <= 20 {
                    let (ma, mc, _) = mlx_mem_mib();
                    tracing::info!(
                        "[local-mlx-native][mem] decode step {} — rss={:.0}(Δ{:.0}) | mlx active={:.0} cache={:.0} MiB",
                        generated_count, rss_mib(), rss_mib() - rss_start, ma, mc
                    );
                }
                let slice: Vec<u32> = buffer.drain(..).map(|t| t.item::<u32>()).collect();
                if generated_count <= 20 {
                    let decoded_preview = tokenizer.decode(&slice, false).unwrap_or_default();
                    tracing::info!(
                        "[local-mlx-native] first {} token ids: {:?} → {:?}",
                        slice.len(), slice, decoded_preview
                    );
                }
                let text = tokenizer
                    .decode(&slice, decode_skip_special)
                    .map_err(|e| anyhow::anyhow!("decode failed: {e:?}"))?;
                if !text.is_empty() && tx.blocking_send(text).is_err() {
                    mlx_release_after_turn();
                    return Ok(());
                }
                // Periodic pool flush during long decode (≈ every 100 tokens) —
                // reclaims per-step intermediates that aren't freed until an eval
                // barrier. The sync (~10-20 ms) pays for itself on 1000+ tokens.
                if generated_count > 0 && generated_count % 100 == 0 {
                    unsafe { mlx_sys::mlx_clear_cache(); }
                }
            }
        };
    }

    // Async-lookahead decode is used whenever no repetition penalty is active
    // (the default for Qwen3/Llama transformer chat). It issues token N+1's
    // forward and `async_eval`s it BEFORE pulling token N to the host, so the
    // GPU runs the next step while we detokenize/commit the current one —
    // hiding the per-token CPU↔GPU sync behind compute. Greedy output is
    // bit-identical to the synchronous path (same logits, same argmax); the
    // only lag is the tool-call stop-mask trailing by one token, corrected at
    // tool-call boundaries via a re-sample. With a repetition penalty the
    // window must reflect every emitted token, so we stay synchronous.
    let pipeline_decode = decode_repetition_penalty <= 1.0;

    if pipeline_decode && !hit_stop && max_tokens > 1 {
        let forbidden_for = |inside: bool| -> &'static [u32] {
            if qwen3_family && inside {
                QWEN3_STOP_TOKENS_TO_MASK
            } else {
                &[]
            }
        };

        // Bootstrap: forward token #1 → its successor, dispatched async.
        let mut pending: Array = {
            let inputs = next_token
                .reshape(&[1, 1])
                .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?;
            let logits = decode_forward!(&inputs);
            rope_offset += 1;
            let src = logits.index((.., -1, ..));
            let tok = sample_decode_token_id(
                &src,
                decode_temperature,
                decode_repetition_penalty,
                decode_truncation,
                &recent_decode_ids,
                forbidden_for(inside_tool_call),
            )?;
            async_eval(std::slice::from_ref(&tok))
                .map_err(|e| anyhow::anyhow!("async_eval failed: {e:?}"))?;
            tok
        };

        while generated_count < max_tokens {
            let tok = pending;
            // Do we need a token after committing `tok`? If not, skip its forward.
            let need_more = generated_count + 1 < max_tokens;

            // Issue the successor's forward NOW (before pulling `tok`) so the GPU
            // overlaps it with the host-side commit below.
            let mut cand: Option<(Array, Array)> = if need_more {
                let inputs = tok
                    .reshape(&[1, 1])
                    .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?;
                let logits = decode_forward!(&inputs);
                rope_offset += 1;
                let src = logits.index((.., -1, ..));
                let ct = sample_decode_token_id(
                    &src,
                    decode_temperature,
                    decode_repetition_penalty,
                    decode_truncation,
                    &recent_decode_ids,
                    forbidden_for(inside_tool_call),
                )?;
                async_eval(std::slice::from_ref(&ct))
                    .map_err(|e| anyhow::anyhow!("async_eval failed: {e:?}"))?;
                Some((ct, src))
            } else {
                None
            };

            // Pull the in-flight token — its GPU work overlapped with `cand`.
            let token_id = tok.item::<u32>();
            mlx_recent_decode_push(&mut recent_decode_ids, token_id, MLX_REPEN_DECODE_WINDOW);
            let was_inside = inside_tool_call;
            if qwen3_family {
                if token_id == QWEN3_TOOL_CALL_OPEN {
                    inside_tool_call = true;
                    tracing::info!(
                        "[local-mlx-native] tool_call OPEN at step {} (token 151657) — masking stops",
                        generated_count
                    );
                } else if token_id == QWEN3_TOOL_CALL_CLOSE {
                    inside_tool_call = false;
                    tracing::info!(
                        "[local-mlx-native] tool_call CLOSE at step {} (token 151658) — un-masking stops",
                        generated_count
                    );
                }
            }
            // Stale-mask correction: `cand` was sampled before we knew `tok`
            // flipped the tool-call state. Re-sample with the right mask (rare —
            // only at `<tool_call>` boundaries, so the lost overlap is moot).
            if qwen3_family && inside_tool_call != was_inside {
                if let Some((ct, src)) = cand.as_mut() {
                    *ct = sample_decode_token_id(
                        src,
                        decode_temperature,
                        decode_repetition_penalty,
                        decode_truncation,
                        &recent_decode_ids,
                        forbidden_for(inside_tool_call),
                    )?;
                }
            }

            if is_stop(token_id) {
                if was_inside {
                    tracing::error!(
                        "[local-mlx-native] STOP TOKEN {} emitted INSIDE <tool_call> at step {}. \
                         Mask must have failed — verify forbidden_ids was non-empty for this step.",
                        token_id,
                        generated_count,
                    );
                }
                hit_stop = true;
                break; // discard `cand` — its forward is harmless wasted work
            }
            buffer.push(tok);
            generated_count += 1;
            flush_buffer!();

            match cand {
                Some((ct, _src)) => pending = ct,
                None => break, // just committed the final requested token
            }
        }
    } else if !pipeline_decode {
        for _step in 1..max_tokens {
            if hit_stop {
                break;
            }
            let inputs = next_token
                .reshape(&[1, 1])
                .map_err(|e| anyhow::anyhow!("reshape failed: {e:?}"))?;
            let logits = match &mut state.model {
                ModelKind::Qwen3(m) => {
                    let decode_input = ModelInput {
                        inputs: &inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    m.forward(decode_input)
                        .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?
                }
                ModelKind::Qwen35(m) => m
                    .forward(&inputs, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Llama(m) => {
                    let decode_input = ModelInput {
                        inputs: &inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    m.forward(decode_input)
                        .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?
                }
                ModelKind::Gemma2(m) => {
                    let decode_input = ModelInput {
                        inputs: &inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    m.forward(decode_input)
                        .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?
                }
                ModelKind::Gemma3(m) => {
                    let decode_input = ModelInput {
                        inputs: &inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    m.forward(decode_input)
                        .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?
                }
                ModelKind::Gemma4(m) => m
                    .forward(&inputs, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::DeepSeekV2(m) => m
                    .forward(&inputs, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Mamba2(m) => m
                    .forward(&inputs, &mut cache, &scan)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::FalconMamba(m) => m
                    .forward(&inputs, &mut cache)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::BonsaiQ1(b) => b
                    .gpu
                    .forward_all_logits_native(&inputs, &mut cache, rope_offset)
                    .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?,
                ModelKind::Ouro(m) => {
                    let decode_input = ModelInput {
                        inputs: &inputs,
                        mask: None,
                        cache: &mut cache,
                        rope_offset,
                    };
                    m.forward(decode_input)
                        .map_err(|e| anyhow::anyhow!("decode forward failed: {e:?}"))?
                }
            };
            rope_offset += 1;
            // Single sync per decode step: `y.item()` below transitively forces
            // the whole graph (forward → cache update → logits index → sample).
            // Previous code emitted 4 separate eval() calls — each ~5–10 ms of
            // CPU↔GPU dispatch overhead — capping decode at ~12 tok/s. Folding
            // them yields ~30–50 % decode speedup on M-series.
            let last_logits = logits.index((.., -1, ..));
            let forbidden_now: &[u32] = if qwen3_family && inside_tool_call {
                QWEN3_STOP_TOKENS_TO_MASK
            } else {
                &[]
            };
            let y = sample_decode_token_id(
                &last_logits,
                decode_temperature,
                decode_repetition_penalty,
                decode_truncation,
                &recent_decode_ids,
                forbidden_now,
            )?;
            let token_id = y.item::<u32>();
            mlx_recent_decode_push(&mut recent_decode_ids, token_id, MLX_REPEN_DECODE_WINDOW);
            if qwen3_family {
                if token_id == QWEN3_TOOL_CALL_OPEN {
                    inside_tool_call = true;
                    tracing::info!(
                        "[local-mlx-native] tool_call OPEN at step {} (token 151657) — masking stops",
                        generated_count
                    );
                } else if token_id == QWEN3_TOOL_CALL_CLOSE {
                    inside_tool_call = false;
                    tracing::info!(
                        "[local-mlx-native] tool_call CLOSE at step {} (token 151658) — un-masking stops",
                        generated_count
                    );
                }
            }
            next_token = y;

            if is_stop(token_id) {
                if inside_tool_call {
                    // Should not happen if logit masking is active. Loud diagnostic so
                    // we can tell whether the binary actually has the mask applied vs.
                    // some other path emitting a stop token mid tool_call.
                    tracing::error!(
                        "[local-mlx-native] STOP TOKEN {} emitted INSIDE <tool_call> at step {}. \
                     Mask must have failed — verify forbidden_ids was non-empty for this step.",
                        token_id,
                        generated_count,
                    );
                }
                hit_stop = true;
                break;
            }
            buffer.push(next_token.clone());
            generated_count += 1;

            let should_flush = buffer.len() >= 20;
            if should_flush {
                eval(&buffer).map_err(|e| anyhow::anyhow!("eval failed: {e:?}"))?;
                if generated_count <= 20 {
                    let (ma, mc, _) = mlx_mem_mib();
                    tracing::info!(
                        "[local-mlx-native][mem] decode step {} — rss={:.0}(Δ{:.0}) | mlx active={:.0} cache={:.0} MiB",
                        generated_count,
                        rss_mib(),
                        rss_mib() - rss_start,
                        ma,
                        mc
                    );
                }
                let slice: Vec<u32> = buffer.drain(..).map(|t| t.item::<u32>()).collect();
                if generated_count <= 20 {
                    let decoded_preview = tokenizer.decode(&slice, false).unwrap_or_default();
                    tracing::info!(
                        "[local-mlx-native] first {} token ids: {:?} → {:?}",
                        slice.len(),
                        slice,
                        decoded_preview
                    );
                }
                let text = tokenizer
                    .decode(&slice, decode_skip_special)
                    .map_err(|e| anyhow::anyhow!("decode failed: {e:?}"))?;
                if !text.is_empty() && tx.blocking_send(text).is_err() {
                    mlx_release_after_turn();
                    return Ok(());
                }
                // Periodic pool flush during long decode. Each forward pass
                // adds small intermediate buffers (attention scores, MLP
                // activations) that aren't reclaimed until the next eval
                // barrier. Without this, decode of 1000+ tokens grows pool by
                // ~1 GB. The flush is a sync barrier (~10-20 ms) but reclaims
                // hundreds of MB → net win for long generation.
                //
                // Every 5 flushes = every 100 decode tokens. Trade-off:
                // - Too frequent → wastes time on syncs that have nothing to free.
                // - Too rare → pool grows; user sees ~1 GB jump mid-decode.
                if generated_count > 0 && generated_count % 100 == 0 {
                    unsafe {
                        mlx_sys::mlx_clear_cache();
                    }
                }
            }
        }
    } // end synchronous (repetition-penalty) decode path
      // Flush any tokens still in the buffer regardless of whether we stopped
      // on EOS or hit max_new_tokens. PREVIOUSLY the `hit_stop` branch returned
      // without flushing, which dropped up to the last 19 tokens (buffer flush
      // threshold). When the model emits `<tool_call>{...}</tool_call><|im_end|>`
      // and the `</tool_call>` happens to sit in the un-flushed tail, the parser
      // never sees the closing tag and rejects the tool call as "truncated".
      // The fix here is structural: drain the buffer on every exit path.
    if !buffer.is_empty() {
        eval(&buffer).map_err(|e| anyhow::anyhow!("final eval failed: {e:?}"))?;
        let slice: Vec<u32> = buffer.drain(..).map(|t| t.item::<u32>()).collect();
        let text = tokenizer
            .decode(&slice, decode_skip_special)
            .map_err(|e| anyhow::anyhow!("final decode failed: {e:?}"))?;
        if !text.is_empty() {
            let _ = tx.blocking_send(text);
        }
    }

    if hit_stop {
        mlx_log_generate_done("stopped (eos/im_end)", rss_start, generated_count);
    } else {
        mlx_log_generate_done("completed", rss_start, generated_count);
    }
    // Throughput summary + final KV size. Help operator compare runs and
    // spot regressions ("decode dropped from 35 to 14 tok/s after I bumped
    // kv_cache_bits → TurboQuant activated on prefill").
    mlx_log_kv_cache("end of turn", &cache);
    if let Some(prefill_at) = prefill_done_at {
        let prefill_elapsed = prefill_at.duration_since(turn_start);
        let decode_elapsed = std::time::Instant::now().duration_since(prefill_at);
        mlx_log_throughput(
            prompt_token_count,
            prefill_elapsed,
            generated_count,
            decode_elapsed,
        );
    }
    if let Ok(mut g) = last_usage.lock() {
        *g = Some((prompt_token_count, generated_count));
    }
    mlx_release_after_turn();

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum Arch {
    Qwen3,
    Qwen35,
    Llama,
    Gemma2,
    Gemma3,
    Gemma4,
    DeepSeekV2,
    Mamba2,
    FalconMamba,
    BonsaiQ1,
    Ouro,
}

/// Strip the same multimodal prefix as [`load_gemma3_any`] / gemma3 model loader.
fn gemma3_loader_strip_prefix(key: &str) -> &str {
    key.strip_prefix("language_model.").unwrap_or(key)
}

/// True when weight files list a standalone `lm_head` (`lm_head.weight`, quant scales, etc.).
/// Used to override `tie_word_embeddings` for QAT checkpoints that ship `lm_head` but omit
/// `tie_word_embeddings: false` in `config.json`.
fn gemma3_safetensors_has_lm_head(model_dir: &Path) -> anyhow::Result<bool> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let json = std::fs::read_to_string(&index)
            .map_err(|e| anyhow::anyhow!("read gemma3 index: {e}"))?;
        let map: super::mlx_lm::models::gemma3::WeightMap =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse gemma3 index: {e}"))?;
        for raw in map.weight_map.keys() {
            let k = gemma3_loader_strip_prefix(raw);
            if k == "lm_head" || k.starts_with("lm_head.") {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        let loaded = mlx_rs::Array::load_safetensors(&single)
            .map_err(|e| anyhow::anyhow!("load_safetensors keys: {e:?}"))?;
        for raw in loaded.keys() {
            let k = gemma3_loader_strip_prefix(raw.as_str());
            if k == "lm_head" || k.starts_with("lm_head.") {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Load a Gemma-3 [`Model`]. Handles three layouts uniformly:
///
/// 1. Plain bf16 / f32 checkpoint — direct `load_safetensors`.
/// 2. 4-bit / 8-bit uniform quantization (`config.quantization.bits + group_size`
///    at the top level) — runs `nn::quantize` first so the param tree exposes
///    `.inner.{weight,bias}` + `.scales` + `.biases` slots, then maps
///    safetensors keys onto them (including the `language_model.` prefix strip
///    for multimodal Gemma-3 4B/12B checkpoints).
/// 3. `QuantizedEmbedding` workaround — mlx-rs 0.25.3 leaves the
///    `inner/scales/biases` fields without `#[param]` annotations, so
///    `parameters_mut()` misses them. We capture them by raw key and apply
///    directly into the struct (same pattern as `load_qwen3_any`).
fn load_gemma3_any(model_dir: &Path) -> anyhow::Result<super::mlx_lm::models::gemma3::Model> {
    use super::mlx_lm::models::gemma3::{get_gemma3_model_args, Model};
    use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
    use std::collections::HashSet;

    let mut args = get_gemma3_model_args(model_dir)
        .map_err(|e| anyhow::anyhow!("read gemma3 args failed: {e:?}"))?;

    // Some checkpoints (e.g. `mlx-community/gemma-3-1b-it-qat-4bit`) ship a
    // separate quantised `lm_head` even though their `config.json` omits
    // `tie_word_embeddings` (so the HF default of `true` would otherwise
    // make us drop those weights and reuse `embed_tokens.as_linear`, which
    // is the WRONG matrix for the output projection after independent
    // QAT — empirically yields gibberish logits).
    //
    // Detection: scan the safetensors index (or single-file weights map)
    // for any `lm_head.weight` key, stripping the multimodal prefix. If
    // present, force `tie_word_embeddings = false` so the model allocates
    // an `lm_head` slot to receive those weights.
    let has_lm_head = gemma3_safetensors_has_lm_head(model_dir).unwrap_or(false);
    if has_lm_head && args.tie_word_embeddings {
        tracing::info!(
            "[local-mlx-native] gemma3: `lm_head.weight` present in safetensors — \
             overriding `tie_word_embeddings = false` (config default was true)"
        );
        args.tie_word_embeddings = false;
    }

    let model = Model::new(args).map_err(|e| anyhow::anyhow!("Model::new failed: {e:?}"))?;

    // Inspect for quantization. Gemma-3 4-bit checkpoints from mlx-community
    // (`gemma-3-*-it-4bit`) ship a top-level `quantization: {bits, group_size}`.
    // Some wrappers also expose `quantization_config` with the same shape.
    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read config.json failed: {e}"))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse config.json failed: {e}"))?;
    let quant = cfg
        .get("quantization")
        .or_else(|| cfg.get("quantization_config"))
        .and_then(|q| {
            let g = q.get("group_size")?.as_i64()? as i32;
            let b = q.get("bits")?.as_i64()? as i32;
            Some((g, b))
        });

    let mut model = if let Some((group_size, bits)) = quant {
        tracing::info!(
            "[local-mlx-native] quantizing Gemma-3 layers: group_size={group_size}, bits={bits}"
        );
        let m = mlx_rs::nn::quantize(model, Some(group_size), Some(bits))
            .map_err(|e| anyhow::anyhow!("nn::quantize failed: {e:?}"))?;
        m.eval()
            .map_err(|e| anyhow::anyhow!("post-quantize eval failed: {e:?}"))?;
        m
    } else {
        model
    };

    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)
            .map_err(|e| anyhow::anyhow!("read index failed: {e}"))?;
        let map: super::mlx_lm::models::gemma3::WeightMap =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse index failed: {e}"))?;
        let files: HashSet<&String> = map.weight_map.values().collect();
        for f in files {
            shard_files.push(model_dir.join(f));
        }
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            anyhow::bail!(
                "no model.safetensors.index.json or model.safetensors in {}",
                model_dir.display()
            );
        }
        shard_files.push(single);
    }

    let is_quant = quant.is_some();
    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    let mut embed_weight: Option<mlx_rs::Array> = None;
    let mut embed_scales: Option<mlx_rs::Array> = None;
    let mut embed_biases: Option<mlx_rs::Array> = None;

    let strip = |key: &str| -> String {
        key.strip_prefix("language_model.")
            .unwrap_or(key)
            .to_string()
    };

    let mut unfilled_slots: std::collections::HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };
    for shard in &shard_files {
        let loaded = mlx_rs::Array::load_safetensors(shard)
            .map_err(|e| anyhow::anyhow!("read shard {}: {e:?}", shard.display()))?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, value) in loaded {
            let key = strip(raw_key.as_str());
            // QuantizedEmbedding workaround (same as qwen3/llama).
            match key.as_str() {
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
            if let Some(slot) = params.get_mut(key.as_str()) {
                **slot = value;
                total_loaded += 1;
                unfilled_slots.remove(&key);
                continue;
            }
            if is_quant {
                // QuantizedLinear: HF stores `proj.weight` but the param tree
                // exposes it under `proj.inner.weight` after `nn::quantize`.
                if let Some(stripped) = key.strip_suffix(".weight") {
                    let remapped = format!("{stripped}.inner.weight");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
                        continue;
                    }
                }
                if let Some(stripped) = key.strip_suffix(".bias") {
                    let remapped = format!("{stripped}.inner.bias");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
                        continue;
                    }
                }
            }
            total_missed += 1;
            if unmatched_samples.len() < 5 {
                unmatched_samples.push(key);
            }
        }
    }

    if embed_weight.is_some() || embed_scales.is_some() || embed_biases.is_some() {
        match &mut model.model.embed_tokens {
            mlx_rs::quantization::MaybeQuantized::Quantized(q) => {
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
            mlx_rs::quantization::MaybeQuantized::Original(e) => {
                if let Some(w) = embed_weight {
                    e.weight.value = w;
                }
            }
        }
    }

    tracing::info!(
        "[local-mlx-native] gemma3 safetensor load: {total_loaded} matched, {total_missed} unmatched (quantized={is_quant})"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[local-mlx-native] gemma3 unmatched key samples: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled_slots.is_empty() {
        let mut samples: Vec<&String> = unfilled_slots.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[local-mlx-native] gemma3 {} parameter slot(s) NOT populated — first few: {}",
            unfilled_slots.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        anyhow::bail!("no safetensor keys matched the Gemma-3 parameter tree");
    }

    model
        .eval()
        .map_err(|e| anyhow::anyhow!("eval after load failed: {e:?}"))?;
    Ok(model)
}

/// Load a Gemma-4 [`Model`]. Mirrors [`load_gemma3_any`] (post-load
/// `nn::quantize` + safetensors key remap, `language_model.` prefix strip) with
/// two extra `QuantizedEmbedding` workarounds: Gemma-4 carries **two** embedding
/// tables — the usual `embed_tokens` and the Per-Layer-Embedding table
/// `embed_tokens_per_layer` — both of which mlx-rs 0.25.3 leaves without
/// `#[param]`-visible `inner/scales/biases` after quantization.
fn load_gemma4_any(model_dir: &Path) -> anyhow::Result<super::mlx_lm::models::gemma4::Model> {
    use super::mlx_lm::models::gemma4::{
        apply_per_module_quantization, get_gemma4_model_args, Model, QuantPlan, WeightMap,
    };
    use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
    use std::collections::HashSet;

    let args = get_gemma4_model_args(model_dir)
        .map_err(|e| anyhow::anyhow!("read gemma4 args failed: {e:?}"))?;
    let first_kv_shared = args.first_kv_shared();
    let mut model = Model::new(args).map_err(|e| anyhow::anyhow!("Model::new failed: {e:?}"))?;

    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read config.json failed: {e}"))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse config.json failed: {e}"))?;

    // Per-module quantization plan — handles both uniform 4-bit
    // (`gemma-4-*-it-4bit`) and OptiQ mixed-bit (`gemma-4-*-it-OptiQ-4bit`).
    // We scan `model.safetensors.index.json` so quantization is driven by
    // what's actually stored quantized (i.e. has `.scales`) rather than only
    // by the OptiQ override list — works whether the checkpoint relies on the
    // top-level default or declares per-module overrides.
    let weights_index = model_dir.join("model.safetensors.index.json");
    let index_keys_owned: Vec<String> = if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)
            .map_err(|e| anyhow::anyhow!("read index failed: {e}"))?;
        let map: WeightMap =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse index failed: {e}"))?;
        map.weight_map.keys().cloned().collect()
    } else {
        // Single-file safetensors — peek at its key list.
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            anyhow::bail!(
                "no model.safetensors.index.json or model.safetensors in {}",
                model_dir.display()
            );
        }
        let loaded = mlx_rs::Array::load_safetensors(&single)
            .map_err(|e| anyhow::anyhow!("read safetensors keys: {e:?}"))?;
        loaded.keys().map(|k| k.as_str().to_string()).collect()
    };

    let plan = QuantPlan::from_config_and_index(&cfg, index_keys_owned.iter().map(|s| s.as_str()));
    let is_quant = !plan.quantized_paths.is_empty();
    if is_quant {
        let override_count = plan.overrides.len();
        let mode = if override_count > 0 {
            "OptiQ mixed-bit"
        } else {
            "uniform"
        };
        tracing::info!(
            "[local-mlx-native] quantizing Gemma-4 ({}): default={}-bit/group{}, {} per-module overrides, {} quantized modules",
            mode,
            plan.default.1,
            plan.default.0,
            override_count,
            plan.quantized_paths.len(),
        );
        apply_per_module_quantization(&mut model, &plan)
            .map_err(|e| anyhow::anyhow!("per-module quantize failed: {e:?}"))?;
        model
            .eval()
            .map_err(|e| anyhow::anyhow!("post-quantize eval failed: {e:?}"))?;
    }

    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)
            .map_err(|e| anyhow::anyhow!("read index failed: {e}"))?;
        let map: WeightMap =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse index failed: {e}"))?;
        let files: HashSet<&String> = map.weight_map.values().collect();
        for f in files {
            shard_files.push(model_dir.join(f));
        }
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            anyhow::bail!(
                "no model.safetensors.index.json or model.safetensors in {}",
                model_dir.display()
            );
        }
        shard_files.push(single);
    }

    // `language_model.model.X` → `model.X`; `language_model.lm_head.X` → `lm_head.X`.
    let strip = |key: &str| -> String {
        key.strip_prefix("language_model.")
            .unwrap_or(key)
            .to_string()
    };

    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut total_skipped_mm = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    // Account for vision/audio tower weights as intentionally-skipped so they
    // don't drown the warning that surfaces a *real* unmatched text-backbone
    // key. `embed_vision.*` is NOT in this list — those 3 keys are routed
    // into the `Model::embed_vision` slot by the per-module quantization +
    // weight-load loop (Phase 1 of vision support — see `gemma4_vision.rs`).
    // The vision tower itself remains skipped until Phase 2 ports it; audio
    // stays skipped indefinitely (out of scope on the text-only inference
    // path).
    let is_multimodal_key = |k: &str| {
        const SKIP_PREFIXES: &[&str] = &[
            "vision_tower.",
            "audio_tower.",
            "embed_audio.",
            "multi_modal_projector.",
        ];
        SKIP_PREFIXES.iter().any(|p| k.starts_with(p))
    };

    // KV-shared layers (idx ≥ first_kv_shared) reuse earlier layers' K/V at
    // runtime, so their own `k_proj`/`v_proj`/`k_norm` weights — which the
    // checkpoint may still carry — are vestigial. The reference `sanitize()`
    // drops them; we simply don't allocate slots for them, so treat them as
    // intentionally-skipped rather than "unmatched".
    let is_vestigial_shared_kv = |k: &str| -> bool {
        let Some(rest) = k.strip_prefix("model.layers.") else {
            return false;
        };
        let Some((idx_str, tail)) = rest.split_once('.') else {
            return false;
        };
        let Ok(idx) = idx_str.parse::<i32>() else {
            return false;
        };
        idx >= first_kv_shared
            && (tail.starts_with("self_attn.k_proj")
                || tail.starts_with("self_attn.v_proj")
                || tail.starts_with("self_attn.k_norm"))
    };

    // QuantizedEmbedding captures (applied after the shard loop).
    let mut embed_weight: Option<mlx_rs::Array> = None;
    let mut embed_scales: Option<mlx_rs::Array> = None;
    let mut embed_biases: Option<mlx_rs::Array> = None;
    let mut ple_weight: Option<mlx_rs::Array> = None;
    let mut ple_scales: Option<mlx_rs::Array> = None;
    let mut ple_biases: Option<mlx_rs::Array> = None;

    let mut unfilled_slots: HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };
    for shard in &shard_files {
        let loaded = mlx_rs::Array::load_safetensors(shard)
            .map_err(|e| anyhow::anyhow!("read shard {}: {e:?}", shard.display()))?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, value) in loaded {
            let key = strip(raw_key.as_str());
            match key.as_str() {
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
                "model.embed_tokens_per_layer.weight" => {
                    ple_weight = Some(value);
                    total_loaded += 1;
                    continue;
                }
                "model.embed_tokens_per_layer.scales" => {
                    ple_scales = Some(value);
                    total_loaded += 1;
                    continue;
                }
                "model.embed_tokens_per_layer.biases" => {
                    ple_biases = Some(value);
                    total_loaded += 1;
                    continue;
                }
                _ => {}
            }
            if let Some(slot) = params.get_mut(key.as_str()) {
                **slot = value;
                total_loaded += 1;
                unfilled_slots.remove(&key);
                continue;
            }
            if is_quant {
                if let Some(stripped) = key.strip_suffix(".weight") {
                    let remapped = format!("{stripped}.inner.weight");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
                        continue;
                    }
                }
                if let Some(stripped) = key.strip_suffix(".bias") {
                    let remapped = format!("{stripped}.inner.bias");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
                        continue;
                    }
                }
            }
            if is_multimodal_key(&key) || is_vestigial_shared_kv(&key) {
                total_skipped_mm += 1;
                continue;
            }
            total_missed += 1;
            if unmatched_samples.len() < 5 {
                unmatched_samples.push(key);
            }
        }
    }

    apply_quantized_embedding(
        &mut model.model.embed_tokens,
        embed_weight,
        embed_scales,
        embed_biases,
    );
    if let Some(eple) = model.model.embed_tokens_per_layer.as_mut() {
        apply_quantized_embedding(eple, ple_weight, ple_scales, ple_biases);
    }

    tracing::info!(
        "[local-mlx-native] gemma4 safetensor load: {total_loaded} matched, {total_missed} unmatched, \
         {total_skipped_mm} vision/audio weights skipped (text-only path; quantized={is_quant})"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[local-mlx-native] gemma4 unmatched key samples: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled_slots.is_empty() {
        let mut samples: Vec<&String> = unfilled_slots.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[local-mlx-native] gemma4 {} parameter slot(s) NOT populated — first few: {}",
            unfilled_slots.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        anyhow::bail!("no safetensor keys matched the Gemma-4 parameter tree");
    }

    model
        .eval()
        .map_err(|e| anyhow::anyhow!("eval after load failed: {e:?}"))?;
    Ok(model)
}

/// Load a DeepSeek-V2 [`Model`] (MLA + DeepSeekMoE). Uniform 4-bit
/// quantization via `nn::quantize` — the router `gate` (plain `nn::Linear`) and
/// the stacked `switch_mlp.*` experts (raw, already-quantized arrays loaded
/// verbatim) are skipped by construction. HF names the MoE block `mlp`; our
/// struct splits dense (`mlp`) vs MoE (`moe`), so MoE-layer keys are remapped
/// `mlp.* → moe.*` during load.
fn load_deepseek_v2_any(
    model_dir: &Path,
) -> anyhow::Result<super::mlx_lm::models::deepseek_v2::Model> {
    use super::mlx_lm::models::deepseek_v2::{get_deepseek_v2_model_args, Model, WeightMap};
    use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
    use std::collections::HashSet;

    let args = get_deepseek_v2_model_args(model_dir)
        .map_err(|e| anyhow::anyhow!("read deepseek_v2 args failed: {e:?}"))?;
    let first_dense = args.first_k_dense_replace;

    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read config.json failed: {e}"))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse config.json failed: {e}"))?;
    let (group_size, bits) = cfg
        .get("quantization")
        .map(|q| {
            (
                q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32,
                q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32,
            )
        })
        .unwrap_or((64, 4));

    let mut model = Model::new(args, group_size, bits)
        .map_err(|e| anyhow::anyhow!("Model::new failed: {e:?}"))?;

    // Uniform quantize: flips every #[quantizable] MaybeQuantized slot
    // (embeddings, attention, dense MLP, shared experts, lm_head) to N-bit. The
    // plain router `gate` and the raw stacked experts are left untouched.
    model = mlx_rs::nn::quantize(model, Some(group_size), Some(bits))
        .map_err(|e| anyhow::anyhow!("nn::quantize failed: {e:?}"))?;
    model
        .eval()
        .map_err(|e| anyhow::anyhow!("post-quantize eval failed: {e:?}"))?;

    let weights_index = model_dir.join("model.safetensors.index.json");
    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)
            .map_err(|e| anyhow::anyhow!("read index failed: {e}"))?;
        let map: WeightMap =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse index failed: {e}"))?;
        let files: HashSet<&String> = map.weight_map.values().collect();
        for f in files {
            shard_files.push(model_dir.join(f));
        }
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            anyhow::bail!("no safetensors in {}", model_dir.display());
        }
        shard_files.push(single);
    }

    let remap_moe = |key: &str| -> String {
        if let Some(rest) = key.strip_prefix("model.layers.") {
            if let Some((idx_str, tail)) = rest.split_once('.') {
                if let Ok(idx) = idx_str.parse::<i32>() {
                    if idx >= first_dense {
                        if let Some(after) = tail.strip_prefix("mlp.") {
                            return format!("model.layers.{idx}.moe.{after}");
                        }
                    }
                }
            }
        }
        key.to_string()
    };

    let mut embed_weight: Option<mlx_rs::Array> = None;
    let mut embed_scales: Option<mlx_rs::Array> = None;
    let mut embed_biases: Option<mlx_rs::Array> = None;

    let mut unfilled: HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };
    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    for shard in &shard_files {
        let loaded = mlx_rs::Array::load_safetensors(shard)
            .map_err(|e| anyhow::anyhow!("read shard {}: {e:?}", shard.display()))?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, value) in loaded {
            let key = remap_moe(raw_key.as_str());
            match key.as_str() {
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
            if let Some(slot) = params.get_mut(key.as_str()) {
                **slot = value;
                total_loaded += 1;
                unfilled.remove(&key);
                continue;
            }
            // Quantized Linear stores the packed weight under `.inner.weight`.
            if let Some(stripped) = key.strip_suffix(".weight") {
                let remapped = format!("{stripped}.inner.weight");
                if let Some(slot) = params.get_mut(remapped.as_str()) {
                    **slot = value;
                    total_loaded += 1;
                    unfilled.remove(&remapped);
                    continue;
                }
            }
            total_missed += 1;
            if unmatched_samples.len() < 8 {
                unmatched_samples.push(key);
            }
        }
    }

    apply_quantized_embedding(
        &mut model.model.embed_tokens,
        embed_weight,
        embed_scales,
        embed_biases,
    );

    tracing::info!(
        "[local-mlx-native] deepseek_v2 safetensor load: {total_loaded} matched, {total_missed} unmatched"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[local-mlx-native] deepseek_v2 unmatched key samples: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled.is_empty() {
        let mut samples: Vec<&String> = unfilled.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[local-mlx-native] deepseek_v2 {} parameter slot(s) NOT populated — first few: {}",
            unfilled.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        anyhow::bail!("no safetensor keys matched the DeepSeek-V2 parameter tree");
    }

    model
        .eval()
        .map_err(|e| anyhow::anyhow!("eval after load failed: {e:?}"))?;
    Ok(model)
}

/// Apply captured `QuantizedEmbedding` weight/scales/biases (mlx-rs 0.25.3 leaves
/// the inner fields unannotated, so `parameters_mut()` misses them).
fn apply_quantized_embedding(
    embed: &mut mlx_rs::quantization::MaybeQuantized<mlx_rs::nn::Embedding>,
    weight: Option<mlx_rs::Array>,
    scales: Option<mlx_rs::Array>,
    biases: Option<mlx_rs::Array>,
) {
    if weight.is_none() && scales.is_none() && biases.is_none() {
        return;
    }
    match embed {
        mlx_rs::quantization::MaybeQuantized::Quantized(q) => {
            if let Some(w) = weight {
                q.inner.weight.value = w;
            }
            if let Some(s) = scales {
                q.scales.value = s;
            }
            if let Some(b) = biases {
                q.biases.value = b;
            }
        }
        mlx_rs::quantization::MaybeQuantized::Original(e) => {
            if let Some(w) = weight {
                e.weight.value = w;
            }
        }
    }
}

/// Load a Mamba-2 [`Model`]. Weight-population uses `load_safetensors` (no
/// post-load quantization in this build — `mlx-community/mamba2-*` ships bf16).
fn load_mamba2_any(model_dir: &Path) -> anyhow::Result<super::mlx_lm::models::mamba2::Model> {
    use super::mlx_lm::models::mamba2::load_mamba2_model;
    load_mamba2_model(model_dir).map_err(|e| anyhow::anyhow!("load_mamba2_model failed: {e:?}"))
}

/// Load a Falcon-Mamba (Mamba-1) [`Model`]. The loader inside the model module
/// handles both plain and quantised (`mlx-community/falcon-mamba-7b-{4,8}bit`)
/// checkpoints, including the `QuantizedEmbedding` workaround.
fn load_falcon_mamba_any(
    model_dir: &Path,
) -> anyhow::Result<super::mlx_lm::models::falcon_mamba::Model> {
    use super::mlx_lm::models::falcon_mamba::load_falcon_mamba_model;
    load_falcon_mamba_model(model_dir)
        .map_err(|e| anyhow::anyhow!("load_falcon_mamba_model failed: {e:?}"))
}

/// Load a Llama [`Model`]. Mirrors `load_qwen3_any`'s logging/quantization
/// support: honours `config.json::quantization` for mlx-community 4-bit
/// variants and surfaces unmatched safetensors keys instead of silently
/// dropping them.
fn load_llama_any(model_dir: &Path) -> anyhow::Result<super::mlx_lm::models::llama::Model> {
    use super::mlx_lm::models::llama::{get_llama_model_args, Model, WeightMap};
    use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
    use std::collections::HashSet;

    let args = get_llama_model_args(model_dir)
        .map_err(|e| anyhow::anyhow!("read llama args failed: {e:?}"))?;
    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read config.json failed: {e}"))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse config.json failed: {e}"))?;
    let quant = cfg.get("quantization").and_then(|q| {
        let g = q.get("group_size")?.as_i64()? as i32;
        let b = q.get("bits")?.as_i64()? as i32;
        Some((g, b))
    });

    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)
            .map_err(|e| anyhow::anyhow!("read index failed: {e}"))?;
        let map: WeightMap =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse index failed: {e}"))?;
        let files: HashSet<&String> = map.weight_map.values().collect();
        for f in files {
            shard_files.push(model_dir.join(f));
        }
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            anyhow::bail!(
                "no model.safetensors.index.json or model.safetensors in {}",
                model_dir.display()
            );
        }
        shard_files.push(single);
    }

    let is_quant = quant.is_some();
    let model = Model::new(args).map_err(|e| anyhow::anyhow!("Model::new failed: {e:?}"))?;
    let mut model = if let Some((group_size, bits)) = quant {
        tracing::info!(
            "[local-mlx-native] quantizing Llama layers: group_size={group_size}, bits={bits}"
        );
        let m = mlx_rs::nn::quantize(model, Some(group_size), Some(bits))
            .map_err(|e| anyhow::anyhow!("nn::quantize failed: {e:?}"))?;
        m.eval()
            .map_err(|e| anyhow::anyhow!("post-quantize eval failed: {e:?}"))?;
        m
    } else {
        model
    };

    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    // Same QuantizedEmbedding workaround as qwen3 (mlx-rs 0.25.3 leaves the
    // inner/scales/biases fields without `#[param]`, so they don't appear in
    // parameters_mut(); capture directly).
    let mut embed_weight: Option<mlx_rs::Array> = None;
    let mut embed_scales: Option<mlx_rs::Array> = None;
    let mut embed_biases: Option<mlx_rs::Array> = None;

    let mut unfilled_slots: std::collections::HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };
    for shard in &shard_files {
        let loaded = mlx_rs::Array::load_safetensors(shard)
            .map_err(|e| anyhow::anyhow!("read shard {}: {e:?}", shard.display()))?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, value) in loaded {
            let key = raw_key.as_str();
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
                unfilled_slots.remove(key);
                continue;
            }
            if is_quant {
                if let Some(stripped) = key.strip_suffix(".weight") {
                    let remapped = format!("{stripped}.inner.weight");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
                        continue;
                    }
                }
                if let Some(stripped) = key.strip_suffix(".bias") {
                    let remapped = format!("{stripped}.inner.bias");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
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
            mlx_rs::quantization::MaybeQuantized::Quantized(q) => {
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
            mlx_rs::quantization::MaybeQuantized::Original(e) => {
                if let Some(w) = embed_weight {
                    e.weight.value = w;
                }
            }
        }
    }

    tracing::info!(
        "[local-mlx-native] llama safetensor load: {total_loaded} matched, {total_missed} unmatched (quantized={is_quant})"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[local-mlx-native] llama unmatched key samples: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled_slots.is_empty() {
        let mut samples: Vec<&String> = unfilled_slots.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[local-mlx-native] llama {} parameter slot(s) NOT populated — first few: {}",
            unfilled_slots.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        anyhow::bail!("no safetensor keys matched the Llama parameter tree");
    }

    model
        .eval()
        .map_err(|e| anyhow::anyhow!("eval after load failed: {e:?}"))?;
    Ok(model)
}

/// Load an Ouro `Model` (looped LM), applying `nn::quantize` when `config.json`
/// declares a `quantization` block. Mirrors [`load_llama_any`] — same quantized
/// embedding workaround and `.weight`→`.inner.weight` remap — but additionally
/// skips the checkpoint's `model.early_exit_gate.*` tensors: the early-exit gate
/// is a no-op at inference (`early_exit_threshold == 1.0` always selects the
/// final UT step), so it isn't modeled and we don't want it logged as a stray
/// unmatched key.
fn load_ouro_any(model_dir: &Path) -> anyhow::Result<super::mlx_lm::models::ouro::Model> {
    use super::mlx_lm::models::llama::WeightMap;
    use super::mlx_lm::models::ouro::{get_ouro_model_args, Model};
    use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
    use std::collections::HashSet;

    let args = get_ouro_model_args(model_dir)
        .map_err(|e| anyhow::anyhow!("read ouro args failed: {e:?}"))?;
    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read config.json failed: {e}"))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse config.json failed: {e}"))?;
    let quant = cfg.get("quantization").and_then(|q| {
        let g = q.get("group_size")?.as_i64()? as i32;
        let b = q.get("bits")?.as_i64()? as i32;
        Some((g, b))
    });

    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)
            .map_err(|e| anyhow::anyhow!("read index failed: {e}"))?;
        let map: WeightMap =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse index failed: {e}"))?;
        let files: HashSet<&String> = map.weight_map.values().collect();
        for f in files {
            shard_files.push(model_dir.join(f));
        }
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            anyhow::bail!(
                "no model.safetensors.index.json or model.safetensors in {}",
                model_dir.display()
            );
        }
        shard_files.push(single);
    }

    let is_quant = quant.is_some();
    let model = Model::new(args).map_err(|e| anyhow::anyhow!("Model::new failed: {e:?}"))?;
    let mut model = if let Some((group_size, bits)) = quant {
        tracing::info!(
            "[local-mlx-native] quantizing Ouro layers: group_size={group_size}, bits={bits}"
        );
        let m = mlx_rs::nn::quantize(model, Some(group_size), Some(bits))
            .map_err(|e| anyhow::anyhow!("nn::quantize failed: {e:?}"))?;
        m.eval()
            .map_err(|e| anyhow::anyhow!("post-quantize eval failed: {e:?}"))?;
        m
    } else {
        model
    };

    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut total_skipped_gate = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    // Quantized-embedding workaround (see load_llama_any): mlx-rs 0.25.3 leaves
    // QuantizedEmbedding's inner/scales/biases without `#[param]`, so capture
    // them directly rather than through parameters_mut().
    let mut embed_weight: Option<mlx_rs::Array> = None;
    let mut embed_scales: Option<mlx_rs::Array> = None;
    let mut embed_biases: Option<mlx_rs::Array> = None;

    let mut unfilled_slots: std::collections::HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };
    for shard in &shard_files {
        let loaded = mlx_rs::Array::load_safetensors(shard)
            .map_err(|e| anyhow::anyhow!("read shard {}: {e:?}", shard.display()))?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, value) in loaded {
            let key = raw_key.as_str();
            // Early-exit gate is intentionally not modeled — skip silently.
            if key.starts_with("model.early_exit_gate.") {
                total_skipped_gate += 1;
                continue;
            }
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
                unfilled_slots.remove(key);
                continue;
            }
            if is_quant {
                if let Some(stripped) = key.strip_suffix(".weight") {
                    let remapped = format!("{stripped}.inner.weight");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
                        continue;
                    }
                }
                if let Some(stripped) = key.strip_suffix(".bias") {
                    let remapped = format!("{stripped}.inner.bias");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
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
            mlx_rs::quantization::MaybeQuantized::Quantized(q) => {
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
            mlx_rs::quantization::MaybeQuantized::Original(e) => {
                if let Some(w) = embed_weight {
                    e.weight.value = w;
                }
            }
        }
    }

    tracing::info!(
        "[local-mlx-native] ouro safetensor load: {total_loaded} matched, {total_missed} unmatched, {total_skipped_gate} early-exit-gate skipped (quantized={is_quant})"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[local-mlx-native] ouro unmatched key samples: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled_slots.is_empty() {
        let mut samples: Vec<&String> = unfilled_slots.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[local-mlx-native] ouro {} parameter slot(s) NOT populated — first few: {}",
            unfilled_slots.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        anyhow::bail!("no safetensor keys matched the Ouro parameter tree");
    }

    model
        .eval()
        .map_err(|e| anyhow::anyhow!("eval after load failed: {e:?}"))?;
    Ok(model)
}

/// Load a Qwen3 `Model`, applying `nn::quantize` when `config.json` declares a
/// `quantization` block (mlx-community 4-bit / 8-bit variants).
fn load_qwen3_any(model_dir: &Path) -> anyhow::Result<super::mlx_lm::models::qwen3::Model> {
    use super::mlx_lm::models::qwen3::{get_qwen3_model_args, Model, WeightMap};
    use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
    use std::collections::HashSet;

    let args = get_qwen3_model_args(model_dir)
        .map_err(|e| anyhow::anyhow!("read model args failed: {e:?}"))?;
    let model = Model::new(args).map_err(|e| anyhow::anyhow!("Model::new failed: {e:?}"))?;

    // Inspect config.json for an optional `quantization: { group_size, bits }`.
    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow::anyhow!("read config.json failed: {e}"))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse config.json failed: {e}"))?;
    let quant = cfg.get("quantization").and_then(|q| {
        let g = q.get("group_size")?.as_i64()? as i32;
        let b = q.get("bits")?.as_i64()? as i32;
        Some((g, b))
    });

    let mut model = if let Some((group_size, bits)) = quant {
        tracing::info!(
            "[local-mlx-native] quantizing Qwen3 layers: group_size={group_size}, bits={bits}"
        );
        let m = mlx_rs::nn::quantize(model, Some(group_size), Some(bits))
            .map_err(|e| anyhow::anyhow!("nn::quantize failed: {e:?}"))?;
        m.eval()
            .map_err(|e| anyhow::anyhow!("post-quantize eval failed: {e:?}"))?;
        m
    } else {
        model
    };

    // Collect every safetensor shard we need to read.
    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)
            .map_err(|e| anyhow::anyhow!("read index failed: {e}"))?;
        let map: WeightMap =
            serde_json::from_str(&json).map_err(|e| anyhow::anyhow!("parse index failed: {e}"))?;
        let files: HashSet<&String> = map.weight_map.values().collect();
        for f in files {
            shard_files.push(model_dir.join(f));
        }
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            anyhow::bail!(
                "no model.safetensors.index.json or model.safetensors in {}",
                model_dir.display()
            );
        }
        shard_files.push(single);
    }

    let n_layers = model.args.num_hidden_layers as usize;
    let _ = n_layers;

    let is_quant = quant.is_some();
    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    // mlx-rs 0.25.3 `QuantizedEmbedding` is missing `#[param]` annotations on
    // its `inner` / `scales` / `biases` fields, so `parameters_mut().flatten()`
    // returns nothing for `model.embed_tokens.*`. Capture those tensors here
    // and apply them directly below.
    let mut embed_weight: Option<mlx_rs::Array> = None;
    let mut embed_scales: Option<mlx_rs::Array> = None;
    let mut embed_biases: Option<mlx_rs::Array> = None;

    let mut unfilled_slots: std::collections::HashSet<String> = {
        let snap = model.parameters_mut().flatten();
        snap.keys().map(|k| k.to_string()).collect()
    };
    for shard in &shard_files {
        let loaded = mlx_rs::Array::load_safetensors(shard)
            .map_err(|e| anyhow::anyhow!("read shard {}: {e:?}", shard.display()))?;
        let mut params = model.parameters_mut().flatten();
        for (raw_key, value) in loaded {
            let key = raw_key.as_str();

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
                unfilled_slots.remove(key);
                continue;
            }
            if is_quant {
                if let Some(stripped) = key.strip_suffix(".weight") {
                    let remapped = format!("{stripped}.inner.weight");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
                        continue;
                    }
                }
                if let Some(stripped) = key.strip_suffix(".bias") {
                    let remapped = format!("{stripped}.inner.bias");
                    if let Some(slot) = params.get_mut(remapped.as_str()) {
                        **slot = value;
                        total_loaded += 1;
                        unfilled_slots.remove(&remapped);
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

    // Apply embed_tokens tensors directly into the QuantizedEmbedding struct.
    if embed_weight.is_some() || embed_scales.is_some() || embed_biases.is_some() {
        match &mut model.model.embed_tokens {
            mlx_rs::quantization::MaybeQuantized::Quantized(q) => {
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
                    "[local-mlx-native] embed_tokens (QuantizedEmbedding) populated via direct mutation"
                );
            }
            mlx_rs::quantization::MaybeQuantized::Original(e) => {
                if let Some(w) = embed_weight {
                    e.weight.value = w;
                }
                tracing::info!(
                    "[local-mlx-native] embed_tokens (Embedding) populated via direct mutation"
                );
            }
        }
    }
    tracing::info!(
        "[local-mlx-native] safetensor load: {total_loaded} matched, {total_missed} unmatched (quantized={is_quant})"
    );
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[local-mlx-native] sample unmatched safetensor keys: {}",
            unmatched_samples.join(", ")
        );
    }
    if !unfilled_slots.is_empty() {
        let mut samples: Vec<&String> = unfilled_slots.iter().collect();
        samples.sort();
        let preview: Vec<&str> = samples.iter().take(8).map(|s| s.as_str()).collect();
        tracing::warn!(
            "[local-mlx-native] {} model parameter slot(s) NOT populated — first few: {}",
            unfilled_slots.len(),
            preview.join(", ")
        );
    }
    if total_loaded == 0 {
        anyhow::bail!(
            "no safetensor keys matched the Qwen3 model parameter tree — \
             this build of mlx-lm may use a different layout than this model expects"
        );
    }

    model
        .eval()
        .map_err(|e| anyhow::anyhow!("eval after load failed: {e:?}"))?;

    Ok(model)
}

/// Best-effort extractor for `"model_type": "..."` that tolerates JSON which
/// `serde_json` would reject — most importantly the `Infinity` value HF writes
/// into `time_step_limit` for Mamba-2 / Mamba-Codestral checkpoints.
///
/// Tries strict `serde_json::Value` first; on failure falls back to a manual
/// scan for the `"model_type"` key. Returns `None` only when the key is truly
/// absent from the file.
fn extract_model_type(raw: &str) -> Option<String> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        if let Some(s) = v.get("model_type").and_then(|x| x.as_str()) {
            return Some(s.to_string());
        }
    }
    let key = "\"model_type\"";
    let idx = raw.find(key)?;
    let rest = &raw[idx + key.len()..];
    let colon = rest.find(':')?;
    let after = &rest[colon + 1..];
    let quote = after.find('"')?;
    let after_quote = &after[quote + 1..];
    let end = after_quote.find('"')?;
    Some(after_quote[..end].to_string())
}

pub fn detect_architecture(model_id: &str, model_dir: &Path) -> anyhow::Result<Arch> {
    let lower = model_id.to_lowercase();
    // config.json::model_type is the source of truth; the id-substring check
    // below is just a hint when config.json hasn't been read yet.
    let cfg_path = model_dir.join("config.json");
    if let Ok(raw) = std::fs::read_to_string(&cfg_path) {
        if let Some(mt_raw) = extract_model_type(&raw) {
            {
                let mt = mt_raw.to_lowercase();
                if mt == "mamba2" || mt.starts_with("mamba2") {
                    return Ok(Arch::Mamba2);
                }
                // Falcon-Mamba is a Mamba-1 backbone with `use_bcdt_rms = true`
                // applied to (delta, B, C). Routed through the dedicated Rust
                // module rather than the Mamba-2 path.
                if mt == "falcon_mamba" || mt.starts_with("falcon_mamba") {
                    return Ok(Arch::FalconMamba);
                }
                // Generic Mamba-1 (`state-spaces/mamba-*`) — same forward path
                // but `use_bcdt_rms = false`. Reuse the FalconMamba loader.
                if mt == "mamba" {
                    return Ok(Arch::FalconMamba);
                }
                // Ouro looped LM (`OuroForCausalLM`, model_type `ouro`) — a
                // Llama-style decoder applied recurrently `total_ut_steps` times.
                if mt == "ouro" || mt.starts_with("ouro") {
                    return Ok(Arch::Ouro);
                }
                if mt.contains("qwen3_moe") || mt.contains("qwen3_5_moe") {
                    anyhow::bail!(
                        "Qwen3-MoE (`model_type={mt}`) is not supported by native MLX in this build — use a dense Qwen3 checkpoint or another supported architecture."
                    );
                }
                if mt.contains("qwen3_next") {
                    anyhow::bail!(
                        "Qwen3-Next (`model_type={mt}`) is not supported by native MLX in this build."
                    );
                }
                if mt.contains("qwen3_5") {
                    return Ok(Arch::Qwen35);
                }
                if mt.contains("qwen3") {
                    return Ok(Arch::Qwen3);
                }
                if mt.contains("qwen2") {
                    return Ok(Arch::Llama);
                }
                if mt == "gemma2" || mt.starts_with("gemma2") {
                    return Ok(Arch::Gemma2);
                }
                // Gemma-3 ships both `gemma3` (multimodal wrapper) and
                // `gemma3_text` (text-only) model_types; both load through
                // the same Rust path.
                if mt == "gemma3" || mt.starts_with("gemma3") {
                    return Ok(Arch::Gemma3);
                }
                // Gemma-4 (`gemma4` multimodal wrapper / `gemma4_text` text-only).
                // Distinct arch from Gemma-3: PLE, cross-layer KV sharing,
                // proportional/partial RoPE — see `gemma4.rs`.
                if mt == "gemma4" || mt.starts_with("gemma4") {
                    return Ok(Arch::Gemma4);
                }
                // DeepSeek-V2 (MLA + DeepSeekMoE). Covers `deepseek_v2` and the
                // Coder-V2 variants which share the same `model_type`.
                if mt == "deepseek_v2" || mt.starts_with("deepseek_v2") {
                    return Ok(Arch::DeepSeekV2);
                }
                // Llama covers `llama`, `llama-3`, `llama3`, Nesso, etc.
                if mt == "llama" || mt.starts_with("llama") {
                    return Ok(Arch::Llama);
                }
                if mt.contains("bonsai") {
                    return Ok(Arch::BonsaiQ1);
                }
            }
        }
    }
    if lower.contains("mamba2") {
        return Ok(Arch::Mamba2);
    }
    if lower.contains("falcon-mamba") || lower.contains("falcon_mamba") {
        return Ok(Arch::FalconMamba);
    }
    if lower.contains("ouro") {
        return Ok(Arch::Ouro);
    }
    if lower.contains("qwen3_moe") || lower.contains("qwen3_5_moe") {
        anyhow::bail!(
            "no native MLX loader for `{model_id}` — Qwen3-MoE is not supported in this build."
        );
    }
    if lower.contains("qwen3_next") {
        anyhow::bail!(
            "no native MLX loader for `{model_id}` — Qwen3-Next is not supported in this build."
        );
    }
    if lower.contains("qwen3.5") || lower.contains("qwen3_5") {
        return Ok(Arch::Qwen35);
    }
    if lower.contains("qwen3") {
        return Ok(Arch::Qwen3);
    }
    if lower.contains("qwen2") {
        return Ok(Arch::Llama);
    }
    if lower.contains("gemma2") || lower.contains("gemma-2") {
        return Ok(Arch::Gemma2);
    }
    if lower.contains("gemma-4") || lower.contains("gemma4") {
        return Ok(Arch::Gemma4);
    }
    if lower.contains("gemma-3") || lower.contains("gemma3") {
        return Ok(Arch::Gemma3);
    }
    if lower.contains("deepseek-coder-v2") || lower.contains("deepseek-v2") {
        return Ok(Arch::DeepSeekV2);
    }
    if lower.contains("llama") || lower.contains("nesso") {
        return Ok(Arch::Llama);
    }
    if lower.contains("bonsai") {
        return Ok(Arch::BonsaiQ1);
    }
    anyhow::bail!(
        "no native loader for `{}` — supported model_type values: qwen3, qwen3_5, llama, gemma2, gemma3, gemma4, deepseek_v2, mamba2, mamba, falcon_mamba, ouro, bonsai / bonsai_q1.",
        model_id
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ensure_installed_errors_when_missing() {
        let eng = MlxNativeEngine::new(
            Path::new("/nonexistent/path"),
            "mlx-community/Qwen3-4B-bf16",
            None,
        );
        let err = eng.ensure_installed().await.unwrap_err().to_string();
        assert!(err.contains("model directory not found"), "got: {err}");
    }

    /// A checkpoint's own recommended sampling is read from
    /// `generation_config.json`, so a new model gets it with no code change.
    #[test]
    fn generation_defaults_read_the_checkpoints_own_sampling() {
        let dir = tempfile::tempdir().unwrap();
        // The real `mlx-community/gemma-4-e2b-it-4bit` file, verbatim.
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"bos_token_id":2,"do_sample":true,"eos_token_id":[1,106,50],
                "pad_token_id":0,"temperature":1.0,"top_k":64,"top_p":0.95}"#,
        )
        .unwrap();
        let d = GenerationDefaults::from_model_dir(dir.path());
        assert_eq!(d.top_k, Some(64));
        assert_eq!(d.top_p, Some(0.95));
        assert_eq!(d.temperature, Some(1.0));
    }

    /// Every failure mode is "no opinion", never an error — sampling defaults
    /// must not be able to fail a model load.
    #[test]
    fn generation_defaults_degrade_to_no_opinion() {
        let dir = tempfile::tempdir().unwrap();

        // Missing file.
        let d = GenerationDefaults::from_model_dir(dir.path());
        assert_eq!((d.top_k, d.top_p, d.temperature), (None, None, None));

        // Unparseable.
        std::fs::write(dir.path().join("generation_config.json"), "{not json").unwrap();
        let d = GenerationDefaults::from_model_dir(dir.path());
        assert_eq!((d.top_k, d.top_p, d.temperature), (None, None, None));

        // Present but wrong-typed / out of range: `as_i64` on a string reads
        // exactly like an absent key, and a top_p above 1 is meaningless.
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"top_k":"64","top_p":1.5,"temperature":-1}"#,
        )
        .unwrap();
        let d = GenerationDefaults::from_model_dir(dir.path());
        assert_eq!((d.top_k, d.top_p, d.temperature), (None, None, None));
    }

    /// `do_sample: false` means the checkpoint asks for greedy decoding; its
    /// temperature/top_k/top_p are then decoration and must not be promoted
    /// into a sampling config.
    #[test]
    fn generation_defaults_honour_do_sample_false() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"do_sample":false,"temperature":0.7,"top_k":50,"top_p":0.9}"#,
        )
        .unwrap();
        let d = GenerationDefaults::from_model_dir(dir.path());
        assert_eq!(d.temperature, Some(0.0));
        assert_eq!(d.top_k, None);
        assert_eq!(d.top_p, None);
    }

    #[test]
    fn preprocess_parses_tool_call_arguments_to_object() {
        // Qwen3.5's chat template iterates `tool_call.arguments|items`; if we feed
        // back the OpenAI JSON-string form it crashes with "cannot convert value
        // into pairs". Preprocessing must turn the string into an object.
        let msgs = vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "t0",
                "type": "function",
                "function": { "name": "ToolSearch", "arguments": "{\"query\":\"gold price\"}" }
            }]
        })];
        let out = preprocess_openai_messages_for_mlx(&msgs);
        let args = &out[0]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_object(), "arguments not an object: {args:?}");
        assert_eq!(args["query"], "gold price");
    }

    #[test]
    fn preprocess_bad_tool_call_arguments_become_empty_object() {
        let msgs = vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": { "name": "X", "arguments": "not json{{" }
            }]
        })];
        let out = preprocess_openai_messages_for_mlx(&msgs);
        assert!(out[0]["tool_calls"][0]["function"]["arguments"].is_object());
    }

    /// End-to-end against the REAL Qwen3.5 chat template: a prior assistant
    /// tool-call message crashes the template when `arguments` is a JSON string
    /// ("cannot convert value into pairs") and renders fine once preprocessed
    /// into an object. Skips when the model weights aren't on disk.
    #[test]
    fn qwen35_tool_call_message_renders_only_after_preprocess() {
        let home = std::env::var("HOME").unwrap_or_default();
        let dir = std::path::PathBuf::from(home)
            .join(".senclaw/local-models/mlx-community__Qwen3.5-0.8B-OptiQ-4bit");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("skip qwen35 template test: model not at {}", dir.display());
            return;
        }
        let model_id = "Qwen/Qwen3.5-0.8B";
        let template = load_mlx_chat_template(&dir, model_id)
            .expect("load template")
            .expect("qwen3.5 has a chat template");
        let mut tok = crate::engine::mlx_lm_utils::tokenizer::Tokenizer::from_file(
            dir.join("tokenizer.json"),
        )
        .expect("load tokenizer");

        let msgs = vec![
            serde_json::json!({"role": "user", "content": "giá vàng?"}),
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "t0", "type": "function",
                    "function": { "name": "ToolSearch", "arguments": "{\"query\":\"gold price\"}" }
                }]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "t0", "content": "results"}),
        ];

        let render = |t: &mut crate::engine::mlx_lm_utils::tokenizer::Tokenizer,
                      m: &[serde_json::Value]| {
            t.apply_chat_template_json_and_encode(
                template.clone(),
                model_id,
                m,
                &[],
                Some(true),
                Some(false),
                Some(""),
                Some(""),
            )
        };

        // Negative control: raw string-args message reproduces the crash.
        assert!(
            render(&mut tok, &msgs).is_err(),
            "expected string-args render to fail (the bug)"
        );

        // After preprocess: arguments is an object → template renders.
        let fixed = preprocess_openai_messages_for_mlx(&msgs);
        assert!(
            render(&mut tok, &fixed).is_ok(),
            "render after preprocess should succeed"
        );
    }

    /// Qwen3.5's chat-turn terminator `<|im_end|>` (248046) is DISTINCT from its
    /// `text_config.eos_token_id` = `<|endoftext|>` (248044). Stopping only on the
    /// config eos lets the model run past `<|im_end|>` into fake `<|im_start|>user`
    /// turns (the "user user assistant" loop + 8192-token runaway). The stop set
    /// must include the tokenizer-resolved `<|im_end|>`.
    #[test]
    fn qwen35_im_end_distinct_from_config_eos() {
        let home = std::env::var("HOME").unwrap_or_default();
        let dir = std::path::PathBuf::from(home)
            .join(".senclaw/local-models/mlx-community__Qwen3.5-0.8B-OptiQ-4bit");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("skip qwen35 stop-token test: model not on disk");
            return;
        }
        let tok = crate::engine::mlx_lm_utils::tokenizer::Tokenizer::from_file(
            dir.join("tokenizer.json"),
        )
        .expect("tokenizer");
        let im_end = tok.token_to_id("<|im_end|>");
        let endoftext = tok.token_to_id("<|endoftext|>");
        assert_eq!(im_end, Some(248_046), "<|im_end|> must resolve to 248046");
        assert_eq!(
            endoftext,
            Some(248_044),
            "<|endoftext|> must resolve to 248044"
        );
        assert_ne!(im_end, endoftext, "im_end must differ from endoftext");
        // The config eos (the value our stop set previously relied on alone) is
        // endoftext, NOT the chat terminator — proving why im_end must be added.
        let cfg_eos = 248_044u32;
        assert_eq!(endoftext, Some(cfg_eos));
        assert_ne!(
            im_end,
            Some(cfg_eos),
            "stopping only on config eos would miss <|im_end|>"
        );
    }

    /// `config.json::model_type == "ouro"` routes to the looped-LM loader even
    /// when the model id carries no architecture hint.
    #[test]
    fn detect_architecture_routes_ouro_by_model_type() {
        let dir = std::env::temp_dir().join(format!("senclaw-ouro-detect-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mk temp dir");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"ouro","num_hidden_layers":48,"total_ut_steps":4}"#,
        )
        .expect("write config");

        let arch = detect_architecture("local/some-checkpoint", &dir).expect("detect");
        assert!(matches!(arch, Arch::Ouro), "got {arch:?}");

        // Id-substring fallback (no config.json on disk) also routes to Ouro.
        let empty = std::env::temp_dir().join(format!("senclaw-ouro-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).expect("mk temp dir");
        let arch2 =
            detect_architecture("mlx-community/Ouro-2.6B-Thinking-4bit", &empty).expect("detect");
        assert!(matches!(arch2, Arch::Ouro), "got {arch2:?}");

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&empty);
    }
}
