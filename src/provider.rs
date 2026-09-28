//! The MLX engine behind an [`LlmProvider`], adapted to this runtime's
//! single-model-per-process shape.
//!
//! `apps/mlx-lm` (the old Space App) could switch between several installed
//! models inside one long-lived process, loading each lazily on first use
//! because the daemon health-gated a newly spawned app on a 30-second budget.
//! A runtime is different on both counts: the daemon launches exactly one
//! process per loaded model (`--model {model_path}`) and health-gates it on
//! `health.startupTimeoutSecs` (minutes, not seconds), so the model loads once,
//! eagerly, at startup — [`Readiness::loading`] until it either
//! [`Readiness::set_ready`]s or [`Readiness::set_failed`]s. There is no model
//! switching and no idle-unload-while-staying-up: the daemon's own
//! `idleTimeoutSecs` stops the whole process when nobody is using it.
//!
//! MLX serialization is still the engine's own concern: [`MlxNativeEngine`]
//! holds a process-wide lock around every load and generation, because
//! concurrent MLX work on separate threads corrupts Metal state. Two MLX
//! *processes* generating concurrently on the same GPU are fine — Metal
//! isolates command queues per process — so this lock only has to cover this
//! one process, same as it always did.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use app_space_sdk::llm::{ChatRequest, Chunk, ChunkSink, LlmProvider, ModelCard};
use sen_runtime_sdk::server::Readiness;
use tokio::sync::watch;

use crate::engine::runtime::LocalModelRuntime;
use crate::engine::MlxNativeEngine;

/// Where the background load task is. `chat()` awaits this instead of
/// re-entering `warm_up()`, so a request arriving before `/health` says ready
/// (a hand-started dev run racing its own first curl, say) waits for the same
/// load rather than kicking off a second one.
#[derive(Clone)]
enum LoadState {
    Loading,
    Ready(Arc<MlxNativeEngine>),
    /// The message from the failed load, so a request arriving after
    /// `Readiness::set_failed()` gets the real reason instead of a generic
    /// "not ready".
    Failed(String),
}

pub struct MlxProvider {
    card: ModelCard,
    state: watch::Receiver<LoadState>,
}

impl MlxProvider {
    /// Validate the checkpoint, build its [`ModelCard`] from `config.json`
    /// (cheap — no weights touched), and start loading weights in the
    /// background. `readiness` flips once the load resolves, which is what
    /// makes `/health` answer 503 until then and 200 (or 500 on failure) after.
    pub fn spawn(model_dir: PathBuf, model_id: String, readiness: Readiness) -> Result<Arc<Self>> {
        if !crate::engine::supports(&model_id, &model_dir) {
            return Err(anyhow!(
                "`{model_id}` at {} is not a checkpoint sen-mlx can load \
                 (missing safetensors, or an unsupported architecture)",
                model_dir.display()
            ));
        }
        let vision = crate::engine::has_vision(&model_dir);
        // The shared settings file sits one level up from any one model's
        // directory — `<SENCLAW_LOCAL_MODELS_DIR>/settings.json` — same lookup
        // `generate_with_cache` uses on every turn.
        let settings_dir = model_dir.parent().unwrap_or(&model_dir).to_path_buf();
        let settings = crate::settings::load(&settings_dir);
        let context_length = crate::engine::context_length(&model_dir)
            .unwrap_or_else(|| settings.max_prompt_tokens());
        let card = ModelCard::new(
            model_id.clone(),
            context_length,
            settings.max_new_tokens(),
            vision,
        );
        let kv_cache_bits = settings.kv_cache_bits.filter(|b| *b > 0);

        let (tx, rx) = watch::channel(LoadState::Loading);
        let provider = Arc::new(Self { card, state: rx });

        tokio::spawn(async move {
            // Weight loading is synchronous, Metal-touching work — off the
            // async reactor, same as every other load/generate call.
            let result = tokio::task::spawn_blocking(move || {
                let engine = MlxNativeEngine::new(&model_dir, &model_id, kv_cache_bits);
                engine.warm_up().map(|()| engine)
            })
            .await;
            let outcome = match result {
                Ok(Ok(engine)) => {
                    readiness.set_ready();
                    LoadState::Ready(Arc::new(engine))
                }
                Ok(Err(e)) => {
                    tracing::error!("model load failed: {e:#}");
                    readiness.set_failed();
                    LoadState::Failed(e.to_string())
                }
                Err(join_err) => {
                    tracing::error!("model load task panicked: {join_err}");
                    readiness.set_failed();
                    LoadState::Failed(join_err.to_string())
                }
            };
            // No receiver left (e.g. in a unit test that dropped the provider)
            // is not an error worth logging — there is nobody left to tell.
            let _ = tx.send(outcome);
        });

        Ok(provider)
    }

    /// The loaded engine, waiting out an in-flight load first. In production
    /// this never actually waits: the daemon does not proxy a request until
    /// `/health` is 200. It exists for hand-started runs and tests, where a
    /// request can race the load.
    async fn engine(&self) -> Result<Arc<MlxNativeEngine>> {
        let mut rx = self.state.clone();
        loop {
            match &*rx.borrow() {
                LoadState::Ready(e) => return Ok(Arc::clone(e)),
                LoadState::Failed(msg) => return Err(anyhow!("model failed to load: {msg}")),
                LoadState::Loading => {}
            }
            rx.changed()
                .await
                .map_err(|_| anyhow!("model load task ended without a result"))?;
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for MlxProvider {
    fn models(&self) -> Vec<ModelCard> {
        vec![self.card.clone()]
    }

    async fn chat(&self, req: ChatRequest, sink: ChunkSink) -> Result<()> {
        let engine = self.engine().await?;

        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);
        let gen = {
            let engine = Arc::clone(&engine);
            let messages = req.messages.clone();
            let tools = req.tools.clone();
            tokio::spawn(async move { engine.generate_stream(messages, tools, tx).await })
        };

        // Buffered, not forwarded token by token. A local model emits its tool
        // calls and its reasoning as *text*, in whatever dialect its chat
        // template uses, and a marker split across two tokens is only
        // recognisable once both have arrived. Streaming the raw tokens through
        // would leak `<|tool_call|>` and half-formed JSON into the visible
        // answer.
        let mut raw = String::new();
        while let Some(chunk) = rx.recv().await {
            raw.push_str(&chunk);
        }
        gen.await??;

        // Parse with the model's *own* config, loaded from its
        // `tokenizer_config.json` at load time. The dialect preset is a
        // fallback for the case where the engine could not surface one, which
        // should not happen after a successful load.
        let (text, reasoning, tool_calls) = match engine.parser_config() {
            Ok(cfg) => crate::engine::stream_parser::parse_complete_with_config(&raw, &cfg),
            Err(e) => {
                tracing::warn!("parser_config unavailable ({e}); falling back to a dialect preset");
                let dialect = crate::engine::stream_parser::dialect_for_model_id(&req.model);
                crate::engine::stream_parser::parse_complete(&raw, dialect)
            }
        };

        if !reasoning.is_empty() {
            sink.send(Chunk::Reasoning(reasoning)).await;
        }
        if !text.is_empty() {
            sink.send(Chunk::Text(text)).await;
        }
        for tc in tool_calls {
            // The parser returns OpenAI-shaped calls; the SDK re-renders them as
            // indexed streaming deltas.
            let id = tc["id"].as_str().unwrap_or_default().to_string();
            let name = tc["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let arguments = tc["function"]["arguments"]
                .as_str()
                .unwrap_or("{}")
                .to_string();
            if name.is_empty() {
                continue;
            }
            sink.send(Chunk::ToolCall {
                id,
                name,
                arguments,
            })
            .await;
        }

        if let Some((prompt_tokens, completion_tokens)) = engine.last_usage() {
            sink.send(Chunk::Usage {
                prompt_tokens: prompt_tokens as u64,
                completion_tokens: completion_tokens as u64,
            })
            .await;
        }

        // Same contract the daemon's own turn loop had: `release_cache_after_session`
        // drops the per-session KV (worth hundreds of MB after a long
        // generation) at the end of a turn while keeping the weights warm.
        let settings_dir = engine.model_dir().parent().map(Path::to_path_buf);
        if let Some(dir) = settings_dir {
            if crate::settings::load(&dir)
                .release_cache_after_session
                .unwrap_or(false)
            {
                engine.release_kv_cache();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory with no `config.json` at all is not a checkpoint this
    /// engine can load — `spawn` must refuse it up front rather than starting
    /// a background load that can only fail.
    #[tokio::test]
    async fn spawn_refuses_a_directory_that_is_not_a_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let readiness = Readiness::loading();
        // `.err().unwrap()` rather than `.unwrap_err()`: the `Ok` side is
        // `Arc<MlxProvider>`, which does not implement `Debug` (nor does the
        // engine it can hold), so `unwrap_err`'s bound on `T: Debug` does not
        // hold here.
        let err = MlxProvider::spawn(dir.path().to_path_buf(), "test/model".into(), readiness)
            .err()
            .unwrap();
        assert!(err.to_string().contains("not a checkpoint"), "{err}");
    }
}
