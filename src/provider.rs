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

use crate::engine::stream_parser::ParserEvent;
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

/// Resolves once the client of this turn has gone away (the SDK side of the
/// sink was dropped). `ChunkSink` only offers the non-blocking check, so poll
/// it; a quarter second is far below any cancel a person would notice.
async fn client_gone(sink: &ChunkSink) {
    while !sink.is_closed() {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

#[async_trait::async_trait]
impl LlmProvider for MlxProvider {
    fn models(&self) -> Vec<ModelCard> {
        vec![self.card.clone()]
    }

    async fn chat(&self, req: ChatRequest, sink: ChunkSink) -> Result<()> {
        let engine = self.engine().await?;

        // Events, not the raw token string. `stream_events_to_channel` runs the
        // chunk-safe `LocalStreamParser` SemaClaw added for this and that the
        // Space App provider never called: a marker split across two tokens
        // stays buffered inside the parser until it completes, then comes out
        // as visible text, reasoning, or one whole tool call. Sending those
        // as they arrive is what resets the daemon's 120s read-stall timer
        // during decode. Buffering the turn and parsing once at the end left
        // the socket silent for the whole prefill (113s on a 37k-token Gemma
        // prompt) plus decode, and the session was reset with
        // "OpenAI stream chunk error".
        let (tx, mut rx) = tokio::sync::mpsc::channel::<ParserEvent>(32);
        let gen = {
            let engine = Arc::clone(&engine);
            let messages = req.messages.clone();
            let tools = req.tools.clone();
            tokio::spawn(
                async move { engine.stream_events_to_channel(&messages, &tools, tx).await },
            )
        };

        let mut made_tool_call = false;
        loop {
            // The daemon cancels a turn (user stop, stall, ACP cancel) by
            // dropping the connection — all this process ever sees is the SDK's
            // side of `sink` going away. Stop reading then: dropping `rx` makes
            // the parser pipe's next send fail, which drops the engine's token
            // channel, and the decode loop ends at its next token. That is the
            // old in-process adapter's `abort()` on cancel; without it an
            // abandoned answer ran to `max_new_tokens` (4 087 tokens, 88s
            // measured) and every later request queued behind it.
            let ev = tokio::select! {
                ev = rx.recv() => ev,
                () = client_gone(&sink) => break,
            };
            let Some(ev) = ev else { break };
            match ev {
                ParserEvent::Reasoning(s) if !s.is_empty() => {
                    sink.send(Chunk::Reasoning(s)).await;
                }
                ParserEvent::Visible(s) if !s.is_empty() => {
                    sink.send(Chunk::Text(s)).await;
                }
                ParserEvent::ToolCall(tc) => {
                    // Emitted only once the closing marker has arrived, so the
                    // name and arguments go out as one delta. The SDK assigns
                    // the stream index.
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
                    made_tool_call = true;
                    sink.send(Chunk::ToolCall {
                        id,
                        name,
                        arguments,
                    })
                    .await;
                }
                _ => {}
            }
        }
        // Closing the receiver is what stops an abandoned generation (see above);
        // for a finished one it is a no-op.
        drop(rx);
        let gen_result = gen.await?;
        if sink.is_closed() {
            // Nobody is listening: whatever the engine returned after its
            // channel closed under it is not an error worth reporting.
            return Ok(());
        }
        gen_result?;

        if let Some((prompt_tokens, completion_tokens)) = engine.last_usage() {
            sink.send(Chunk::Usage {
                prompt_tokens: prompt_tokens as u64,
                completion_tokens: completion_tokens as u64,
            })
            .await;
        }

        // Same contract the daemon's own turn loop had: `release_cache_after_session`
        // drops the per-session KV (worth hundreds of MB after a long
        // generation) at the end of a turn while keeping the weights warm —
        // but only on a turn with no tool call. A tool call means the agent
        // loop is coming straight back with this prompt plus the result, and
        // dropping the prefix cache there would re-prefill it in full.
        let settings_dir = engine.model_dir().parent().map(Path::to_path_buf);
        if let (false, Some(dir)) = (made_tool_call, settings_dir) {
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
