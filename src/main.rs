//! `sen-mlx` — MLX LLM runtime for SenClaw (chat + vision, Apple Silicon only).
//!
//! `sen-mlx serve --host {host} --port {port} --model {model_path}` loads
//! exactly that one checkpoint at startup (also from `SENCLAW_MODEL_PATH` when
//! `--model` is omitted) and serves it as an OpenAI-compatible provider
//! (`GET /v1/models`, `POST /v1/chat/completions`) plus the runtime protocol's
//! common routes (`/health`, `/runtime/info`, `/runtime/shutdown`) from
//! [`sen_runtime_sdk::server::serve`]. Full contract:
//! `senclaw/docs/runtime-protocol.md` §4.2.
//!
//! ## The startup rule this binary is shaped around
//!
//! Unlike the old Space App (health-gated on a 30-second budget, so it had to
//! load lazily on first request), a runtime is health-gated on
//! `health.startupTimeoutSecs` — minutes, not seconds. So the model loads once,
//! eagerly, right here: [`Readiness::loading`] until [`provider::MlxProvider`]'s
//! background task either [`Readiness::set_ready`]s or
//! [`Readiness::set_failed`]s, and `/health` answers 503 until then.

mod engine;
mod provider;
mod settings;

use std::sync::Arc;

use app_space_sdk::llm::LlmProvider;
use sen_runtime_sdk::env::LaunchEnv;
use sen_runtime_sdk::manifest::{Capability, RunMode};
use sen_runtime_sdk::server::{serve, Readiness, ServeArgs, ServeOptions};

use provider::MlxProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    sen_runtime_sdk::server::init_tracing();

    // `serve` is the only subcommand this runtime has. Accept a bare flag list
    // too (`sen-mlx --port 4970 --model …`), the way `cargo run --` is used
    // during development, but refuse anything else by name rather than
    // silently mis-parsing a typo as a flag.
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("serve") => {
            argv.remove(0);
        }
        Some(other) if !other.starts_with('-') => {
            anyhow::bail!("unknown subcommand `{other}` (sen-mlx has only `serve`)");
        }
        _ => {}
    }
    let parsed = ServeArgs::parse(argv).map_err(|e| anyhow::anyhow!(e))?;

    let env = LaunchEnv::from_env("sen-mlx", env!("CARGO_PKG_VERSION"));
    let model_path = parsed
        .model
        .clone()
        .or_else(|| env.model_path.clone())
        .ok_or_else(|| anyhow::anyhow!("no model: pass --model <path> or set SENCLAW_MODEL_PATH"))?;
    if !model_path.is_dir() {
        anyhow::bail!("--model {} is not a directory", model_path.display());
    }
    // The directory name is the fallback id: for the shared model root's
    // `<org>__<repo>` layout this is close enough to the real id for a picker
    // label, and it is all a hand-started dev run has to go on.
    let model_id = env
        .model_id
        .clone()
        .or_else(|| model_path.file_name().map(|n| n.to_string_lossy().into_owned()))
        .ok_or_else(|| anyhow::anyhow!("could not derive a model id from {}", model_path.display()))?;

    let readiness = Readiness::loading();
    let provider = MlxProvider::spawn(model_path, model_id, readiness.clone())?;
    let routes = app_space_sdk::llm::openai_router(Arc::clone(&provider));

    serve(
        routes,
        ServeOptions {
            env,
            mode: RunMode::Model,
            capabilities: vec![Capability::Chat, Capability::Vision],
            readiness,
            info_detail: Some(Arc::new(move || serde_json::json!({ "models": provider.models() }))),
            args: parsed,
        },
    )
    .await
}
