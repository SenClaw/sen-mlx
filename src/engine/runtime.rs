//! Runtime trait and shared types for local LLM backends.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEndpoint {
    /// OpenAI-compatible base URL when the runtime exposes HTTP (sidecar mode).
    /// `None` for in-process native runtimes.
    pub base_url: Option<String>,
    pub model_name: String,
    pub adapt: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    NotInstalled,
    Installing,
    DownloadingModel,
    Starting,
    Ready,
    Stopped,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeHealth {
    pub status: RuntimeStatus,
    pub message: Option<String>,
}

/// Abstraction over local model runtimes (native mlx-rs, future llama.cpp, etc.).
#[async_trait]
pub trait LocalModelRuntime: Send + Sync {
    /// Ensure binaries / weights / venvs the runtime needs are present.
    async fn ensure_installed(&self) -> anyhow::Result<()>;

    /// Bring the runtime up for a given model id.
    async fn start(&self, model: &str) -> anyhow::Result<RuntimeEndpoint>;

    /// Tear down the runtime; idempotent.
    async fn stop(&self) -> anyhow::Result<()>;

    /// Liveness / readiness probe.
    async fn health(&self) -> anyhow::Result<RuntimeHealth>;

    /// True when the runtime can stream tokens via an in-process channel
    /// rather than HTTP. Native engines override this.
    fn supports_native_stream(&self) -> bool {
        false
    }

    /// Stream generation directly into `tx`. Default impl is unsupported.
    ///
    /// `messages` and `tools` follow the OpenAI Chat Completions JSON shape
    /// (same as HuggingFace `apply_chat_template` / many Jinja chat templates).
    async fn generate_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Vec<Value>,
        _tx: tokio::sync::mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("runtime does not support native streaming")
    }
}
