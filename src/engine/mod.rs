//! The MLX inference engine, moved out of the SenClaw daemon and then out of
//! its Space App into this standalone runtime.
//!
//! These modules are the daemon's old `src/local_model/` MLX half, then the
//! `apps/mlx-lm` Space App engine, carried over close to verbatim: the only
//! rewrite is that the three settings lookups which used to reach into
//! `local_model_core::settings` now read [`crate::settings`], the same file on
//! disk (`<SENCLAW_LOCAL_MODELS_DIR>/settings.json`) with the same struct.
//!
//! What deliberately did **not** come along:
//!
//! - `mlx_lm/models/whisper.rs` — ASR, not an LLM. `sen-whisper` serves
//!   speech-to-text; Whisper's only dependency on this tree was
//!   `mlx_lm::error::Error`.
//! - The known-models catalog and HF downloader/store (`apps/local-model-core`'s
//!   `store`/`download`/`api` modules) — the daemon owns the model library now
//!   (§6.1 of the runtime protocol); this runtime is handed exactly one
//!   `--model` path at startup and never lists or fetches anything.

pub mod chat_template_openai;
pub mod image_input;
pub mod mlx_lm;
pub mod mlx_lm_utils;
pub mod mlx_native;
pub mod mlx_prompt;
pub mod runtime;
pub mod stream_parser;
pub mod thinking_parse;

use std::path::Path;

pub use mlx_native::MlxNativeEngine;

/// Does this checkpoint ship weights this engine can actually read?
///
/// Both MLX and llama.cpp checkpoints can live under the shared model root, and
/// MLX reads **safetensors only** — a repo whose weights are `pytorch_model.bin`
/// is a complete, valid download of a supported architecture that MLX cannot
/// load. Checked before `detect_architecture` so a missing-weights checkpoint
/// fails with a clear reason instead of a loader error deep in the load path.
fn has_safetensors(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
}

/// Can this engine run the checkpoint in `dir`? Used at startup to fail fast
/// (`Readiness::set_failed`) with a clear reason instead of a loader error deep
/// in the load path.
pub fn supports(model_id: &str, dir: &Path) -> bool {
    if !has_safetensors(dir) {
        return false;
    }
    let Ok(raw) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    // A Whisper checkpoint sits in the same directory tree and is not an LLM.
    // Loading it here would produce a "model" that starts and answers nothing,
    // which reads as sen-mlx being broken rather than the wrong checkpoint.
    if serde_json::from_str::<serde_json::Value>(&raw).is_ok_and(|v| v.get("n_mels").is_some()) {
        return false;
    }
    mlx_native::detect_architecture(model_id, dir).is_ok()
}

/// The checkpoint's own context window, from `config.json`.
///
/// `text_config.max_position_embeddings` is checked first: it is where a
/// multimodal wrapper (Gemma 4) keeps the *language model's* window, and the
/// top-level field on those checkpoints describes the wrapper instead. `None`
/// when the config has neither — the caller falls back to the shared
/// `settings.json`'s `max_prompt_tokens`.
pub fn context_length(dir: &Path) -> Option<u32> {
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    v["text_config"]["max_position_embeddings"]
        .as_u64()
        .or_else(|| v["max_position_embeddings"].as_u64())
        .map(|n| n as u32)
}

/// Does this checkpoint take image input?
///
/// From the config, never from the model id. A local checkpoint is named things
/// like `mlx-community/Qwen3.5-2B-OptiQ-4bit`, which matches no vendor pattern —
/// a name-based guess is right or wrong by accident, and the wrong direction is
/// expensive: a text-only endpoint answers an image block with a hard 400 that
/// fails the whole turn.
pub fn has_vision(dir: &Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    v.get("vision_config").is_some()
        || v.get("vision_tower").is_some()
        || v["architectures"]
            .as_array()
            .is_some_and(|a| a.iter().any(|s| s.as_str().is_some_and(|s| s.contains("Vision") || s.contains("Conditional"))))
}
