pub mod bonsai_q1;
/// DeepSeek-V2 (MLA + DeepSeekMoE) — `mlx-community/DeepSeek-Coder-V2-Lite-*`.
pub mod deepseek_v2;
pub mod falcon_mamba;
pub mod gated_delta;
pub mod gemma2;
pub mod gemma3;
pub mod gemma4;
/// Gemma-4 vision-side modules (MultimodalEmbedder now; vision tower
/// pending). Lives alongside `gemma4` and inherits its `local-mlx` gate via
/// the parent `mlx_lm` module — no extra feature flag needed.
pub mod gemma4_vision;
pub mod llama;
pub mod mamba2;
/// Ouro looped language model (LoopLM) — ByteDance Ouro-2.6B / -Thinking.
/// Llama-style decoder with sandwich norm, looped `total_ut_steps` times.
pub mod ouro;
pub mod qwen3;
pub mod qwen3_5;
/// Shared Qwen-family parser primitives (used by `qwen3` and `qwen3_5`).
pub mod qwen_common;
// `whisper` stays in the daemon: it is ASR, not an LLM, and the daemon
// still serves it. Its only dependency here was `error::Error`.
// pub mod whisper;
