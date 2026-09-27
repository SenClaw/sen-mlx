//! Inference settings, stored next to the models.
//!
//! One `settings.json` at the shared model root (`<SENCLAW_LOCAL_MODELS_DIR>/settings.json`),
//! written by the daemon for as long as local models have existed and now owned
//! by this runtime. The struct and its field names are unchanged from the old
//! in-daemon / Space-App `local_model_core::settings::Settings` — same file,
//! same shape, so nothing already on a user's disk is invalidated by the split.
//!
//! Every field is `Option`, and `None` never means "zero" — it means *defer*.
//! The precedence is **user setting → the checkpoint's own
//! `generation_config.json` → off**, which is why a `top_k` of `None` is not the
//! same as a `top_k` of `0`: `None` lets Gemma 4's shipped `64` apply, `0`
//! overrides it with an untruncated draw.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Prompt ceiling after chat-template encoding. Older turns are dropped from
/// the start.
pub const DEFAULT_MAX_PROMPT_TOKENS: u32 = 128_000;
/// Generated-token ceiling for one turn.
pub const DEFAULT_MAX_NEW_TOKENS: u32 = 8192;
/// Rolling KV window: past this, the oldest tokens are evicted so memory stays
/// bounded. RoPE positions stay absolute, so the retained window keeps its
/// quality.
pub const DEFAULT_KV_WINDOW_TOKENS: u32 = 20_480;
/// Drop weights after this long without a request.
///
/// Five minutes, not the 60-second Space-App default: reloading is gigabytes of
/// disk, and two messages two minutes apart would otherwise pay it twice.
pub const DEFAULT_IDLE_UNLOAD_SECS: u32 = 300;

fn default_enable_thinking() -> Option<bool> {
    Some(false)
}

/// **Field names are the JSON keys, snake_case.** No `rename_all`: this file
/// already exists on every machine that has used a local model, written by the
/// daemon in snake_case. A camelCase rename would parse every one of those into
/// all-`None` — not an error, just every setting silently back to its default,
/// which is the worst way for a config format to change.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    /// `None` → [`DEFAULT_MAX_PROMPT_TOKENS`].
    #[serde(default)]
    pub max_prompt_tokens: Option<u32>,
    /// `None` → [`DEFAULT_MAX_NEW_TOKENS`].
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    /// `Some(0.0)` is greedy `argmax`, not "unset".
    #[serde(default)]
    pub temperature: Option<f32>,
    /// HF repetition penalty over recent ids. `1.0` disables it.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// Keep the `top_k` most likely tokens. **`0` disables truncation**;
    /// `None` defers to the checkpoint.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Nucleus mass. **`1.0` disables it**; `None` defers to the checkpoint.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Tokens per prefill chunk. Clamped to 32-4096.
    #[serde(default)]
    pub prefill_chunk_tokens: Option<u32>,
    /// `enable_thinking` for the chat template. Defaults to `false` — an
    /// unbounded reasoning block on a small local model is mostly latency.
    #[serde(default = "default_enable_thinking")]
    pub enable_thinking: Option<bool>,
    /// `None` → [`DEFAULT_KV_WINDOW_TOKENS`]. Range 128-262 144.
    #[serde(default)]
    pub max_kv_tokens: Option<u32>,
    /// Packed KV on Metal: `4` or `8` bits, `None`/`0` for FP16. MLX only.
    #[serde(default)]
    pub mlx_kv_cache_bits: Option<u8>,
    /// TurboQuant KV total bit budget: `3` = TQ3, `4` = TQ4. `2` is accepted for
    /// compatibility and remapped to 3. `0` or `None` disables it (FP16 KV).
    #[serde(default)]
    pub kv_cache_bits: Option<u8>,
    /// Context length at which TurboQuant quantization starts. `0` quantizes
    /// from the first decode step; `None` → 2048.
    #[serde(default)]
    pub tq_activate_at: Option<u32>,
    /// Free the KV cache once process RSS passes this many MiB. `None` disables
    /// the check.
    #[serde(default)]
    pub kv_release_rss_mib: Option<u32>,
    /// Recurrent sweeps for looped LMs (Ouro). `None` uses the checkpoint's own
    /// value.
    #[serde(default)]
    pub recurrence_steps: Option<u32>,
    /// Which engine app the user prefers on this machine: `"mlx"` or
    /// `"candle"`. Historical — candle left the daemon, but the field stays so
    /// an old settings file round-trips instead of losing an unrelated key.
    #[serde(default)]
    pub preferred_backend: Option<String>,
    /// `None` → [`DEFAULT_IDLE_UNLOAD_SECS`]. `0` keeps weights resident.
    /// Not read operationally by this runtime (one model per process; the
    /// daemon's own `idleTimeoutSecs` stops the whole process), kept for
    /// round-tripping the shared file.
    #[serde(default)]
    pub idle_unload_secs: Option<u32>,
    /// Drop the per-session KV cache when a turn ends without a tool call.
    #[serde(default)]
    pub release_cache_after_session: Option<bool>,
}

impl Settings {
    pub fn max_prompt_tokens(&self) -> u32 {
        self.max_prompt_tokens.unwrap_or(DEFAULT_MAX_PROMPT_TOKENS)
    }
    pub fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS)
    }
    pub fn max_kv_tokens(&self) -> u32 {
        self.max_kv_tokens.unwrap_or(DEFAULT_KV_WINDOW_TOKENS).clamp(128, 262_144)
    }
    pub fn prefill_chunk_tokens(&self) -> u32 {
        self.prefill_chunk_tokens.unwrap_or(512).clamp(32, 4096)
    }
    pub fn idle_unload_secs(&self) -> u32 {
        self.idle_unload_secs.unwrap_or(DEFAULT_IDLE_UNLOAD_SECS)
    }
}

pub fn settings_path(root: &Path) -> PathBuf {
    root.join("settings.json")
}

/// Read the stored settings, or defaults.
///
/// A malformed file reads as defaults rather than an error: this is called on
/// the inference path, and refusing to generate because a settings file has a
/// stray comma would be a worse failure than ignoring it.
pub fn load(root: &Path) -> Settings {
    std::fs::read_to_string(settings_path(root)).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

pub fn save(root: &Path, s: &Settings) -> std::io::Result<()> {
    std::fs::create_dir_all(root)?;
    let body = serde_json::to_vec_pretty(s)?;
    // Write-then-rename: a reader on the inference path sees either the old
    // settings or the new ones, never a truncated file it would silently
    // interpret as defaults.
    let path = settings_path(root);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &body)?;
    std::fs::rename(&tmp, &path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file the daemon has been writing for as long as local models have
    /// existed. Reading it must not quietly reset anyone's settings.
    #[test]
    fn the_settings_file_the_daemon_already_wrote_still_parses() {
        let on_disk = r#"{
            "kv_cache_bits": null, "max_prompt_tokens": 128000, "max_new_tokens": null,
            "temperature": null, "repetition_penalty": null, "enable_thinking": true,
            "tq_activate_at": null, "max_kv_tokens": 128000, "mlx_kv_cache_bits": null,
            "preferred_backend": "mlx", "idle_unload_secs": 60,
            "kv_release_rss_mib": null, "release_cache_after_session": false,
            "recurrence_steps": null
        }"#;
        let s: Settings = serde_json::from_str(on_disk).unwrap();
        assert_eq!(s.max_prompt_tokens, Some(128_000));
        assert_eq!(s.max_kv_tokens, Some(128_000));
        assert_eq!(s.enable_thinking, Some(true));
        assert_eq!(s.preferred_backend.as_deref(), Some("mlx"));
        assert_eq!(s.idle_unload_secs, Some(60));
    }

    /// Round-tripping must keep the same keys, or the daemon and this runtime
    /// would each write a file the other reads as empty.
    #[test]
    fn the_written_keys_are_the_ones_that_were_read() {
        let s = Settings { max_new_tokens: Some(1), ..Default::default() };
        let v: serde_json::Value = serde_json::to_value(&s).unwrap();
        for key in [
            "max_prompt_tokens", "max_new_tokens", "enable_thinking", "max_kv_tokens",
            "kv_cache_bits", "idle_unload_secs", "preferred_backend",
        ] {
            assert!(v.get(key).is_some(), "`{key}` must be written in snake_case");
        }
    }

    #[test]
    fn zero_and_none_are_different_answers() {
        let s: Settings = serde_json::from_str(r#"{"top_k": 0}"#).unwrap();
        assert_eq!(s.top_k, Some(0), "0 means untruncated, not unset");

        let s: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(s.top_k, None, "unset must stay None so the checkpoint's own value applies");
    }

    #[test]
    fn thinking_is_off_unless_the_user_turned_it_on() {
        let s: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(s.enable_thinking, Some(false));
    }

    #[test]
    fn out_of_range_values_are_clamped_not_obeyed() {
        let s = Settings { max_kv_tokens: Some(1), prefill_chunk_tokens: Some(999_999), ..Default::default() };
        assert_eq!(s.max_kv_tokens(), 128);
        assert_eq!(s.prefill_chunk_tokens(), 4096);
    }

    #[test]
    fn weights_are_kept_five_minutes_not_the_space_app_default_of_sixty_seconds() {
        assert_eq!(Settings::default().idle_unload_secs(), 300);
        // 0 is an explicit "never unload", not a fallback to the default.
        let s = Settings { idle_unload_secs: Some(0), ..Default::default() };
        assert_eq!(s.idle_unload_secs(), 0);
    }

    #[test]
    fn a_malformed_file_reads_as_defaults_rather_than_stopping_inference() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(settings_path(dir.path()), "{ not json").unwrap();
        assert_eq!(load(dir.path()).max_new_tokens(), DEFAULT_MAX_NEW_TOKENS);
    }

    #[test]
    fn settings_round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let s = Settings { temperature: Some(0.0), top_k: Some(0), max_new_tokens: Some(1234), ..Default::default() };
        save(dir.path(), &s).unwrap();
        let back = load(dir.path());
        assert_eq!(back.temperature, Some(0.0), "greedy must survive the trip");
        assert_eq!(back.top_k, Some(0));
        assert_eq!(back.max_new_tokens(), 1234);
    }
}
