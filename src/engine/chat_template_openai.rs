//! OpenAI-shaped `messages` / `tools` → HuggingFace Jinja chat templates.
//!
//! Layout:
//! - **Renderer** (this file): Jinja env setup, OpenAI message normalisation,
//!   `apply_chat_template_openai_shape` — model-agnostic.
//! - **`ChatTemplateModel` trait** (this file): per-model dispatch for
//!   `bos_token` / `eos_token` resolution. Each model in `mlx_lm::models::*`
//!   implements it (the model owns the only authoritative source for its
//!   special token ids: `args.bos_token_id`, tokenizer string ids, …).
//! - **Loader** (this file): `load_chat_template_from_dir` reads
//!   `tokenizer_config.json` → `chat_template.jinja` → caller-supplied fallback.

use minijinja::Value as MjValue;
use minijinja::{context, Environment, Template};
use serde_json::Value;
use std::path::Path;

use super::mlx_lm_utils::tokenizer::Tokenizer;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    RenderTemplate(#[from] minijinja::Error),
    #[error(transparent)]
    Encode(#[from] tokenizers::tokenizer::Error),
}

// ── Special tokens ─────────────────────────────────────────────────────────

/// `bos_token` / `eos_token` strings injected into HF chat templates.
///
/// Templates like Gemma‑3 prefix the prompt with `{{ bos_token }}`; Mistral V1
/// `[INST]` puts `{{ eos_token }}` after every assistant turn. Resolution is
/// model-specific — see [`ChatTemplateModel::resolve_special_tokens`].
#[derive(Debug, Default, Clone)]
pub struct SpecialTokens {
    pub bos: Option<String>,
    pub eos: Option<String>,
}

impl SpecialTokens {
    pub fn empty() -> Self {
        Self::default()
    }
}

/// True when the rendered template body literally mentions the named Jinja
/// variable (e.g. `bos_token`, `eos_token`). Cheap substring scan — used to
/// skip BPE-decode round-trips when the template doesn't need them.
pub fn template_mentions(template: &str, var: &str) -> bool {
    template.contains(var)
}

/// Per-model dispatch for special-token resolution.
///
/// Each MLX model (`mlx_lm::models::*`) implements this so it can decode its
/// own `bos_token_id` / `eos_token_ids` (which live on the model's own
/// `args` struct, or are derived from tokenizer string lookups for Mamba-class
/// models). Common renderer ([`apply_chat_template_openai_shape`]) consumes
/// the resolved [`SpecialTokens`] without caring about model arch.
pub trait ChatTemplateModel {
    /// The output-stream dialect this model emits. Each MLX model owns its own
    /// answer to "what structural markers do I produce" so adding a new arch
    /// just means implementing this trait in `mlx_lm::models::<arch>`.
    ///
    /// Default = no structured markers (base models that emit plain text).
    /// Override when the model is trained to emit `<think>` / `<tool_call>` /
    /// `<|channel>` / `<|tool_call>` / etc. The engine uses this as the
    /// authoritative [`super::stream_parser::MarkerSet`] for the loaded model
    /// (with `tokenizer_config.json`-driven detection as a fallback for archs
    /// that haven't declared their dialect here yet).
    fn markers(&self) -> super::stream_parser::MarkerSet {
        super::stream_parser::MarkerSet::empty()
    }

    /// Decode `bos_token` / `eos_token` strings the template asks for.
    ///
    /// Implementations should:
    /// 1. Check `template_mentions(template, "bos_token")` / `"eos_token"`
    ///    before doing any decode — most templates use neither.
    /// 2. Decode through `tokenizer` so the returned `String` is exactly the
    ///    token piece the BPE/SentencePiece encoder will produce when fed
    ///    back through `encode`.
    fn resolve_special_tokens(&self, template: &str, tokenizer: &Tokenizer) -> SpecialTokens;

    /// Token ids that terminate generation for this model.
    ///
    /// Each model owns this because the terminator set is config-specific: it
    /// comes from the model's own `args` (`eos_token_id` / `eos_token_ids`)
    /// and/or named special tokens resolved through `tokenizer`
    /// (`<|im_end|>`, `<|eot_id|>`, `</s>`, …) whose ids differ per vocabulary.
    /// Returning an empty `Vec` means "no stop token" — decode then runs to
    /// `max_new_tokens`. Prefer [`resolve_token_ids`] for the named-lookup part.
    fn stop_token_ids(&self, tokenizer: &Tokenizer) -> Vec<u32>;
}

/// Resolve a list of named special tokens (e.g. `<|im_end|>`) to ids via the
/// tokenizer, skipping misses and de-duplicating. Helper for `stop_token_ids`.
pub fn resolve_token_ids(tokenizer: &Tokenizer, names: &[&str]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    for name in names {
        if let Some(id) = tokenizer.token_to_id(name) {
            if !out.contains(&id) {
                out.push(id);
            }
        }
    }
    out
}

// ── Loader ─────────────────────────────────────────────────────────────────

/// Load the model's Jinja chat template.
///
/// Probe order (matches HF transformers behaviour):
/// 1. `tokenizer_config.json` → `chat_template` field.
/// 2. `chat_template.jinja` (newer HF layout — Qwen 3, Llama 3.x).
/// 3. `chat_template.json` → `chat_template` field (standalone template file
///    shipped by some mlx-community / multimodal checkpoints, e.g. Qwen3-ASR).
/// 4. `fallback(model_dir)` — caller supplies arch-specific fallbacks
///    (e.g. Mistral `[INST]` for Mamba-Codestral that ships no template).
///
/// Returns `Ok(None)` when no template is found; callers may then fall back
/// to a plain `role: content\n` transcript for base models.
pub fn load_chat_template_from_dir<F>(
    model_dir: &Path,
    model_id: &str,
    fallback: F,
) -> std::io::Result<Option<String>>
where
    F: FnOnce(&Path) -> std::io::Result<Option<String>>,
{
    let tokenizer_config = model_dir.join("tokenizer_config.json");
    if let Some(t) = load_chat_template_from_tokenizer_config(&tokenizer_config)? {
        return Ok(Some(t));
    }
    let jinja_path = model_dir.join("chat_template.jinja");
    if jinja_path.exists() {
        let t = std::fs::read_to_string(&jinja_path)?;
        tracing::info!("[local-mlx-native] loaded chat_template.jinja for {model_id}");
        return Ok(Some(t));
    }
    // Standalone `chat_template.json` (`{"chat_template": "…"}`) — same shape as
    // the tokenizer_config field, so reuse the extractor. Used by some
    // mlx-community / multimodal checkpoints (e.g. Qwen3-ASR) that ship the
    // template separately rather than embedding it in tokenizer_config.json.
    let json_path = model_dir.join("chat_template.json");
    if let Some(t) = load_chat_template_from_tokenizer_config(&json_path)? {
        tracing::info!("[local-mlx-native] loaded chat_template.json for {model_id}");
        return Ok(Some(t));
    }
    fallback(model_dir)
}

/// Pull `chat_template` out of `tokenizer_config.json` if present.
pub fn load_chat_template_from_tokenizer_config(path: &Path) -> std::io::Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str::<Value>(&content).ok().and_then(|v| {
        v.get("chat_template")
            .and_then(|x| x.as_str())
            .map(ToString::to_string)
    }))
}

// ── Jinja env / filters ────────────────────────────────────────────────────

/// Register filters required by HF chat templates (`tojson` for tool schemas
/// — `{{- tool | tojson }}` is how Qwen3 / Llama-3 templates emit tool args).
pub fn configure_jinja_env(env: &mut Environment<'static>) {
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    let _ = env.add_filter(
        "tojson",
        |value: MjValue| -> Result<String, minijinja::Error> {
            serde_json::to_string(&value).map_err(|e| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("tojson: {e}"),
                )
            })
        },
    );
}

// ── OpenAI message normalisation ───────────────────────────────────────────

/// HF templates index `message.content` like a Python `str` (`.startswith`,
/// `.endswith`, …). OpenAI multimodal payloads ship `content` as an array of
/// `{"type":"text","text":"…"}` blocks; minijinja exposes that as a sequence
/// and the template explodes. Collapse text parts into a single string —
/// same practical outcome as HF transformers in tool / text-only paths.
fn normalize_openai_messages_for_hf_jinja(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(normalize_one_openai_message_for_hf_jinja)
        .collect()
}

fn normalize_one_openai_message_for_hf_jinja(msg: &Value) -> Value {
    let Some(obj) = msg.as_object() else {
        return msg.clone();
    };
    let mut out = obj.clone();
    if let Some(content) = obj.get("content") {
        if let Some(plain) = openai_message_content_to_plain_string(content) {
            out.insert("content".to_string(), Value::String(plain));
        }
    }
    Value::Object(out)
}

fn openai_message_content_to_plain_string(content: &Value) -> Option<String> {
    match content {
        Value::String(_) => None,
        Value::Null => Some(String::new()),
        Value::Array(parts) => Some(flatten_openai_content_parts(parts)),
        Value::Object(block) => {
            content_block_text(block).or_else(|| serde_json::to_string(content).ok())
        }
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
    }
}

fn flatten_openai_content_parts(parts: &[Value]) -> String {
    let mut out = String::new();
    for p in parts {
        let piece = match p {
            Value::Object(o) => content_block_text(o),
            Value::String(s) => Some(s.clone()),
            _ => None,
        };
        let Some(piece) = piece.filter(|s| !s.is_empty()) else {
            continue;
        };
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&piece);
    }
    out
}

fn content_block_text(block: &serde_json::Map<String, Value>) -> Option<String> {
    match block.get("text") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(chunks)) => {
            let joined = chunks.iter().filter_map(|c| c.as_str()).collect::<String>();
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        _ => None,
    }
}

// ── Renderer ───────────────────────────────────────────────────────────────

/// Render one conversation (`messages` + optional `tools`) via Jinja.
///
/// Template-cache key is `chat_template_id` if provided, else `model_id`; the
/// caller is responsible for invalidating when the template body changes
/// (e.g. swap the model). `bos_token` / `eos_token` are passed through as
/// strings even if the template doesn't reference them (cost: one no-op
/// context slot) — keeps the call site uniform across archs.
pub fn apply_chat_template_openai_shape(
    env: &mut Environment<'static>,
    model_template: String,
    model_id: &str,
    chat_template_id: Option<&str>,
    messages: &[Value],
    tools: &[Value],
    add_generation_prompt: Option<bool>,
    enable_thinking: Option<bool>,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<String, Error> {
    let add_generation_prompt = add_generation_prompt.unwrap_or(false);
    let bos_slot = bos_token.unwrap_or("");
    let eos_slot = eos_token.unwrap_or("");

    let template = match chat_template_id {
        Some(chat_template_id) => env.get_template(chat_template_id)?,
        None => match env.get_template(model_id) {
            Ok(template) => template,
            Err(_) => {
                env.add_template_owned(model_id.to_owned(), model_template)?;
                env.get_template(model_id)
                    .expect("Newly added template must be present")
            }
        },
    };

    let messages = normalize_openai_messages_for_hf_jinja(messages);
    render_openai_template(
        &template,
        &messages,
        tools,
        add_generation_prompt,
        enable_thinking,
        bos_slot,
        eos_slot,
    )
}

fn render_openai_template(
    template: &Template,
    messages: &[Value],
    tools: &[Value],
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
    bos_token: &str,
    eos_token: &str,
) -> Result<String, Error> {
    match enable_thinking {
        Some(thinking) => template
            .render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => add_generation_prompt,
                enable_thinking => thinking,
                bos_token => bos_token,
                eos_token => eos_token,
            })
            .map_err(Into::into),
        None => template
            .render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => add_generation_prompt,
                bos_token => bos_token,
                eos_token => eos_token,
            })
            .map_err(Into::into),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: Gemma-4's chat template indexes tool definitions as
    /// `tool['function']['name']`. A flattened HF-style tool
    /// (`{name, description, parameters}`) raises an UndefinedError at the
    /// `format_function_declaration` macro; the OpenAI-nested form
    /// (`{type:"function", function:{…}}`) renders. Uses the real on-disk
    /// template when present (skips otherwise so CI without the model passes).
    #[test]
    fn gemma4_template_renders_nested_tool_not_flattened() {
        let path = std::path::Path::new(env!("HOME"))
            .join(".senclaw/local-models/mlx-community__gemma-4-e2b-it-4bit/chat_template.jinja");
        let Ok(template) = std::fs::read_to_string(&path) else {
            eprintln!("skip: gemma-4 template not present at {}", path.display());
            return;
        };

        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "What's the weather?"}),
        ];
        let flattened = vec![serde_json::json!({
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        })];
        let nested = vec![serde_json::json!({
            "type": "function",
            "function": flattened[0].clone()
        })];

        // Flattened → must fail (this is the bug the engine fix avoids).
        let mut env1 = Environment::new();
        configure_jinja_env(&mut env1);
        let flat_res = apply_chat_template_openai_shape(
            &mut env1,
            template.clone(),
            "gemma4-flat",
            None,
            &messages,
            &flattened,
            Some(true),
            None,
            Some("<bos>"),
            None,
        );
        assert!(
            flat_res.is_err(),
            "flattened tool should raise UndefinedError"
        );

        // Nested (what the engine now produces for Gemma-4) → must render.
        let mut env2 = Environment::new();
        configure_jinja_env(&mut env2);
        let nested_res = apply_chat_template_openai_shape(
            &mut env2,
            template,
            "gemma4-nested",
            None,
            &messages,
            &nested,
            Some(true),
            None,
            Some("<bos>"),
            None,
        );
        let rendered = nested_res.expect("nested tool must render without error");
        assert!(
            rendered.contains("get_weather"),
            "tool name should appear in render"
        );
        assert!(
            rendered.contains("model"),
            "generation prompt should be present"
        );
    }

    #[test]
    fn template_mentions_finds_bos() {
        assert!(template_mentions(
            "{{ bos_token }}{% for m in messages %}",
            "bos_token"
        ));
        assert!(!template_mentions("{% for m in messages %}", "bos_token"));
    }

    #[test]
    fn load_chat_template_from_tokenizer_config_extracts_field() {
        let tmp = std::env::temp_dir().join("chat_template_test_tokcfg.json");
        std::fs::write(&tmp, r#"{"chat_template": "hello {{ messages }}"}"#).unwrap();
        let t = load_chat_template_from_tokenizer_config(&tmp).unwrap();
        assert_eq!(t.as_deref(), Some("hello {{ messages }}"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn load_chat_template_from_dir_uses_fallback() {
        let tmp = std::env::temp_dir().join("chat_template_test_dir_fb");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let t = load_chat_template_from_dir(&tmp, "test-model", |_dir| {
            Ok(Some("fallback-template".to_string()))
        })
        .unwrap();
        assert_eq!(t.as_deref(), Some("fallback-template"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_chat_template_from_dir_reads_standalone_json() {
        // A checkpoint that ships only `chat_template.json` (no embedded
        // tokenizer_config template, no `.jinja`) — e.g. Qwen3-ASR.
        let tmp = std::env::temp_dir().join("chat_template_test_dir_json");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            tmp.join("chat_template.json"),
            r#"{"chat_template": "JSON-TPL {{ messages }}"}"#,
        )
        .unwrap();
        let t = load_chat_template_from_dir(&tmp, "test-model", |_dir| {
            Ok(Some("fallback-template".to_string()))
        })
        .unwrap();
        assert_eq!(
            t.as_deref(),
            Some("JSON-TPL {{ messages }}"),
            "chat_template.json must be read in preference to the fallback"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
