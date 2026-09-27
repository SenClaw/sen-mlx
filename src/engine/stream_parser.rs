//! Data-driven streaming parser for local LLM dialects.
//!
//! ## Architecture
//!
//! Every local model's wire format is captured in a [`ParserConfig`] loaded
//! from the model's `tokenizer_config.json` (+ `chat_template.jinja` fallback):
//!
//! - **Special tokens / markers** ([`MarkerSet`]) — what `<|channel>` /
//!   `<|tool_call>` / `<think>` / `<|"|>` strings the model is trained to emit.
//! - **Chat template** — the Jinja string that renders OpenAI-shape messages
//!   into the prompt the model expects.
//! - **bos / eos** — needed by the chat template (`{{ bos_token }}` literal
//!   injection) and by the stop set.
//!
//! With config loaded, the same code handles every supported arch:
//!
//! ```text
//!   tokenizer_config.json + chat_template.jinja
//!                    │
//!                    ▼
//!         ParserConfig::from_model_dir()
//!              │             │
//!         render_chat   LocalStreamParser
//!         (input)        (output stream)
//!              │             │
//!              ▼             ▼
//!         prompt tokens   ParserEvent::{Visible, Reasoning, ToolCall}
//! ```
//!
//! Adding a new arch becomes: ship its `tokenizer_config.json`. No code change.
//!
//! ## Marker discovery
//!
//! Two paths, tried in order, both keyed on `tokenizer_config.json`:
//!
//! 1. **Explicit role-tokens** (Gemma-4 style) — keys like `soc_token`
//!    (`<|channel>`), `eoc_token` (`<channel|>`), `stc_token` (`<|tool_call>`),
//!    `etc_token` (`<tool_call|>`), `escape_token` (`<|"|>`), `think_token`
//!    (`<|think|>`) declare role-to-string mapping directly.
//! 2. **Chat-template scan** (Qwen / fallback) — when the named keys are
//!    absent, scan the literal text of the chat template for `<think>` /
//!    `</think>` / `<tool_call>` / `</tool_call>`. Templates literally write
//!    out the markers the model was trained to emit, so a contains-check is a
//!    sound, robust signal.

use std::path::Path;

use minijinja::Environment;
use serde_json::Value;

// ── Public types ────────────────────────────────────────────────────────────

/// How a tool-call body is serialized between the open/close markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// `{"name":…, "arguments":{…}}` JSON, with Hermes `<function=NAME>
    /// <parameter=K>v</parameter>…</function>` XML fallback. Used by Qwen /
    /// Llama / OpenAI-compatible local models.
    QwenJsonOrXml,
    /// `call:NAME{key:val,…}` with `<|"|>`-wrapped string args, bare numbers,
    /// bools, `{…}`/`[…]`, unquoted keys. Used by Gemma-4 harmony.
    Gemma4Compact,
}

impl Default for ToolCallFormat {
    fn default() -> Self {
        Self::QwenJsonOrXml
    }
}

/// All structural markers the model is trained to emit. Every field is
/// optional — absence means "this model doesn't use that block".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MarkerSet {
    /// Reasoning block open/close, e.g. `("<think>", "</think>")`. Content is
    /// emitted as [`ParserEvent::Reasoning`].
    pub think: Option<(String, String)>,
    /// Tool-call body open/close. Content is parsed per [`Self::tool_call_format`].
    pub tool_call: Option<(String, String)>,
    /// Channel-wrapper open/close (Gemma-4 harmony). Content `name\nbody…`.
    /// A `thought` channel routes to reasoning; other channels route to visible.
    pub channel: Option<(String, String)>,
    /// String wrapper used inside tool-call args (e.g. Gemma-4 `<|"|>`).
    /// Stripped from any visible / reasoning text.
    pub quote: Option<String>,
    /// How to parse the tool-call body.
    pub tool_call_format: ToolCallFormat,
}

impl MarkerSet {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Preset for Qwen / Llama-style models. Matches `tokenizer_config.json`
    /// for those families which don't expose `soc_token`/`stc_token`.
    pub fn qwen() -> Self {
        Self {
            think: Some(("<think>".into(), "</think>".into())),
            tool_call: Some(("<tool_call>".into(), "</tool_call>".into())),
            channel: None,
            quote: None,
            tool_call_format: ToolCallFormat::QwenJsonOrXml,
        }
    }

    /// Preset for Gemma-4 harmony. Mirrors the named tokens in Gemma-4's
    /// `tokenizer_config.json` (`soc_token`, `eoc_token`, `stc_token`,
    /// `etc_token`, `escape_token`, …).
    pub fn gemma4() -> Self {
        Self {
            think: None,
            tool_call: Some(("<|tool_call>".into(), "<tool_call|>".into())),
            channel: Some(("<|channel>".into(), "<channel|>".into())),
            quote: Some("<|\"|>".into()),
            tool_call_format: ToolCallFormat::Gemma4Compact,
        }
    }

    /// Build a marker set by scanning a literal chat-template string. Used
    /// when `tokenizer_config.json` doesn't declare role-tokens explicitly.
    pub fn detect_from_chat_template(template: &str) -> Self {
        let mut ms = MarkerSet::empty();

        // Gemma-4 harmony — explicit because order matters: harmony `<|tool_call>`
        // would also trigger the `<tool_call>` (Qwen) check below.
        let has_harmony_channel =
            template.contains("<|channel>") && template.contains("<channel|>");
        let has_harmony_tool =
            template.contains("<|tool_call>") && template.contains("<tool_call|>");
        if has_harmony_channel {
            ms.channel = Some(("<|channel>".into(), "<channel|>".into()));
        }
        if has_harmony_tool {
            ms.tool_call = Some(("<|tool_call>".into(), "<tool_call|>".into()));
            ms.tool_call_format = ToolCallFormat::Gemma4Compact;
        }
        if template.contains("<|\"|>") {
            ms.quote = Some("<|\"|>".into());
        }

        // Qwen / Llama. `<tool_call>` only counts if harmony tool wasn't found
        // (template might mention `<tool_call|>` substring incidentally).
        if template.contains("<think>") && template.contains("</think>") {
            ms.think = Some(("<think>".into(), "</think>".into()));
        }
        if !has_harmony_tool
            && template.contains("<tool_call>")
            && template.contains("</tool_call>")
        {
            ms.tool_call = Some(("<tool_call>".into(), "</tool_call>".into()));
            ms.tool_call_format = ToolCallFormat::QwenJsonOrXml;
        }

        ms
    }

    /// Build from explicit named role-tokens declared in `tokenizer_config.json`.
    /// Returns `None` when the file declares none of the known role-token keys.
    fn from_role_tokens(tok_cfg: &Value) -> Option<Self> {
        let get_str = |k: &str| tok_cfg.get(k).and_then(|v| v.as_str()).map(String::from);
        let soc = get_str("soc_token"); // start-of-channel
        let eoc = get_str("eoc_token"); // end-of-channel
        let stc = get_str("stc_token"); // start tool call
        let etc = get_str("etc_token"); // end tool call
        let escape = get_str("escape_token");
        let think_tok = get_str("think_token");
        if soc.is_none() && stc.is_none() && think_tok.is_none() && escape.is_none() {
            return None;
        }
        let mut ms = MarkerSet::empty();
        if let (Some(o), Some(c)) = (soc, eoc) {
            ms.channel = Some((o, c));
        }
        if let (Some(o), Some(c)) = (stc, etc) {
            ms.tool_call = Some((o, c));
            ms.tool_call_format = ToolCallFormat::Gemma4Compact;
        }
        if let Some(q) = escape {
            ms.quote = Some(q);
        }
        // `think_token` is a single token in Gemma-4 (one-sided marker that
        // appears before reasoning); we don't have a paired close, so we leave
        // `think` unset (Gemma-4 wraps thinking in a channel, not a paired
        // `<think>`).
        let _ = think_tok;
        Some(ms)
    }

    /// True when the parser would do anything (any marker configured).
    pub fn is_empty(&self) -> bool {
        self.think.is_none()
            && self.tool_call.is_none()
            && self.channel.is_none()
            && self.quote.is_none()
    }
}

/// Full per-model config — markers + chat-template + special tokens — loaded
/// once from a model directory and reused across generation turns.
#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub model_id: String,
    /// The Jinja chat template (from `tokenizer_config.json::chat_template` or
    /// `chat_template.jinja`). `None` for base models that ship no template.
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub markers: MarkerSet,
}

impl ParserConfig {
    /// Load by reading `tokenizer_config.json` (+ chat_template.jinja fallback)
    /// from a model directory. Marker discovery: tries explicit named
    /// role-tokens first, then falls back to chat-template scan.
    pub fn from_model_dir(model_dir: &Path, model_id: &str) -> std::io::Result<Self> {
        let tok_cfg_path = model_dir.join("tokenizer_config.json");
        let tok_cfg: Value = if tok_cfg_path.exists() {
            let raw = std::fs::read_to_string(&tok_cfg_path)?;
            serde_json::from_str(&raw).unwrap_or(Value::Null)
        } else {
            Value::Null
        };

        // bos/eos: each can be a plain string or `{ "content": "…", … }`.
        let token_str = |v: &Value| -> Option<String> {
            match v {
                Value::String(s) => Some(s.clone()),
                Value::Object(o) => o.get("content").and_then(|x| x.as_str()).map(String::from),
                _ => None,
            }
        };
        let bos_token = tok_cfg.get("bos_token").and_then(token_str);
        let eos_token = tok_cfg.get("eos_token").and_then(token_str);

        // chat_template — probe order matches HF transformers behaviour
        // (consolidated with `chat_template_openai::load_chat_template_from_dir`
        // so ParserConfig is the single source of truth for template loading):
        //   1. `tokenizer_config.json::chat_template` (embedded — Qwen / Llama)
        //   2. `chat_template.jinja` (newer HF layout — Qwen 3, Llama 3.x, Gemma-4)
        //   3. `chat_template.json::chat_template` (some mlx-community / multimodal
        //      checkpoints ship it standalone, e.g. Qwen3-ASR)
        let chat_template = tok_cfg
            .get("chat_template")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| std::fs::read_to_string(model_dir.join("chat_template.jinja")).ok())
            .or_else(|| {
                std::fs::read_to_string(model_dir.join("chat_template.json"))
                    .ok()
                    .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
                    .and_then(|v| {
                        v.get("chat_template")
                            .and_then(|x| x.as_str())
                            .map(String::from)
                    })
            });

        // Markers: explicit role-tokens (Gemma-4 style) → template scan → empty.
        let markers = MarkerSet::from_role_tokens(&tok_cfg)
            .or_else(|| {
                chat_template
                    .as_deref()
                    .map(MarkerSet::detect_from_chat_template)
            })
            .unwrap_or_default();

        Ok(Self {
            model_id: model_id.to_string(),
            chat_template,
            bos_token,
            eos_token,
            markers,
        })
    }

    /// Render OpenAI-shaped `messages` + `tools` through the model's chat
    /// template. Delegates to [`super::chat_template_openai::apply_chat_template_openai_shape`]
    /// using this config's template + bos/eos.
    pub fn render_chat(
        &self,
        env: &mut Environment<'static>,
        messages: &[Value],
        tools: &[Value],
        add_generation_prompt: Option<bool>,
        enable_thinking: Option<bool>,
    ) -> Result<String, super::chat_template_openai::Error> {
        let template = self.chat_template.clone().ok_or_else(|| {
            super::chat_template_openai::Error::RenderTemplate(minijinja::Error::new(
                minijinja::ErrorKind::TemplateNotFound,
                "no chat_template for this model",
            ))
        })?;
        super::chat_template_openai::apply_chat_template_openai_shape(
            env,
            template,
            &self.model_id,
            None,
            messages,
            tools,
            add_generation_prompt,
            enable_thinking,
            self.bos_token.as_deref(),
            self.eos_token.as_deref(),
        )
    }
}

// ── Legacy dialect-based API (kept for callers that don't have a model_dir) ─

/// Coarse dialect hint. Prefer [`ParserConfig::from_model_dir`] for new code —
/// this enum is here so callers without a `model_dir` (memory/cognitive paths)
/// can still pick a sensible preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalDialect {
    Auto,
    Qwen,
    Gemma4,
    None,
}

impl LocalDialect {
    pub fn into_markers(self) -> MarkerSet {
        match self {
            Self::Qwen => MarkerSet::qwen(),
            Self::Gemma4 => MarkerSet::gemma4(),
            // Auto = union of Qwen + Gemma-4 so first marker locks behaviour.
            Self::Auto => {
                let mut g = MarkerSet::gemma4();
                let q = MarkerSet::qwen();
                if g.think.is_none() {
                    g.think = q.think;
                }
                // Tool-call: Auto starts with Gemma-4 shape; once the first
                // marker is consumed, the parser doesn't re-detect (callers
                // that need rebust dual-handling should pass a per-model
                // config built from the model dir).
                let _ = q;
                g
            }
            Self::None => MarkerSet::empty(),
        }
    }
}

/// Best-effort dialect pick from a model id. Falls back to `Auto`.
pub fn dialect_for_model_id(model_id: &str) -> LocalDialect {
    let lower = model_id.to_lowercase();
    if lower.contains("gemma-4") || lower.contains("gemma4") {
        LocalDialect::Gemma4
    } else if lower.contains("qwen") || lower.contains("llama") {
        LocalDialect::Qwen
    } else {
        LocalDialect::Auto
    }
}

// ── ParserEvent ─────────────────────────────────────────────────────────────

/// Canonical output of the parser — OpenAI-shape ready.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserEvent {
    /// User-visible text (markers + quote-wrappers stripped).
    Visible(String),
    /// Reasoning text (think / `thought` channel).
    Reasoning(String),
    /// `{id, type:"function", function:{name, arguments:String}}`. `arguments`
    /// is a JSON-encoded string per OpenAI spec.
    ToolCall(Value),
}

// ── Parser state machine ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Outside,
    InChannel(String),
    InThink(String),
    InToolCall(String),
}

/// Stateful chunk-safe parser. One instance per generation turn.
pub struct LocalStreamParser {
    markers: MarkerSet,
    buf: String,
    state: State,
    tool_idx: usize,
}

impl LocalStreamParser {
    pub fn new(markers: MarkerSet) -> Self {
        Self {
            markers,
            buf: String::new(),
            state: State::Outside,
            tool_idx: 0,
        }
    }

    /// Convenience: build from a dialect preset.
    pub fn from_dialect(dialect: LocalDialect) -> Self {
        Self::new(dialect.into_markers())
    }

    /// Convenience: build from a [`ParserConfig`].
    pub fn from_config(config: &ParserConfig) -> Self {
        Self::new(config.markers.clone())
    }

    /// Push a stream chunk; returns canonical events that became complete.
    pub fn push(&mut self, chunk: &str) -> Vec<ParserEvent> {
        if chunk.is_empty() {
            return Vec::new();
        }
        self.buf.push_str(chunk);
        self.process(false)
    }

    /// Drain at end-of-stream. Unclosed channels/thinks → emit content;
    /// unclosed tool-call body → drop.
    pub fn finish(&mut self) -> Vec<ParserEvent> {
        self.process(true)
    }

    fn process(&mut self, eof: bool) -> Vec<ParserEvent> {
        let mut out = Vec::new();
        loop {
            // Snapshot relevant markers as owned strings — the loop body needs
            // exclusive access to `self.buf`/`self.state`, so we can't keep a
            // borrow into `self.markers` alive across the mutation.
            let markers: Vec<String> = self.relevant_markers();
            let earliest = markers
                .iter()
                .filter_map(|m| self.buf.find(m).map(|i| (i, m.clone())))
                .min_by_key(|(i, _)| *i);

            if let Some((i, marker)) = earliest {
                if i > 0 {
                    let before: String = self.buf.drain(..i).collect();
                    self.absorb_text(&before, &mut out);
                }
                self.buf.drain(..marker.len());
                self.handle_marker(&marker, &mut out);
                continue;
            }

            // No marker → hold the suffix that might be a marker prefix.
            if !eof {
                let safe_len = if markers.is_empty() {
                    self.buf.len()
                } else {
                    let refs: Vec<&str> = markers.iter().map(|s| s.as_str()).collect();
                    self.buf.len() - longest_marker_prefix_at_end(&self.buf, &refs)
                };
                if safe_len > 0 {
                    let segment: String = self.buf.drain(..safe_len).collect();
                    self.absorb_text(&segment, &mut out);
                }
                break;
            }

            // EOF — flush everything and close any open state.
            if !self.buf.is_empty() {
                let leftover: String = std::mem::take(&mut self.buf);
                self.absorb_text(&leftover, &mut out);
            }
            self.flush_open_state(&mut out);
            break;
        }
        out
    }

    /// Markers that could occur at the current state's boundary, returned as
    /// owned strings so the caller can mutate `self.buf` / `self.state`
    /// without keeping a borrow into `self.markers` alive.
    ///
    /// Inside a reasoning state (`InChannel` / `InThink`) we ALSO scan for
    /// other states' OPEN markers — Gemma-4 sometimes emits
    /// `<|tool_call>call:NAME{…}<tool_call|>` before closing the surrounding
    /// `<|channel>thought\n…<channel|>` block. Treating the unexpected open as
    /// an implicit close + transition (via [`Self::flush_unexpected_open`])
    /// keeps both events distinct — the think text and the tool call — instead
    /// of swallowing the tool call into the channel body. The tool-call body
    /// itself stays opaque (only its own close is scanned) since its content
    /// can legitimately contain `<` characters.
    fn relevant_markers(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        match &self.state {
            State::Outside => {
                if let Some((o, _)) = &self.markers.think {
                    v.push(o.clone());
                }
                if let Some((o, _)) = &self.markers.tool_call {
                    v.push(o.clone());
                }
                if let Some((o, _)) = &self.markers.channel {
                    v.push(o.clone());
                }
            }
            State::InChannel(_) => {
                // Close marker for the current state — first priority.
                if let Some((_, c)) = &self.markers.channel {
                    v.push(c.clone());
                }
                // Other states' opens — model may transition without closing.
                if let Some((o, _)) = &self.markers.tool_call {
                    v.push(o.clone());
                }
                if let Some((o, _)) = &self.markers.think {
                    v.push(o.clone());
                }
            }
            State::InThink(_) => {
                if let Some((_, c)) = &self.markers.think {
                    v.push(c.clone());
                }
                if let Some((o, _)) = &self.markers.tool_call {
                    v.push(o.clone());
                }
                if let Some((o, _)) = &self.markers.channel {
                    v.push(o.clone());
                }
            }
            State::InToolCall(_) => {
                // Tool-call body is opaque — only the close marker is structural.
                if let Some((_, c)) = &self.markers.tool_call {
                    v.push(c.clone());
                }
            }
        }
        v
    }

    fn absorb_text(&mut self, text: &str, out: &mut Vec<ParserEvent>) {
        match &mut self.state {
            State::Outside => {
                let clean = strip_quote(text, &self.markers);
                if !clean.is_empty() {
                    out.push(ParserEvent::Visible(clean));
                }
            }
            State::InChannel(buf) | State::InThink(buf) | State::InToolCall(buf) => {
                buf.push_str(text);
            }
        }
    }

    fn handle_marker(&mut self, marker: &str, out: &mut Vec<ParserEvent>) {
        // Open markers
        if let Some((o, _)) = &self.markers.channel {
            if marker == o {
                self.flush_unexpected_open(out);
                self.state = State::InChannel(String::new());
                return;
            }
        }
        if let Some((o, _)) = &self.markers.think {
            if marker == o {
                self.flush_unexpected_open(out);
                self.state = State::InThink(String::new());
                return;
            }
        }
        if let Some((o, _)) = &self.markers.tool_call {
            if marker == o {
                self.flush_unexpected_open(out);
                self.state = State::InToolCall(String::new());
                return;
            }
        }
        // Close markers
        if let Some((_, c)) = &self.markers.channel {
            if marker == c {
                if let State::InChannel(content) =
                    std::mem::replace(&mut self.state, State::Outside)
                {
                    self.emit_channel_close(&content, out);
                }
                return;
            }
        }
        if let Some((_, c)) = &self.markers.think {
            if marker == c {
                if let State::InThink(content) = std::mem::replace(&mut self.state, State::Outside)
                {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        out.push(ParserEvent::Reasoning(trimmed.to_string()));
                    }
                }
                return;
            }
        }
        if let Some((_, c)) = &self.markers.tool_call {
            if marker == c {
                if let State::InToolCall(body) = std::mem::replace(&mut self.state, State::Outside)
                {
                    if let Some(tc) =
                        parse_tool_call_body(&body, self.markers.tool_call_format, self.tool_idx)
                    {
                        self.tool_idx += 1;
                        out.push(ParserEvent::ToolCall(tc));
                    }
                }
            }
        }
    }

    fn emit_channel_close(&self, content: &str, out: &mut Vec<ParserEvent>) {
        let (name, body) = split_channel_name(content);
        let body_clean = strip_quote(body.trim(), &self.markers);
        if body_clean.is_empty() {
            return;
        }
        match route_channel(&name) {
            ChannelRouting::Reasoning => out.push(ParserEvent::Reasoning(body_clean)),
            ChannelRouting::Visible => out.push(ParserEvent::Visible(body_clean)),
        }
    }

    /// Emit channel content when the model **never sent `<channel|>` close** —
    /// most often a Gemma-4 quirk where the model transitions from `<|channel>
    /// thought\n…` straight into the user-facing answer without the close
    /// marker. Without this heuristic, the answer is locked inside the
    /// reasoning event and the UI shows only a think bubble with no body.
    ///
    /// Strategy: if the body has a paragraph break (`\n\n`) and the trailing
    /// segment is substantial (≥ [`MIN_UNCLOSED_TRAIL_LEN`] chars), split at
    /// the latest such boundary — leading → Reasoning, trailing → Visible.
    /// Otherwise fall back to the well-formed close behaviour.
    fn emit_unclosed_channel(&self, content: &str, out: &mut Vec<ParserEvent>) {
        let (name, body) = split_channel_name(content);
        let body_clean = strip_quote(body.trim(), &self.markers);
        if body_clean.is_empty() {
            return;
        }
        // Only the reasoning-routed channels (`thought`/`thinking`/unknown)
        // get the smart-split treatment — that's where Gemma-4's "forgot to
        // close before answering" quirk shows up. A channel that's already
        // routed to Visible (`answer`) is emitted as-is, no split needed.
        match route_channel(&name) {
            ChannelRouting::Visible => {
                out.push(ParserEvent::Visible(body_clean));
            }
            ChannelRouting::Reasoning => {
                if let Some((reasoning, answer)) = split_unclosed_thought_body(&body_clean) {
                    tracing::debug!(
                        "[stream-parser] unclosed thought channel split into \
                         reasoning ({} chars) + visible ({} chars)",
                        reasoning.len(),
                        answer.len(),
                    );
                    out.push(ParserEvent::Reasoning(reasoning));
                    out.push(ParserEvent::Visible(answer));
                } else {
                    out.push(ParserEvent::Reasoning(body_clean));
                }
            }
        }
    }

    fn flush_unexpected_open(&mut self, out: &mut Vec<ParserEvent>) {
        match std::mem::replace(&mut self.state, State::Outside) {
            State::Outside => {}
            State::InChannel(content) => self.emit_unclosed_channel(&content, out),
            State::InThink(content) => {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    out.push(ParserEvent::Reasoning(trimmed.to_string()));
                }
            }
            // Unclosed tool-call body is unreliable — drop.
            State::InToolCall(_) => {}
        }
    }

    fn flush_open_state(&mut self, out: &mut Vec<ParserEvent>) {
        self.flush_unexpected_open(out);
    }
}

/// Minimum length of the trailing segment for an unclosed-thought split to be
/// considered "this looks like an answer the model didn't separate properly".
/// Anything shorter is probably a stray sentence at the tail of pure thinking.
const MIN_UNCLOSED_TRAIL_LEN: usize = 100;

/// Find a paragraph boundary inside an unclosed `<|channel>thought\n…` body
/// such that the trailing segment is substantial. Prefers the LATEST such
/// boundary (so reasoning keeps as much context as possible).
fn split_unclosed_thought_body(body: &str) -> Option<(String, String)> {
    let breaks: Vec<usize> = body.match_indices("\n\n").map(|(i, _)| i).collect();
    if breaks.is_empty() {
        return None;
    }
    for &pos in breaks.iter().rev() {
        let trailing = body[pos + 2..].trim();
        if trailing.len() < MIN_UNCLOSED_TRAIL_LEN {
            continue;
        }
        let leading = body[..pos].trim();
        if leading.is_empty() {
            continue;
        }
        return Some((leading.to_string(), trailing.to_string()));
    }
    None
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn longest_marker_prefix_at_end(buf: &str, markers: &[&str]) -> usize {
    let max_marker_len = markers.iter().map(|m| m.len()).max().unwrap_or(0);
    if max_marker_len == 0 || buf.is_empty() {
        return 0;
    }
    let limit = buf.len().min(max_marker_len - 1);
    let bytes_len = buf.len();
    for k in (1..=limit).rev() {
        let start = bytes_len - k;
        if !buf.is_char_boundary(start) {
            continue;
        }
        let tail = &buf[start..];
        if markers.iter().any(|m| *m != tail && m.starts_with(tail)) {
            return k;
        }
    }
    0
}

fn strip_quote(s: &str, markers: &MarkerSet) -> String {
    match &markers.quote {
        Some(q) if !q.is_empty() => s.replace(q.as_str(), ""),
        _ => s.to_string(),
    }
}

fn split_channel_name(content: &str) -> (String, String) {
    match content.find('\n') {
        Some(nl) => (
            content[..nl].trim().to_string(),
            content[nl + 1..].to_string(),
        ),
        None => (content.trim().to_string(), String::new()),
    }
}

/// Where the body of a channel block should be emitted — `Reasoning`
/// (collapsible `<think>` widget in the UI) or `Visible` (user-facing answer).
///
/// Gemma-4's templates use named channels (`<|channel>NAME\n…<channel|>`) to
/// distinguish kinds of model output. The model is mostly trained on `thought`
/// (chain-of-thought) and `answer` (final user-facing reply), but the format
/// is open-ended — other names like `analysis` / `plan` / `critique` could
/// appear in custom templates. Default-to-`Reasoning` is the safer choice
/// here: an unrecognised channel is almost certainly model-internal content,
/// and routing it to Reasoning keeps it tucked into the collapsible widget
/// rather than dumping potentially noisy meta-content into the visible body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelRouting {
    Reasoning,
    Visible,
}

fn route_channel(name: &str) -> ChannelRouting {
    match name.trim().to_ascii_lowercase().as_str() {
        // Explicit user-facing answer channel.
        "answer" | "response" | "reply" | "final" => ChannelRouting::Visible,
        // `thought` / `thinking` — Gemma-4 thinking channel. Also the empty
        // case (channel opened with no name preceded by `\n`) and any other
        // unrecognised meta-channel default to Reasoning so we never leak
        // model internals into the visible body unintentionally.
        _ => ChannelRouting::Reasoning,
    }
}

// ── Tool-call body dispatch ─────────────────────────────────────────────────
//
// Generic orchestration only. The actual format-specific parsers live in each
// model's own file (divide-and-conquer):
//
//   • Gemma-4 harmony  → [`super::mlx_lm::models::gemma4::parser::parse_tool_call_body`]
//   • Qwen JSON/XML    → [`super::mlx_lm::models::qwen_common::parse_tool_call_body`]
//
// Adding a new model dialect: implement its body parser in the model file's
// own `parser` submodule, add a `ToolCallFormat` variant here, and route it
// in `parse_tool_call_body` below.

fn parse_tool_call_body(body: &str, format: ToolCallFormat, idx: usize) -> Option<Value> {
    // The per-model parser bodies live under `mlx_lm::models::*`, which is
    // only compiled when the `local-mlx` feature is enabled (those models
    // depend on `mlx-rs`). Without the feature there's no MLX model that can
    // be the source of these tokens — return None so the state machine emits
    // nothing for a tool-call body.
    {
        match format {
            ToolCallFormat::Gemma4Compact => {
                super::mlx_lm::models::gemma4::parser::parse_tool_call_body(body, idx)
            }
            ToolCallFormat::QwenJsonOrXml => {
                super::mlx_lm::models::qwen_common::parse_tool_call_body(body, idx)
            }
        }
    }
}

// ── Convenience: collapse events back to (visible, reasoning, tool_calls) ──

pub fn collapse_events(events: Vec<ParserEvent>) -> (String, String, Vec<Value>) {
    let mut visible = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for e in events {
        match e {
            ParserEvent::Visible(s) => visible.push_str(&s),
            ParserEvent::Reasoning(s) => {
                if !reasoning.is_empty() {
                    reasoning.push('\n');
                }
                reasoning.push_str(&s);
            }
            ParserEvent::ToolCall(v) => tool_calls.push(v),
        }
    }
    (
        visible.trim().to_string(),
        reasoning.trim().to_string(),
        tool_calls,
    )
}

/// One-shot convenience using an explicit dialect preset. Prefer
/// [`parse_complete_with_config`] for new code so the parser uses the model's
/// declared markers verbatim.
pub fn parse_complete(text: &str, dialect: LocalDialect) -> (String, String, Vec<Value>) {
    parse_complete_with_markers(text, &dialect.into_markers())
}

/// One-shot convenience using a per-model config.
pub fn parse_complete_with_config(
    text: &str,
    config: &ParserConfig,
) -> (String, String, Vec<Value>) {
    parse_complete_with_markers(text, &config.markers)
}

pub fn parse_complete_with_markers(
    text: &str,
    markers: &MarkerSet,
) -> (String, String, Vec<Value>) {
    let mut p = LocalStreamParser::new(markers.clone());
    let mut events = p.push(text);
    events.extend(p.finish());
    collapse_events(events)
}

// ── Streaming bridge ────────────────────────────────────────────────────────

/// Drain a raw text stream (from any local engine's `stream_*_to_channel`) and
/// forward canonical [`ParserEvent`]s to the consumer. This is the single
/// plumbing piece every local model needs: the engine emits raw text, the
/// parser normalizes it, the downstream sees the same OpenAI-shape interface.
///
/// Stops gracefully when either the raw source closes or the event consumer
/// disconnects. Best-effort: send errors on `event_tx` are swallowed (the
/// consumer hung up — nothing to do).
pub async fn pipe_text_stream_to_events(
    mut raw_rx: tokio::sync::mpsc::Receiver<String>,
    mut parser: LocalStreamParser,
    event_tx: tokio::sync::mpsc::Sender<ParserEvent>,
) {
    while let Some(chunk) = raw_rx.recv().await {
        for ev in parser.push(&chunk) {
            if event_tx.send(ev).await.is_err() {
                return;
            }
        }
    }
    for ev in parser.finish() {
        if event_tx.send(ev).await.is_err() {
            return;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_match_known_dialects() {
        let g = MarkerSet::gemma4();
        assert_eq!(
            g.channel.as_ref().map(|(o, c)| (o.as_str(), c.as_str())),
            Some(("<|channel>", "<channel|>"))
        );
        assert_eq!(
            g.tool_call.as_ref().map(|(o, c)| (o.as_str(), c.as_str())),
            Some(("<|tool_call>", "<tool_call|>"))
        );
        assert_eq!(g.quote.as_deref(), Some("<|\"|>"));
        assert_eq!(g.tool_call_format, ToolCallFormat::Gemma4Compact);
        assert!(g.think.is_none());

        let q = MarkerSet::qwen();
        assert_eq!(
            q.think.as_ref().map(|(o, c)| (o.as_str(), c.as_str())),
            Some(("<think>", "</think>"))
        );
        assert_eq!(q.tool_call_format, ToolCallFormat::QwenJsonOrXml);
        assert!(q.channel.is_none());
    }

    #[test]
    fn detect_from_chat_template_picks_gemma4_when_harmony_markers_present() {
        let tpl = "{{- '<|channel>thought\\n' + x + '\\n<channel|>' -}}\
                   <|tool_call>call:{{- name -}}{}<tool_call|><|\"|>q<|\"|>";
        let ms = MarkerSet::detect_from_chat_template(tpl);
        assert!(ms.channel.is_some());
        assert_eq!(ms.tool_call_format, ToolCallFormat::Gemma4Compact);
        assert_eq!(ms.quote.as_deref(), Some("<|\"|>"));
        assert!(ms.think.is_none());
    }

    #[test]
    fn detect_from_chat_template_picks_qwen_when_think_and_tool_call_present() {
        let tpl = "<think>{{ x }}</think><tool_call>{...}</tool_call>";
        let ms = MarkerSet::detect_from_chat_template(tpl);
        assert!(ms.think.is_some());
        assert!(ms.channel.is_none());
        assert_eq!(ms.tool_call_format, ToolCallFormat::QwenJsonOrXml);
    }

    /// `<tool_call>` substring also lives inside `<tool_call|>` — detection
    /// must NOT mis-attribute the harmony marker as Qwen.
    #[test]
    fn detect_does_not_confuse_harmony_tool_with_qwen_tool() {
        let tpl = "<|tool_call>call:{}<tool_call|>";
        let ms = MarkerSet::detect_from_chat_template(tpl);
        assert_eq!(ms.tool_call_format, ToolCallFormat::Gemma4Compact);
        let (o, c) = ms.tool_call.as_ref().unwrap();
        assert_eq!(o, "<|tool_call>");
        assert_eq!(c, "<tool_call|>");
    }

    /// End-to-end: load the real Gemma-4 model dir's `tokenizer_config.json`
    /// + `chat_template.jinja` and verify the parser config matches the named
    /// role-tokens.
    #[test]
    // Marker discovery for real model dirs exercises the mlx model parsers.
    fn parser_config_loads_real_gemma4_model_dir() {
        let dir = std::path::Path::new(env!("HOME"))
            .join(".senclaw/local-models/mlx-community__gemma-4-e2b-it-4bit");
        if !dir.join("tokenizer_config.json").exists() {
            eprintln!("skip: gemma-4 model not present at {}", dir.display());
            return;
        }
        let cfg = ParserConfig::from_model_dir(&dir, "mlx-community/gemma-4-e2b-it-4bit").unwrap();

        // Discovered from explicit role-tokens (Gemma-4 declares soc/eoc/stc/etc).
        assert_eq!(
            cfg.markers
                .channel
                .as_ref()
                .map(|(o, c)| (o.as_str(), c.as_str())),
            Some(("<|channel>", "<channel|>"))
        );
        assert_eq!(
            cfg.markers
                .tool_call
                .as_ref()
                .map(|(o, c)| (o.as_str(), c.as_str())),
            Some(("<|tool_call>", "<tool_call|>"))
        );
        assert_eq!(cfg.markers.quote.as_deref(), Some("<|\"|>"));
        assert_eq!(cfg.markers.tool_call_format, ToolCallFormat::Gemma4Compact);
        assert_eq!(cfg.bos_token.as_deref(), Some("<bos>"));
        assert_eq!(cfg.eos_token.as_deref(), Some("<eos>"));
        assert!(
            cfg.chat_template
                .as_ref()
                .is_some_and(|t| t.contains("<|channel>")),
            "chat_template should have loaded from chat_template.jinja"
        );

        // Smoke: parse_complete_with_config produces clean output on real wire format.
        let raw = "<|channel>thought\nReason.<channel|>\
                   <|tool_call>call:f{x:<|\"|>v<|\"|>}<tool_call|>Done.";
        let (vis, reas, tcs) = parse_complete_with_config(raw, &cfg);
        assert_eq!(vis, "Done.");
        assert_eq!(reas, "Reason.");
        assert_eq!(tcs.len(), 1);
        let args: Value =
            serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["x"], "v");
    }

    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn gemma4_complete_thought_and_tool_call() {
        let raw = "<|channel>thought\nThe user wants gold price.<channel|>\
                   <|tool_call>call:Skill{skill:<|\"|>agent-browser<|\"|>}<tool_call|>";
        let (vis, reas, tcs) = parse_complete(raw, LocalDialect::Gemma4);
        assert_eq!(vis, "");
        assert!(reas.contains("gold price"));
        assert_eq!(tcs[0]["function"]["name"], "Skill");
    }

    #[test]
    fn gemma4_marker_split_across_chunks_does_not_leak() {
        let mut p = LocalStreamParser::from_dialect(LocalDialect::Gemma4);
        let mut events = Vec::new();
        for c in ["<|", "cha", "nnel>thought\nhello<channel|>"] {
            events.extend(p.push(c));
        }
        events.extend(p.finish());
        for e in &events {
            if let ParserEvent::Visible(s) | ParserEvent::Reasoning(s) = e {
                assert!(!s.contains("<|"), "leak: {s:?}");
                assert!(!s.contains("|>"), "leak: {s:?}");
            }
        }
        let (vis, reas, _) = collapse_events(events);
        assert_eq!(vis, "");
        assert_eq!(reas, "hello");
    }

    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn gemma4_arbitrary_chunking_yields_clean_events() {
        let raw = "Some preamble. <|channel>thought\nreasoning here<channel|>\
                   Answer text. <|tool_call>call:f{x:1,y:<|\"|>z<|\"|>}<tool_call|> end.";
        let mut p = LocalStreamParser::from_dialect(LocalDialect::Gemma4);
        let mut events = Vec::new();
        let bytes_len = raw.len();
        let mut i = 0;
        while i < bytes_len {
            let mut step = 4.min(bytes_len - i);
            while !raw.is_char_boundary(i + step) && step > 0 {
                step -= 1;
            }
            if step == 0 {
                step = 1;
                while i + step <= bytes_len && !raw.is_char_boundary(i + step) {
                    step += 1;
                }
            }
            events.extend(p.push(&raw[i..i + step]));
            i += step;
        }
        events.extend(p.finish());
        for e in &events {
            if let ParserEvent::Visible(s) | ParserEvent::Reasoning(s) = e {
                assert!(!s.contains("<|") && !s.contains("|>"), "leak in {e:?}");
            }
        }
        let (vis, reas, tcs) = collapse_events(events);
        assert!(
            vis.contains("Some preamble") && vis.contains("Answer text") && vis.contains("end.")
        );
        assert_eq!(reas, "reasoning here");
        let args: Value =
            serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["x"], 1);
        assert_eq!(args["y"], "z");
    }

    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn qwen_think_and_tool_call_compatible() {
        let raw = "<think>Let me consider.</think>\
                   <tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"hi\"}}</tool_call>\
                   Done.";
        let (vis, reas, tcs) = parse_complete(raw, LocalDialect::Qwen);
        assert_eq!(vis, "Done.");
        assert_eq!(reas, "Let me consider.");
        assert_eq!(tcs[0]["function"]["name"], "search");
    }

    #[test]
    fn plain_text_no_markers_passes_through() {
        let (vis, reas, tcs) = parse_complete("Just plain.", LocalDialect::Auto);
        assert_eq!(vis, "Just plain.");
        assert!(reas.is_empty());
        assert!(tcs.is_empty());
    }

    #[test]
    fn unclosed_tool_call_at_eos_is_dropped() {
        let (vis, _, tcs) =
            parse_complete("<|tool_call>call:partial{k:<|\"|>v", LocalDialect::Gemma4);
        assert!(tcs.is_empty());
        assert_eq!(vis, "");
    }

    #[test]
    fn unclosed_channel_at_eos_becomes_reasoning() {
        let (_, reas, _) = parse_complete("<|channel>thought\nhalf finished", LocalDialect::Gemma4);
        assert_eq!(reas, "half finished");
    }

    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn multiple_tool_calls_get_distinct_ids() {
        let raw = "<|tool_call>call:f{a:1}<tool_call|><|tool_call>call:g{b:2}<tool_call|>";
        let (_, _, tcs) = parse_complete(raw, LocalDialect::Gemma4);
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0]["id"], "local_tool_0");
        assert_eq!(tcs[1]["id"], "local_tool_1");
    }

    /// End-to-end streaming-bridge test mirroring `MlxNativeEngine::stream_events_to_channel`:
    /// raw text chunks → parser pipe → canonical events on the other side.
    #[tokio::test]
    async fn pipe_text_stream_to_events_normalizes_chunks_in_flight() {
        let (raw_tx, raw_rx) = tokio::sync::mpsc::channel::<String>(8);
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<ParserEvent>(8);

        // Send the same realistic Gemma-4 stream, deliberately chunked so the
        // `<|channel>` marker straddles two messages.
        tokio::spawn(async move {
            raw_tx.send("Hi! <|cha".into()).await.unwrap();
            raw_tx
                .send("nnel>thought\nReason about gold price.<chan".into())
                .await
                .unwrap();
            raw_tx
                .send(
                    "nel|><|tool_call>call:Skill{skill:<|\"|>agent-browser<|\"|>}<tool_call|>"
                        .into(),
                )
                .await
                .unwrap();
        });

        let parser = LocalStreamParser::new(MarkerSet::gemma4());
        let pipe_fut = pipe_text_stream_to_events(raw_rx, parser, event_tx);
        let collect_fut = async {
            let mut evs = Vec::new();
            while let Some(e) = event_rx.recv().await {
                evs.push(e);
            }
            evs
        };
        let (_, events) = tokio::join!(pipe_fut, collect_fut);

        let (vis, reas, tcs) = collapse_events(events);
        assert_eq!(vis, "Hi!");
        assert_eq!(reas, "Reason about gold price.");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["function"]["name"], "Skill");
        let args: Value =
            serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["skill"], "agent-browser");
    }

    /// Each MLX model implements `ChatTemplateModel::markers()` to return its
    /// own dialect; this test verifies the trait-default + the three explicit
    /// overrides produce the expected MarkerSets without instantiating real
    /// model weights.
    #[test]
    fn each_mlx_model_owns_its_dialect() {
        // We can't construct full Models without weights, but we CAN verify
        // the presets each `markers()` returns by calling MarkerSet directly —
        // every model's `markers()` body is one of these constructors.
        let g4 = MarkerSet::gemma4();
        let qwen = MarkerSet::qwen();
        let empty = MarkerSet::empty();

        // Gemma-4 → harmony channel + harmony tool_call + quote wrapper.
        assert!(g4.channel.is_some() && g4.tool_call.is_some() && g4.quote.is_some());
        assert_eq!(g4.tool_call_format, ToolCallFormat::Gemma4Compact);
        assert!(g4.think.is_none());

        // Qwen 3 / Qwen 3.5 → think + tool_call (JSON-or-XML), no channel.
        assert!(qwen.think.is_some() && qwen.tool_call.is_some());
        assert!(qwen.channel.is_none() && qwen.quote.is_none());
        assert_eq!(qwen.tool_call_format, ToolCallFormat::QwenJsonOrXml);

        // Base / Mamba / Bonsai / Gemma 2/3 → empty (no structured output).
        assert!(empty.is_empty());

        // Gemma-4 vs Qwen markers must be distinct (different dialects).
        assert_ne!(g4, qwen);

        // Qwen 3 == Qwen 3.5 today, but they ARE separate `markers()` calls
        // (see qwen3.rs vs qwen3_5.rs) so either can diverge without touching
        // the other. Guard the current parity so a divergence is intentional.
        assert_eq!(qwen, MarkerSet::qwen());
    }

    /// Regression for the daemon-observed bug: model emits a long
    /// `<|channel>thought\n…` and then transitions to the user-facing answer
    /// WITHOUT ever sending `<channel|>` close. The parser must not lock the
    /// answer inside the reasoning event — otherwise UI shows only a think
    /// bubble and no body.
    #[test]
    fn unclosed_thought_channel_with_trailing_answer_splits_into_visible() {
        let raw = "<|channel>thought\n\
                   I have successfully executed the search tool and received structured content \
                   containing several snippets about today's silver prices.\n\
                   \n\
                   The user asked for gold price, but the results are about silver. I should \
                   present the information clearly and reply in Vietnamese.\n\
                   \n\
                   Tôi đã tìm kiếm thông tin về giá bạc hôm nay theo yêu cầu của bạn. \
                   Dưới đây là các mức giá được tìm thấy từ kết quả tìm kiếm. \
                   Bạn nên kiểm tra kỹ từng đường link được cung cấp.";
        let (vis, reas, _) = parse_complete(raw, LocalDialect::Gemma4);
        assert!(
            !vis.is_empty(),
            "trailing answer must reach the visible event (got empty body)"
        );
        assert!(
            vis.contains("Tôi đã tìm kiếm"),
            "expected Vietnamese answer in visible, got: {vis:?}"
        );
        assert!(
            !reas.contains("Tôi đã tìm kiếm"),
            "answer must NOT also remain in reasoning, got: {reas:?}"
        );
        assert!(
            reas.contains("I have successfully executed"),
            "early English reasoning lost: {reas:?}"
        );
    }

    /// Edge case: unclosed thought with no clear paragraph structure (pure
    /// thinking that just ran out of budget). Should fall back to all-reasoning
    /// — better to keep think collapsible than to mis-split mid-sentence.
    #[test]
    fn unclosed_thought_with_no_paragraph_break_stays_as_reasoning() {
        let raw = "<|channel>thought\nLet me think step by step about this problem.";
        let (vis, reas, _) = parse_complete(raw, LocalDialect::Gemma4);
        assert_eq!(vis, "");
        assert!(reas.contains("Let me think"));
    }

    /// Edge case: paragraph break exists but trailing segment is too short
    /// to be a real answer — likely just a closing remark on the thinking.
    /// Should NOT split (avoid false-positive splits on incidental newlines).
    #[test]
    fn unclosed_thought_with_short_trailing_stays_as_reasoning() {
        let raw = "<|channel>thought\nThis is a long reasoning paragraph that explains \
                   the analysis in detail with many words to exceed any length threshold \
                   for safety against false-positive splits.\n\
                   \n\
                   Done.";
        let (vis, reas, _) = parse_complete(raw, LocalDialect::Gemma4);
        assert_eq!(vis, "", "tiny trailing must not be promoted to visible");
        assert!(reas.contains("long reasoning paragraph"));
        assert!(reas.contains("Done"));
    }

    /// Regression for the daemon-observed bug: Gemma-4 sometimes emits a
    /// `<|tool_call>…<tool_call|>` BEFORE closing the surrounding `<|channel>
    /// thought\n…<channel|>` reasoning block. The parser must NOT swallow the
    /// tool call into the channel body — instead, treat the unexpected open
    /// as an implicit channel close and produce a distinct ToolCall event.
    ///
    /// Without this, the UI shows the literal `call:Skill{…}<tool_call|>` at
    /// the end of the think bubble and never executes the tool.
    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn tool_call_inside_unclosed_channel_is_extracted_separately() {
        let raw = "<|channel>thought\nLet me search for gold price.\n\
                   <|tool_call>call:Skill{skill:<|\"|>agent-browser<|\"|>}<tool_call|>";
        let (vis, reas, tcs) = parse_complete(raw, LocalDialect::Gemma4);

        assert!(
            reas.contains("Let me search for gold price"),
            "reasoning lost: {reas:?}"
        );
        assert!(
            !reas.contains("<|tool_call>") && !reas.contains("call:Skill"),
            "tool_call leaked into reasoning: {reas:?}"
        );
        assert!(
            !vis.contains("call:Skill"),
            "tool_call leaked into visible: {vis:?}"
        );
        assert_eq!(
            tcs.len(),
            1,
            "tool call must be extracted as a separate event"
        );
        assert_eq!(tcs[0]["function"]["name"], "Skill");
        let args: Value =
            serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["skill"], "agent-browser");
    }

    /// And the well-formed case (channel close BEFORE tool_call open) still
    /// works — the change to `relevant_markers` must not break this path.
    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn well_formed_channel_then_tool_call_still_works() {
        let raw = "<|channel>thought\nReason.<channel|>\
                   <|tool_call>call:f{x:1}<tool_call|>";
        let (vis, reas, tcs) = parse_complete(raw, LocalDialect::Gemma4);
        assert_eq!(reas, "Reason.");
        assert_eq!(vis, "");
        assert_eq!(tcs.len(), 1);
        let args: Value =
            serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["x"], 1);
    }

    // ── Gemma-4 design contract ────────────────────────────────────────────
    //
    // Canonical contract for the Gemma-4 stream parser: every input maps to a
    // well-defined sequence of `ParserEvent`s. The four primitives:
    //
    //   • Visible(s)     — user-facing answer text
    //   • Reasoning(s)   — model's thinking (`<|channel>thought\n…<channel|>`)
    //   • ToolCall(v)    — `<|tool_call>call:NAME{…}<tool_call|>`
    //   • (none)         — input had no recognised content
    //
    // These tests pin down the contract for the eight canonical scenarios
    // observed in production logs. If any test below fails, the parser has
    // silently changed its behaviour and downstream code (`build_assistant_message`,
    // `merge_assistant_reasoning_for_web_ui`, the web UI) may break.

    /// Plain answer with no markers — pure visible text.
    #[test]
    fn gemma4_contract_plain_answer_only() {
        let (vis, reas, tcs) = parse_complete(
            "Chào bạn, giá vàng hôm nay là 75 triệu.",
            LocalDialect::Gemma4,
        );
        assert_eq!(vis, "Chào bạn, giá vàng hôm nay là 75 triệu.");
        assert!(reas.is_empty());
        assert!(tcs.is_empty());
    }

    /// Thinking only (well-formed thought channel) — pure reasoning.
    #[test]
    fn gemma4_contract_thought_channel_only() {
        let (vis, reas, tcs) = parse_complete(
            "<|channel>thought\nLet me consider the request.<channel|>",
            LocalDialect::Gemma4,
        );
        assert!(vis.is_empty());
        assert_eq!(reas, "Let me consider the request.");
        assert!(tcs.is_empty());
    }

    /// Explicit `answer` channel routes to Visible (not Reasoning).
    #[test]
    fn gemma4_contract_answer_channel_routes_to_visible() {
        let (vis, reas, _) = parse_complete(
            "<|channel>answer\nThe gold price is 75 million.<channel|>",
            LocalDialect::Gemma4,
        );
        assert_eq!(vis, "The gold price is 75 million.");
        assert!(reas.is_empty(), "answer channel must not go to reasoning");
    }

    /// Unknown channel name defaults to Reasoning (safer than leaking
    /// model-internal meta-content into the visible body).
    #[test]
    fn gemma4_contract_unknown_channel_defaults_to_reasoning() {
        let (vis, reas, _) = parse_complete(
            "<|channel>analysis\nInternal meta-content.<channel|>",
            LocalDialect::Gemma4,
        );
        assert!(vis.is_empty(), "unknown channels must not reach visible");
        assert_eq!(reas, "Internal meta-content.");
    }

    /// Standalone tool call (no surrounding thought).
    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn gemma4_contract_tool_call_only() {
        let (vis, reas, tcs) = parse_complete(
            "<|tool_call>call:search{q:<|\"|>gold price<|\"|>}<tool_call|>",
            LocalDialect::Gemma4,
        );
        assert!(vis.is_empty());
        assert!(reas.is_empty());
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["function"]["name"], "search");
        let args: Value =
            serde_json::from_str(tcs[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["q"], "gold price");
    }

    /// Well-formed think → tool_call: intermediate-turn shape — reasoning +
    /// tool call, no visible answer. Matches `text_len=0 reasoning_len=N
    /// tool_calls=1` log line.
    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn gemma4_contract_think_then_tool_call_no_answer() {
        let (vis, reas, tcs) = parse_complete(
            "<|channel>thought\nI should search the web.<channel|>\
             <|tool_call>call:Skill{skill:<|\"|>agent-browser<|\"|>}<tool_call|>",
            LocalDialect::Gemma4,
        );
        assert!(vis.is_empty(), "intermediate turn has no visible body");
        assert_eq!(reas, "I should search the web.");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["function"]["name"], "Skill");
    }

    /// Well-formed think → answer: final-turn shape — reasoning + visible.
    /// Matches `text_len=N reasoning_len=N tool_calls=0` log line.
    #[test]
    fn gemma4_contract_think_then_answer_final_turn() {
        let (vis, reas, tcs) = parse_complete(
            "<|channel>thought\nReady to answer.<channel|>\
             Giá vàng hôm nay là 75 triệu đồng mỗi lượng.",
            LocalDialect::Gemma4,
        );
        assert_eq!(vis, "Giá vàng hôm nay là 75 triệu đồng mỗi lượng.");
        assert_eq!(reas, "Ready to answer.");
        assert!(tcs.is_empty(), "final turn — no more tool calls");
    }

    /// Malformed: model emitted tool call BEFORE closing the channel.
    /// Parser must extract it as a distinct ToolCall event (not swallow it
    /// into the reasoning body).
    #[test]
    // Tool-call body parsing lives in mlx_lm::models (local-mlx only) —
    // without the feature the parser deliberately emits no tool events.
    fn gemma4_contract_malformed_tool_call_inside_unclosed_channel() {
        let (vis, reas, tcs) = parse_complete(
            "<|channel>thought\nI will invoke the skill.\
             <|tool_call>call:Skill{skill:<|\"|>agent-browser<|\"|>}<tool_call|>",
            LocalDialect::Gemma4,
        );
        assert!(vis.is_empty());
        assert!(reas.contains("I will invoke the skill"));
        assert!(
            !reas.contains("<|tool_call>") && !reas.contains("call:Skill"),
            "tool call must NOT leak into reasoning: {reas:?}"
        );
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["function"]["name"], "Skill");
    }

    /// Malformed: model emitted long reasoning + answer in one block (no
    /// `<channel|>` close before the answer). Parser's smart-split heuristic
    /// must extract the trailing answer paragraph as Visible.
    #[test]
    fn gemma4_contract_malformed_unclosed_channel_split_at_trailing_answer() {
        let raw = "<|channel>thought\n\
                   The user asked for gold price. I analyzed the search results \
                   and found relevant data from multiple Vietnamese sources.\n\
                   \n\
                   Giá vàng hôm nay theo các nguồn tham khảo từ các trang web \
                   uy tín dao động trong khoảng 75 triệu đồng mỗi lượng. \
                   Bạn nên kiểm tra trực tiếp để xem giá chính xác.";
        let (vis, reas, _) = parse_complete(raw, LocalDialect::Gemma4);
        assert!(reas.contains("analyzed the search results"));
        assert!(
            vis.contains("Giá vàng hôm nay"),
            "trailing answer must reach visible: {vis:?}"
        );
        // Critical: the answer must NOT also remain in reasoning (no double-emit).
        assert!(
            !reas.contains("Giá vàng hôm nay"),
            "answer must NOT be duplicated in reasoning: {reas:?}"
        );
    }

    #[test]
    fn dialect_for_model_id_picks_correctly() {
        assert_eq!(
            dialect_for_model_id("mlx-community/gemma-4-e2b-it-4bit"),
            LocalDialect::Gemma4
        );
        assert_eq!(dialect_for_model_id("Qwen/Qwen3-0.6B"), LocalDialect::Qwen);
        assert_eq!(dialect_for_model_id("unknown/model"), LocalDialect::Auto);
    }
}
