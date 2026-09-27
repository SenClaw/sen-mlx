//! Prompt preprocessing for native MLX inference (RAM / token control).

use serde_json::Value;

use super::mlx_lm_utils::tokenizer::{Conversation, Role};

pub use super::thinking_parse::{split_thinking_blocks, strip_thinking_blocks};

/// Drop the oldest non-system + non-final-user message (one assistant or
/// `tool` reply, or one user turn in the middle of the history). Used to
/// shrink an OpenAI-shaped prompt that overshoots the KV budget — strictly
/// preserves the system block and the most recent user turn.
///
/// Returns `true` when a message was removed, `false` when nothing could be
/// trimmed (only system + last user remain, or no system + only one user).
pub fn drop_oldest_openai_middle_message(messages: &mut Vec<Value>) -> bool {
    let last_user_idx = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| m.get("role").and_then(|v| v.as_str()) == Some("user"))
        .map(|(i, _)| i);
    let leading_system = messages
        .first()
        .is_some_and(|m| m.get("role").and_then(|v| v.as_str()) == Some("system"));
    let start = if leading_system { 1 } else { 0 };
    let end = last_user_idx.unwrap_or(messages.len());
    if end <= start {
        return false;
    }
    messages.remove(start);
    true
}

/// Last-resort token-level fit, used only when even the system block + the
/// final user turn (after all middle turns and tools have been dropped) still
/// overflow the KV budget.
///
/// A plain head-truncation (`tokens.drain(..overflow)`) would delete the system
/// prompt and the chat template's role / tool-schema framing — the model then
/// loses its grounding and spirals (tool-less repetition, ignored skills,
/// infinite fallback loops). Instead we keep the **head** (system prompt / tool
/// schemas) AND the **tail** (most recent turn + assistant generation prompt),
/// dropping tokens from the *middle*. Both the instructions and the current
/// question survive.
///
/// Returns the input unchanged when already within budget; otherwise a vector
/// of exactly `budget` tokens. ~40% of the budget is reserved for the tail so
/// the current question + generation prompt are never cut.
pub fn splice_head_tail_to_budget(tokens: Vec<u32>, budget: usize) -> Vec<u32> {
    if budget == 0 || tokens.len() <= budget {
        return tokens;
    }
    let tail_keep = (budget * 2 / 5).min(tokens.len());
    let head_keep = budget.saturating_sub(tail_keep);
    let tail_start = tokens.len() - tail_keep;
    let mut spliced = Vec::with_capacity(head_keep + tail_keep);
    spliced.extend_from_slice(&tokens[..head_keep]);
    spliced.extend_from_slice(&tokens[tail_start..]);
    spliced
}

/// User-assistant turns capped before templating; a leading `Role::System` is preserved with the first user turn when present.
pub const MLX_MAX_HISTORY_TURNS: usize = 4;

/// Keep the last `max_turns` user messages and all messages from the first kept user onward.
pub fn trim_conversation_history<T>(convs: &mut Vec<Conversation<Role, T>>, max_turns: usize) {
    if max_turns == 0 || convs.is_empty() {
        return;
    }

    let mut user_seen = 0usize;
    let mut start = 0usize;
    for (i, c) in convs.iter().enumerate().rev() {
        if c.role == Role::User {
            user_seen += 1;
            if user_seen == max_turns {
                start = i;
                break;
            }
        }
    }

    // Keep an optional leading system plus the first user chunk (Gemma-3 Jinja), or
    // the first merged user turn (no system), matching the old single-prefix rule.
    let trim_prefix = if convs.first().is_some_and(|c| c.role == Role::System) {
        convs
            .iter()
            .enumerate()
            .find(|(i, c)| *i > 0 && c.role == Role::User)
            .map(|(i, _)| i + 1)
            .unwrap_or(convs.len())
            .max(1)
    } else {
        1
    };

    if start > trim_prefix {
        let before = convs.len();
        convs.drain(trim_prefix..start);
        tracing::info!(
            "[local-mlx-native] trimming chat history: {before} → {} messages (keep last {max_turns} user turns, preserve {} leading message(s))",
            convs.len(),
            trim_prefix
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splice_noop_when_within_budget() {
        let toks: Vec<u32> = (0..10).collect();
        assert_eq!(splice_head_tail_to_budget(toks.clone(), 10), toks);
        assert_eq!(splice_head_tail_to_budget(toks.clone(), 100), toks);
    }

    #[test]
    fn splice_keeps_head_and_tail_exact_budget() {
        // 0..100, budget 50 → tail_keep = 50*2/5 = 20, head_keep = 30.
        let toks: Vec<u32> = (0..100).collect();
        let out = splice_head_tail_to_budget(toks, 50);
        assert_eq!(out.len(), 50, "spliced output is exactly budget");
        // Head: first 30 tokens preserved (system prompt / tool framing).
        assert_eq!(&out[..30], &(0..30).collect::<Vec<u32>>()[..]);
        // Tail: last 20 tokens preserved (recent turn + generation prompt).
        assert_eq!(&out[30..], &(80..100).collect::<Vec<u32>>()[..]);
    }

    #[test]
    fn splice_preserves_first_and_last_token() {
        let toks: Vec<u32> = (0..1000).collect();
        let out = splice_head_tail_to_budget(toks, 200);
        assert_eq!(out.len(), 200);
        assert_eq!(*out.first().unwrap(), 0, "system-prompt head survives");
        assert_eq!(*out.last().unwrap(), 999, "generation-tail survives");
    }

    #[test]
    fn splice_zero_budget_is_noop() {
        let toks: Vec<u32> = (0..10).collect();
        assert_eq!(splice_head_tail_to_budget(toks.clone(), 0), toks);
    }

    #[test]
    fn trim_keeps_last_n_user_turns() {
        let mut convs: Vec<Conversation<Role, String>> = vec![
            Conversation {
                role: Role::User,
                content: "1".into(),
            },
            Conversation {
                role: Role::Assistant,
                content: "a".into(),
            },
            Conversation {
                role: Role::User,
                content: "2".into(),
            },
            Conversation {
                role: Role::Assistant,
                content: "b".into(),
            },
            Conversation {
                role: Role::User,
                content: "3".into(),
            },
        ];
        trim_conversation_history(&mut convs, 2);
        assert_eq!(convs[0].content, "1", "first user turn preserved");
        assert!(convs.len() >= 2);
        assert_eq!(convs.last().unwrap().content, "3");
    }

    #[test]
    fn trim_preserves_system_and_first_user_when_capping() {
        let mut convs: Vec<Conversation<Role, String>> = vec![
            Conversation {
                role: Role::System,
                content: "sys".into(),
            },
            Conversation {
                role: Role::User,
                content: "u0".into(),
            },
            Conversation {
                role: Role::Assistant,
                content: "a0".into(),
            },
            Conversation {
                role: Role::User,
                content: "u1".into(),
            },
            Conversation {
                role: Role::Assistant,
                content: "a1".into(),
            },
            Conversation {
                role: Role::User,
                content: "u2".into(),
            },
        ];
        trim_conversation_history(&mut convs, 2);
        assert_eq!(convs.len(), 5);
        assert_eq!(convs[0].role, Role::System);
        assert_eq!(convs[0].content, "sys");
        assert_eq!(convs[1].content, "u0");
        assert_eq!(convs.last().unwrap().content, "u2");
    }

    #[test]
    fn drop_oldest_middle_preserves_system_and_last_user() {
        let mut msgs: Vec<Value> = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u0"}),
            serde_json::json!({"role": "assistant", "content": "a0"}),
            serde_json::json!({"role": "user", "content": "u1"}),
        ];
        assert!(drop_oldest_openai_middle_message(&mut msgs));
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs.last().unwrap()["content"], "u1");
    }

    #[test]
    fn drop_oldest_middle_refuses_when_only_system_and_last_user() {
        let mut msgs: Vec<Value> = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u0"}),
        ];
        assert!(!drop_oldest_openai_middle_message(&mut msgs));
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn drop_oldest_middle_refuses_when_single_user_no_system() {
        let mut msgs: Vec<Value> = vec![serde_json::json!({"role": "user", "content": "u0"})];
        assert!(!drop_oldest_openai_middle_message(&mut msgs));
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn drop_oldest_middle_walks_history_until_only_essentials_left() {
        let mut msgs: Vec<Value> = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "u0"}),
            serde_json::json!({"role": "assistant", "content": "a0"}),
            serde_json::json!({"role": "user", "content": "u1"}),
            serde_json::json!({"role": "assistant", "content": "a1"}),
            serde_json::json!({"role": "user", "content": "u2"}),
        ];
        while drop_oldest_openai_middle_message(&mut msgs) {}
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["content"], "u2");
    }
}
