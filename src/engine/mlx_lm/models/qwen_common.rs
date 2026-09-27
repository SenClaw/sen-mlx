//! Shared output-stream parser primitives for the Qwen family (Qwen3,
//! Qwen3.5, and any future Qwen variant that emits the same envelope).
//!
//! Colocated with the model files so all Qwen-format knowledge lives in one
//! place. The generic state machine in [`crate::engine::stream_parser`]
//! dispatches to [`parse_tool_call_body`] via the
//! `ToolCallFormat::QwenJsonOrXml` discriminant.
//!
//! Wire format envelope: `<tool_call>BODY</tool_call>`.
//!
//! Body has two accepted shapes:
//!
//! 1. **HF JSON** (Qwen 3 prefers): `{"name": "...", "arguments": {...}}`.
//!    Also tolerated: a top-level `{"tool_calls": [...]}` wrapper.
//!
//! 2. **Hermes XML fallback** (Qwen 3.5 prefers):
//!    `<function=NAME><parameter=K1>v1</parameter>…</function>`.
//!    Parameter values are type-coerced (JSON / bool / int / float, fallback
//!    string).
//!
//! Both Qwen models share this module via `MarkerSet::qwen()`; if a future
//! Qwen variant needs to diverge, point its `ChatTemplateModel::markers()`
//! impl at a different [`crate::engine::stream_parser::MarkerSet`] with
//! its own `ToolCallFormat` variant.

use serde_json::{Map, Value};

/// Parse a Qwen `<tool_call>…</tool_call>` body into the OpenAI-shape
/// tool_call object (`{id, type:"function", function:{name, arguments:String}}`).
/// Tries the JSON form first, falls back to Hermes XML.
pub fn parse_tool_call_body(body: &str, idx: usize) -> Option<Value> {
    let body = body.trim();
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let parsed = parsed.or_else(|| parse_xml_tool_call_body(body))?;

    // Body may be an object, an array, or `{"tool_calls":[…]}`. Normalise.
    let items: Vec<Value> = if let Some(arr) = parsed
        .as_object()
        .and_then(|o| o.get("tool_calls"))
        .and_then(|v| v.as_array())
    {
        arr.clone()
    } else {
        match parsed {
            Value::Array(a) => a,
            one => vec![one],
        }
    };

    for item in items {
        let Some(name) = item.get("name").and_then(|x| x.as_str()) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let args = item
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Default::default()));
        let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
        return Some(serde_json::json!({
            "id": format!("local_tool_{idx}"),
            "type": "function",
            "function": { "name": name, "arguments": args_str }
        }));
    }
    None
}

/// Hermes-style `<function=NAME><parameter=K>v</parameter>…</function>` body
/// (Qwen 3.5's preferred wire format). Type-coerces parameter values.
fn parse_xml_tool_call_body(body: &str) -> Option<Value> {
    const FN_OPEN: &str = "<function=";
    const PARAM_OPEN: &str = "<parameter=";
    const PARAM_CLOSE: &str = "</parameter>";

    let fstart = body.find(FN_OPEN)? + FN_OPEN.len();
    let rest = &body[fstart..];
    let gt = rest.find('>')?;
    let name = rest[..gt].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut cursor = &rest[gt + 1..];

    let mut args = Map::new();
    while let Some(p_open) = cursor.find(PARAM_OPEN) {
        let after = &cursor[p_open + PARAM_OPEN.len()..];
        let Some(k_end) = after.find('>') else { break };
        let key = after[..k_end].trim().to_string();
        let body_start = &after[k_end + 1..];
        let Some(p_close) = body_start.find(PARAM_CLOSE) else {
            break;
        };
        let raw = body_start[..p_close].trim();
        let val: Value = if let Ok(v) = serde_json::from_str::<Value>(raw) {
            v
        } else if raw == "true" {
            Value::Bool(true)
        } else if raw == "false" {
            Value::Bool(false)
        } else if let Ok(i) = raw.parse::<i64>() {
            Value::from(i)
        } else if let Ok(f) = raw.parse::<f64>() {
            Value::from(f)
        } else {
            Value::String(raw.to_string())
        };
        args.insert(key, val);
        cursor = &body_start[p_close + PARAM_CLOSE.len()..];
    }

    Some(serde_json::json!({ "name": name, "arguments": Value::Object(args) }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_json_body() {
        let tc = parse_tool_call_body(
            "{\"name\": \"search\", \"arguments\": {\"q\": \"vàng\"}}",
            0,
        )
        .unwrap();
        assert_eq!(tc["function"]["name"], "search");
        let args: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["q"], "vàng");
    }

    #[test]
    fn parses_hermes_xml_body_with_type_coercion() {
        // Qwen 3.5's XML wire format — `days` should coerce to a number.
        let body = "<function=weather>\n\
                    <parameter=city>\nHanoi\n</parameter>\n\
                    <parameter=days>\n3\n</parameter>\n\
                    </function>";
        let tc = parse_tool_call_body(body, 0).unwrap();
        assert_eq!(tc["function"]["name"], "weather");
        let args = tc["function"]["arguments"].as_str().unwrap();
        assert!(args.contains("\"city\":\"Hanoi\""), "args: {args}");
        assert!(args.contains("\"days\":3"), "days→number: {args}");
    }

    #[test]
    fn unwraps_tool_calls_array() {
        let body = "{\"tool_calls\":[{\"name\":\"first\",\"arguments\":{}}]}";
        let tc = parse_tool_call_body(body, 0).unwrap();
        assert_eq!(tc["function"]["name"], "first");
    }

    #[test]
    fn returns_none_for_empty_or_nameless() {
        assert!(parse_tool_call_body("", 0).is_none());
        assert!(parse_tool_call_body("{}", 0).is_none());
        assert!(parse_tool_call_body("{\"name\":\"\"}", 0).is_none());
    }
}
