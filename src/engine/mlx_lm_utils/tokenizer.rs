use std::{
    collections::HashMap,
    fs::read_to_string,
    ops::{Deref, DerefMut},
    path::Path,
    str::FromStr,
};

use minijinja::{context, Environment, Template};
use serde::Serialize;
use serde_json::Value;
use tokenizers::Encoding;

use super::error::Error;
use crate::engine::chat_template_openai;

/// Wrapper around [`tokenizers::Tokenizer`] and [`minijinja::Environment`]
/// providing more utilities.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    env: Environment<'static>,
}

impl FromStr for Tokenizer {
    type Err = tokenizers::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        tokenizers::Tokenizer::from_str(s).map(Self::from_tokenizer)
    }
}

impl Tokenizer {
    pub fn from_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        let mut env = Environment::new();
        chat_template_openai::configure_jinja_env(&mut env);
        Self {
            inner: tokenizer,
            env,
        }
    }

    pub fn from_file(file: impl AsRef<Path>) -> tokenizers::Result<Self> {
        tokenizers::Tokenizer::from_file(file).map(Self::from_tokenizer)
    }

    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> tokenizers::Result<Self> {
        tokenizers::Tokenizer::from_bytes(bytes).map(Self::from_tokenizer)
    }

    pub fn apply_chat_template<'a, I, R, T>(
        &'a mut self,
        model_template: String,
        args: ApplyChatTemplateArgs<'a, I, R, T>,
    ) -> Result<Vec<String>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + ToString + 'a,
    {
        apply_chat_template(&mut self.env, model_template, args)
    }

    pub fn apply_chat_template_and_encode<'a, I, R, T>(
        &mut self,
        model_template: String,
        args: ApplyChatTemplateArgs<'a, I, R, T>,
    ) -> Result<Vec<Encoding>, Error>
    where
        I: IntoIterator<Item = Chat<'a, R, T>>,
        R: Serialize + 'a,
        T: Serialize + ToString + 'a,
    {
        let Self { inner, env } = self;

        let rendered_chats = apply_chat_template(env, model_template, args)?;
        inner
            .encode_batch(rendered_chats, false)
            .map_err(Into::into)
    }

    /// OpenAI / HF-shaped `messages` + `tools` for Qwen3-style Jinja templates.
    pub fn apply_chat_template_json_and_encode(
        &mut self,
        model_template: String,
        model_id: &str,
        messages: &[Value],
        tools: &[Value],
        add_generation_prompt: Option<bool>,
        enable_thinking: Option<bool>,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<Vec<Encoding>, Error> {
        let rendered = chat_template_openai::apply_chat_template_openai_shape(
            &mut self.env,
            model_template,
            model_id,
            None,
            messages,
            tools,
            add_generation_prompt,
            enable_thinking,
            bos_token,
            eos_token,
        )
        .map_err(|e| match e {
            chat_template_openai::Error::RenderTemplate(err) => Error::RenderTemplate(err),
            chat_template_openai::Error::Encode(err) => Error::Encode(err),
        })?;
        self.inner
            .encode(rendered, false)
            .map_err(Into::into)
            .map(|e| vec![e])
    }
}

impl Deref for Tokenizer {
    type Target = tokenizers::Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Tokenizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
#[derive(PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    /// Used by Gemma / HF chat templates that branch on `messages[0].role == "system"`.
    System,
}

#[derive(Debug, Clone, Serialize)]
pub enum Content {
    String(String),
    Map(HashMap<String, String>),
}

#[derive(Debug, Clone, Serialize)]
pub struct Conversation<R, T> {
    pub role: R,
    pub content: T,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Chat<'a, R, T> {
    Borrowed(&'a [Conversation<R, T>]),
    Owned(Vec<Conversation<R, T>>),
}

impl<R, T> Deref for Chat<'_, R, T> {
    type Target = [Conversation<R, T>];

    fn deref(&self) -> &Self::Target {
        match self {
            Chat::Borrowed(conversations) => conversations,
            Chat::Owned(conversations) => conversations,
        }
    }
}

impl<R, T> From<Vec<Conversation<R, T>>> for Chat<'_, R, T> {
    fn from(value: Vec<Conversation<R, T>>) -> Self {
        Chat::Owned(value)
    }
}

impl<'a, R, T> From<&'a [Conversation<R, T>]> for Chat<'a, R, T> {
    fn from(value: &'a [Conversation<R, T>]) -> Self {
        Chat::Borrowed(value)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Document {
    pub title: String,
    pub text: String,
}

pub enum Padding {
    Longest,
    MaxLength,
}

pub enum Truncation {
    MaxLength(usize),
}

#[derive(Default)]
pub struct ApplyChatTemplateArgs<'a, I, R = Role, T = String>
where
    I: IntoIterator<Item = Chat<'a, R, T>>,
    R: Serialize + 'a,
    T: Serialize + ToString + 'a,
{
    pub conversations: I,
    pub documents: Option<&'a [Document]>,
    pub model_id: &'a str,
    pub chat_template_id: Option<&'a str>,
    pub add_generation_prompt: Option<bool>,
    pub continue_final_message: Option<bool>,
    /// Qwen3 templates: `false` pre-fills an empty thinking block to skip reasoning.
    pub enable_thinking: Option<bool>,
    /// Injected into Jinja as `bos_token`. Gemma-3 templates open with `{{ bos_token }}`.
    pub bos_token: Option<String>,
    /// Injected as `eos_token` when templates reference it (`None` → empty string passed).
    pub eos_token: Option<String>,
}

pub fn load_model_chat_template_from_str(content: &str) -> std::io::Result<Option<String>> {
    serde_json::from_str::<serde_json::Value>(content)
        .map(|value| {
            value
                .get("chat_template")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
        })
        .map_err(Into::into)
}

pub fn load_model_chat_template_from_file(
    file: impl AsRef<Path>,
) -> std::io::Result<Option<String>> {
    let content = read_to_string(file)?;
    load_model_chat_template_from_str(&content)
}

pub fn apply_chat_template<'a, I, R, T>(
    env: &mut Environment<'static>,
    model_template: String,
    args: ApplyChatTemplateArgs<'a, I, R, T>,
) -> Result<Vec<String>, Error>
where
    I: IntoIterator<Item = Chat<'a, R, T>>,
    R: Serialize + 'a,
    T: Serialize + ToString + 'a,
{
    let ApplyChatTemplateArgs {
        conversations,
        documents,
        model_id,
        chat_template_id,
        add_generation_prompt,
        continue_final_message,
        enable_thinking,
        bos_token,
        eos_token,
    } = args;

    let add_generation_prompt = add_generation_prompt.unwrap_or(false);
    let continue_final_message = continue_final_message.unwrap_or(false);

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

    render_jinja_tempalte(
        template,
        conversations,
        documents,
        Some(add_generation_prompt),
        Some(continue_final_message),
        enable_thinking,
        bos_token.as_deref(),
        eos_token.as_deref(),
    )
}

fn render_jinja_tempalte<'a, R, T>(
    template: Template,
    conversations: impl IntoIterator<Item = Chat<'a, R, T>>,
    documents: Option<&'a [Document]>,
    add_generation_prompt: Option<bool>,
    continue_final_message: Option<bool>,
    enable_thinking: Option<bool>,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<Vec<String>, Error>
where
    R: Serialize + 'a,
    T: Serialize + ToString + 'a,
{
    let add_generation_prompt = add_generation_prompt.unwrap_or(false);
    let continue_final_message = continue_final_message.unwrap_or(false);
    let bos_slot = bos_token.unwrap_or("");
    let eos_slot = eos_token.unwrap_or("");

    let mut rendered = Vec::new();
    for chat in conversations {
        let mut rendered_chat = match enable_thinking {
            Some(thinking) => template.render(context! {
                messages => chat,
                documents => documents,
                add_generation_prompt => add_generation_prompt,
                enable_thinking => thinking,
                bos_token => bos_slot,
                eos_token => eos_slot,
            })?,
            None => template.render(context! {
                messages => chat,
                documents => documents,
                add_generation_prompt => add_generation_prompt,
                bos_token => bos_slot,
                eos_token => eos_slot,
            })?,
        };

        if continue_final_message {
            let Some(final_message) = chat.last().map(|chat| &chat.content) else {
                continue;
            };

            let final_message_str = final_message.to_string();

            if !rendered_chat.contains(final_message_str.trim()) {
                return Err(Error::FinalMsgNotInChat);
            }

            let final_msg_loc = rendered_chat.rfind(&final_message_str.trim()).unwrap();
            let final_msg_len = final_message_str.trim_start().len();
            rendered_chat = if rendered_chat[final_msg_loc..final_msg_loc + final_msg_len]
                == final_message_str
            {
                rendered_chat[..final_msg_loc + final_msg_len].to_string()
            } else {
                rendered_chat[..final_msg_loc + final_message_str.trim().len()].to_string()
            };
        }
        rendered.push(rendered_chat);
    }

    Ok(rendered)
}
