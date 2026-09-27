//! Gemma-4 vision-side modules — the projection layer that sits between the
//! vision tower's image features and the text decoder's hidden space, plus
//! (in Phase 2) the vision tower itself.
//!
//! ## Phase 1 (this file, today)
//!
//! Just [`MultimodalEmbedder`]: the small RMSNorm + Linear that projects
//! `[B, num_image_tokens, vision_hidden]` features into the text decoder's
//! `[B, num_image_tokens, text_hidden]` embedding space, ready to be
//! scattered into the text-token embedding stream at `<|image|>` positions.
//!
//! ## Phase 2 (planned)
//!
//! `VisionModel` — the 16-layer SigLIP2-variant transformer (ClippableLinear,
//! multidim RoPE, fp32 RMSNorm variants) that turns `[B, 3, 224, 224]`
//! pixel tensors into `[B, 280, 768]` image features. See
//! `~/.claude/projects/-Users-benji-Projects-SenClaw/memory/gemma4-vision-research.md`
//! for the architecture details.
//!
//! ## Why a separate module instead of inside `gemma4.rs`?
//!
//! Vision input is an optional feature — gating the entire module instead of
//! sprinkling `#[cfg]` throughout the text decoder keeps the text path
//! readable. Other vision-capable models could land here without forcing
//! everyone who reads `gemma4.rs` to wade through vision code.

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt},
    nn,
    ops::ones,
    quantization::MaybeQuantized,
    Array,
};

/// Project a `[B, N, embedding_dim]` feature tensor (image / audio / other
/// modality) into the text decoder's hidden space. Matches the
/// `MultimodalEmbedder` class in `mlx-vlm`'s gemma4.py:
///
/// ```python
/// class MultimodalEmbedder(nn.Module):
///     def __init__(self, embedding_dim, text_hidden_size, eps=1e-6):
///         self.embedding_pre_projection_norm = RMSNormNoScale(embedding_dim, eps=eps)
///         self.embedding_projection = nn.Linear(embedding_dim, text_hidden_size, bias=False)
///
///     def __call__(self, inputs_embeds):
///         normed = self.embedding_pre_projection_norm(inputs_embeds)
///         return self.embedding_projection(normed)
/// ```
///
/// Weight paths in the safetensors checkpoint:
/// - `embed_vision.embedding_pre_projection_norm.weight` — though it's
///   "NoScale" in the math, the checkpoint may still ship a scale buffer
///   (loaded but multiplied by `1` effectively). The loader treats it as a
///   parameter slot; the forward pass uses scale-free RMSNorm regardless.
/// - `embed_vision.embedding_projection.weight` — `[text_hidden, vision_hidden]`
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MultimodalEmbedder {
    pub embedding_dim: i32,
    pub text_hidden_size: i32,
    pub eps: f32,
    /// Linear projection (`vision_hidden → text_hidden`, no bias). Quantizable
    /// because OptiQ checkpoints may store this 8-bit.
    #[quantizable]
    #[param]
    pub embedding_projection: MaybeQuantized<nn::Linear>,
}

impl MultimodalEmbedder {
    pub fn new(embedding_dim: i32, text_hidden_size: i32, eps: f32) -> Result<Self, Exception> {
        let embedding_projection = nn::LinearBuilder::new(embedding_dim, text_hidden_size)
            .bias(false)
            .build()?;
        Ok(Self {
            embedding_dim,
            text_hidden_size,
            eps,
            embedding_projection: MaybeQuantized::Original(embedding_projection),
        })
    }

    /// `inputs_embeds`: `[B, N, embedding_dim]` features from the vision (or
    /// audio) tower. Returns `[B, N, text_hidden_size]` ready to be scattered
    /// into the text decoder's embedding stream at `<|image|>` token
    /// positions (see `gemma4.rs::forward_with_images`, Phase 2/3 wiring).
    ///
    /// Two-step:
    /// 1. Scale-free RMSNorm over the last axis — `x / sqrt(mean(x², axis=-1) + eps)`.
    /// 2. Linear projection to `text_hidden_size`.
    pub fn forward(&mut self, inputs_embeds: &Array) -> Result<Array, Exception> {
        // Step 1 — scale-free RMSNorm. mlx-rs's `fast::rms_norm` requires a
        // scale tensor; pass all-ones to get the no-scale behaviour.
        let dim = inputs_embeds.dim(-1);
        let ones_scale = ones::<f32>(&[dim])?;
        let normed = mlx_rs::fast::rms_norm(inputs_embeds, &ones_scale, self.eps)?;

        // Step 2 — linear projection.
        self.embedding_projection.forward(&normed)
    }
}

// ── Loader helper for the embed_vision.* safetensor keys ────────────────────

/// Module path prefix for these weights in the safetensors index — set by
/// the wrapper checkpoint (`Gemma4ForConditionalGeneration`).
pub const EMBED_VISION_PREFIX: &str = "embed_vision";

/// `true` when the given (stripped of `language_model.`) safetensor key
/// belongs to the multimodal-embedder projection (`embed_vision.*`).
///
/// Used by the loader to route these weights into the [`MultimodalEmbedder`]
/// slots instead of skipping them along with the rest of the multimodal
/// tower. Audio (`embed_audio.*`) is intentionally NOT routed here — it
/// stays skipped until the audio tower is ported.
pub fn is_embed_vision_key(stripped_key: &str) -> bool {
    stripped_key.starts_with("embed_vision.")
}

/// Total parameter slots in a `MultimodalEmbedder` (used by the loader to
/// assert weight coverage after the load loop).
pub fn parameter_slot_count(emb: &mut MultimodalEmbedder) -> usize {
    use mlx_rs::module::ModuleParameters;
    emb.parameters_mut().flatten().keys().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: the projection has the right shape and one parameter slot
    /// (`embedding_projection.weight` — RMSNormNoScale has no learnable
    /// parameters, matching the `with_scale=False` PyTorch variant).
    #[test]
    fn embedder_constructs_with_expected_shape() {
        let emb = MultimodalEmbedder::new(768, 1536, 1e-6).unwrap();
        assert_eq!(emb.embedding_dim, 768);
        assert_eq!(emb.text_hidden_size, 1536);
        assert_eq!(emb.eps, 1e-6);
        // Confirm the linear has the right in/out dims (post-quantize this
        // would be `inner.weight`; pre-quantize we test the Original path).
        if let MaybeQuantized::Original(ref lin) = emb.embedding_projection {
            assert_eq!(lin.weight.value.shape(), &[1536, 768]);
        } else {
            panic!("expected unquantized linear before nn::quantize");
        }
    }

    /// The forward pass on a small `[1, 4, 768]` input produces the expected
    /// `[1, 4, 1536]` output shape — and runs without error (validates the
    /// rms_norm + linear chain compiles + executes).
    #[test]
    fn forward_produces_text_hidden_shape() {
        let mut emb = MultimodalEmbedder::new(768, 1536, 1e-6).unwrap();
        let x = mlx_rs::ops::zeros::<f32>(&[1, 4, 768]).unwrap();
        let y = emb.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 1536]);
    }

    #[test]
    fn key_routing_picks_embed_vision_but_not_embed_audio() {
        assert!(is_embed_vision_key(
            "embed_vision.embedding_projection.weight"
        ));
        assert!(is_embed_vision_key(
            "embed_vision.embedding_pre_projection_norm.weight"
        ));
        // These should be classified as NOT belonging to the embedder, so
        // the existing skip-paths still apply to them.
        assert!(!is_embed_vision_key(
            "embed_audio.embedding_projection.weight"
        ));
        assert!(!is_embed_vision_key(
            "vision_tower.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_embed_vision_key(
            "audio_tower.layers.0.feed_forward1.linear.weight"
        ));
        assert!(!is_embed_vision_key(
            "model.layers.0.self_attn.q_proj.weight"
        ));
    }
}
