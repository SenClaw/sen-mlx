//! Image preprocessing for vision-capable local models (currently Gemma-4).
//!
//! Pure-Rust, CPU-only image-side helpers. Bundled into the `local-mlx`
//! feature so the preprocessor ships with the Gemma-4 model that consumes
//! it. (Earlier iterations had a separate `gemma4-image` feature for
//! standalone testability; that turned out not to be worth the extra
//! feature-surface knob, since you can't run the model without MLX anyway.)
//!
//! ## Pipeline (matches `mlx-vlm` Gemma 4 `Gemma4ImageProcessor` defaults)
//!
//! 1. **Decode** the bytes / path (PNG / JPEG / WebP / BMP / GIF).
//! 2. **Convert to RGB** — drop alpha, expand grayscale.
//! 3. **Resize** to 224×224 with bicubic interpolation. The reference uses
//!    PIL's `BICUBIC`; we use `image::imageops::FilterType::CatmullRom`,
//!    which is the bicubic implementation in the `image` crate. Minor
//!    numeric drift vs PIL is expected and irrelevant to model output.
//! 4. **Rescale** to `[0, 1]` by multiplying by `1.0 / 255.0`. NO mean/std
//!    normalisation — the processor config sets `do_normalize=false` and
//!    `image_mean=[0,0,0]`, `image_std=[1,1,1]`.
//! 5. **Reorder** HWC → CHW (channels-first, matching `pixel_values` of
//!    shape `[B, 3, H, W]` consumed by the vision tower).
//!
//! Output: 1×3×224×224 = **150 528** f32 values, CHW layout.
//!
//! ## Why not normalise?
//!
//! Per `mlx-community/gemma-4-e2b-it-4bit/processor_config.json`:
//! ```text
//!   image_processor.do_normalize  = false
//!   image_processor.image_mean    = [0, 0, 0]
//!   image_processor.image_std     = [1, 1, 1]
//!   image_processor.rescale_factor = 0.00392156862745098   // = 1/255
//! ```
//! The vision tower's own input RMSNorm handles whitening internally.

use image::{DynamicImage, ImageError};

/// Default target size for Gemma-4 vision input (square 224×224 per
/// `processor_config.json::image_processor.size`).
pub const TARGET_SIZE: u32 = 224;

/// Rescale factor applied to each pixel — `1 / 255` per processor config.
pub const RESCALE_FACTOR: f32 = 1.0 / 255.0;

/// Number of colour channels — RGB, always 3 after `to_rgb8()`.
pub const CHANNELS: usize = 3;

/// Preprocessed image in CHW (channels-first) float32 layout, ready to be
/// wrapped into a `[1, 3, height, width]` MLX `Array` by the forward path.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// `channels * height * width` f32 values, layout `[c0_h0_w0, c0_h0_w1,
    /// …, c0_h1_w0, …, c1_h0_w0, …]` (row-major within each channel,
    /// channels-first across them). Values in `[0, 1]`.
    pub pixels: Vec<f32>,
    /// 3 for RGB.
    pub channels: usize,
    /// 224 for the Gemma-4 default.
    pub height: usize,
    /// 224 for the Gemma-4 default.
    pub width: usize,
}

impl PreprocessedImage {
    /// Total number of f32 values — useful for sanity-checking output shape.
    pub fn numel(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// `[1, channels, height, width]` — the MLX array shape consumed by the
    /// vision tower's patch embedding.
    pub fn nchw_shape(&self) -> [i32; 4] {
        [
            1,
            self.channels as i32,
            self.height as i32,
            self.width as i32,
        ]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error reading image: {0}")]
    Io(#[from] std::io::Error),
    #[error("image decode failed: {0}")]
    Decode(#[from] ImageError),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Preprocess from a file path.
pub fn preprocess_path<P: AsRef<std::path::Path>>(path: P) -> Result<PreprocessedImage> {
    let img = image::open(path.as_ref())?;
    Ok(preprocess(img, TARGET_SIZE, TARGET_SIZE))
}

/// Preprocess from an in-memory byte buffer (useful for base64-decoded image
/// payloads from OpenAI-shape `image_url` content blocks).
pub fn preprocess_bytes(bytes: &[u8]) -> Result<PreprocessedImage> {
    let img = image::load_from_memory(bytes)?;
    Ok(preprocess(img, TARGET_SIZE, TARGET_SIZE))
}

/// Core preprocessing on a decoded [`DynamicImage`].
///
/// Steps:
/// 1. Force RGB (drops alpha, expands grayscale).
/// 2. Resize to (target_w, target_h) with bicubic interpolation.
/// 3. Rescale `u8` → `f32 / 255`.
/// 4. Reorder HWC → CHW into a single contiguous `Vec<f32>`.
pub fn preprocess(img: DynamicImage, target_w: u32, target_h: u32) -> PreprocessedImage {
    use image::imageops::FilterType;

    // (1) RGB-only. `to_rgb8` drops alpha; for grayscale it broadcasts to 3
    // channels via the image crate's standard conversion.
    let rgb = img.to_rgb8();

    // (2) Resize. `CatmullRom` is bicubic (matches PIL.BICUBIC up to minor
    // boundary handling — the Python reference uses BICUBIC too).
    let resized = if rgb.width() == target_w && rgb.height() == target_h {
        rgb
    } else {
        image::imageops::resize(&rgb, target_w, target_h, FilterType::CatmullRom)
    };

    let w = resized.width() as usize;
    let h = resized.height() as usize;

    // (3) + (4) Rescale + HWC → CHW in one pass. Output layout: channels-first.
    //
    // Source HWC: `resized.as_raw()` is `[h0_w0_r, h0_w0_g, h0_w0_b, h0_w1_r,
    // …]` (interleaved RGB rows). We iterate by channel-then-row-then-col.
    let src = resized.as_raw();
    let mut pixels: Vec<f32> = Vec::with_capacity(CHANNELS * h * w);
    for c in 0..CHANNELS {
        for y in 0..h {
            let row_start = y * w * CHANNELS;
            for x in 0..w {
                let pixel_start = row_start + x * CHANNELS;
                pixels.push(src[pixel_start + c] as f32 * RESCALE_FACTOR);
            }
        }
    }

    PreprocessedImage {
        pixels,
        channels: CHANNELS,
        height: h,
        width: w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid_rgb(w: u32, h: u32, color: [u8; 3]) -> DynamicImage {
        let mut img = RgbImage::new(w, h);
        for px in img.pixels_mut() {
            *px = Rgb(color);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn output_has_correct_shape_and_size() {
        let img = solid_rgb(100, 50, [128, 128, 128]);
        let p = preprocess(img, TARGET_SIZE, TARGET_SIZE);
        assert_eq!(p.channels, 3);
        assert_eq!(p.height, 224);
        assert_eq!(p.width, 224);
        assert_eq!(p.numel(), 3 * 224 * 224);
        assert_eq!(p.pixels.len(), p.numel());
        assert_eq!(p.nchw_shape(), [1, 3, 224, 224]);
    }

    #[test]
    fn rescale_to_unit_interval() {
        // Solid white image — every pixel should be exactly 1.0 after rescale.
        let img = solid_rgb(224, 224, [255, 255, 255]);
        let p = preprocess(img, TARGET_SIZE, TARGET_SIZE);
        for v in &p.pixels {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "expected 1.0 after × 1/255, got {v}"
            );
        }
        // Solid black image — every pixel exactly 0.0.
        let img = solid_rgb(224, 224, [0, 0, 0]);
        let p = preprocess(img, TARGET_SIZE, TARGET_SIZE);
        for v in &p.pixels {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn no_normalize_means_value_range_is_zero_to_one() {
        // Mid-grey 128 → 128/255 ≈ 0.502. Without mean/std this is the
        // output (NOT centred around 0 like ImageNet preprocessing).
        let img = solid_rgb(224, 224, [128, 128, 128]);
        let p = preprocess(img, TARGET_SIZE, TARGET_SIZE);
        for v in &p.pixels {
            assert!(
                (*v - 0.501_960_8).abs() < 1e-4,
                "expected ~0.502 (no mean/std), got {v}"
            );
        }
    }

    #[test]
    fn chw_layout_has_separated_channels() {
        // R=255, G=0, B=0 — channel 0 should be all 1.0, channels 1 and 2 all 0.0.
        let img = solid_rgb(224, 224, [255, 0, 0]);
        let p = preprocess(img, TARGET_SIZE, TARGET_SIZE);
        let plane = 224 * 224;
        for v in &p.pixels[0..plane] {
            assert_eq!(*v, 1.0, "R channel should be 1.0");
        }
        for v in &p.pixels[plane..2 * plane] {
            assert_eq!(*v, 0.0, "G channel should be 0.0");
        }
        for v in &p.pixels[2 * plane..3 * plane] {
            assert_eq!(*v, 0.0, "B channel should be 0.0");
        }
    }

    #[test]
    fn rgba_input_is_converted_to_rgb() {
        // 4-channel RGBA input must be flattened to RGB (alpha dropped).
        let mut img = image::RgbaImage::new(224, 224);
        for px in img.pixels_mut() {
            *px = image::Rgba([200, 100, 50, 128]); // alpha 128 should be ignored
        }
        let p = preprocess(DynamicImage::ImageRgba8(img), TARGET_SIZE, TARGET_SIZE);
        // First pixel of channel 0 should be 200/255 (NOT 100/255 from
        // alpha-blending — `to_rgb8` discards alpha, doesn't multiply).
        assert!(
            (p.pixels[0] - 200.0 / 255.0).abs() < 1e-4,
            "expected R = 200/255, got {}",
            p.pixels[0]
        );
    }

    #[test]
    fn grayscale_is_broadcast_to_three_rgb_channels() {
        let gray = image::GrayImage::from_fn(224, 224, |_x, _y| image::Luma([200]));
        let p = preprocess(DynamicImage::ImageLuma8(gray), TARGET_SIZE, TARGET_SIZE);
        // All three channels should be (200/255).
        for v in &p.pixels {
            assert!((v - 200.0 / 255.0).abs() < 1e-4);
        }
    }

    /// Non-square input is resized to the target square — useful sanity that
    /// aspect ratio is squashed (Gemma-4's v1 doesn't preserve aspect; that
    /// would require the soft-token bucketing logic, deferred).
    #[test]
    fn non_square_input_is_resized_to_target_square() {
        let img = solid_rgb(640, 320, [64, 64, 64]);
        let p = preprocess(img, TARGET_SIZE, TARGET_SIZE);
        assert_eq!(p.height, 224);
        assert_eq!(p.width, 224);
        // Still solid colour after resize.
        for v in &p.pixels {
            assert!((v - 64.0 / 255.0).abs() < 1e-3);
        }
    }

    #[test]
    fn already_target_size_skips_resize_but_still_rescales_and_reorders() {
        let img = solid_rgb(224, 224, [255, 128, 0]);
        let p = preprocess(img, TARGET_SIZE, TARGET_SIZE);
        // Channel 0 = 1.0, Channel 1 ≈ 0.502, Channel 2 = 0.0.
        let plane = 224 * 224;
        assert_eq!(p.pixels[0], 1.0);
        assert!((p.pixels[plane] - 128.0 / 255.0).abs() < 1e-4);
        assert_eq!(p.pixels[2 * plane], 0.0);
    }

    /// Round-trip through `preprocess_bytes` — exercises the decode path.
    #[test]
    fn preprocess_bytes_decodes_png() {
        // Encode a solid PNG, then preprocess it from bytes.
        let img = solid_rgb(100, 100, [200, 100, 50]);
        let mut buf: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
        let p = preprocess_bytes(&buf).unwrap();
        assert_eq!(p.height, 224);
        assert_eq!(p.width, 224);
    }
}
