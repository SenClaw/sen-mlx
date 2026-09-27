use mlx_rs::{
    arange,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    ops::{
        expand_dims,
        indexing::{IndexOp, NewAxis},
        quantized_matmul, reshape, softmax_axis,
    },
    Array, Dtype,
};

use super::cache::KeyValueCache;
pub use super::cache::{QuantizedKeys, QuantizedValues};

pub mod moe;
pub mod rope;
pub mod tokenizer;
pub mod turboquant_attn;
pub mod yarn;

#[allow(unused_macros)]
macro_rules! try_unwrap {
    ($expr:expr) => {
        match $expr {
            core::result::Result::Ok(val) => val,
            core::result::Result::Err(e) => return Some(Err(e.into())),
        }
    };
}

fn index_out_of_bound_exception() -> Exception {
    Exception::custom("index out of bound")
}

#[allow(non_snake_case)]
pub(crate) fn quantized_scaled_dot_product_attention(
    queries: Array,
    mut q_keys: QuantizedKeys,
    mut q_values: QuantizedValues,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    let q_shape = queries.shape();
    let B = *q_shape.first().ok_or_else(index_out_of_bound_exception)?;
    let n_q_heads = *q_shape.get(1).ok_or_else(index_out_of_bound_exception)?;
    let L = *q_shape.get(2).ok_or_else(index_out_of_bound_exception)?;
    let D = *q_shape.get(3).ok_or_else(index_out_of_bound_exception)?;

    let q_keys_shape = q_keys.keys.shape();
    let n_kv_heads = q_keys_shape[q_keys_shape.len() - 3];
    let n_repeats = n_q_heads / n_kv_heads;

    let mut queries = queries * scale;

    if n_repeats > 1 {
        queries = reshape(&queries, &[B, n_kv_heads, n_repeats, L, D])?;

        q_keys.keys = expand_dims(q_keys.keys, -3)?;
        q_keys.scales = expand_dims(q_keys.scales, -3)?;
        q_keys.biases = expand_dims(q_keys.biases, -3)?;

        q_values.values = expand_dims(q_values.values, -3)?;
        q_values.scales = expand_dims(q_values.scales, -3)?;
        q_values.biases = expand_dims(q_values.biases, -3)?;
    }

    let mut scores = quantized_matmul(
        &queries,
        q_keys.keys,
        q_keys.scales,
        &q_keys.biases,
        true,
        group_size,
        bits,
    )?;

    if let Some(mask) = mask {
        if mask.dtype() == Dtype::Bool {
            let finfo_min = scores.dtype().finfo_min()? as f32;
            let fill = Array::from_f32(finfo_min).as_dtype(scores.dtype())?;
            scores = mlx_rs::ops::r#where(mask, scores, fill)?;
        } else {
            scores += mask;
        }
    }
    scores = softmax_axis(scores, -1, true)?;
    let mut out = quantized_matmul(
        scores,
        q_values.values,
        q_values.scales,
        &q_values.biases,
        false,
        group_size,
        bits,
    )?;

    if n_repeats > 1 {
        out = reshape(out, &[B, n_q_heads, L, D])?;
    }

    Ok(out)
}

pub enum MaybeQuantizedKeys {
    Original(Array),
    Quantized(QuantizedKeys),
}

impl From<Array> for MaybeQuantizedKeys {
    fn from(value: Array) -> Self {
        Self::Original(value)
    }
}

impl From<QuantizedKeys> for MaybeQuantizedKeys {
    fn from(value: QuantizedKeys) -> Self {
        Self::Quantized(value)
    }
}

pub enum MaybeQuantizedValues {
    Original(Array),
    Quantized(QuantizedValues),
}

impl From<Array> for MaybeQuantizedValues {
    fn from(value: Array) -> Self {
        Self::Original(value)
    }
}

impl From<QuantizedValues> for MaybeQuantizedValues {
    fn from(value: QuantizedValues) -> Self {
        Self::Quantized(value)
    }
}

pub(crate) fn scaled_dot_product_attention<C>(
    queries: Array,
    keys: impl Into<MaybeQuantizedKeys>,
    values: impl Into<MaybeQuantizedValues>,
    cache: Option<C>,
    scale: f32,
    mask: Option<&Array>,
) -> Result<Array, Exception>
where
    C: KeyValueCache,
{
    let keys = keys.into();
    let values = values.into();

    if let Some(cache) = cache {
        if cache.is_quantized() {
            let group_size = cache
                .group_size()
                .ok_or_else(|| Exception::custom("Cache is quantized but group size is not set"))?;
            let bits = cache
                .bits()
                .ok_or_else(|| Exception::custom("Cache is quantized but bits are not set"))?;

            let (keys, values) = match (keys, values) {
                (MaybeQuantizedKeys::Quantized(keys), MaybeQuantizedValues::Quantized(values)) => {
                    (keys, values)
                }
                _ => {
                    return Err(Exception::custom(
                        "Both keys and values must be quantized when KV cache is quantized",
                    ));
                }
            };

            return quantized_scaled_dot_product_attention(
                queries, keys, values, scale, mask, group_size, bits,
            );
        }
    }

    let (keys, values) = match (keys, values) {
        (MaybeQuantizedKeys::Original(keys), MaybeQuantizedValues::Original(values)) => {
            (keys, values)
        }
        _ => {
            return Err(Exception::custom(
                "Both keys and values must NOT be quantized when KV cache is NOT quantized",
            ));
        }
    };

    mlx_rs::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale,
        mask.map(ScaledDotProductAttentionMask::Array),
    )
}

#[derive(Debug, Clone)]
pub(crate) enum AttentionMask {
    Array(Array),
    Causal,
}

impl<'a> From<&'a AttentionMask> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a AttentionMask) -> Self {
        match mask {
            AttentionMask::Array(array) => ScaledDotProductAttentionMask::Array(array),
            AttentionMask::Causal => ScaledDotProductAttentionMask::Causal,
        }
    }
}

#[allow(non_snake_case)]
pub(crate) fn create_causal_mask(
    N: i32,
    offset: Option<i32>,
    window_size: Option<i32>,
    lengths: Option<Array>,
) -> Result<Array, Exception> {
    let offset = offset.unwrap_or(0);

    let rinds = arange!(stop = offset + N)?;
    let linds = arange!(start = offset, stop = offset + N)?;
    let linds = linds.index((.., NewAxis));
    let rinds = rinds.index(NewAxis);

    let mut mask = linds.ge(&rinds)?;
    if let Some(window_size) = window_size {
        // Align with `mlx_lm.models.base.create_causal_mask`: `linds < rinds + window`
        // (strict `<`, not `<=`).
        mask = mask.logical_and(&linds.lt(&(rinds + window_size))?)?;
    }

    if let Some(lengths) = lengths {
        let lengths = lengths.index((.., NewAxis, NewAxis, NewAxis));
        mask = mask.logical_and(&linds.lt(&lengths)?)?;
    }

    Ok(mask)
}

#[allow(non_snake_case)]
pub(crate) fn create_attention_mask<C>(
    h: &Array,
    cache: &[Option<C>],
    rope_offset: usize,
    return_array: Option<bool>,
) -> Result<Option<AttentionMask>, Exception>
where
    C: KeyValueCache,
{
    let mut return_array = return_array.unwrap_or(false);
    let T = h.shape()[1];
    if T > 1 {
        let mut offset = rope_offset as i32;
        let mut window_size = None;
        if let Some(c) = cache.first().and_then(|c| c.as_ref()) {
            if let Some(window_size_) = c.max_size() {
                window_size = Some(window_size_);
                offset = offset.min(window_size_);
                return_array = return_array || (offset + T) > window_size_;
            }
        }

        if return_array {
            create_causal_mask(T, Some(offset), window_size, None)
                .map(AttentionMask::Array)
                .map(Some)
        } else {
            Ok(Some(AttentionMask::Causal))
        }
    } else {
        Ok(None)
    }
}
