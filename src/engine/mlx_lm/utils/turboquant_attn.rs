//! GQA attention against a per-layer [`turboquant::attention::QuantizedKVCache`].

use mlx_rs::{error::Exception, transforms::eval, Array, Dtype};
use turboquant::attention::QuantizedKVCache;

use super::super::cache::TurboQuantKeyValueCache;

fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / scores.len() as f32; scores.len()];
    }
    exp.iter().map(|e| e / sum).collect()
}

fn weighted_values_one_head(
    tq: &QuantizedKVCache,
    layer: usize,
    kv_head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    weights: &[f32],
) -> Result<Vec<f32>, Exception> {
    let values = tq
        .dequantize_all_values(layer)
        .map_err(|e| Exception::custom(format!("turboquant dequantize values: {e}")))?;
    let num_tokens = weights.len();
    let expected = num_tokens * n_kv_heads;
    if values.len() != expected {
        return Err(Exception::custom(format!(
            "turboquant value count {} != {num_tokens} tokens × {n_kv_heads} heads",
            values.len()
        )));
    }
    let mut out = vec![0f32; head_dim];
    for (t, w) in weights.iter().enumerate() {
        let idx = t * n_kv_heads + kv_head;
        for (o, &vi) in out.iter_mut().zip(values[idx].iter()) {
            *o += w * vi;
        }
    }
    Ok(out)
}

fn attention_scores_one_head(
    tq: &QuantizedKVCache,
    layer: usize,
    kv_head: usize,
    n_kv_heads: usize,
    query: &[f32],
) -> Result<Vec<f32>, Exception> {
    let all = tq
        .attention_scores(layer, query)
        .map_err(|e| Exception::custom(format!("turboquant attention_scores: {e}")))?;
    Ok(all
        .iter()
        .enumerate()
        .filter(|(i, _)| i % n_kv_heads == kv_head)
        .map(|(_, s)| *s)
        .collect())
}

fn apply_mask_row(scores: &mut [f32], mask_row: &[f32]) {
    if mask_row.len() != scores.len() {
        return;
    }
    for (s, m) in scores.iter_mut().zip(mask_row) {
        if *m < 0.5 {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// `queries`: `[B, n_heads, L, D]` after RoPE. Returns same shape.
#[allow(non_snake_case)]
pub fn turboquant_gqa_attention(
    queries: &Array,
    cache: &TurboQuantKeyValueCache,
    scale: f32,
    mask: Option<&Array>,
    n_heads: i32,
    n_kv_heads: i32,
) -> Result<Array, Exception> {
    let tq = cache.tq();
    let layer = 0usize;
    let head_dim = cache.head_dim() as usize;
    let n_kv_heads = n_kv_heads as usize;
    let n_heads = n_heads as usize;

    let q = queries.as_dtype(Dtype::Float32)?;
    eval(&[q.clone()])?;
    let shape = q.shape();
    if shape.len() != 4 {
        return Err(Exception::custom("turboquant attention expects 4D queries"));
    }
    let b = shape[0] as usize;
    let h = shape[1] as usize;
    let l = shape[2] as usize;
    let d = shape[3] as usize;
    if d != head_dim || h != n_heads {
        return Err(Exception::custom(
            "turboquant attention: query shape mismatch",
        ));
    }
    let q_flat = q.as_slice::<f32>();

    let mask_flat = if let Some(m) = mask {
        let m = m.as_dtype(Dtype::Float32)?;
        eval(&[m.clone()])?;
        Some((m.as_slice::<f32>().to_vec(), m.shape().to_vec()))
    } else {
        None
    };

    let mut out = vec![0f32; b * h * l * d];

    for bi in 0..b {
        for hi in 0..h {
            let kv_h = hi * n_kv_heads / n_heads;
            for li in 0..l {
                let base = ((bi * h + hi) * l + li) * d;
                let q_start = ((bi * h + hi) * l + li) * d;
                let q_vec = &q_flat[q_start..q_start + d];
                let mut scores = attention_scores_one_head(tq, layer, kv_h, n_kv_heads, q_vec)?;
                for s in &mut scores {
                    *s *= scale;
                }
                if let Some((ref m_flat, ref m_shape)) = mask_flat {
                    if m_shape.len() == 2 {
                        let cols = m_shape[1] as usize;
                        let row: Vec<f32> =
                            (0..scores.len()).map(|ki| m_flat[li * cols + ki]).collect();
                        apply_mask_row(&mut scores, &row);
                    }
                }
                let weights = softmax(&scores);
                let v_out =
                    weighted_values_one_head(tq, layer, kv_h, n_kv_heads, head_dim, &weights)?;
                out[base..base + d].copy_from_slice(&v_out);
            }
        }
    }

    Ok(Array::from_slice(
        &out,
        &[b as i32, h as i32, l as i32, d as i32],
    ))
}
