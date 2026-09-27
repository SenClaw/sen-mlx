//! Token sampling for the native MLX path.
//!
//! The engine used to sample straight off the full logit row: scale by
//! `1/temperature`, then `categorical!` over the whole vocabulary. On a
//! ~150–260 K-entry vocabulary that leaves every token in the tail reachable,
//! and at `temperature > 0` the tail is where incoherent output comes from.
//! [`sample_with`] adds the two standard truncations — **top-k** and **top-p
//! (nucleus)** — before the draw.
//!
//! ## Order of operations
//!
//! Nucleus mass is defined against the model's *own* distribution, so `top_p`
//! is evaluated on `softmax(logits)` **before** temperature is applied; the
//! temperature then only shapes the draw among the survivors. Applying
//! temperature first would make the nucleus grow and shrink with it, which is
//! not what `top_p = 0.95` is understood to mean.
//!
//! ## Why top-k runs first
//!
//! A textbook nucleus filter sorts the whole vocabulary (`O(V log V)` per
//! token). We instead take the k largest logits with `argpartition` (`O(V)`)
//! and evaluate the nucleus on those k. That is not an approximation: the
//! nucleus is by construction a prefix of the probability-sorted order, so
//! `nucleus ∩ top-k` is the first `min(|nucleus|, k)` tokens — exactly the set
//! the sort-everything version leaves after its own top-k truncation. The
//! cumulative sums are taken over the **unrenormalized** full-vocabulary
//! probabilities, so the threshold keeps its usual meaning.
//!
//! ## Contract
//!
//! With `top_k = None` / `top_p = None`, or with values that cannot truncate,
//! this is the previous path unchanged — including the greedy
//! `temperature == 0` branch. Defaults must not move existing models' output.

use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    ops::{argpartition_axis, argsort_axis, indexing::take_along_axis, softmax_axis, which},
    Array,
};

/// Sample one token id from `logits` (shape `[..., vocab]`), returning an
/// array shaped like `logits` minus its last axis — the same shape the plain
/// `argmax` / `categorical` path returns.
///
/// - `temp <= 0.0` → greedy `argmax`. Truncation is skipped because neither
///   filter can change which entry is the maximum.
/// - `top_k` → keep only the k highest-probability tokens. `None`, `<= 0`, or
///   `>= vocab` disables it.
/// - `top_p` → keep the shortest prefix of the probability-sorted tokens whose
///   cumulative mass reaches `top_p`. `None`, `<= 0.0`, or `>= 1.0` disables
///   it. The highest-probability token always survives, so the filter can
///   never empty the candidate set.
pub fn sample_with(
    logits: &Array,
    temp: f32,
    top_k: Option<i32>,
    top_p: Option<f32>,
) -> Result<Array, Exception> {
    if temp <= 0.0 {
        return argmax_axis!(logits, -1);
    }

    let shape = logits.shape();
    let vocab = *shape
        .last()
        .ok_or_else(|| Exception::custom("sample: logits have no axes"))?;

    let k = top_k.filter(|k| *k > 0 && *k < vocab);
    let p = top_p.filter(|p| *p > 0.0 && *p < 1.0);
    if k.is_none() && p.is_none() {
        return categorical!(&logits.multiply(array!(1.0 / temp))?);
    }

    // Work in a flat `[rows, vocab]` view so the index arithmetic below is
    // plain 2-D, then restore the caller's leading shape at the end.
    let rows: i32 = shape.iter().take(shape.len() - 1).product();
    let out_shape: Vec<i32> = shape[..shape.len() - 1].to_vec();
    let flat = logits.reshape(&[rows, vocab])?;

    // ── Candidate set ────────────────────────────────────────────────────
    // With a top-k, `argpartition` on the negated logits puts the k largest in
    // the first k slots (in unspecified order — the nucleus step sorts them,
    // and plain top-k does not care). Without one, a nucleus filter still
    // needs the probability-sorted order, which costs the full sort.
    let neg = flat.negative()?;
    let cand = match k {
        Some(k) => {
            use mlx_rs::ops::indexing::IndexOp;
            argpartition_axis(&neg, k - 1, -1)?.index((.., ..k))
        }
        None => argsort_axis(&neg, -1)?,
    };
    let cand_logits = take_along_axis(&flat, &cand, -1)?;

    let (cand, cand_logits) = match p {
        None => (cand, cand_logits),
        Some(p) => {
            // Cumulative mass over the *full-vocabulary* softmax, restricted to
            // the candidates and read in descending-probability order.
            let probs = softmax_axis(&flat, -1, None)?;
            let cand_probs = take_along_axis(&probs, &cand, -1)?;
            let order = argsort_axis(&cand_probs.negative()?, -1)?;

            let sorted_probs = take_along_axis(&cand_probs, &order, -1)?;
            let sorted_cand = take_along_axis(&cand, &order, -1)?;
            let sorted_logits = take_along_axis(&cand_logits, &order, -1)?;

            // Keep token i when the mass *before* it has not yet reached `p`.
            // The exclusive prefix (`cumsum − own`) is what guarantees the
            // top-1 token always survives however peaked the distribution:
            // its exclusive prefix is 0.
            let cum = sorted_probs.cumsum(-1, None, None)?;
            let keep = cum.subtract(&sorted_probs)?.lt(array!(p))?;

            let neg_inf = Array::full::<f32>(sorted_logits.shape(), array!(f32::NEG_INFINITY))?
                .as_dtype(sorted_logits.dtype())?;
            (sorted_cand, which(&keep, &sorted_logits, &neg_inf)?)
        }
    };

    // ── Draw among the survivors, map the local index back to a vocab id ──
    let scaled = cand_logits.multiply(array!(1.0 / temp))?;
    let pick = categorical!(&scaled)?.reshape(&[rows, 1])?;
    take_along_axis(&cand, &pick, -1)?.reshape(&out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[1, vocab]` logits from a slice.
    fn logits(v: &[f32]) -> Array {
        Array::from_slice(v, &[1, v.len() as i32])
    }

    fn draw(l: &Array, temp: f32, k: Option<i32>, p: Option<f32>) -> u32 {
        sample_with(l, temp, k, p).unwrap().item::<u32>()
    }

    /// Greedy is unaffected by either filter — neither can move the argmax.
    #[test]
    fn temperature_zero_is_greedy_regardless_of_truncation() {
        let l = logits(&[0.1, 5.0, 0.2, 0.3]);
        assert_eq!(draw(&l, 0.0, None, None), 1);
        assert_eq!(draw(&l, 0.0, Some(1), Some(0.5)), 1);
    }

    /// `top_k = 1` is greedy by construction: one survivor, so every draw
    /// returns it however high the temperature.
    #[test]
    fn top_k_one_collapses_to_the_argmax() {
        let l = logits(&[0.0, 1.0, 9.0, 2.0, 0.5]);
        for _ in 0..32 {
            assert_eq!(draw(&l, 2.0, Some(1), None), 2);
        }
    }

    /// The tail must become *unreachable*, not merely unlikely — that is the
    /// whole point of the filter on a large vocabulary.
    #[test]
    fn top_k_never_draws_outside_the_k_largest() {
        // Ranks: 4 (9.0) > 0 (8.0) > 2 (7.0) > everything else.
        let l = logits(&[8.0, -20.0, 7.0, -30.0, 9.0, -25.0, -40.0]);
        for _ in 0..200 {
            let id = draw(&l, 5.0, Some(3), None);
            assert!(matches!(id, 0 | 2 | 4), "drew {id} from outside the top-3");
        }
    }

    /// One token holding nearly all the mass is a nucleus of size 1, so a
    /// nucleus filter alone must reproduce greedy output even at high
    /// temperature.
    #[test]
    fn top_p_keeps_only_the_dominant_token_when_mass_is_concentrated() {
        let l = logits(&[20.0, 0.0, 0.0, 0.0]);
        for _ in 0..64 {
            assert_eq!(draw(&l, 3.0, None, Some(0.9)), 0);
        }
    }

    /// A tiny `top_p` must not leave an empty candidate set: the exclusive
    /// prefix sum keeps the top-1 token whatever the threshold.
    #[test]
    fn top_p_always_keeps_at_least_one_token() {
        let l = logits(&[1.0, 1.0, 1.0, 1.0, 5.0]);
        for _ in 0..64 {
            assert_eq!(draw(&l, 1.0, None, Some(0.0001)), 4);
        }
    }

    /// Combined filters intersect, and on a flat distribution `top_k` is the
    /// binding constraint.
    #[test]
    fn top_k_and_top_p_intersect() {
        let l = logits(&[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        // Eight equal tokens: the 0.95 nucleus spans all of them, so the k = 2
        // cap decides. *Which* two is up to `argpartition`, but no draw may
        // ever span more than two distinct ids.
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..200 {
            seen.insert(draw(&l, 1.0, Some(2), Some(0.95)));
        }
        assert!(seen.len() <= 2, "top-k cap leaked: saw {seen:?}");
    }

    /// Values that cannot truncate must leave the legacy path in place, so a
    /// full-vocabulary draw still reaches every token.
    #[test]
    fn disabled_filters_leave_the_whole_vocabulary_reachable() {
        let l = logits(&[1.0, 1.0, 1.0, 1.0]);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..400 {
            seen.insert(draw(&l, 1.0, Some(0), Some(1.0)));
            seen.insert(draw(&l, 1.0, None, None));
        }
        assert_eq!(seen.len(), 4, "some token became unreachable: {seen:?}");
    }
}
