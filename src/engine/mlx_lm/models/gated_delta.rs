//! Gated-delta (linear attention) recurrence for Qwen3.5 linear-attn layers.
//!
//! Two equivalent implementations of the SAME recurrence:
//! - [`gated_delta_update_sequential`] — the O(T) per-token scan (compiled step).
//!   Used for decode (`T=1`) and the rare masked path.
//! - [`gated_delta_update_chunked`] — the chunked-parallel form (chunk size 64),
//!   which resolves the intra-chunk delta dependency with the UT transform
//!   (`tri_inv`) and does the bulk work as batched matmuls. This keeps the GPU
//!   saturated during prefill (the sequential scan is CPU-dispatch-bound: one
//!   tiny kernel per token per layer → the GPU starves). Ported from the
//!   HuggingFace `torch_chunk_gated_delta_rule` reference (Qwen3-Next) — the
//!   authoritative form that FLA's Triton `chunk_gated_delta_rule` also computes.
//!
//! [`gated_delta_update`] dispatches: chunked for the no-mask prefill path with
//! `T >= CHUNK`, sequential otherwise. Both produce identical `(y, state)`
//! (guarded by the `chunked_matches_sequential` test).

use mlx_rs::{
    array,
    error::Exception,
    linalg::tri_inv_device,
    nn,
    ops::{
        arange, concatenate_axis, cumsum, expand_dims,
        indexing::{IndexOp, NewAxis},
        matmul, repeat_axis, sigmoid, stack_axis, sum_axis, tril, zeros_dtype,
    },
    transforms::compile::compile,
    Array, Dtype, StreamOrDevice,
};

/// Exponentiated per-token decay `g = exp(-exp(A_log) * softplus(a + dt_bias))`.
/// Used by the sequential scan (applied as a multiplicative decay each step).
fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    compute_g_log(a_log, a, dt_bias)?.exp()
}

/// LOG-space per-token decay `g_log = -exp(A_log) * softplus(a + dt_bias)` (≤ 0).
/// The chunked form needs the log form (it cumsums it and only ever exponentiates
/// non-positive differences — the key numerical-stability invariant).
fn compute_g_log(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let inner = a.add(dt_bias)?;
    let soft = nn::softplus(&inner)?;
    let neg_a = a_log.exp()?.multiply(&array!(-1.0_f32))?;
    soft.multiply(&neg_a)
}

fn gated_delta_step(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let old_state = state;
    let decay = if g.shape().len() == 2 {
        g.index((.., .., NewAxis, NewAxis))
    } else {
        g.index((.., .., .., NewAxis))
    };
    let mut state = state.multiply(&decay)?;
    let kv_mem = sum_axis(&state.multiply(&k.index((.., .., NewAxis, ..)))?, -1, false)?;
    let delta = v
        .subtract(&kv_mem)?
        .multiply(&beta.index((.., .., NewAxis)))?;
    state = state.add(&k.index((.., .., NewAxis, ..)).multiply(&delta.index((
        ..,
        ..,
        ..,
        NewAxis,
    )))?)?;
    let y = sum_axis(&state.multiply(&q.index((.., .., NewAxis, ..)))?, -1, false)?;
    let state = if let Some(mask) = mask {
        let mask = expand_dims(mask, 1)?.expand_dims(2)?.expand_dims(3)?;
        mlx_rs::ops::r#where(&mask, &state, old_state)?
    } else {
        state
    };
    Ok((y, state))
}

/// Chunk size for the parallel scan (HF/FLA default).
const CHUNK: i32 = 64;

/// Run gated-delta over a sequence (prefill / decode).
///
/// Shapes: `q,k: [B,T,Hk,Dk]`, `v: [B,T,Hv,Dv]`, `a,b: [B,T,Hv]`,
/// `a_log,dt_bias: [Hv]`. Returns `y: [B,T,Hv,Dv]`, `state: [B,Hv,Dv,Dk]`.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_update(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let shape = q.shape();
    let b_size = shape[0];
    let seq_len = shape[1];
    let n_kv = shape[2];
    let d_k = shape[3];
    let h_v = v.shape()[2];
    let d_v = v.shape()[3];

    let beta = sigmoid(b)?;
    let g_log = compute_g_log(a_log, a, dt_bias)?;

    // GQA: broadcast the shared k-heads out to the v-head count.
    let repeat = h_v / n_kv;
    let (q_use, k_use) = if repeat > 1 {
        (
            repeat_axis::<f32>(q.clone(), repeat as i32, 2)?,
            repeat_axis::<f32>(k.clone(), repeat as i32, 2)?,
        )
    } else {
        (q.clone(), k.clone())
    };

    let state0 = match state {
        Some(s) => s.clone(),
        None => zeros_dtype(&[b_size, h_v, d_v, d_k], Dtype::Float32)?,
    };

    // Chunked-parallel for the no-mask prefill path (keeps the GPU busy);
    // sequential scan for decode (T=1) and the masked path.
    if mask.is_none() && seq_len >= CHUNK {
        return gated_delta_update_chunked(&q_use, &k_use, v, &g_log, &beta, &state0, CHUNK);
    }
    let g = g_log.exp()?;
    gated_delta_update_sequential(&q_use, &k_use, v, &g, &beta, &state0, mask)
}

/// O(T) per-token scan. `q,k: [B,T,Hv,Dk]` (GQA already expanded), `v: [B,T,Hv,Dv]`,
/// `g` (EXPONENTIATED decay), `beta: [B,T,Hv]`, `state: [B,Hv,Dv,Dk]`.
fn gated_delta_update_sequential(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let seq_len = q.shape()[1];
    let mut ys = Vec::with_capacity(seq_len as usize);
    let mut state = state.clone();

    // The recurrence is sequential (each step feeds the next), so this dispatches
    // `seq_len` tiny kernels — CPU-dispatch-bound. `compile` fuses each step's ~7
    // ops into one kernel (mlx caches the traced graph by closure type). Only the
    // no-mask path is compiled; the rare masked path uses the plain step.
    if mask.is_none() {
        let mut step = compile(
            |inp: &[Array]| -> Result<Vec<Array>, Exception> {
                let (q, k, v, g, beta, st) = (&inp[0], &inp[1], &inp[2], &inp[3], &inp[4], &inp[5]);
                let decay = if g.shape().len() == 2 {
                    g.index((.., .., NewAxis, NewAxis))
                } else {
                    g.index((.., .., .., NewAxis))
                };
                let mut s = st.multiply(&decay)?;
                let kv_mem = sum_axis(&s.multiply(&k.index((.., .., NewAxis, ..)))?, -1, false)?;
                let delta = v
                    .subtract(&kv_mem)?
                    .multiply(&beta.index((.., .., NewAxis)))?;
                s = s.add(&k.index((.., .., NewAxis, ..)).multiply(&delta.index((
                    ..,
                    ..,
                    ..,
                    NewAxis,
                )))?)?;
                let y = sum_axis(&s.multiply(&q.index((.., .., NewAxis, ..)))?, -1, false)?;
                Ok(vec![y, s])
            },
            None,
        );
        for t in 0..seq_len {
            let q_t = q.index((.., t, .., ..));
            let k_t = k.index((.., t, .., ..));
            let v_t = v.index((.., t, .., ..));
            let g_t = g.index((.., t, ..));
            let beta_t = beta.index((.., t, ..));
            let out = step(&[q_t, k_t, v_t, g_t, beta_t, state.clone()])?;
            ys.push(out[0].clone());
            state = out[1].clone();
        }
    } else {
        for t in 0..seq_len {
            let q_t = q.index((.., t, .., ..));
            let k_t = k.index((.., t, .., ..));
            let v_t = v.index((.., t, .., ..));
            let g_t = g.index((.., t, ..));
            let beta_t = beta.index((.., t, ..));
            let mask_t = mask.map(|m| m.index((.., t)));
            let (y_t, s) =
                gated_delta_step(&q_t, &k_t, &v_t, &g_t, &beta_t, &state, mask_t.as_ref())?;
            state = s;
            ys.push(y_t);
        }
    }
    let y = stack_axis(&ys, 1)?;
    Ok((y, state))
}

/// `[C,C]` identity (mlx-rs has no `eye`).
fn eye_c(c: i32) -> Result<Array, Exception> {
    let r = arange::<_, f32>(0, c, 1)?;
    r.reshape(&[c, 1])?
        .eq(&r.reshape(&[1, c])?)?
        .as_dtype(Dtype::Float32)
}

/// `(I + L)^{-1}` for a strictly-lower `L` of trailing shape `[C,C]` — the UT /
/// WY transform inverse. Computed with `tri_inv` on a **CPU stream** (mlx's
/// triangular inverse is CPU-only in this build).
///
/// We deliberately do NOT use a GPU matrix-power method (Neumann sum or the
/// doubling product `∏(I ± L^{2^k})`): although `L` is nilpotent, its powers
/// `L^{2^k}` grow like `C^k` (up to `C^32` for `C=64`) before hitting `L^C=0`,
/// which **overflows to inf/NaN** at real decay scales even though the final
/// inverse is bounded. `tri_inv`'s forward substitution never forms high powers,
/// so it's numerically stable. The per-layer CPU sync is negligible (~a few
/// small batched inverses against a multi-second prefill).
fn ut_inverse(l: &Array, c: i32, bhn: i32) -> Result<Array, Exception> {
    // (I + L) is unit-lower-triangular; invert it. Batch as [bhn, C, C].
    let i_plus_l = eye_c(c)?.add(l)?.reshape(&[bhn, c, c])?;
    tri_inv_device(&i_plus_l, Some(false), StreamOrDevice::cpu())
}

/// Chunked-parallel gated delta rule. Inputs match the sequential helper except
/// `g_log` is the LOG-space decay (not exponentiated). Produces identical
/// `(y: [B,T,Hv,Dv], state: [B,Hv,Dv,Dk])` to the sequential scan.
///
/// Ported from HF `torch_chunk_gated_delta_rule`. Internally uses the FLA
/// `[B,H,K,V]` (k-major) state; the caller's `[B,Hv,Dv,Dk]` (v-major) state is
/// transposed at the boundary. All math is float32.
#[allow(non_snake_case, clippy::similar_names)]
fn gated_delta_update_chunked(
    q: &Array,
    k: &Array,
    v: &Array,
    g_log: &Array,
    beta: &Array,
    state_in: &Array,
    chunk: i32,
) -> Result<(Array, Array), Exception> {
    let bsz = q.shape()[0];
    let t = q.shape()[1];
    let h = q.shape()[2];
    let kdim = q.shape()[3];
    let vdim = v.shape()[3];

    // Pad T up to a multiple of `chunk`. Padded tokens carry beta=0 (→ no delta
    // write, no output) and g_log=0 (decay 1); their outputs are sliced off.
    let pad = (chunk - t % chunk) % chunk;
    let pad_time = |x: &Array| -> Result<Array, Exception> {
        if pad == 0 {
            return Ok(x.clone());
        }
        let mut sh = x.shape().to_vec();
        sh[1] = pad;
        concatenate_axis(&[x.clone(), zeros_dtype(&sh, x.dtype())?], 1)
    };
    let (q, k, v, g_log, beta) = (
        pad_time(q)?,
        pad_time(k)?,
        pad_time(v)?,
        pad_time(g_log)?,
        pad_time(beta)?,
    );
    let tp = t + pad;
    let n = tp / chunk;

    // [B,T,H,D] -> [B,H,T,D]; scalars [B,T,H] -> [B,H,T]. Everything f32.
    let f = |x: &Array| x.as_dtype(Dtype::Float32);
    let q = f(&q.transpose_axes(&[0, 2, 1, 3])?)?;
    let k = f(&k.transpose_axes(&[0, 2, 1, 3])?)?;
    let v = f(&v.transpose_axes(&[0, 2, 1, 3])?)?;
    let g_log = f(&g_log.transpose_axes(&[0, 2, 1])?)?;
    let beta = f(&beta.transpose_axes(&[0, 2, 1])?)?;

    let v_beta = v.multiply(&beta.index((.., .., .., NewAxis)))?;
    let k_beta = k.multiply(&beta.index((.., .., .., NewAxis)))?;

    // Reshape to chunks [B,H,n,C,·].
    let rc = |x: &Array, d: i32| x.reshape(&[bsz, h, n, chunk, d]);
    let q = rc(&q, kdim)?;
    let k = rc(&k, kdim)?;
    let v_beta = rc(&v_beta, vdim)?;
    let k_beta = rc(&k_beta, kdim)?;
    let g = cumsum(&g_log.reshape(&[bsz, h, n, chunk])?, -1, false, true)?; // in-chunk log-cumsum

    // Pairwise decay: decay_mask[i,j] = exp(g_i - g_j) for i>=j, else 0.
    // (tril BEFORE exp so upper stays 0 not exp(+); tril AFTER to re-zero the
    //  exp(0)=1 that lands in the strict-upper triangle.)
    let diff = g
        .index((.., .., .., .., NewAxis))
        .subtract(&g.index((.., .., .., NewAxis, ..)))?;
    let decay_mask = tril(&tril(&diff, 0)?.exp()?, 0)?;

    // UT transform T = (I - tril(k_beta·kᵀ · decay_mask, -1))^{-1}.
    let kt = k.transpose_axes(&[0, 1, 2, 4, 3])?;
    let l = tril(&matmul(&k_beta, &kt)?.multiply(&decay_mask)?, -1)?; // strictly lower
    let tmat = ut_inverse(&l, chunk, bsz * h * n)?.reshape(&[bsz, h, n, chunk, chunk])?; // (I+L)^{-1}

    // WY: pseudo-values u = T·(β⊙v); decayed keys w = T·(β⊙k⊙exp(g)).
    let u = matmul(&tmat, &v_beta)?;
    let kbg = k_beta.multiply(&g.exp()?.index((.., .., .., .., NewAxis)))?;
    let w = matmul(&tmat, &kbg)?;

    // Chunk scan. State S: [B,H,K,V] (transpose caller's [B,H,V,K]).
    let mut s = f(&state_in.transpose_axes(&[0, 1, 3, 2])?)?;
    let mut outs: Vec<Array> = Vec::with_capacity(n as usize);
    for i in 0..n {
        let qi = q.index((.., .., i, .., ..));
        let ki = k.index((.., .., i, .., ..));
        let ui = u.index((.., .., i, .., ..));
        let wi = w.index((.., .., i, .., ..));
        let dmi = decay_mask.index((.., .., i, .., ..));
        let gi = g.index((.., .., i, ..));

        let kit = ki.transpose_axes(&[0, 1, 3, 2])?;
        let attn_intra = tril(&matmul(&qi, &kit)?.multiply(&dmi)?, 0)?;
        let v_new = ui.subtract(&matmul(&wi, &s)?)?;
        let qi_dec = qi.multiply(&gi.exp()?.index((.., .., .., NewAxis)))?;
        let out_i = matmul(&qi_dec, &s)?.add(&matmul(&attn_intra, &v_new)?)?;
        outs.push(out_i);

        // Carry state: decay by the chunk-total, add residual-decayed kᵀ·v_new.
        let g_last = gi.index((.., .., -1));
        let decay_s = g_last.index((.., .., NewAxis, NewAxis)).exp()?;
        let residual = g_last.index((.., .., NewAxis)).subtract(&gi)?.exp()?;
        let k_dec_t = ki
            .multiply(&residual.index((.., .., .., NewAxis)))?
            .transpose_axes(&[0, 1, 3, 2])?;
        s = s.multiply(&decay_s)?.add(&matmul(&k_dec_t, &v_new)?)?;
    }

    // y: [B,H,n,C,V] -> [B,H,T,V] -> drop pad -> [B,T,H,V].
    let y = stack_axis(&outs, 2)?
        .reshape(&[bsz, h, tp, vdim])?
        .index((.., .., 0..t, ..))
        .transpose_axes(&[0, 2, 1, 3])?;
    // State back to caller layout [B,H,V,K].
    let state_out = s.transpose_axes(&[0, 1, 3, 2])?;
    Ok((y, state_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random;

    /// L2-normalize over the last axis (as the real model does to q,k — the
    /// delta-rule recurrence is numerically unstable on un-normalized keys).
    fn l2norm(x: &Array) -> Array {
        let ss = sum_axis(&x.multiply(x).unwrap(), -1, true).unwrap();
        x.divide(&ss.add(&array!(1e-6_f32)).unwrap().sqrt().unwrap())
            .unwrap()
    }

    /// The chunked-parallel scan must reproduce the sequential scan bit-close on
    /// the SAME inputs — the correctness gate before it touches the real model.
    #[test]
    fn chunked_matches_sequential() {
        random::seed(0).unwrap();
        // T = 150 exercises padding (not a multiple of 64) and 3 chunks; GQA off.
        let (b, t, h, dk, dv) = (1, 150, 2, 16, 16);
        let q = l2norm(&random::uniform::<_, f32>(-1.0, 1.0, &[b, t, h, dk], None).unwrap());
        let k = l2norm(&random::uniform::<_, f32>(-1.0, 1.0, &[b, t, h, dk], None).unwrap());
        let v = random::uniform::<_, f32>(-1.0, 1.0, &[b, t, h, dv], None).unwrap();
        // g_log ≤ 0 (log-decay); beta ∈ (0,1).
        let g_log = random::uniform::<_, f32>(-0.5, 0.0, &[b, t, h], None).unwrap();
        let beta = random::uniform::<_, f32>(0.0, 1.0, &[b, t, h], None).unwrap();
        let state = zeros_dtype(&[b, h, dv, dk], Dtype::Float32).unwrap();

        let g = g_log.exp().unwrap();
        let (y_seq, s_seq) =
            gated_delta_update_sequential(&q, &k, &v, &g, &beta, &state, None).unwrap();
        let (y_chk, s_chk) =
            gated_delta_update_chunked(&q, &k, &v, &g_log, &beta, &state, 64).unwrap();

        let dy = (&y_seq - &y_chk)
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let ds = (&s_seq - &s_chk)
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(dy < 1e-3, "y max diff {dy} (chunked vs sequential)");
        assert!(ds < 1e-3, "state max diff {ds} (chunked vs sequential)");
    }

    /// A non-zero carried-in state must also match (exercises the inter-chunk
    /// state read `v_prime`/`attn_inter` and the carry update).
    #[test]
    fn chunked_matches_sequential_with_initial_state() {
        random::seed(3).unwrap();
        let (b, t, h, dk, dv) = (2, 128, 4, 16, 32);
        let q = l2norm(&random::uniform::<_, f32>(-1.0, 1.0, &[b, t, h, dk], None).unwrap());
        let k = l2norm(&random::uniform::<_, f32>(-1.0, 1.0, &[b, t, h, dk], None).unwrap());
        let v = random::uniform::<_, f32>(-1.0, 1.0, &[b, t, h, dv], None).unwrap();
        let g_log = random::uniform::<_, f32>(-0.3, 0.0, &[b, t, h], None).unwrap();
        let beta = random::uniform::<_, f32>(0.0, 1.0, &[b, t, h], None).unwrap();
        let state = random::uniform::<_, f32>(-0.2, 0.2, &[b, h, dv, dk], None).unwrap();

        let g = g_log.exp().unwrap();
        let (y_seq, s_seq) =
            gated_delta_update_sequential(&q, &k, &v, &g, &beta, &state, None).unwrap();
        let (y_chk, s_chk) =
            gated_delta_update_chunked(&q, &k, &v, &g_log, &beta, &state, 64).unwrap();

        let dy = (&y_seq - &y_chk)
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let ds = (&s_seq - &s_chk)
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(dy < 2e-3, "y max diff {dy}");
        assert!(ds < 2e-3, "state max diff {ds}");
    }
}
