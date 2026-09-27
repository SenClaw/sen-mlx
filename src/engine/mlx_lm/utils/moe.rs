//! Mixture-of-Experts primitives for DeepSeek-style models.
//!
//! The make-or-break operation for DeepSeekMoE is a **grouped/batched quantized
//! expert matmul**: given a stacked quantized weight `w` of shape
//! `[E, out, in]` (one expert per leading slice) and a per-token list of
//! selected expert indices, compute `x @ w[idx].T` for each selection without
//! looping over the (e.g. 64) experts on the host. MLX exposes this as
//! `mx.gather_qmm`; the oxiglade/mlx-rs fork ships it only at the raw C-FFI
//! layer (`mlx_sys::mlx_gather_qmm`) with no safe Rust wrapper.
//!
//! [`gather_qmm`] is that wrapper. It replicates the `Guarded::try_from_op`
//! pattern (which is `pub(crate)` inside mlx-rs and therefore unreachable from
//! this crate): allocate a result handle with `mlx_array_new`, invoke the C op,
//! check the returned status int manually, and wrap the result with the public
//! `Array::from_ptr`. A failing status frees the handle and returns an error
//! rather than leaking or relying on mlx-rs's `pub(crate)` error channel.

use mlx_rs::{error::Exception, Array, StreamOrDevice};

/// Grouped quantized expert matmul: `out[b] = x[b] @ dequant(w[rhs_indices[b]]).T`.
///
/// Mirrors `mx.gather_qmm(x, w, scales, biases, rhs_indices=indices,
/// transpose=true, group_size, bits, sorted_indices)`. `lhs_indices` is always
/// absent (we never gather along `x`). Typical MoE use: `x` shaped
/// `[..., 1, in]` broadcasts against `rhs_indices` shaped `[..., top_k]`,
/// producing `[..., top_k, out]`.
///
/// `w` is the stacked quantized weight `[E, out, in_packed]` (uint32) with
/// `scales`/`biases` shaped `[E, out, in/group_size]`. Set `sorted_indices` only
/// when `rhs_indices` is sorted ascending by expert (enables MLX's faster decode
/// kernel); `false` always produces correct results.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    rhs_indices: &Array,
    transpose: bool,
    group_size: i32,
    bits: i32,
    sorted_indices: bool,
) -> Result<Array, Exception> {
    let stream = StreamOrDevice::default();
    // A zeroed `mlx_array` has a null `ctx`, which the C side treats as an
    // absent optional (`lhs_indices.ctx ? ... : std::nullopt`). The raw
    // `mlx_sys::mlx_array` is plain POD with no `Drop`, so this never frees.
    let lhs_null: mlx_sys::mlx_array = unsafe { std::mem::zeroed() };

    unsafe {
        let mut res = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_gather_qmm(
            &mut res,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            biases.as_ptr(),
            lhs_null,
            rhs_indices.as_ptr(),
            transpose,
            group_size,
            bits,
            sorted_indices,
            stream.as_ref().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(res);
            return Err(Exception::custom(format!(
                "mlx_gather_qmm failed (status {status}); inputs x={:?} w={:?} idx={:?}",
                x.shape(),
                w.shape(),
                rhs_indices.shape(),
            )));
        }
        Ok(Array::from_ptr(res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{dequantize, indexing::IndexOp, matmul, quantize};

    /// The grouped kernel must agree with an explicit per-expert
    /// dequantize+matmul on the SAME quantized weights — this isolates kernel
    /// correctness (and our shape/broadcast understanding) from quantization
    /// error, since both paths dequantize identically. Also exercises the
    /// unsafe FFI shim end-to-end on the Metal device.
    #[test]
    fn gather_qmm_matches_per_expert_reference() {
        let (e, out, k_in) = (4i32, 32i32, 64i32); // in divisible by group_size
        let (group, bits) = (64i32, 4i32);

        // Stacked float expert weights [E, out, in], deterministic small values.
        let wf: Vec<f32> = (0..(e * out * k_in) as usize)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let w_float = Array::from_slice(&wf, &[e, out, k_in]);
        let (wq, scales, biases) = quantize(&w_float, group, bits).unwrap();

        // x: [M, in]; selected experts inds: [M, K].
        let (m, k) = (3i32, 2i32);
        let xf: Vec<f32> = (0..(m * k_in) as usize)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();
        let x = Array::from_slice(&xf, &[m, k_in]);
        let inds_v: Vec<i32> = vec![0, 1, 2, 3, 1, 0]; // [M=3, K=2]
        let inds = Array::from_slice(&inds_v, &[m, k]);

        // MLX gather_qmm: x's leading dims (all but the trailing [M, K_in]) are
        // the batch and must broadcast against rhs_indices. The SwitchGLU idiom
        // expands x to [..., 1, 1, in] so batch dims [M, 1] broadcast against
        // inds [M, K] -> output [M, K, 1, out].
        let x_e = x.reshape(&[m, 1, 1, k_in]).unwrap();
        let got = gather_qmm(&x_e, &wq, &scales, &biases, &inds, true, group, bits, false).unwrap();
        assert_eq!(got.shape(), &[m, k, 1, out], "gather_qmm output shape");

        for mi in 0..m {
            for ki in 0..k {
                let e_idx = inds_v[(mi * k + ki) as usize];
                let w_de = dequantize(
                    &wq.index((e_idx, .., ..)),
                    &scales.index((e_idx, .., ..)),
                    &biases.index((e_idx, .., ..)),
                    group,
                    bits,
                )
                .unwrap(); // [out, in]
                let x_m = x.index((mi, ..)).reshape(&[k_in, 1]).unwrap();
                let ref_out = matmul(&w_de, &x_m).unwrap().reshape(&[out]).unwrap();
                let got_mk = got.index((mi, ki, 0, ..));
                let diff = got_mk
                    .subtract(&ref_out)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max(None)
                    .unwrap()
                    .item::<f32>();
                assert!(
                    diff < 1e-2,
                    "expert ({mi},{ki}) idx {e_idx}: max abs diff {diff}"
                );
            }
        }
    }
}
