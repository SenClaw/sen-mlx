# Gemma 4 on the native MLX path — optimizations and rejected ones

This is the record for the Gemma-4 work driven by a read of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), a Swift +
Metal runtime for `gemma-4-26b-a4b` whose value to us is not its code (different
stack entirely) but its experiment log: `docs/OPTIMIZATION_JOURNEY.md` and
`docs/experiments/summaries/01..09`, which publish **negative results** with
end-to-end measurements and state why each candidate was rejected.

Two of our own deferred optimizations were settled by it without us having to
measure them, and one was confirmed worth building.

## What changed

| Change | Where | Status |
|---|---|---|
| Ring buffer for sliding-window KV during decode | [`cache.rs`](../src/local_model/mlx_lm/cache.rs) | Landed; **neutral** — <1% decode, no CPU/GPU/RAM change (E2B + E4B) |
| Top-k / top-p sampling | [`sampling.rs`](../src/local_model/mlx_lm/sampling.rs) | Landed |
| Sampling defaults read from the checkpoint | [`mlx_native.rs`](../src/local_model/mlx_native.rs) `GenerationDefaults` | Landed |
| `prefill_chunk_tokens` setting | [`mlx_native.rs`](../src/local_model/mlx_native.rs) | Landed (default unchanged at 512) |
| `gemma-4-26b-a4b` MoE architecture | [`gemma4.rs`](../src/local_model/mlx_lm/models/gemma4.rs) | Implemented; **not runtime-verified** |
| TurboQuant 4-bit KV for Gemma-4 | — | **Rejected, permanently** |
| Deeper async decode lookahead | — | **Rejected, permanently** |

## The sliding-window KV ring

Sliding-window layers keep a decode-time window (`SteppingKeyValueCache::
with_decode_window`). The eviction used to be a tail slice, which left the
buffer one row short of the required slots and therefore re-entered the grow
branch — a trim, a `zeros` pad and a `concatenate` — on **every decode step past
the window**. Two full copies of K and two of V, per sliding layer, per token.

The ring replaces all of that with a single `slice_update` of one row. Whether
that update actually writes *in place* is the part that did not survive
measurement — see the resource benchmark below. Rotation needs no
change at the model, because attention is permutation-invariant along the key
axis (softmax over keys, then a weighted sum of the matching values — permuting
the `(k, v)` pairs consistently leaves the result unchanged) and keys carry
their RoPE phase from when they were written. The one requirement is that these
layers pass **no mask** on decode, which `Gemma4TextModel::forward` already does
at `seq <= 1`.

Three paths need chronological order back and call `unrotate` first: a
multi-token write landing on a cache that already decoded, `trim_by`, and
`snapshot_clone` (the prefix cache replays a snapshot as a *positional* prefix).

**Measured result: flat on throughput.** Interleaved A/B on
`gemma-4-e2b-it-4bit` (greedy, 6 K-token prompt, alternating base/ring rounds
against separately-built binaries):

| Decode median, tok/s | round 1 | round 2 | round 3 |
|---|---:|---:|---:|
| 400-token generations — baseline | 50.1 | 51.1 | 51.7 |
| 400-token generations — ring | 50.7 | 52.8 | 49.8 |
| **1500-token generations — baseline** | **54.2** | **53.7** | **53.9** |
| **1500-token generations — ring** | **54.2** | **53.7** | **54.2** |

The 400-token rows are too noisy to read (2–6% run-to-run spread swamps
everything). The 1500-token rows are the answer: the spread collapses to ~1%,
between-round drift is thermal and hits both variants alike, and the ring lands
+0.2% — i.e. **flat**. Steady-state memory is unchanged as well, because the
pre-ring path *also* held exactly `window` rows; what the ring removes is
transient allocation the MLX pool was already absorbing.

A follow-up CPU/GPU/RAM benchmark
([mlx-resource-benchmark.md](mlx-resource-benchmark.md)) looked for the cost
side. On E2B it found one — peak RSS ~68 MiB higher, consistently — but that gap
**did not reproduce on E4B**, whose ranges overlap completely, so it was an E2B
artifact rather than a property of the ring. Across both models the honest
summary is no measurable difference in throughput, CPU, GPU or memory.

That still falsifies the premise. Eliminating four buffer copies per layer per
token would not look like a sub-1% decode change with no CPU or GPU movement;
the likely explanation is that `slice_update_axis2` clones the handle before
mutating, so the buffer still has a live reference from `self.keys` at
op-construction time and MLX cannot donate it. **Keep the ring for the simpler
eviction path or revert it, but do not describe it as an optimization.** If it is revisited, the fix to look for is
making the update donate its buffer, and the model to test on is one with a
1024-token window (Gemma-4 26B, Gemma-3) where the copy volume doubles.

[METH-07](https://github.com/drumih/turbo-fieldfare/blob/main/docs/experiments/summaries/09-validation-and-measurement-lessons.md)
is the lesson that applies twice over here: mechanism counters explain results,
they do not replace them — and "fewer copies" was itself only a counter.

What *was* verified is that it is safe: `MLX_BENCH_EXT_DETERMINISM=1` with
temperature pinned to 0 gives `HIT output == full-prefill output ✓`, so the
prefix cache's byte-determinism survives. Note the bench's determinism check is
only meaningful with greedy settings pinned — at the Gemma default temperature
of 0.65 it reports "OUTPUTS DIFFER" for every build, including unmodified ones.

Reordering the key axis does change floating-point accumulation order in SDPA,
so the contract is **token parity, not bit identity**.

## Sampling

The engine sampled off the full logit row: scale by `1/temperature`, then
`categorical!` over the whole vocabulary. On Gemma-4's 262 144-entry vocabulary
that leaves the entire tail reachable at `temperature > 0`.

`sample_with` adds top-k and top-p. Top-k runs first via `argpartition`
(`O(V)`), and the nucleus is then evaluated on those k — not an approximation,
since the nucleus is by construction a prefix of the probability-sorted order,
so `nucleus ∩ top-k` is exactly what the sort-everything version leaves after
its own truncation. Cumulative sums use the **unrenormalized** full-vocabulary
probabilities so the threshold keeps its usual meaning, and `top_p` is evaluated
before temperature is applied — otherwise the nucleus would grow and shrink with
the temperature, which is not what `top_p = 0.95` means.

Defaults come from the **checkpoint's own `generation_config.json`**
(`GenerationDefaults`), not a per-architecture table here, so a new checkpoint
gets its own recommended values with no code change. Precedence is user setting
→ checkpoint → off, where "off" is the historical untruncated draw. Every
failure mode of reading that file (missing, unparseable, wrong-typed) is "no
opinion", never an error, and `do_sample: false` is read as "this checkpoint
wants greedy" rather than as a sampling config.

**This changes sampled output for every local checkpoint that ships those
fields, not just Gemma.** Of the models on this machine:

| Checkpoint | `top_k` | `top_p` |
|---|---:|---:|
| `gemma-4-e2b-it-4bit` (and OptiQ) | 64 | 0.95 |
| `Qwen3-1.7B` | 20 | 0.95 |
| `Qwen3.5-0.8B/2B-OptiQ-4bit` | 20 | 0.80 |

Each is that model card's own recommendation, which is the argument for
following it — but it is a behaviour change, and anyone comparing output against
a pre-change transcript should expect a difference at `temperature > 0`. Greedy
decoding is untouched: `sample_with` returns `argmax` before either filter is
considered, so prefix-cache byte-determinism is unaffected. Setting `top_k: 0`
or `top_p: 1.0` in `settings.json` restores the old untruncated draw.

## What turbo-fieldfare settled

**TurboQuant 4-bit KV — do not build it.** Our note estimated a 4–8× KV saving,
but that was against an *unbounded* FP16 cache. Measured against a windowed one
(KV-07/09): packed K4/V4 was still slower than FP16 after optimization, saved
only ~82 MiB at 4 K, and *grew larger* at longer context because it expands on
all 30 layers while the window is fixed on 25. Quality failed every split — mean
ΔNLL +0.0152, p95 +0.287, top-1 agreement −5.08 pp, top-8 −5.60 pp. The bail in
`gemma4.rs` (`"Gemma-4 KV cache must be FP16"`) is now a decision, not a gap.

**Deeper decode lookahead — do not build it.** PF-03: two pending tiles and four
experts per tile regressed a 527-token prefill from 26.61 s to 33.38 s because
tile dispatches doubled. This matches our own ~1% finding.

Their whole expert-streaming half (mmap vs `pread`, LFU expert cache,
`F_RDADVISE`, persistent MoE workgroups, batched routed MoE) does not transfer:
it exists to run a 14.3 GB model in 2 GB of RAM, and nothing on our path streams
weights.

The measurement discipline is worth borrowing outright: gate by the *contract*
of the change (bit identity for a claimed-lossless change, ΔNLL + top-k
agreement for reordered floating-point work), interleave A/B runs for anything
under ~5% because run order and thermal state can manufacture a 2× swing, and
compute "local speedup × current share" before starting.

## `gemma-4-26b-a4b` (MoE) — implemented, unverified

Structurally a different model from E2B, not a bigger one. Verified against the
published `config.json` and safetensors index:

- **MoE**: 128 routed experts per layer, top-8, `moe_intermediate_size` 704,
  beside a dense shared FFN (`intermediate_size` 2112). The two branches run in
  **parallel** off the same residual and are summed — the routed branch is not
  chained after the dense one. Three extra norms
  (`post_feedforward_layernorm_1`, `pre_feedforward_layernorm_2`,
  `post_feedforward_layernorm_2`) wrap them; the original
  `post_feedforward_layernorm` still wraps the sum.
- **Router**: `rms_norm(x, scale · hidden^-0.5)` → project → top-k → **softmax
  over the k selected scores only** → multiply by `per_expert_scale[idx]`. It
  reads the *unnormalized* block input, not either branch's normalized one.
  `router.proj` is 8-bit where everything around it is 4-bit.
- **k-eq-V**: full-attention layers carry no `v_proj` at all — V is the raw K
  projection, which then takes the no-scale norm and skips RoPE. Confirmed
  against the index: full layers ship no `v_proj` tensor, sliding layers do.
  Those layers also use their own KV head count
  (`num_global_key_value_heads` = 2 against 8).
- **No PLE** (`hidden_size_per_layer_input` = 0) and **no cross-layer KV
  sharing** (`num_kv_shared_layers` = 0) — both of which E2B relies on.

The routed weights arrive already stacked *and* already quantized, so they are
plain parameters excluded from the runtime per-module quantize pass; `gather_qmm`
only needs their group size and bit width, which come from the checkpoint's
top-level `quantization`. Getting either wrong is not a load error — it silently
misreads packed words.

Config parsing is unit-tested against the real 26B field set, and a
half-declared MoE deliberately reads as "no MoE" rather than building a
half-configured one. **Nothing beyond the config has been run**: the text-only
weights are ~14.3 GB and were not downloaded, so the forward pass, the weight
loader's key matching, and the expert matmul shapes are unverified on real
tensors. Treat the 26B entry in `KNOWN_MODELS` as untested until someone runs it.
