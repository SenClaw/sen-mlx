# MLX native path — CPU / GPU / RAM benchmark

`scripts/mlx_bench.sh` answers *how fast*. This one answers *at what cost*, and
is the record of the first run of it, taken to check what the turbo-fieldfare-
driven changes (see [gemma4-local-optimizations.md](gemma4-local-optimizations.md))
actually cost in resources rather than only in tokens per second.

Harness: [`scripts/mlx_resource_bench.py`](../scripts/mlx_resource_bench.py). It
runs `examples/mlx_bench` under a 2 Hz sampler that reads process RSS, CPU% and
cumulative CPU time from `ps`, and GPU busy-percent plus GPU-resident memory
from the Apple accelerator's IOKit counters (`ioreg -c AGXAccelerator`) — which,
unlike `powermetrics`, need no root. It also scrapes the MLX allocator figures
the bench prints. Variants are interleaved *within* each scenario so thermal
drift cannot be assigned to one of them.

```bash
scripts/mlx_resource_bench.py \
  --bin noring=/path/to/mlx_bench_noring --bin ring=target/release/examples/mlx_bench \
  --model-dir ~/.senclaw/local-models/mlx-community__gemma-4-e2b-it-4bit \
  --model-id mlx-community/gemma-4-e2b-it-4bit \
  --iters 2 --rounds 2 --out bench-out/
```

## Run

- **Host** Mac16,8 (Apple M4 Pro, 12 CPU cores / 16 GPU cores)
- **Models** `mlx-community/gemma-4-e2b-it-4bit` and
  `mlx-community/gemma-4-e4b-it-4bit`, greedy decode, `max_kv_tokens` 32 000,
  `mlx_kv_cache_bits` 0
- **Variants** `ring` = current build; `noring` = same tree with only the
  sliding-window ring dispatch compiled out
- 2 timed turns per run (plus a warm-up), 2 interleaved rounds, 2026-08-15
- **GPU idle floor before the run**: 0.0% busy, 532 MiB GPU-resident

## Resource profile of the current build — Gemma 4 E2B

Means across both rounds, `ring` variant:

| scenario | prompt→gen | decode tok/s | TTFT s | CPU % | GPU busy % | RSS peak MiB | MLX active / peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| short | 500→300 | 56.9 | 1.02 | 50.5 | 74.4 | 2 920 | 2 622 / 4 216 |
| medium | 4 000→800 | 55.2 | 0.40 | 38.8 | 81.2 | 2 923 | 2 928 / 4 534 |
| long | 12 000→1 500 | 51.5 | 0.44 | 20.0 | 90.6 | 2 938 | 3 168 / 4 930 |

Three things worth reading off this table, none of which throughput alone shows:

**The GPU is not saturated at short prompts.** Busy-percent climbs 74% → 81% →
91% with prompt length, so on a short prompt the GPU is idle roughly a quarter
of the time. That is the host-dispatch cost of a per-token decode loop, not a
kernel that is too slow, and it is where a decode-side win would have to come
from. Note this counter is whole-device and system-wide, so it is only readable
against the idle floor above.

**CPU falls as the GPU rises** — 50% → 39% → 20% of a single core. Even at its
worst the process uses half a core out of twelve, so the host is nowhere near a
bottleneck; the short-prompt figure is higher only because tokenization, prompt
synthesis and template rendering are a larger share of a short run.

**Memory is flat in prompt length, and it is the weights that dominate.** RSS
moves 2 920 → 2 938 MiB across a 24× change in prompt size, because MLX active
memory (2 622 → 3 168 MiB) is mostly the 4-bit weights; KV growth is the ~550
MiB difference. The MLX *peak* figure (4 216 → 4 930 MiB) is the number that
matters for "will this fit", and it is roughly 1.7 GB above steady state.

## Resource profile — Gemma 4 E4B

Same harness, same scenarios, `mlx-community/gemma-4-e4b-it-4bit` (5.2 GB on
disk, 42 layers, hidden 2560, 2 KV heads, 18 KV-shared layers). Means across
both rounds, `ring` variant:

| scenario | prompt→gen | decode tok/s | TTFT s | CPU % | GPU busy % | RSS peak MiB | MLX active / peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| short | 500→300 | 34.0 | 2.18 | 43.0 | 79.8 | 4 391 | 4 191 / 5 813 |
| medium | 4 000→800 | 32.6 | 0.69 | 33.4 | 85.4 | 4 432 | 5 108 / 6 257 |
| long | 12 000→1 500 | 29.9 | 0.80 | 20.3 | 91.6 | 4 407 | 5 813 / 7 259 |

Against E2B: **1.7× slower** (34 vs 57 tok/s short, 30 vs 52 long) for **+1.5 GB
peak MLX memory** (7 259 vs 4 930 MiB at long context) and +1.5 GB RSS. It loads
with **zero unmatched keys** and produces fluent, byte-deterministic output, so
the existing dense Gemma-4 path covers E4B with no code change; the only edit
was adding it to `KNOWN_MODELS`.

The two shapes that carry over from E2B unchanged are the interesting ones:

- **GPU busy-percent climbs the same way** — 80% → 85% → 92% against E2B's 74% →
  81% → 91%. The short-prompt idle gap is therefore structural (per-token host
  dispatch), not a property of one model's size.
- **CPU falls the same way** — 43% → 33% → 20% of one core, against E2B's 50% →
  39% → 20%. The host is not the bottleneck at either size.

## Does the KV ring pay for itself?

Interleaved `noring` vs `ring`, mean of two rounds, on both models:

| scenario | decode (E2B / E4B) | CPU % (E2B / E4B) | GPU % (E2B / E4B) | RSS peak (E2B / E4B) |
|---|---:|---:|---:|---:|
| short | +0.44% / +0.89% | +5.1% / +2.0% | −0.4% / +0.8% | **+2.5%** / −0.8% |
| medium | +0.73% / +0.93% | −4.6% / −2.0% | +0.5% / −0.3% | **+1.3%** / +0.9% |
| long | +0.29% / −0.66% | −2.8% / −2.4% | −0.1% / −0.9% | **+2.6%** / +0.9% |

- **Decode**: small and not stable — five of six E2B pairs positive at ~+0.4%,
  E4B positive twice and negative once. Consistent with the separate
  1 500-token throughput A/B (+0.2%). Whatever is there is under 1%.
- **CPU and GPU**: no signal on either model. Both columns change sign between
  scenarios, which is what noise looks like.
- **RSS**: this is why the second model was worth running. On E2B the ring cost
  a consistent **+68 MiB** at peak (medians 2 923 vs 2 855, ranges barely
  overlapping) and it looked like the one real finding. On E4B it **does not
  reproduce at all** — medians 4 430 vs 4 399, ranges fully overlapping
  (4 353–4 444 against 4 355–4 431), and the sign flips between scenarios. E4B
  has *more* sliding KV than E2B (20 owned sliding layers × 2 KV heads against
  12 × 1), so if the gap were caused by the ring's buffers it should have grown,
  not vanished.

### What that says about the mechanism

The ring was built on the premise that replacing "tail-slice, trim, allocate
zeros, concatenate" with a single `slice_update` removes the per-token copies.
Nothing in either model's numbers looks like eliminating four buffer copies per
layer per token: decode moves under 1%, CPU and GPU not at all.

The likely reason the copies did not disappear is copy-on-write.
`slice_update_axis2` clones the array handle and then mutates through it, so at
op-construction time the underlying buffer still has a live reference from
`self.keys` — which means MLX cannot donate it and allocates a fresh buffer
anyway. That is a hypothesis consistent with the data, not something these runs
prove; proving it needs allocator instrumentation.

**Conclusion: the ring is neutral, not an optimization.** The E2B RSS penalty
that first looked like a real cost did not survive contact with a second model,
so the fair summary across both is "no measurable difference in any of
throughput, CPU, GPU or memory". Keep it for the simpler eviction path, or
revert it — but do not describe it as fast. If it is revisited, the place to
look is making the update actually donate its buffer, and the model to test on
is one with a 1 024-token window (Gemma-4 26B, Gemma-3), where the copy volume
doubles.

### A measurement bug this run exposed

The first E4B run reported `-0.0%` CPU for one cell. `ps` can still answer for a
pid that is mid-exit, with every field zeroed; as the *last* sample that makes
the cumulative CPU-time delta negative. The harness now drops any sample with
`rss <= 0` and refuses non-monotonic CPU-time readings. The E4B figures above
are recomputed from the stored per-sample CSVs with that filter applied, which
is the reason to keep the raw CSVs rather than only the summary.

## Reproducing

Raw per-sample CSVs, per-run bench logs, `results.json` and the generated
`report.md` are written to `--out`. The harness needs no root and no
instrumentation build; a plain `cargo build --release --features local-mlx
--example mlx_bench` is enough.
