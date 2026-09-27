# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

`sen-mlx` is SenClaw's MLX LLM runtime: a standalone binary the SenClaw daemon
(`../senclaw`) launches as a child process and drives over loopback HTTP —
never in-process, never linked into the daemon. It loads exactly **one**
model per process, given on the command line (`--model {model_path}`), and
serves it as an OpenAI-compatible provider.

This repo is the continuation of the old SemaClaw monorepo's `apps/mlx-lm`
Space App, which was itself the continuation of the daemon's original
`src/local_model/`. The engine (`src/engine/`) is a close-to-verbatim port;
what changed each time is the surrounding shape:

1. `src/local_model/` (daemon) → `apps/mlx-lm` (Space App): model management
   REST, a manifest, lazy multi-model loading with an idle sweeper.
2. `apps/mlx-lm` → this repo (runtime): no manifest, no model-management REST
   (the daemon owns the model library — protocol §6.1), no model switching —
   one process, one model, loaded eagerly at startup because the runtime
   protocol's health gate is minutes wide, not the Space-App's 30 seconds.

Contract: [`../senclaw/docs/runtime-protocol.md`](../senclaw/docs/runtime-protocol.md)
§4.2. SDK: [`../senclaw/crates/sen-runtime-sdk`](../senclaw/crates/sen-runtime-sdk)
(path dependency, frozen — read-only from here). OpenAI wire:
[`../senclaw/app-space-sdk`](../senclaw/app-space-sdk)'s `llm` module (path
dependency, also read-only) — `app_space_sdk::llm::openai_router` renders
`/v1/models` + `/v1/chat/completions` from the semantic `LlmProvider` trait,
so this repo never hand-writes OpenAI SSE framing.

## Build & run

```bash
cargo build --release                        # or `make build`
cargo test                                   # or `make test` — see below
make package                                 # dist/sen-mlx-<version>-darwin-arm64.tar.gz
make run-dev MODEL=<path to an MLX checkpoint dir>
```

Share `CARGO_TARGET_DIR` with a `sen-whisper` checkout while developing both:
they pin `mlx-rs`/`mlx-sys` to the same tag, and a shared target dir reuses
one compile of every *pure-Rust* dependency (tokio, candle, tokenizers, …)
between them. It does **not** reuse the MLX C++ build itself — `sen-mlx` and
`sen-whisper` are separate workspaces with separate `Cargo.lock` files, so
`mlx-sys`'s build-script fingerprint (and therefore its `mlx-sys-<hash>`
output directory) differs between the two even on the same tag. Budget one
full `mlx-sys` build (~1.2 GB, several minutes) per repo, not one total.

`[profile.dev.build-override] debug = false` (in `Cargo.toml`) keeps that one
build stable across `build`/`check`/`test`: Cargo's default debuginfo
handling for build-script units otherwise varies by *which* of those three you
run, which flips `mlx-sys`'s fingerprint and silently triggers a fresh
from-scratch C++ rebuild on the next invocation even when nothing changed. Do
not remove this override.

In a disk-constrained environment, prefix a routine (non-rebuild-intending)
`cargo build`/`test`/`check` with `CMAKE=/usr/bin/false`: if the cached
`mlx-sys` build is still valid it is reused and the override is never
consulted; if Cargo decides a rebuild is needed, the build script fails fast
against the fake `cmake` instead of silently spending ~1.2 GB. Unset it (or
just don't set it) for the one invocation where a real `mlx-sys` build is
actually intended.

## Architecture

- `src/main.rs` — the `serve` subcommand: parses `--host`/`--port`/`--model`
  (also `SENCLAW_MODEL_PATH`), builds a [`Readiness`], hands the router from
  `provider.rs` to `sen_runtime_sdk::server::serve`.
- `src/provider.rs` — `MlxProvider`, the single-model `LlmProvider`. Validates
  the checkpoint and builds its `ModelCard` synchronously (cheap — reads
  `config.json`, no weights), then loads weights in the background and flips
  `Readiness` when the load resolves. `chat()` awaits that load rather than
  re-triggering it, so a request racing a hand-started dev server waits
  instead of double-loading.
- `src/settings.rs` — `Settings`: sampling/KV/prefill knobs read from
  `<SENCLAW_LOCAL_MODELS_DIR>/settings.json`. Same struct, same on-disk shape
  as the old `local_model_core::settings::Settings` — **never add
  `rename_all`**, see "Local models left the daemon" below.
- `src/engine/` — the MLX engine itself, ported close to verbatim:
  - `mlx_native.rs` — `MlxNativeEngine`: architecture dispatch, load, generate,
    the process-wide MLX serial lock, KV-window sizing.
  - `mlx_lm/models/` — one file per supported architecture (Qwen3, Qwen3.5,
    Llama/Qwen2, Gemma-2/3/4, DeepSeek-V2, Mamba-2, Falcon-Mamba, Bonsai-Q1,
    Ouro).
  - `mlx_lm/{cache,sampling,prefix_cache}.rs` — KV cache (incl. TurboQuant),
    top-k/nucleus sampling, the prefix cache that turns turn-2+ prefill from
    tens of seconds into a few.
  - `chat_template_openai.rs`, `mlx_prompt.rs`, `stream_parser.rs`,
    `thinking_parse.rs`, `image_input.rs` — chat-template rendering and
    special-token resolution, output-stream marker parsing (think/tool_call
    per model dialect), Gemma-4 vision preprocessing.
  - `mod.rs` — `supports()` / `has_vision()` / `context_length()`: read a
    checkpoint's `config.json` to decide if this engine can load it, without
    touching weights.

  Dropped on the way in from `apps/mlx-lm`: the known-models catalog and the
  HF downloader/store (`local_model_core::{store,download,api}`) — the daemon
  owns the model library now, and this runtime is handed exactly one path.

## Testing

`cargo test` — unit tests co-located in `#[cfg(test)]` modules, plus
`tests/manifest.rs` (the manifest parses with the SDK and its version tracks
`Cargo.toml`). A few tests read an existing checkpoint from the real
`~/.senclaw/local-models/` (read-only) and skip themselves when it is absent.

### MLX and test concurrency

Rules for Claude:

- **MLX's Metal command queue is not safe for concurrent dispatch from
  multiple OS threads in one process.** `mlx_native::mlx_serial_lock` covers
  the engine's own load/generate/unload path; it does **not** cover a unit
  test that builds and runs a small MLX graph directly (most of
  `engine::mlx_lm::models::*::tests` and `engine::mlx_lm::{cache,sampling}::tests`).
  The default parallel test harness intermittently **SIGSEGVs** running
  several such tests on different threads at once — reproduced while porting
  this repo. [`.cargo/config.toml`](.cargo/config.toml) sets
  `RUST_TEST_THREADS=1` so a plain `cargo test` is safe; do not remove it, and
  do not "fix" a flaky-looking MLX test by parallelizing.
- **This is a same-process constraint, not a same-machine one.** Two MLX
  *processes* generating concurrently on the same GPU are fine — Metal
  isolates command queues per process — which is exactly what the daemon does
  running this binary and another model's `sen-mlx` process side by side.

## Local models left the daemon (background, carried from the old monorepo)

The full history: `src/local_model/` (daemon, ~30k lines) → `apps/mlx-lm`
(Space App) → here. Rules that still apply at this stop:

- **Do not reintroduce an MLX dependency into the daemon.** The measurement
  that unlocked every step of this split: two MLX processes generating
  concurrently on the same `Device(gpu, 0)` run clean — Metal isolates command
  queues per process. In-process concurrency is still unsafe (see "MLX and
  test concurrency" above); each MLX binary keeps its own process-wide serial
  lock (`mlx_serial_lock` here, a separate copy in `sen-whisper`).
- **`settings.json` is snake_case, and this runtime and the daemon's other
  local-model history read the same file.** A `rename_all` on
  `settings::Settings` would parse every existing file into all-`None`
  silently — not an error, every setting quietly back to default, the worst
  way for a config format to change. A test pins the exact on-disk shape.
- **`ModelCard::vision` is required and comes from the checkpoint's config**
  (`engine::has_vision`), never from the model id. A local id like
  `mlx-community/Qwen3.5-2B-OptiQ-4bit` matches no vendor pattern, so a
  name-based guess is right or wrong by accident — and the wrong direction is
  a hard 400 that fails the whole turn upstream.
- **Weights load once, eagerly, at startup — not lazily on first request.**
  This is the one behavior that changed shape from the Space-App generation:
  there the daemon's 30-second health-gate budget forced a lazy load on first
  `/v1/chat/completions`; the runtime protocol's `health.startupTimeoutSecs`
  is minutes wide, so `main.rs` starts the load in the background immediately
  and `/health` answers 503 until `Readiness::set_ready()`.
- **The workspace pin that must never drift is `mlx-rs`/`mlx-sys`.** This repo
  and `sen-whisper` both build the MLX C++ runtime, and each pays for its own
  full build regardless (see "Build & run" above — a shared `CARGO_TARGET_DIR`
  only reuses the pure-Rust dependencies between them, not this one). Drifted
  off the same tag, the two repos' `mlx-c` sources themselves diverge too,
  which is a correctness risk on top of the disk cost.

## Gemma 4 on the native MLX path

Sliding-window layers keep a decode-time KV **ring**
([`engine/mlx_lm/cache.rs`](src/engine/mlx_lm/cache.rs) `ring_head`): decode
writes one row via `slice_update` instead of evicting with a tail slice and
re-growing, which the pre-ring path did on every token past the window.
Measured on both Gemma-4 E2B and E4B it is **neutral, not an optimization**
(<1% decode, no CPU or GPU change), so treat it as a simpler eviction path, not
a fast one. Rotation is safe because attention is permutation-invariant along
the key axis — but *only* while these layers pass **no mask** on decode, which
`Gemma4TextModel::forward` does at `seq <= 1`. Three paths need chronological
order back and call `unrotate` first: a multi-token write on a cache that
already decoded, `trim_by`, and `snapshot_clone` (the prefix cache replays a
snapshot as a positional prefix). Reordering the key axis changes
floating-point accumulation order, so the contract is **token parity, not bit
identity**.

Sampling goes through [`engine/mlx_lm/sampling.rs`](src/engine/mlx_lm/sampling.rs)
`sample_with` (top-k then nucleus on the survivors). Defaults come from the
**checkpoint's own `generation_config.json`**, never a per-architecture table:
precedence is user setting → checkpoint → off, where off is the historical
untruncated full-vocabulary draw. This moves sampled output for **every**
local checkpoint shipping those fields, not just Gemma — Qwen3 ships
`top_k: 20`, Qwen3.5 ships `20 / 0.80`. Greedy is untouched (`argmax`
short-circuits before either filter), so prefix-cache determinism is
unaffected.

Rules for Claude:

- **`MLX_BENCH_EXT_DETERMINISM=1` is only meaningful with `temperature: 0`
  pinned in the bench cell's own `settings.json`.** At the Gemma default of
  0.65 it reports "OUTPUTS DIFFER" for every build including unmodified ones —
  that is the sampler being stochastic, not a determinism regression.
- **Never claim a decode win from a single ordered pair of runs, and generate
  enough tokens to see past the noise.** At 400-token generations the KV-ring
  A/B spread 2–6% and read as inconclusive; at 1500 tokens the spread
  collapsed to ~1% and the answer appeared — flat.
- **Measure RAM as well as tok/s, and on more than one model.** The KV ring
  looked neutral on throughput, then appeared to cost ~68 MiB of peak RSS on
  E2B — consistently, across six runs. On E4B that gap did not reproduce at
  all, which is what demoted it from "a real cost" to "an E2B artifact". A
  finding from one checkpoint is a hypothesis:
  [docs/mlx-resource-benchmark.md](docs/mlx-resource-benchmark.md).
- **E4B needs no code of its own.** `gemma-4-e4b-it-4bit` is the same dense
  Gemma-4 path as E2B (42 layers, hidden 2560, 2 KV heads, 18 KV-shared) and
  loads with zero unmatched keys — ~1.7× slower than E2B for ~1.5 GB more peak
  MLX memory.
- **TurboQuant 4-bit KV for Gemma-4 is rejected, not missing.** The
  `Exception` in `gemma4.rs` is a decision: measured against a windowed FP16
  cache it is slower, saves ~82 MiB at 4 K, grows *larger* at long context,
  and fails quality (top-1 agreement −5.08 pp).
- **`gemma-4-26b-a4b` is implemented but never run.** Config parsing is
  tested; the forward pass, loader key matching and expert matmul shapes are
  unverified on real tensors (~14.3 GB, not downloaded).

Full record, including what transfers from
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) and what
does not: [docs/gemma4-local-optimizations.md](docs/gemma4-local-optimizations.md).
TurboQuant KV design: [docs/mlx-rs-turboquant-native-runtime.md](docs/mlx-rs-turboquant-native-runtime.md).
The two earlier moves this repo continues:
[docs/local-gemma-mlx-runtime.md](docs/local-gemma-mlx-runtime.md) (daemon →
Space App) and [docs/local-model-space-app-extraction.md](docs/local-model-space-app-extraction.md).

## Porting conventions

- `anyhow::Result` for fallible functions; `thiserror` only where the SDK or a
  caller needs a typed error (none of this engine's own code does today).
- Comments explain *why*, not *what* — match that density in new code.
- No plan ids, phase numbers, or finding codes in code, comments, test names,
  or commit messages: explain the invariant or behavior directly.
