# sen-mlx

MLX LLM runtime for [SenClaw](https://github.com/SenClaw/senclaw): chat and
vision inference on Apple Silicon, served as an OpenAI-compatible HTTP API.
The daemon launches this as a child process and talks to it over loopback —
the way LM Studio runs its engines — so the daemon itself links no MLX at all.

Contract: [`senclaw/docs/runtime-protocol.md`](../senclaw/docs/runtime-protocol.md)
(esp. §4.2), built on [`sen-runtime-sdk`](../senclaw/crates/sen-runtime-sdk).

## Requirements

- macOS, Apple Silicon (darwin-arm64). MLX is the only supported backend.
- Rust (stable), Xcode command line tools (Metal, cmake) for the `mlx-sys`
  native build.

## Build

```bash
make build                    # cargo build --release
make test                     # cargo test (single-threaded — see CLAUDE.md)
make package                  # dist/sen-mlx-<version>-darwin-arm64.tar.gz + .sha256
```

Building `mlx-sys` compiles MLX's C++ core and is slow the first time. Point
`CARGO_TARGET_DIR` at a directory shared with a checkout of
[`sen-whisper`](../sen-whisper) (same `mlx-rs`/`mlx-sys` tag) to reuse every
*pure-Rust* dependency between the two — this does not reuse the `mlx-sys`
C++ build itself (separate `Cargo.lock` per repo means separate build-script
fingerprints), so budget one full `mlx-sys` build per repo either way:

```bash
make build CARGO_TARGET_DIR=/path/to/.cargo-target-mlx CARGO_BUILD_JOBS=4
```

## Run

The daemon launches this with `sen-mlx serve --host {host} --port {port}
--model {model_path}` plus the environment in
[`sen_runtime_sdk::env`](../senclaw/crates/sen-runtime-sdk/src/env.rs)
(token, data dir, parent pid, …). Every variable has a standalone default, so
it also runs by hand:

```bash
make run-dev MODEL=~/.senclaw/local-models/mlx-community__Qwen2.5-0.5B-Instruct-4bit
# -> serves on 127.0.0.1:4970, no auth token, no parent watchdog
```

`GET /health` answers 503 while the model loads, 200 once it can generate.
`GET /v1/models` lists the one loaded model; `POST /v1/chat/completions`
(JSON or SSE) serves it — tools, usage, and Gemma-4 vision input all included.

## Install into a daemon's runtime directory

```bash
make install-local            # `senclaw runtime install-local dist/…` if senclaw
                               # is on PATH, else extracts to
                               # ~/.senclaw/runtimes/sen-mlx/<version>/ by hand
```

## Layout

| Path | What |
|---|---|
| `src/main.rs` | CLI, env resolution, wires the model into the SDK's server scaffold |
| `src/provider.rs` | `MlxProvider` — loads the one model eagerly at startup, implements `LlmProvider` |
| `src/settings.rs` | Sampling/KV/prefill settings shared with the daemon (`<models dir>/settings.json`) |
| `src/engine/` | The MLX engine: model architectures, KV cache, sampler, chat templates, vision input, output-stream parsing |
| `senclaw-runtime.json` | The package manifest the daemon reads |
| `docs/` | Design records carried over from the old monorepo (Gemma-4 optimizations, TurboQuant KV, the Space-App extraction this repo continues) |

See [`CLAUDE.md`](CLAUDE.md) for the rules this port depends on, especially
around MLX and test concurrency.
