//! Qwen3 inference stack vendored from [oxiglade/mlx-rs `mlx-lm`](https://github.com/oxiglade/mlx-rs/tree/main/mlx-lm).
//! Built on **`mlx-rs` only** as an external ML dependency (no `mlx-lm` crate in `Cargo.toml`).

pub mod cache;
pub mod error;
pub mod models;
pub mod prefix_cache;
pub mod sampling;
pub mod utils;
