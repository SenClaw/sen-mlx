# sen-mlx — MLX LLM runtime for SenClaw.
#
# While developing alongside sen-whisper, point CARGO_TARGET_DIR at one shared
# directory: both repos pin mlx-rs/mlx-sys to the same tag, and a shared
# target dir reuses one compile of every pure-Rust dependency between them.
# It does NOT reuse the MLX C++ build itself — separate workspaces, separate
# Cargo.lock files, so mlx-sys's build-script fingerprint differs between the
# two even on the same tag. Budget one full mlx-sys build (~1.2 GB, several
# minutes) per repo, not one total.
#
#   make build CARGO_TARGET_DIR=/path/to/.cargo-target-mlx CARGO_BUILD_JOBS=4

CARGO_TARGET_DIR ?= target
PROFILE ?= release
CARGO_PROFILE_FLAG := $(if $(filter release,$(PROFILE)),--release,)
# Cargo's default profile is *named* `dev` but *outputs* to a `debug/`
# directory (a long-standing Cargo naming quirk) — this is the actual
# directory name under CARGO_TARGET_DIR, used everywhere below instead of
# $(PROFILE) so `make package PROFILE=dev` finds the right binary.
OUT_DIR := $(if $(filter release,$(PROFILE)),release,debug)

ID := sen-mlx
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
# darwin-arm64 only: MLX is Apple Silicon only, and that is the one platform
# this runtime's manifest declares.
PLATFORM := darwin-arm64

DIST := dist
PKG_NAME := $(ID)-$(VERSION)-$(PLATFORM)
PKG_DIR := $(DIST)/$(PKG_NAME)
ARCHIVE := $(DIST)/$(PKG_NAME).tar.gz

export CARGO_TARGET_DIR

.PHONY: build test package install-local run-dev clean

build:
	cargo build $(CARGO_PROFILE_FLAG)

test:
	cargo test

# The C++ build script writes mlx.metallib beside the compiled MLX library,
# not somewhere cargo tracks as a crate output — it has to be found under the
# target dir's build output. Most-recently-modified match wins, in case a
# shared dev target dir holds more than one mlx-sys build directory.
#
# `=` (recursive), not `:=` — this must re-run every time $(METALLIB) is
# referenced, not once when Make parses the file. A `:=` binds at parse time,
# which is *before* the `build` prerequisite below has actually run, so
# `package` would always see the pre-build (missing or stale) state.
METALLIB = $(shell find $(CARGO_TARGET_DIR)/$(OUT_DIR)/build -path '*/out/build/lib/mlx.metallib' 2>/dev/null | xargs -I{} stat -f '%m %N' {} 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

package: build
	@if [ -z "$(METALLIB)" ]; then \
		echo "error: mlx.metallib not found under $(CARGO_TARGET_DIR)/$(OUT_DIR)/build/*/out/build/lib/ — the MLX build did not produce it" >&2; \
		exit 1; \
	fi
	@echo "packaging with metallib: $(METALLIB)"
	rm -rf "$(PKG_DIR)"
	mkdir -p "$(PKG_DIR)/bin"
	cp "$(CARGO_TARGET_DIR)/$(OUT_DIR)/sen-mlx" "$(PKG_DIR)/bin/sen-mlx"
	cp "$(METALLIB)" "$(PKG_DIR)/bin/mlx.metallib"
	cp senclaw-runtime.json "$(PKG_DIR)/senclaw-runtime.json"
	mkdir -p $(DIST)
	tar -C $(DIST) -czf "$(ARCHIVE)" "$(PKG_NAME)"
	# `<hex>  <file name>`, the format `shasum -a 256 -c` reads — the same as every
	# other runtime package, so one command verifies any of them.
	cd $(DIST) && shasum -a 256 "$(notdir $(ARCHIVE))" > "$(notdir $(ARCHIVE)).sha256"
	@echo "packaged $(ARCHIVE)"

# `senclaw runtime install-local dist/<archive>` when a `senclaw` binary is on
# PATH, else extract by hand into ~/.senclaw/runtimes/<id>/<version>/, manifest
# written LAST so an interrupted install has no manifest (never "installed").
install-local: package
	@if command -v senclaw >/dev/null 2>&1; then \
		senclaw runtime install-local "$(ARCHIVE)"; \
	else \
		echo "senclaw not on PATH — installing into ~/.senclaw/runtimes/$(ID)/$(VERSION)/ by hand"; \
		dest="$$HOME/.senclaw/runtimes/$(ID)/$(VERSION)"; \
		rm -rf "$$dest" && mkdir -p "$$dest"; \
		cp -r "$(PKG_DIR)/bin" "$$dest/bin"; \
		cp "$(PKG_DIR)/senclaw-runtime.json" "$$dest/senclaw-runtime.json"; \
		echo "installed to $$dest"; \
	fi

# Standalone serve on a fixed dev port — no token (SENCLAW_RUNTIME_TOKEN
# unset), no parent watchdog (SENCLAW_PARENT_PID unset). Requires a model:
#   make run-dev MODEL=~/.senclaw/local-models/mlx-community__Qwen2.5-0.5B-Instruct-4bit
run-dev: build
	cargo run $(CARGO_PROFILE_FLAG) -- serve --host 127.0.0.1 --port 4970 $(if $(MODEL),--model "$(MODEL)",)

clean:
	rm -rf $(DIST)
