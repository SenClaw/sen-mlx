//! In-memory prefix cache for multi-turn KV reuse.
//!
//! Multi-turn agent chats repeatedly send prompts of the form
//! `[system + N tools + turn1 + turn2 + … + turnK]`. For a tools-heavy
//! deployment (≥ 100 MCP tools), the **system + tools** block alone is 12–15 K
//! tokens — repeating prefill for it every turn is the dominant latency cost.
//!
//! `PrefixCache` stores the post-prefill [`KvCache`] snapshot keyed by the
//! exact token-id prefix that produced it. On the next turn the engine looks
//! for the **longest** cached prefix that matches the incoming prompt and
//! restores those KV writes verbatim — prefill then only runs for the
//! *suffix* (the new user / assistant / tool turn).
//!
//! ## Why "in-memory only" (not persisted to disk)
//!
//! - KV is 2–6 GB per cached prefix (`max_kv_tokens × layers × heads × dim
//!   × 2 B`). Disk IO on Apple Silicon swaps a 60 ms prefill for a 200–500 ms
//!   read — net negative.
//! - On `idle_unload`, the whole [`Loaded`] is dropped, which drops the
//!   prefix cache too. Re-loading after a long idle means re-running prefill
//!   once; that's correct behaviour (model weights aren't in memory either).
//! - Cross-restart persistence would require versioning the cache against
//!   model weights + tokenizer + chat-template hashes — not worth the
//!   complexity for a per-machine convenience.
//!
//! ## Cache invalidation
//!
//! Entries are evicted LRU when [`MAX_ENTRIES`] is exceeded. Each generation
//! turn stores its post-prefill state under the prompt prefix (excluding the
//! final `<|im_start|>assistant\n` generation suffix, so different turns of
//! the same conversation share their common prefix).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::cache::KvCache;

/// Maximum number of cached prefix entries.
///
/// **RAM cost per entry** — Each entry pins the per-layer KV `Array` handles
/// from the live cache. Because `trim_by` slices via `slice_axis2` (a view
/// op, zero-copy in mlx), the snapshot SHARES storage with the live cache's
/// preallocated buffer (`max_kv_tokens × layers × 2 × 2 B`). For typical
/// Qwen3-4B at `max_kv_tokens=32000`: each entry pins ~4.4 GB until evicted.
///
/// **Why 2** — linear multi-turn chat only needs the latest snapshot;
/// previous-turn snapshots are strict prefixes of the new one and get
/// dropped by [`PrefixCache::store`]'s semantic eviction. The second slot
/// is reserved as a safety net for one-step-back retries / branching
/// conversations.
///
/// **Memory ceiling** — bounded by [`MAX_TOTAL_BYTES`] regardless of count;
/// snapshots over budget are evicted oldest-first.
pub const MAX_ENTRIES: usize = 2;

/// Hard ceiling on combined snapshot bytes. Once exceeded, oldest entries are
/// evicted until under budget — protects from runaway RAM when multiple
/// distinct conversations cycle through the cache.
///
/// 6 GB sized for a typical M-series unified-memory setup with Qwen3-4B-4bit:
/// `model 2.3 GB + 2 × snapshot 2.2 GB = 6.7 GB` covers two parallel chats.
/// Raise if you see frequent `evict_for_bytes_budget` warnings.
pub const MAX_TOTAL_BYTES: usize = 6 * 1024 * 1024 * 1024;

/// Idle TTL — entries with no `find_longest_match` hit for this long are
/// evicted by [`PrefixCache::evict_idle`]. The point is to release the
/// ~2.5 GB per-entry KV pin when a conversation has gone quiet but the
/// model itself is still loaded (model unload runs on a separate timer
/// configured via `idle_unload_secs`).
///
/// 5 minutes covers normal "user reads, then replies" pauses without
/// rebuilding the prefix on every reply, but evicts forgotten chats.
pub const IDLE_TTL_SECS: u64 = 5 * 60;

/// Minimum prefix length to bother caching. Short prefixes (a few hundred
/// tokens) prefill in well under a second — the cache lookup/clone overhead
/// would dwarf the savings.
pub const MIN_PREFIX_LEN: usize = 1024;

/// One cached `(prompt-prefix, post-prefill KV state)` pair.
pub struct PrefixCacheEntry {
    /// The exact prompt token ids that produced `caches`.
    pub tokens: Vec<u32>,
    /// Per-layer KV snapshot taken immediately after `forward_chunked` (or
    /// the single-shot prefill) finished. Same layer order as the live
    /// `Vec<Option<KvCache>>` that callers pass into the model's forward.
    pub caches: Vec<Option<KvCache>>,
    /// RoPE absolute position after the cached prefill — caller resumes
    /// decode at `rope_offset = tokens.len()` after restore.
    pub rope_offset: usize,
    /// Wall-clock timestamp of the most recent `store` or
    /// `find_longest_match` hit. Drives [`PrefixCache::evict_idle`].
    pub last_used: Instant,
}

/// Small fixed-capacity LRU. We could use the `lru` crate but a 4-entry
/// `VecDeque` walked linearly is fast enough and avoids the dependency.
pub struct PrefixCache {
    entries: VecDeque<PrefixCacheEntry>,
    /// Entry-count cap (defaults to [`MAX_ENTRIES`]). Lowered for looped LMs
    /// whose snapshots each pin `total_ut_steps`× the KV of a dense model.
    max_entries: usize,
    /// Byte cap (defaults to [`MAX_TOTAL_BYTES`]). Lowered for looped LMs.
    max_total_bytes: usize,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::with_limits(MAX_ENTRIES, MAX_TOTAL_BYTES)
    }

    /// Construct with explicit caps. Used to shrink the cache for architectures
    /// with oversized per-snapshot KV (e.g. Ouro's 192-slot looped cache), so
    /// the prefix cache doesn't stack several 4× snapshots on top of live KV.
    pub fn with_limits(max_entries: usize, max_total_bytes: usize) -> Self {
        let max_entries = max_entries.max(1);
        Self {
            entries: VecDeque::with_capacity(max_entries),
            max_entries,
            max_total_bytes,
        }
    }

    /// Find the entry whose cached `tokens` is the **longest** prefix of
    /// `prompt`. Returns `None` when no entry shares ≥ [`MIN_PREFIX_LEN`]
    /// tokens with `prompt` (cache miss). Takes `&mut self` so the
    /// matched entry's `last_used` can be touched (drives idle TTL
    /// eviction so frequently-hit prefixes survive while stale ones age out).
    ///
    /// O(`entries × min(prompt_len, cached_len)`) — fine for ≤ 4 entries.
    pub fn find_longest_match(&mut self, prompt: &[u32]) -> Option<&PrefixCacheEntry> {
        let mut best_idx: Option<usize> = None;
        let mut best_len = MIN_PREFIX_LEN.saturating_sub(1);
        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.tokens.len() > prompt.len() {
                continue;
            }
            if entry.tokens.len() <= best_len {
                continue;
            }
            if entry.tokens.as_slice() == &prompt[..entry.tokens.len()] {
                best_len = entry.tokens.len();
                best_idx = Some(idx);
            }
        }
        best_idx.map(|i| {
            self.entries[i].last_used = Instant::now();
            &self.entries[i]
        })
    }

    /// Drop entries whose `last_used` is older than [`IDLE_TTL_SECS`].
    /// Returns the number of evictions + bytes freed (for logging).
    ///
    /// Call at the start of every `generate_with_cache` so RAM is reclaimed
    /// the moment the operator returns after a long pause (5+ min) — without
    /// this, a single forgotten prefix would pin ~2.5 GB until the model
    /// itself unloaded. Cheap (O(entries), no allocation when nothing
    /// expires).
    pub fn evict_idle(&mut self) -> (usize, usize) {
        let cutoff = Instant::now() - Duration::from_secs(IDLE_TTL_SECS);
        let mut freed_bytes = 0usize;
        let mut evicted = 0usize;
        self.entries.retain(|e| {
            if e.last_used < cutoff {
                freed_bytes += entry_bytes(e);
                evicted += 1;
                false
            } else {
                true
            }
        });
        if evicted > 0 {
            tracing::info!(
                "[prefix-cache] idle eviction: dropped {} entry/entries (TTL {}s), freed {}",
                evicted,
                IDLE_TTL_SECS,
                fmt_bytes(freed_bytes),
            );
        }
        (evicted, freed_bytes)
    }

    /// Insert `(tokens, caches, rope_offset)`. Three eviction passes:
    ///
    /// 1. **Exact-duplicate drop** — same key replaces (LRU touch).
    /// 2. **Semantic eviction** — entries whose `tokens` are a strict prefix
    ///    of the new entry are redundant (the new entry dominates them) and
    ///    get evicted. This is the dominant case for linear multi-turn chat:
    ///    turn N+1's prefix is turn N's prefix + tool/user content, so the
    ///    turn N entry is always strictly contained. Without this, two
    ///    consecutive turns of a 14 K-token tools-heavy prompt would pin
    ///    ~8.8 GB of KV state in the cache.
    /// 3. **Count + byte budget** — fall back to LRU eviction when over
    ///    [`MAX_ENTRIES`] or [`MAX_TOTAL_BYTES`].
    pub fn store(&mut self, tokens: Vec<u32>, caches: Vec<Option<KvCache>>, rope_offset: usize) {
        // 1. Drop exact duplicate (will re-insert at front below as LRU touch).
        self.entries.retain(|e| e.tokens != tokens);
        // 2. Semantic eviction: any existing entry whose token list is a
        //    strict prefix of the new one is dominated and can be dropped.
        //    Counts entries removed for logging.
        let before = self.entries.len();
        self.entries
            .retain(|e| !is_strict_prefix_of(&e.tokens, &tokens));
        let dropped_dominated = before - self.entries.len();
        if dropped_dominated > 0 {
            tracing::debug!(
                "[prefix-cache] semantic eviction dropped {} dominated entry/entries",
                dropped_dominated
            );
        }
        // 3a. Count cap.
        while self.entries.len() >= self.max_entries {
            self.entries.pop_back();
        }
        self.entries.push_front(PrefixCacheEntry {
            tokens,
            caches,
            rope_offset,
            last_used: Instant::now(),
        });
        // 3b. Bytes cap — evict oldest until under budget. Guard against
        //     the lone newest entry being itself over budget (don't loop forever).
        while self.entries.len() > 1 && self.total_bytes() > self.max_total_bytes {
            if let Some(evicted) = self.entries.pop_back() {
                tracing::info!(
                    "[prefix-cache] bytes budget exceeded ({} > {}), evicted entry of {} tokens",
                    fmt_bytes(self.total_bytes() + entry_bytes(&evicted)),
                    fmt_bytes(self.max_total_bytes),
                    evicted.tokens.len(),
                );
            }
        }
    }

    /// Combined byte footprint of all entries (approx — sum of layer caches'
    /// `approx_bytes` which counts the **pinned** buffer storage, even for
    /// snapshots that logically only need stored_len tokens).
    pub fn total_bytes(&self) -> usize {
        self.entries.iter().map(entry_bytes).sum()
    }

    /// Drop every cached snapshot, releasing all pinned KV `Array` handles.
    /// Returns the (approximate) bytes freed. Used to reclaim KV on demand —
    /// memory-pressure eviction or an explicit `release_kv_cache()` — while the
    /// model weights stay loaded. The dropped `Array`s return their MLX buffers
    /// to the pool; the caller should follow with `mlx_clear_cache()` to push
    /// the freed pool back to the OS.
    pub fn clear(&mut self) -> usize {
        let freed = self.total_bytes();
        self.entries.clear();
        freed
    }

    /// Snapshot all cache variants in `live` via [`KvCache::try_snapshot`].
    /// Returns `None` if any layer's cache cannot be cloned (e.g. TurboQuant
    /// in active state) — in that case the caller skips caching this turn.
    pub fn snapshot_layers(live: &[Option<KvCache>]) -> Option<Vec<Option<KvCache>>> {
        live.iter()
            .map(|opt| match opt {
                Some(c) => c.try_snapshot().map(Some),
                None => Some(None),
            })
            .collect()
    }

    /// Snapshot + trim the last `trim` tokens from each layer's KV. Used to
    /// store a prefix cache entry whose key excludes the assistant
    /// generation suffix — so the next turn's prompt (which lacks that
    /// suffix at the same position) still hits the cache.
    ///
    /// **Compact** — each layer's KV is then materialized into independent
    /// storage sized exactly to `stored_len` (via
    /// [`KvCache::compact_to_stored_len`]). Without this, the snapshot would
    /// share the live cache's full preallocated buffer (~4.4 GB for Qwen3-4B
    /// + 32 K KV) and the snapshot would pin ~4.4 GB even though it
    /// logically only references the first ~14 K tokens. Post-compact each
    /// snapshot pins only `stored_len × 4096 × 36 × 2 B` ≈ 2.2 GB.
    pub fn snapshot_layers_trimmed(
        live: &[Option<KvCache>],
        trim: usize,
    ) -> Option<Vec<Option<KvCache>>> {
        let mut snap = Self::snapshot_layers(live)?;
        if trim > 0 {
            for slot in snap.iter_mut().flatten() {
                slot.trim_by(trim);
            }
        }
        for slot in snap.iter_mut().flatten() {
            slot.compact_to_stored_len();
        }
        Some(snap)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

/// `a` is a STRICT prefix of `b` (shorter and matches all of `a`'s positions).
fn is_strict_prefix_of(a: &[u32], b: &[u32]) -> bool {
    a.len() < b.len() && a == &b[..a.len()]
}

fn entry_bytes(e: &PrefixCacheEntry) -> usize {
    e.caches
        .iter()
        .filter_map(|c| c.as_ref())
        .map(|c| c.approx_bytes())
        .sum()
}

fn fmt_bytes(n: usize) -> String {
    const KB: f64 = 1024.0;
    let n = n as f64;
    if n < KB * KB {
        format!("{:.0} KB", n / KB)
    } else if n < KB * KB * KB {
        format!("{:.1} MB", n / (KB * KB))
    } else {
        format!("{:.2} GB", n / (KB * KB * KB))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(toks: Vec<u32>) -> PrefixCacheEntry {
        PrefixCacheEntry {
            tokens: toks,
            caches: vec![],
            rope_offset: 0,
            last_used: Instant::now(),
        }
    }

    #[test]
    fn longest_prefix_wins_over_shorter_match() {
        let mut pc = PrefixCache::new();
        // Make MIN_PREFIX_LEN-token vec for the test.
        let short: Vec<u32> = (0..MIN_PREFIX_LEN as u32).collect();
        let long: Vec<u32> = (0..(MIN_PREFIX_LEN as u32 + 100)).collect();
        let prompt: Vec<u32> = (0..(MIN_PREFIX_LEN as u32 + 500)).collect();
        pc.entries.push_back(entry(short.clone()));
        pc.entries.push_back(entry(long.clone()));
        let hit = pc.find_longest_match(&prompt).expect("should hit");
        assert_eq!(hit.tokens.len(), long.len());
    }

    #[test]
    fn short_prefix_below_min_is_ignored() {
        let mut pc = PrefixCache::new();
        let tiny: Vec<u32> = (0..100).collect();
        let prompt: Vec<u32> = (0..200).collect();
        pc.entries.push_back(entry(tiny));
        assert!(pc.find_longest_match(&prompt).is_none());
    }

    #[test]
    fn divergent_prefix_is_a_miss() {
        let mut pc = PrefixCache::new();
        let cached: Vec<u32> = (0..(MIN_PREFIX_LEN as u32 + 50)).collect();
        let mut prompt = cached.clone();
        prompt[10] = 99_999; // diverge inside the prefix
        prompt.extend(0..100);
        pc.entries.push_back(entry(cached));
        assert!(pc.find_longest_match(&prompt).is_none());
    }

    #[test]
    fn lru_evicts_oldest_when_full() {
        let mut pc = PrefixCache::new();
        for i in 0..(MAX_ENTRIES + 2) {
            // Use DISTINCT (non-prefix) keys so semantic eviction doesn't drop them.
            let toks: Vec<u32> = std::iter::repeat(i as u32)
                .take(MIN_PREFIX_LEN + 10)
                .collect();
            pc.store(toks, vec![], 0);
        }
        assert_eq!(pc.len(), MAX_ENTRIES);
    }

    #[test]
    fn semantic_eviction_drops_strict_prefix_entries() {
        let mut pc = PrefixCache::new();
        let short: Vec<u32> = (0..(MIN_PREFIX_LEN as u32 + 100)).collect();
        let long: Vec<u32> = (0..(MIN_PREFIX_LEN as u32 + 500)).collect();
        // Turn 1 stores `short`.
        pc.store(short.clone(), vec![], short.len());
        assert_eq!(pc.len(), 1);
        // Turn 2 stores `long` (superset of `short`). Semantic eviction drops short.
        pc.store(long.clone(), vec![], long.len());
        assert_eq!(pc.len(), 1, "old prefix should have been dropped");
        assert_eq!(pc.entries[0].tokens, long);
    }

    #[test]
    fn evict_idle_drops_stale_entries() {
        let mut pc = PrefixCache::new();
        let toks: Vec<u32> = (0..(MIN_PREFIX_LEN as u32 + 50)).collect();
        pc.entries.push_back(PrefixCacheEntry {
            tokens: toks,
            caches: vec![],
            rope_offset: 0,
            // Simulate an entry from 10 minutes ago (well past TTL).
            last_used: Instant::now() - Duration::from_secs(IDLE_TTL_SECS + 60),
        });
        assert_eq!(pc.len(), 1);
        let (evicted, _) = pc.evict_idle();
        assert_eq!(evicted, 1);
        assert_eq!(pc.len(), 0);
    }

    #[test]
    fn evict_idle_keeps_fresh_entries() {
        let mut pc = PrefixCache::new();
        let toks: Vec<u32> = (0..(MIN_PREFIX_LEN as u32 + 50)).collect();
        pc.store(toks, vec![], 1024);
        assert_eq!(pc.len(), 1);
        let (evicted, _) = pc.evict_idle();
        assert_eq!(evicted, 0, "fresh entry should not be evicted");
        assert_eq!(pc.len(), 1);
    }

    #[test]
    fn divergent_branches_both_kept() {
        let mut pc = PrefixCache::new();
        let base: Vec<u32> = (0..(MIN_PREFIX_LEN as u32 + 50)).collect();
        let mut branch_a = base.clone();
        branch_a.extend((0..100).map(|i| 1000 + i));
        let mut branch_b = base.clone();
        branch_b.extend((0..100).map(|i| 2000 + i));
        // Note: `base` is a strict prefix of both branches and would be
        // semantically dropped after either branch is stored. Branches don't
        // contain each other → both retained.
        pc.store(branch_a.clone(), vec![], branch_a.len());
        pc.store(branch_b.clone(), vec![], branch_b.len());
        assert_eq!(pc.len(), 2);
    }
}
