# probing.py

## Purpose

Implements the first late-insertion Flash Cache probe for an all-attention model. Cold blocks retain original RoPE positions, recent KV is computed without candidates, and each speculative branch is isolated from committed state.

## Components

### `TokenizedNeedleTask`
- **Does**: Stores section token IDs and their original logical positions.

### `Rollout`
- **Does**: Carries horizon-by-vocabulary logits and selected tokens.

### `CacheStep`
- **Does**: Carries one next-token distribution plus the corresponding advanced private cache branch.

### `PreparedProbeCaches`
- **Does**: Carries the baseline cache, cold candidate caches, and each candidate's effective encoding positions.

### `tokenize_task`
- **Does**: Tokenizes raw sections or Qwen's non-thinking chat serialization and assigns full-history positions.

### `prepare_probe_caches`
- **Does**: Builds pinned+recent baseline KV and candidate blocks conditioned only on pinned context, using original positions or a shared hot slot.
- **Rationale**: This prevents candidate information from leaking into the baseline recent cache.

### `flash_candidate`
- **Does**: Physically inserts candidate KV between pinned and recent KV without recomputing recent tokens.

### `advance_cache`
- **Does**: Processes one input token on an isolated cache clone and returns both logits and advanced cache.
- **Interacts with**: Iterative sentinel search, which discards rejected candidate branches and commits clean no-candidate steps.
- **Rationale**: The caller must be able to retain a winning candidate branch without mutating the clean committed state.

### `rollout`
- **Does**: Runs a greedy baseline or fixed-token speculative branch without mutating the source cache.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Needle experiment | Candidate cache order is pinned, candidate, recent | Changing concatenation order |
| Position experiment | Cached keys retain original logical RoPE positions | Compacting positions silently |
| Hot-slot control | All candidates end immediately before the fixed recent-context positions | Moving recent/probe positions between candidates |
| Prompt control | Chat format uses the tokenizer's official template with one block placeholder | Hand-writing model-specific special tokens |
| Metrics | Candidate rollout can follow the exact baseline token path | Removing `forced_tokens` |
| Iterative search | One-step candidate branches can be discarded or retained independently | Mutating the source cache in `advance_cache` |

## Notes

- Candidate deeper-layer KV is computed from pinned context plus that candidate, not from preceding cold blocks.
- Recent KV is deliberately not recomputed after a flash; the final probe token can attend to the newly inserted block at every layer.
- The `hot_slot` policy remaps candidates only; recent and probe positions remain fixed across baseline and every candidate.
