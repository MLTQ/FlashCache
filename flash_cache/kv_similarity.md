# `kv_similarity.py`

## Purpose

Tests whether already-cached value vectors can serve as a cheap semantic page index. It compares the no-page query-prefix values with each independently encoded cold page without another model forward or position-sensitive attention sweep.

## Mechanism

1. Slice recent query KV from the ordinary pinned-plus-query baseline cache.
2. L2-normalize query and page value vectors per layer and KV head.
3. Compute query-token by page-token cosine matrices.
4. Reduce them with several predeclared all-layer and tail-layer metrics.
5. Rank pages by one globally fixed metric and retain top K.

Values are used rather than keys because cached keys include RoPE position rotations. Value vectors are position-independent and should retain lexical/semantic content, although they were never trained as retrieval embeddings.

## Components

### `KVSimilarityScore`

- **Does**: Stores top-pair, query-max, page-max, and global-max cosine reductions across all layers and the configured tail layers.

### `scan_kv_value_similarity`

- **Does**: Scores all independent pages directly from prepared caches.
- **Rationale**: Online work is only vector normalization and dot products; cold-page encoding remains offline.

### `rank_kv_similarity_page_ids`

- **Does**: Applies a named global metric and deterministic page-ID tie breaking.

### `build_packed_cold_value_index`

- **Does**: Concatenates and normalizes all cold-page values per layer once, with a page-length segment table.
- **Rationale**: Packing is an offline archive operation and removes page-wise concatenation and normalization from query latency.

### `scan_packed_value_max_similarity`

- **Does**: Computes the selected all-layer maximum-cosine metric with one query/page matrix multiplication and one segmented maximum per layer.
- **Rationale**: The discovery implementation calculated eight metrics with page/layer Python loops and repeated GPU synchronization. The packed path computes only the globally selected rule and synchronizes once.

### `rank_packed_value_page_ids`

- **Does**: Ranks packed scalar scores with deterministic ties.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Similarity suite | Baseline cache order is pinned then recent query | Reusing a baseline containing cold pages |
| Vector comparison | Every layer exposes dense value tensors with compatible head/dimension axes | Hybrid or non-token-addressable cache layers |
| Position-independence | Values do not receive RoPE rotations | Switching silently to cached keys |
| Answer-free claim | Page ranking receives only cached vectors and fixed reductions | Using relevance labels or answers online |
| Offline cost | Cold page preparation is excluded from query-time scan timing | Re-encoding pages inside `scan_kv_value_similarity` |
| Packed online scan | Page normalization and concatenation occur before online timing | Rebuilding `PackedColdValueIndex` per query |

## Notes

- Maximum pair similarity can be noisy; top-pair and mean-of-max reductions are recorded as alternatives.
- The discovery implementation loops over pages and layers. The packed maximum-cosine path batches every page within each layer and uses half-precision matrix multiplication by default.
- A successful retrieval score still needs a reconstruction path. Current evidence favors replaying the selected original token IDs over directly concatenating independent KV.
