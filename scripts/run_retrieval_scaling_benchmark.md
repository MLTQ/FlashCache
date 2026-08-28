# `run_retrieval_scaling_benchmark.py`

## Purpose

Measures retrieval latency and memory as a deterministic cold archive grows from dozens to hundreds of pages, while keeping query length, coarse M, and final K fixed.

## Paths

- **Packed scan**: offline-normalized value index and all-layer maximum cosine.
- **Subset attention**: query-attention reranking over packed top-M only.
- **Hybrid scan**: packed plus subset-attention latency.
- **Full attention**: query-attention scan over every page.

The benchmark also records relevant-page coverage at packed top-M, packed top-K, hybrid top-K, and full-attention top-K.

## Timing design

- Model loading, cold-page encoding, and packed-index construction are excluded from online scan latency and reported separately.
- Each path receives the configured warmup count, followed by synchronized repeated measurements.
- Mean, median, minimum, and maximum are retained; median is the primary latency signal.
- Full and subset attention return eager layer/head attention tensors, matching the research implementation.

## Memory

- `cold_kv_bytes` counts key and value tensors for every independent page.
- `packed_value_index_bytes` counts the normalized half-precision value-only retrieval index.
- The packed index currently duplicates values for speed. A deployed design could store the normalized retrieval view on host memory or derive a smaller layer subset offline.

## Interpretation

At small page counts, a second model forward has a fixed launch cost and hybrid retrieval can be slower than full attention. The scaling question is whether packed scan grows slowly enough that packed plus constant-size subset attention crosses below full-archive attention as page count rises.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S_UUID> python scripts/run_retrieval_scaling_benchmark.py \
  --page-counts 12,32,64,128 --coarse-k 16 --final-k 4 --repeats 5 \
  --output-dir outputs/phase11/retrieval_scaling
```
