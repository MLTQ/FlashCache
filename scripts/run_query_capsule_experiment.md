# `run_query_capsule_experiment.py`

## Purpose

Measures a compact latent Flash Cache path: independently flash the ordinary query over each reusable page, retain a fixed-width query-KV capsule per page, discard raw pages, integrate the capsule bank with one final query refresh, and decode normally.

## Conditions

1. No-page baseline.
2. Stale-query simultaneous cold pages.
3. Query refresh over raw independent pages.
4. Query-conditioned compact capsules plus final query integration.
5. Ordinary contiguous full prefill.

Every condition uses the same unmodified task prompt, corpus, final answer horizon, and original page encoding positions.

## Timing

Cold page preparation is reported separately. Capsule online latency includes all per-page query forwards, KV slicing/concatenation, the final query integration forward, and answer decode. The current implementation processes pages sequentially; batching is deferred until correctness warrants optimization.

## Output

`summary.json` records answers, correctness, synchronized latency, capsule width/count, compact final cache size, and the capsule-to-raw-page token ratio.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S-UUID> python scripts/run_query_capsule_experiment.py \
  --hop-depth 2 \
  --blocks 12 \
  --capsule-width 4 \
  --output-dir outputs/phase7/capsule_depth2_blocks12_width4
```

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU safety | Visible device matches the requested 2070 SUPER | Removing the refusal check |
| Latent-only claim | Capsule path generates no per-page tokens or text | Adding decoded summaries |
| Runtime selection | Every page receives identical width with no labels | Relevance-adaptive widths or filtering |
| Compression metric | Capsule count is `page_count × capsule_width` | Retaining raw page KV in the final bank |
