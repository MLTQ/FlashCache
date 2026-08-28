# `run_query_refresh_experiment.py`

## Purpose

Measures whether recomputing only the short query prefix over reusable independent cold-page KV recovers useful cross-page reasoning without paying archive-prefill cost at request time or changing autoregressive decode speed.

## Conditions

1. No-page baseline from stale query KV.
2. All cold pages inserted while retaining stale no-page query KV.
3. All cold pages inserted after discarding and recomputing the query prefix.
4. Ordinary contiguous full-corpus prefill.

All conditions use the same task, unmodified prompt, final probe, greedy answer horizon, and original logical page positions.

## Timing contract

- `cold_prepare_latency_ms` reports baseline plus independent page precomputation and is treated as reusable/offline work.
- Each condition's `online_latency_ms` covers its answer path after cold KV is available.
- Query refresh includes archive-cache assembly, one batched query-prefix forward, and answer decode.
- Full prefill includes contiguous prompt construction, archive/query prefill, and answer decode.

These are synchronized wall-clock measurements on the visible GPU, intended for relative screening rather than final throughput benchmarking.

## Output

`summary.json` stores task metadata, token counts, correctness, generated answers, cache sizes, and synchronized latency for every control.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S-UUID> python scripts/run_query_refresh_experiment.py \
  --hop-depth 3 \
  --blocks 12 \
  --output-dir outputs/phase6/query_refresh_depth3_blocks12
```

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU safety | Visible CUDA device name matches the requested 2070 SUPER | Removing the refusal check |
| Correctness comparison | All conditions use the ordinary task prompt and same answer horizon | Condition-specific prompting |
| Speed interpretation | Cold-page encoding is reported separately from online refresh | Charging page precompute to only one condition |
| Context mechanism | Refreshed path reprocesses query tokens but never archive tokens | Re-encoding pages during the online path |
