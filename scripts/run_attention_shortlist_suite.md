# `run_attention_shortlist_suite.py`

## Purpose

Evaluates the globally fixed attention-ranking policy end to end across a deterministic task suite. It measures whether selected pages improve actual answer generation, and whether success comes from intact independent KV or compact exact-text replay.

## Conditions

- **No page**: clean baseline cache and ordinary query.
- **Attention independent KV**: fixed-metric top-K intact cold KV, one query refresh, ordinary decode.
- **Attention exact replay**: fixed-metric top-K original page text replayed in a compact prompt.
- **Fixed-prefix exact replay**: the first K pages replayed regardless of the query.
- **Full prefill**: every page replayed as the positive capability control.

The attention metric and K are command-line constants fixed before the suite. Ground-truth page IDs and answer strings are used only after generation for evaluation.

## Correctness

Two signals are retained:

- permissive expected-phrase presence for continuity with earlier phases;
- strict answer assertion, which requires a direct answer, explicit answer marker/conclusion, or a query-subject food assertion.

Strict scoring prevents a refusal that merely repeats an answer-bearing page from counting as multi-hop success. Per-trial text remains in JSONL for manual audit.

## Timing contract

- Cold-page construction is reported separately from online work.
- The attention scan, shortlist reconstruction, and fixed-horizon decode are synchronized.
- End-to-end attention plus exact-replay latency is scan time plus compact replay/decode time.
- Every decode path uses the same fixed-horizon Python loop, so timing is comparable within this runner.
- A production implementation can preserve normal steady-state token speed after shortlist reconstruction; only time to first answer token gains the scan/replay overhead.

## Outputs

- `trials.jsonl`: page selection, coverage, generated text, strict/permissive correctness, and timing per condition.
- `summary.json`: all-trial and full-prefill-capable aggregates plus scan/end-to-end latency.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S_UUID> python scripts/run_attention_shortlist_suite.py \
  --seed-base 212 --trial-count 12 --blocks 12 --top-k 4 --metric all_query_mass \
  --output-dir outputs/phase8/attention_shortlist_suite_holdout
```
