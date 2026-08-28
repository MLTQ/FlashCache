# `run_attention_ranking_suite.py`

## Purpose

Tests whether query-to-page attention supports one global, answer-free cold-page ranking rule rather than retrospectively selecting a different metric for every question.

## Trial design

- Load Qwen3-1.7B once on the required GPU.
- Generate deterministic shuffled archives across the requested seeds, variants, and one- through four-hop depths.
- Assign the first half of the predeclared task sequence to development and the second half to holdout before looking at scores. The depth/variant schedule restarts at the holdout boundary, giving both halves the same parameter pattern with different seeds.
- Build independent cold page KV and run one query-attention scan per trial.
- Record all predeclared attention rankings, top-K relevant-page coverage, fixed-prefix coverage, and a full-prefill generation control.
- Select one global metric using development labels only.
- Report that unchanged metric on the holdout split, both across every trial and across trials where full prefill can answer.

## Global metric selection

Metrics are ordered as declared in `attention_shortlist.ATTENTION_METRICS`. The development objective is lexicographic:

1. all-required-pages top-K rate;
2. mean relevant-page recall;
3. mean reciprocal rank of the worst-ranked required page;
4. declared metric order for an exact tie.

This selection is offline experiment design. Online page ranking receives only attention scores, the globally fixed metric, and fixed K.

## Timing contract

- Model loading is excluded.
- Cold-page construction, attention scanning, and full-prefill generation are synchronized and reported separately.
- The scan timing includes returning full eager attention tensors. A production streaming implementation should reduce each group to page scalars immediately.
- Decode throughput is unchanged because this suite evaluates retrieval only; shortlist reconstruction is measured separately by the Phase 8 runner.

## Outputs

- `trials.jsonl`: complete page scores, rankings, labels, full-prefill control, and timings per task.
- `summary.json`: selected metric, development and holdout aggregates, every metric aggregate, fixed-prefix baseline, and scan latency.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S_UUID> python scripts/run_attention_ranking_suite.py \
  --seed-base 100 --trial-count 24 --blocks 12 --hop-depths 1,2,3,4 --top-k 4 \
  --output-dir outputs/phase8/attention_ranking_suite_24
```
