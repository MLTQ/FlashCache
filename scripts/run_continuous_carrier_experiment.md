# `run_continuous_carrier_experiment.py`

## Purpose

Runs the uninterrupted free-decode experiment requested after separating it from the page-summarization follow-up. One response is generated under a deterministic rotating cold-page schedule, with no visible page boundaries or per-page resets.

By default the rotating path uses the ordinary question prompt, with no carrier-specific behavioral instructions at all. `--carrier-prompt continuous_instruction` retains an exploratory instruction that describes the continuous stream, but both modes permit the answer immediately. The runner never gates or stops on generated text; the fixed final cue exists only for equal final-answer scoring.

## Controls

Each summary reports:

1. Ordinary question with no pages.
2. Continuous-carrier prompt with no pages and no working-response tokens.
3. Normal contiguous full-corpus prefill.
4. All independently encoded pages inserted simultaneously at original positions.
5. Page-conditioned uninterrupted decode.
6. Exact clean replay of the uninterrupted decode's original probe and visible tokens.

The last two paths use identical token IDs, positions, final cue, and answer horizon. Only the page-conditioned path inserts pages while processing the response.

## Parameters

- `--carrier-tokens`: Number of visible free-form working-response tokens.
- `--page-window-tokens`: Number of processed response tokens before rotating to the next physical page. `1` is literal per-token rotation; small values such as `4` or `8` test short windows.
- `--carrier-prompt`: `ordinary` uses the unmodified task prompt; `continuous_instruction` adds a generic description of the stream without page summaries.
- `--position-policy`: Original or exploratory hot-slot encoding for sequentially flashed pages. The simultaneous all-page control always uses original positions.

## Output

- `continuous_steps.jsonl`: Exact token/page schedule, token text, evaluation-only relevance telemetry, cache lengths, and page-conditioned KV deltas. The last row is the commit-only step for the final selected token.
- `summary.json`: All control answers and correctness values, exact visible transcript, token-identity check, latency, and aggregate KV deltas.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S-UUID> python scripts/run_continuous_carrier_experiment.py \
  --hop-depth 2 \
  --blocks 4 \
  --carrier-tokens 48 \
  --page-window-tokens 1 \
  --carrier-prompt ordinary \
  --output-dir outputs/phase5/continuous_depth2_window1
```

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU safety | Visible CUDA device name contains the requested 2070 SUPER marker | Removing the device refusal check |
| Mechanism interpretation | Ordinary mode has no summaries, resets, page markers, relevance gates, answer oracle, or carrier instruction | Adding page-specific textual prompts or selection |
| Causal comparison | Exact replay uses the same processed tokens and positions | Comparing independently generated responses |
| Capacity control | Full prefill remains separately reported | Treating carrier failure as model incapacity without the control |
