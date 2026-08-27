# `run_semantic_carrier_experiment.py`

## Purpose

Runs the first richer-carrier experiment for Qwen3-1.7B. Every cold page generates a fixed-width semantic scratchpad fragment, after which the runner asks the original question from the accumulated page-conditioned KV. It then replays the exact token transcript without pages.

`--note-selection-mode sequential` lets prior carried state influence later note choices. `--note-selection-mode isolated` proposes each note using only the static semantic prompt and the current page, then commits those exact tokens under the accumulating page-conditioned cache. Neither mode uses answer or relevance metadata.

## Controls

The summary records five accuracy paths:

1. Ordinary question with no pages.
2. Semantic instructions with no pages and no generated notes.
3. Normal contiguous full-corpus prefill.
4. All independently encoded pages inserted simultaneously at their original positions.
5. Semantic carrier and exact clean replay of its visible tokens.

The simultaneous-cache control always uses original logical positions, even when the sequential carrier is explicitly run with the exploratory hot-slot policy.

## Output

- `semantic_steps.jsonl`: page order, source/relevance telemetry, generated note text and token IDs, positions, retained-cache lengths, and per-page page-conditioned KV deltas.
- `summary.json`: task metadata, all control answers and correctness values, visible carrier transcript, latency, and exact-replay token identity.

Correctness is evaluated only on the final generated answer. Note formatting does not affect the score.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S-UUID> python scripts/run_semantic_carrier_experiment.py \
  --hop-depth 2 \
  --blocks 4 \
  --note-tokens 16 \
  --note-selection-mode isolated \
  --output-dir outputs/phase4/semantic_depth2_seed62
```

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU safety | Visible CUDA device name contains the requested 2070 SUPER marker | Removing the device-name refusal check |
| Causal interpretation | Poisoned and clean replay use identical token IDs, positions, final cue, and horizon | Decode/re-tokenize or different answer prompts |
| Capacity check | Full prefill and simultaneous independently cached pages remain separately reported | Treating either as equivalent to sequential carrier accumulation |
