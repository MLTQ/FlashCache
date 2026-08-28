# `run_iterative_navigation_suite.py`

## Purpose

Evaluates the fixed top‑1 rare-token navigation controller across many shuffled 128-page tasks and depths one through four with one model load. It supports both the original personal-preference chains and a mixed set whose answers are historical names, literary speakers, and treaty locations.

## Fixed controller

- Offline: retain immutable page token IDs, build an IDF token-posting sidecar, and prepare the no-page control baseline. The working controller does not encode or retain cold-page KV.
- Online: retrieve top K from the current carrier, replay exact selected source text, and let Qwen emit one lookup rewrite or final answer.
- Canonicalize only unique one-edit entity spellings against that step's selected source values.
- Stop on model answer, invalid action, repeated carrier, or the fixed global step budget.
- If a lookup repeats, allow one same-note repair generation before stopping; the repair receives no answer feedback.
- `--task-set diverse` schedules the three non-food domains across the requested depths; the default `preference` set preserves the original benchmark.
- `--variant-offset` rotates the deterministic content variants, allowing a prompt/controller to be frozen before a disjoint per-family content split.
- `--max-document-fraction` controls which document-frequency bands enter the token sidecar. Setting it to `1.0` retains low-weight relation words that disambiguate two pages sharing the same entity key.
- A quoted intermediate value returned to an explicit who-question triggers the same one bounded, answer-free repair used for repeated lookups.

The controller receives neither hop depth, relevant page IDs, nor the expected answer. Those fields exist only in per-trial evaluation output.

## Controls

- **No page**: original baseline cache and fixed-horizon decode.
- **Full prefill**: all page text and fixed-horizon decode.

Both permissive answer-phrase presence and strict asserted-answer correctness are retained because long full-prefill reasoning can contain the correct fact near the horizon without phrasing it as a concise final answer.

## Timing contract

- Model loading, no-page baseline preparation, and token-index construction are outside iterative latency. Cold-page KV encoding is unnecessary for this controller.
- Iterative latency is the sum of sub-millisecond token retrieval and early-stopping navigation generations.
- Controls use a fixed token horizon, while navigation generation stops at EOS; absolute latency is informative, but generation-throughput parity requires a separate benchmark.

## Outputs

- `trials.jsonl`: full decisions, selected/relevant IDs, the evaluation-only expected next page, generated carriers, correctness, and timing.
- `summary.json`: overall, per-depth, and per-family accuracy, phrase presence, latency, stop counts, loose relevance coverage, strict logical-order coverage, sidecar build/scan latency, and repair count.
