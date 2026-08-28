# `run_iterative_navigation_experiment.py`

## Purpose

Tests unknown-depth multi-hop retrieval over a large cold archive. A selectable packed-KV, query-attention, hybrid, or token-sidecar scan retrieves notes for the current question; Qwen writes either a rewritten next question or final answer, and the controller repeats without consulting hop depth or expected-answer correctness. The final large-scale controller uses `--retriever token`.

## Online loop

1. Build a no-page baseline for the current question only.
2. Score the immutable packed cold-value index.
3. Optionally compute query-attention ranking and take a fixed union of half packed/half attention candidates, filling duplicates deterministically.
4. Replay the fixed top-K original notes in a short navigation prompt.
5. Parse `LOOKUP: <rewritten question>` or `ANSWER: <value>` leniently.
6. Continue on a novel lookup; stop on model answer, invalid action, repeated question, or fixed safety budget.

The ground-truth hop depth, relevant page IDs, and answer are logged only after each decision for evaluation. They never affect retrieval, rewriting, or stopping.

## Controls

- **No page**: original query from the clean baseline cache.
- **Full prefill**: all original page text with the original query.

## Timing contract

- Cold-page encoding and packed-index construction are offline.
- Iterative online latency includes each rewritten-query prefill, packed scan, and early-stopping navigation generation.
- The default union also includes one full-archive query-attention scan per step. Packed-only and attention-only ablations are available.
- The `kv_union` ablation takes a fixed quota from every predeclared position-independent KV similarity reduction. It is slower discovery code, but tests metric complementarity before implementing another packed kernel.
- The `token` path uses a fixed IDF-weighted inverted index over retained page token IDs. It requires no rewritten-query model prefill or KV scan and is the rare-key scale control.
- One packed-scan warmup occurs after offline index construction so kernel initialization does not inflate the first online step.
- The navigation output is a short explicit reasoning carrier; ordinary final-generation token speed is not modified.
- This prototype uses `generate` for navigation and fixed-horizon rollout for controls, so latency comparisons focus on iterative absolute overhead rather than cross-path throughput parity.

## Safety and answer-free behavior

- Selected notes are exact source text, not summaries.
- The controller does not know how many relations are required.
- Format fallbacks never inspect whether text matches the expected answer.
- Repeated questions and a fixed step budget prevent loops.
- One no-progress repair is allowed before a repeated-question stop; the repair receives no correctness feedback.
- The same bounded answer-free repair handles an obvious target-type mismatch, such as returning a quotation to a who-question.

## Example

```bash
CUDA_VISIBLE_DEVICES=<2070S_UUID> PYTHONPATH=. python scripts/run_iterative_navigation_experiment.py \
  --blocks 128 --hop-depth 2 --retriever token --retrieval-k 1 \
  --max-navigation-steps 5 \
  --output-dir outputs/phase12/navigation_depth2_blocks128
```
