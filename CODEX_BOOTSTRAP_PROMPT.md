# Codex bootstrap prompt — Flash Cache prototype

You are taking over an exploratory research/code project called **Flash Cache**.

Read `FLASH_CACHE_RESEARCH.md` in this repository first. Treat it as the research handoff and source of truth for the current hypothesis, but **do not assume the proposed mechanism works**. The point of the codebase is to falsify or support it experimentally.

## Context

The core idea is to investigate whether an autoregressive transformer can operate over a logical history much larger than its active KV-cache/attention budget by dynamically swapping historical KV blocks into and out of the active cache.

The idea evolved through several stages:

1. Start with a huge logical context whose KV state cannot all remain active.
2. Keep system/anchor/recent tokens pinned and place old KV blocks in a cold store.
3. Randomly flash historical blocks into the active KV set as an exploration baseline.
4. Do **not** necessarily generate/commit a token after each flash. Instead, use the flashed block as a probe.
5. Compare logits or speculative multi-token continuations with and without the block.
6. Treat the magnitude/structure of that perturbation as a possible relevance signal.
7. Promote high-value blocks into a bounded hot set and evict low-value blocks.
8. Eventually compare random exploration, influence-guided exploration, hierarchical search, semantic candidate retrieval, draft-model candidate selection, and hybrids.

The key question is not merely whether a block is semantically similar to the current query. It is whether exposing that block **materially and usefully changes the model's predicted future**.

## Immediate objective

Build the smallest scientifically useful Python/PyTorch test harness that can answer:

> When a known relevant historical block is flashed into a model's KV cache, can we detect it from the resulting change in logits or short speculative continuations, and rank it above irrelevant historical blocks?

Do not optimize performance yet. Do not write custom CUDA kernels yet. Favor inspectability, deterministic experiments, logging, plots, and unit tests.

## First implementation phases

### 1. Repository inspection

Inspect the repo. If it is empty, create a minimal clean Python project structure. Use the environment/package tooling already present if any; otherwise keep dependencies minimal.

### 2. Model/cache adapter

Use a small Hugging Face causal language model that exposes KV cache cleanly and fits easily on available hardware.

Create an adapter that can:

- tokenize source blocks,
- prefill and obtain cache state,
- preserve block provenance and token positions,
- clone/restore a baseline cache,
- construct an active cache from pinned/recent/historical blocks,
- run a one-token logit probe,
- run a short multi-token speculative probe without mutating the committed generation state.

Be very careful about modern Transformers cache APIs, RoPE/position IDs, attention masks, GQA/MQA tensor shapes, and per-layer consistency.

### 3. Cache-surgery validation FIRST

Before testing Flash Cache retrieval, write tests showing that cache manipulation itself is correct.

At minimum:

- prefill a prompt normally and record next-token logits,
- reconstruct the exact same cache through the adapter and verify logits match within numerical tolerance,
- split a prompt into blocks and reassemble an equivalent cache while preserving valid positional semantics,
- detect and loudly fail if our cache surgery changes logits when it should not.

If arbitrary KV concatenation is invalid for the chosen model/API, investigate and document the reason rather than hacking around it silently. We need to know whether we are testing Flash Cache or merely broken positional/cache semantics.

### 4. Synthetic needle benchmark

Generate controlled long-context examples composed of blocks. Exactly one block initially contains a fact required to answer the final query.

Example structure:

- many irrelevant blocks,
- one block such as `Valve X has a maximum rated pressure of 400 psi.`,
- recent context says the rig reached 620 psi,
- query asks for the likely cause of failure.

Keep the active budget intentionally too small to hold all historical blocks.

### 5. Exhaustive probing baseline

For each historical block:

1. establish baseline active cache = pinned + recent (+ any fixed hot state),
2. record baseline logits and/or speculative continuation,
3. flash candidate block,
4. probe again,
5. calculate influence metrics,
6. restore the exact baseline state,
7. continue to the next block.

Initially probe **every** candidate. Search efficiency comes later.

Implement at least:

- tokenwise KL divergence,
- Jensen-Shannon divergence,
- top-k token/rank changes,
- baseline and candidate entropy,
- sequence log probability differences for a short speculative horizon.

Make speculative horizon configurable, including `k=1`.

### 6. Result logging

Every experiment should emit structured data containing at least:

- experiment seed,
- model identifier,
- candidate block ID,
- token span / source provenance,
- whether that block is ground-truth relevant,
- each influence metric,
- speculative continuation(s),
- baseline/candidate entropy,
- latency,
- final ranking.

Prefer JSONL or Parquet plus a human-readable summary.

### 7. First visualization

Produce the first decisive plot:

- x-axis: candidate block ID
- y-axis: influence score
- mark the ground-truth relevant block

Then aggregate over many randomized trials and report MRR, recall@k, ROC-AUC/PR-AUC where appropriate.

## Baselines to support early

Implement or leave clean interfaces for:

- recent-only truncation,
- fixed random historical subset,
- random flashing,
- exhaustive influence ranking,
- embedding retrieval,
- embedding candidate proposal + influence reranking,
- full-context evaluation when the model/context size permits it.

Remember: the full-context run is an evaluation reference, **not** a selection oracle available to the algorithm.

## Research discipline

`FLASH_CACHE_RESEARCH.md` contains many open questions labeled `FC-###`. Treat these as candidate beads/issues/experiments.

As you discover implementation constraints or experimental results:

- do not quietly resolve uncertain design choices,
- document them,
- create/update beads where appropriate,
- keep hypotheses separable,
- preserve failed experiments,
- prefer controlled comparisons to intuition.

Especially do not collapse these distinctions prematurely:

- random flashing vs directed search,
- one-token vs multi-token speculative influence,
- maximum perturbation vs useful perturbation,
- semantic similarity vs causal influence on continuation,
- original position preservation vs position remapping,
- independent single-block relevance vs multi-block synergy,
- inference-only cache manipulation vs learned persistent memory.

## Major suspected failure modes

Design the harness so we can explicitly test:

- surprising but irrelevant chunks receiving huge KL scores,
- contradictory/adversarial chunks dominating influence,
- context switching destabilizing generation,
- relevant chunks that matter only jointly,
- self-reinforcing incorrect hot-memory attractors,
- RoPE/position semantics making naive KV splicing invalid,
- transfer/recompute costs eventually eliminating any systems benefit.

## Coding style

Keep the first version small and legible. Favor modules such as:

```text
flash_cache/
    cache_adapter.py
    blocks.py
    probing.py
    metrics.py
    policies.py
    synthetic.py
    experiment.py
    logging_utils.py

tests/
    test_cache_equivalence.py
    test_probe_restore.py
    test_synthetic_task.py

scripts/
    run_needle_experiment.py
    plot_influence.py
```

This is only a suggestion; adapt to the repository if there is already structure.

## What I want you to do now

1. Read the research document.
2. Inspect the repository and current environment.
3. Produce a brief implementation plan tied to the Phase 0/Phase 1 experiments.
4. Create beads for the immediate work and the highest-priority unresolved scientific questions.
5. Implement the cache-equivalence/cache-surgery validation before implementing retrieval claims.
6. Implement the smallest exhaustive single-block synthetic probe experiment.
7. Run it and inspect the actual data.
8. Report what worked, what failed, and what the next experiment should be based on evidence.

The first milestone is not "build a giant context system." It is:

> **Demonstrate or falsify that a relevant cold KV block can be identified from its measured effect on the model's predicted continuation.**

Proceed experimentally.
