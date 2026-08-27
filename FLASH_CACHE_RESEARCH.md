# Flash Cache

## Research design notes and prototype plan

**Status:** exploratory / hypothesis-generating  
**Goal:** determine whether an LLM can behave as though it has access to a context substantially larger than its active attention budget by dynamically swapping blocks of precomputed KV cache and using the model's own speculative predictions as a signal for which blocks matter.

---

## 1. Motivation

A decoder-only transformer normally processes a prompt during **prefill**, producing a per-layer KV cache. During autoregressive decoding, newly generated tokens attend to that cached representation rather than recomputing the full prompt from raw text on every step.

This suggests a useful reframing of long-context inference:

> The scarce resource is not necessarily the raw text context. It is the amount of KV state that can be made available to attention at once.

Suppose a model can comfortably attend to roughly 100k tokens of KV state, but we want it to operate over a logical history of roughly 1M tokens. Instead of forcing all 1M tokens to remain active, maintain a much larger **cold KV store** and expose only a subset of it at any given time.

The initial intuition was deliberately simple:

> Randomly flash blocks of old KV state into the active context during inference. Perhaps the statistical effect of repeatedly sampling the larger history can approximate having more of that history present simultaneously.

That idea immediately leads to a stronger one:

> If flashed blocks measurably perturb the model's predicted continuation, use that perturbation as a relevance signal. Random flashing then becomes exploration, while large prediction changes identify blocks worth keeping resident.

This is the core research direction described below.

---

## 2. Core hypothesis

Let the complete logical context be divided into blocks:

\[
C = \{C_1, C_2, \dots, C_N\}
\]

where only a subset can be active at one time.

Maintain three conceptual regions:

1. **Pinned context** — system prompt, essential instructions, sink/anchor tokens if required, and the most recent conversational/generation history.
2. **Hot context** — older KV blocks currently believed to be useful.
3. **Cold context** — the full remaining KV store, held outside the active attention budget.

At decode step \(t\), the model receives:

\[
A_t = P \cup H_t \cup X_t
\]

where:

- \(P\) is pinned context,
- \(H_t\) is the current hot set,
- \(X_t\) is one or more temporarily flashed candidate blocks.

Instead of immediately committing a generated token, the system can use the candidate block as a **probe** and compare the predicted distribution with and without it.

If a candidate materially changes the model's expected continuation, promote it into the hot set.

The resulting system is a form of active, model-driven memory selection.

---

## 3. Important clarification: we are manipulating KV state, not raw context text

The first version of the idea was phrased as "flashing parts of the context window in and out." Operationally, after prefill the more useful abstraction is:

> **Flash blocks of KV cache into and out of the active attention set.**

The prototype should therefore distinguish between:

- source token spans,
- their positional metadata,
- and the corresponding per-layer key/value tensors.

The initial experiment does **not** require training or weight modification.

We should first determine whether the effect exists at all using a normal pretrained causal language model and explicit KV-cache manipulation.

---

## 4. The progression of ideas so far

These should be preserved as separate hypotheses rather than collapsed prematurely into one design.

### H0 — Random Flash Cache

Maintain a large cold KV store. At each probe opportunity, select old blocks uniformly or pseudo-randomly and flash them into the active attention set.

Question:

> Does stochastic exposure to a much larger context produce any measurable improvement over a fixed truncated context, even with no intelligent selection policy?

This is the minimum viable experiment and an important baseline.

---

### H1 — Influence-scored Flash Cache

For each flashed candidate block \(C_i\), compare the model's continuation distribution against a baseline active context.

A simple one-token score would be:

\[
I_i = D_{KL}(p_i(x_{t+1}) \parallel p_0(x_{t+1}))
\]

where:

- \(p_0\) is the distribution without the candidate block,
- \(p_i\) is the distribution after flashing \(C_i\).

Blocks producing larger perturbations are considered more influential and can be promoted into hot memory.

This turns random sampling into **stochastic search over context blocks**.

---

### H2 — Multi-token / speculative influence scoring

A candidate may barely change the next-token distribution while substantially changing the trajectory several tokens later.

Therefore score its influence over a speculative continuation of length \(k\):

\[
I_i^{(k)} = D\big(P_i(x_{t+1:t+k}),\ P_0(x_{t+1:t+k})\big)
\]

The exact divergence measure is an open question.

Possible signals include:

- cumulative tokenwise KL divergence,
- Jensen-Shannon divergence,
- change in candidate sequence log probability,
- top-k rank displacement,
- branch divergence / edit distance between speculative continuations,
- entropy reduction,
- consistency of perturbation over multiple sampled speculative paths.

This was one of the most important conceptual shifts in the discussion:

> **Use speculative decoding not merely to accelerate generation, but as a longer-horizon sensor for context influence.**

---

### H3 — Promote influential blocks into a hot set

Instead of probing independently forever, maintain a bounded resident set of old blocks.

One possible update rule:

\[
s_i(t+1) = \lambda s_i(t) + I_i(t)
\]

where:

- \(s_i\) is the block's accumulated influence score,
- \(I_i(t)\) is its measured influence at the current probe,
- \(\lambda\) controls decay.

Then:

- high-score blocks remain hot,
- low-score blocks decay back to cold memory,
- new candidates compete for residency.

This makes Flash Cache resemble a memory hierarchy:

```text
full logical history
        |
        v
+---------------------+
|    cold KV store    |   CPU RAM / disk / remote store
+----------+----------+
           |
      candidate flash
           |
           v
+---------------------+
| speculative probe   |
+----------+----------+
           |
      influence score
           |
           v
+---------------------+
|      hot KV set     |   active attention budget
| + pinned + recent   |
+----------+----------+
           |
           v
        decoding
```

---

### H4 — Hierarchical or directed search

Uniform random search may be too expensive when the cold store contains thousands of blocks.

Possible refinements:

- random candidate sampling,
- weighted random sampling based on historical influence,
- coarse-to-fine search,
- locality-based expansion around a high-influence chunk,
- Thompson-sampling / multi-armed-bandit framing,
- semantic retrieval as a proposal distribution followed by logit-perturbation verification,
- draft-model prediction of useful chunks followed by perturbation verification,
- learned selection policy after collecting enough probe data.

Important philosophical distinction:

> The selector should ideally optimize **effect on the model's future prediction**, not merely semantic similarity to the query.

A semantically distant block may still have enormous causal relevance to the current continuation.

---

### H5 — Persistent compressed memory

A more radical extension is to reserve persistent memory tokens/state and repeatedly expose them to different cold blocks:

```text
memory + block A -> updated memory
memory + block B -> updated memory
memory + block C -> updated memory
```

This could allow information discovered during one flash to persist after that block leaves active attention.

However, this likely crosses from inference-only cache manipulation into architecture/training territory, depending on implementation.

It should therefore be treated as a later experiment, not part of the first proof of concept.

---

## 5. What would constitute success?

The strongest claim would be:

> With an active KV budget of \(B\), Flash Cache can recover information or reasoning behavior from a logical history much larger than \(B\), with quality significantly above a fixed-window baseline and at a cost substantially below full-context attention.

We should decompose that into progressively easier claims.

### Claim A — Detectability

A relevant cold block produces a measurable and reproducible perturbation in logits or speculative continuations.

### Claim B — Discriminability

Relevant blocks produce a stronger or more useful perturbation signal than irrelevant blocks often enough to rank them.

### Claim C — Retrieval

A search policy using that signal identifies useful blocks better than random chance.

### Claim D — Generation improvement

Promoting those blocks improves downstream answer accuracy or generation quality compared with the same active-context budget without Flash Cache.

### Claim E — Effective-context extension

The system continues to improve as the logical cold context becomes many times larger than the active KV budget.

---

## 6. First prototype

### Philosophy

Do **not** begin by modifying CUDA kernels or production inference engines.

First prove the scientific primitive in ordinary Python/PyTorch using a small model where we can inspect and manipulate `past_key_values` directly.

The first prototype should optimize for observability and correctness, not speed.

### Suggested stack

- Python
- PyTorch
- Hugging Face Transformers or another implementation exposing KV cache cleanly
- A small causal model that fits comfortably in local VRAM
- deterministic seeds where possible
- experiment outputs saved as JSONL/Parquet/CSV plus plots

### Basic experiment

Construct a synthetic long-context task containing many blocks and one or more buried facts.

Example:

```text
block 0000: irrelevant material
block 0001: irrelevant material
...
block 0183: valve X is rated to 400 psi
...
block 0999: irrelevant material

query: Why did the test rig fail after reaching 620 psi?
```

Run:

1. Prefill pinned + recent context.
2. Establish baseline logits / speculative continuation.
3. Flash one candidate cold block into the active KV set.
4. Re-run the probe without committing generated tokens.
5. Measure the perturbation.
6. Restore baseline state.
7. Repeat for other candidates.
8. Rank candidates by influence.
9. Promote one or more top candidates.
10. Generate the final answer.

Initially, brute-force all candidate blocks on small tasks. This gives us ground truth about whether the scoring signal works before solving scalable search.

---

## 7. Critical KV-cache engineering questions

These are not incidental details; they may determine whether the idea works at all.

### Position handling

Can arbitrary cached blocks be inserted while preserving their original positional representation?

For RoPE-based models, test at least:

- preserving original position IDs,
- remapping flashed blocks into compact active positions,
- block-relative positions,
- whether large positional gaps matter.

We must avoid accidentally testing a broken-position implementation and concluding that Flash Cache itself fails.

### Layer consistency

A block is not a single tuple. It has key/value tensors for every attention layer (and potentially implementation-specific cache metadata).

Candidate swaps must be consistent across layers.

### Attention masks

The model must see the intended causal relationships after blocks are inserted or removed.

### Cache APIs

Modern Transformers implementations may use cache abstractions rather than raw tuples. Build a small adapter layer so experiment logic does not depend excessively on one version of one library.

### Block boundaries

Start with token-aligned contiguous blocks. Later test whether block size or semantic boundaries matter.

---

## 8. Baselines

At minimum compare against:

1. **Recent-only truncation**  
   Same active token budget; no access to older context.

2. **Random fixed subset**  
   Choose old chunks once and keep them resident.

3. **Random flashing**  
   H0.

4. **Brute-force influence ranking**  
   Probe every candidate and retain the highest-scoring chunks.

5. **Embedding retrieval**  
   Retrieve blocks by semantic similarity, under the same hot-memory budget.

6. **Hybrid retrieval + influence verification**  
   Embeddings generate candidates; Flash Cache influence scoring reranks them.

7. **Full-context run**, where computationally possible  
   This is an evaluation ceiling / reference, **not** the chooser used by Flash Cache.

The distinction above matters: a full-context "oracle" does not solve selection. It only tells us how much performance is theoretically being left on the table.

---

## 9. Evaluation tasks

Do not start with only natural conversational data. Synthetic tasks let us control relevance precisely.

### Stage 1 — Needle retrieval

One exact fact in one cold chunk.

Measure whether influence ranks the correct chunk.

### Stage 2 — Distractor needles

Multiple semantically related but incorrect facts.

Determine whether semantic similarity and causal influence diverge.

### Stage 3 — Multi-hop reasoning

The answer requires two or more different cold chunks.

This directly tests whether independent flashing is enough or whether useful chunks must become simultaneously resident.

### Stage 4 — Temporal / narrative state

Old events modify interpretation of recent events.

### Stage 5 — Codebase-style tasks

Definitions, implementations, and call sites spread over many chunks.

### Stage 6 — Real long conversations / documents

Only after the controlled experiments establish the primitive.

---

## 10. Measurements

Record much more than final task accuracy.

For every probe, save:

- candidate block ID,
- token span,
- baseline logits,
- candidate logits,
- KL / JS divergence,
- top-k token changes,
- speculative sequence(s),
- sequence log probabilities,
- entropy change,
- candidate's known relevance label in synthetic tests,
- probe latency,
- KV transfer volume,
- hot-set membership before/after,
- final generated answer.

Useful summary metrics:

- MRR of relevant chunks,
- recall@k,
- ROC-AUC / PR-AUC for relevant-block detection,
- final answer accuracy,
- logical-context-size / active-context-size ratio,
- probes per generated token,
- bytes transferred per generated token,
- wall-clock slowdown relative to normal decoding.

---

## 11. Major failure modes

### 11.1 High influence is not the same as useful influence

A malicious, contradictory, surprising, or stylistically bizarre chunk could strongly perturb logits without helping answer the question.

Therefore raw KL maximization may preferentially retrieve disruptive context.

We need to test relevance-sensitive objectives rather than assume maximum perturbation is ideal.

### 11.2 Context switching may destabilize generation

Repeatedly changing active KV blocks could cause inconsistent next-token distributions.

Mitigations to test:

- probe without committing tokens,
- only update the hot set between committed decode blocks,
- hysteresis in hot-set replacement,
- retain pinned and recent context at all times.

### 11.3 Independent flashes cannot perform joint reasoning

If chunk A and chunk B are each weakly useful alone but decisive together, single-block perturbation may miss both.

This motivates pairwise probes, promotion accumulation, or iterative search.

### 11.4 Transfer cost may dominate

KV tensors are large. CPU/GPU or SSD/GPU swapping may destroy latency even if the algorithm works statistically.

This is an engineering question **after** establishing the primitive, but it eventually determines practicality.

### 11.5 Position encoding may invalidate naive swapping

If cached blocks cannot be recombined cleanly under a model's positional scheme, we need a model-specific cache surgery method or an alternate experimental construction.

### 11.6 Self-reinforcing wrong context

Once a misleading block enters the hot set, it may alter predictions so that nearby misleading blocks appear increasingly influential.

This creates possible attractors / memory poisoning dynamics.

---

## 12. Open questions / candidate beads

Each item below should be filed as a discrete experiment or research issue rather than decided by intuition.

### FC-001 — Does purely random flashing help at all?

Compare recent-only truncation, fixed random subsets, and continuously random flashed chunks.

### FC-002 — Can logit perturbation identify a relevant chunk?

Brute-force all chunks in a controlled needle task and measure ranking quality.

### FC-003 — What is the best influence metric?

Compare raw KL, JS, top-k rank displacement, entropy change, sequence-level log-probability changes, and other candidates.

### FC-004 — Is one-token influence sufficient?

Compare \(k = 1\) against speculative horizons such as 2, 4, 8, 16 tokens.

### FC-005 — Does influence over speculative tokens predict final-answer usefulness?

Measure correlation between probe score and downstream answer improvement.

### FC-006 — How should influence be aggregated across speculative tokens?

Sum, mean, discounted sum, max, trajectory divergence, etc.

### FC-007 — Maximum perturbation versus useful perturbation

Construct strongly distracting chunks and test whether naive KL maximization retrieves them incorrectly.

### FC-008 — Promotion policy

Top-k replacement, thresholding, decayed scores, LRU+influence, hysteresis, or bandit policies.

### FC-009 — Hot-set size

Sweep the fraction of the active KV budget reserved for historical hot memory.

### FC-010 — Flash candidate block size

Compare small token blocks, larger contiguous blocks, and semantic segment boundaries.

### FC-011 — Probe budget

How many candidate probes are required per decode step or per decode block?

### FC-012 — Probe frequency

Probe every token, every speculative block, only at high-entropy points, only after topic shifts, or adaptively.

### FC-013 — Random versus directed exploration

Uniform sampling versus weighted sampling using prior influence scores.

### FC-014 — Hierarchical search

Probe coarse super-blocks first, then subdivide high-influence regions.

### FC-015 — Semantic proposal + influence reranking

Use embeddings only to generate a candidate pool; let model perturbation select among them.

### FC-016 — Draft-model proposal + target-model verification

Can a cheap model predict promising KV blocks, with the target model measuring actual influence?

### FC-017 — Neighborhood expansion

If block \(i\) is influential, should neighboring blocks \(i-1, i+1\) receive higher probe priority?

### FC-018 — Multi-block synergy

Find cases where neither A nor B scores highly alone but A+B is decisive.

### FC-019 — Iterative promotion

After promoting one influential block, rerun the search. Does another previously weak block become detectable?

### FC-020 — Position-preserving cache surgery

Determine the correct method for recombining RoPE-positioned cached blocks.

### FC-021 — Position remapping

Test whether remapping historical blocks into compact active positions works better or worse than preserving original positions.

### FC-022 — Pinned anchors

Determine exactly which context must never be evicted: system prompt, BOS, sink tokens, recent window, tool state, etc.

### FC-023 — Generation stability

Measure whether hot-set changes cause incoherence, oscillation, or distribution drift.

### FC-024 — Commit timing

Compare probing before every token versus probing before a multi-token committed generation block.

### FC-025 — Cold-store hierarchy

GPU hot / CPU warm / SSD cold tiers.

### FC-026 — KV compression

Quantized or compressed cold KV blocks to reduce storage and transfer cost.

### FC-027 — Recompute versus store

For very cold blocks, is it cheaper to store raw tokens and recompute KV than to persist full KV tensors?

### FC-028 — Persistent memory tokens/state

Can the system accumulate information from flashes into a compact recurrent state?

### FC-029 — Million-token scaling curve

At what cold-store/active-context ratio does search cost overwhelm benefit?

### FC-030 — Adversarial / distracting influence

How easy is it for irrelevant content to win the influence metric simply by being surprising?

### FC-031 — Influence normalization

Normalize perturbation relative to baseline entropy, chunk length, candidate surprisal, or token frequency.

### FC-032 — Objective based on confidence gain

Instead of maximizing distribution change, prefer candidates that reduce uncertainty or stabilize a continuation.

### FC-033 — Objective based on answer consistency

Probe multiple speculative paths and prefer blocks whose effect is consistent rather than chaotic.

### FC-034 — Query-conditioned versus continuously running memory

Should the system search cold memory only when answering a query, or continuously while generating ordinary conversation?

### FC-035 — Cache identity and provenance

Track source token span and document/conversation location for every KV block so promoted memory remains inspectable.

### FC-036 — Active-memory attractors

Study whether the promotion mechanism can lock onto an incorrect cluster of mutually reinforcing chunks.

### FC-037 — Better-than-retrieval cases

Design tasks where ordinary embedding similarity is misleading but model-impact scoring should succeed.

### FC-038 — Worse-than-retrieval cases

Characterize tasks where semantic retrieval clearly beats Flash Cache.

### FC-039 — Model dependence

Does the phenomenon differ across model sizes, architectures, RoPE scaling methods, GQA/MQA variants, etc.?

### FC-040 — Layer-specific influence

Do all layers need flashed KV blocks, or could a cheaper approximation probe only selected layers?

---

## 13. Recommended implementation phases

### Phase 0 — Cache surgery sanity check

Before studying retrieval:

- prefill a prompt,
- remove and reinsert a known KV block,
- verify that an unchanged reconstructed cache reproduces logits to numerical tolerance,
- verify that controlled replacement causes expected deterministic changes.

If this fails, do not proceed to higher-level conclusions.

### Phase 1 — Exhaustive single-block probing

Small synthetic contexts. Probe every block. No search policy yet.

Primary result:

> Can influence score rank the known relevant block?

### Phase 2 — Multi-token speculative probes

Sweep speculative horizon and scoring metrics.

### Phase 3 — Bounded hot memory

Introduce promotion/eviction policies.

### Phase 4 — Search

Stop exhaustively testing every block. Compare random, hierarchical, semantic-proposal, and bandit-like candidate selection.

### Phase 5 — Multi-hop and synergy

Require multiple historical blocks.

### Phase 6 — Systems work

Only after the algorithm earns it:

- asynchronous transfers,
- pinned host memory,
- quantized KV,
- paged-attention integration,
- inference-engine integration,
- SSD or distributed cold stores.

---

## 14. First concrete milestone

A convincing first graph would be extremely simple:

**x-axis:** candidate historical block ID  
**y-axis:** measured speculative influence score

One block contains the fact required by the query.

If that block produces a clear spike, we have evidence for the primitive.

Then repeat over hundreds of randomized synthetic tasks and report retrieval statistics.

That result determines whether the rest of Flash Cache is worth pursuing.

---

## 15. Working definition

**Flash Cache** is an inference-time memory scheme in which only a bounded subset of a larger precomputed KV history is active at once. Historical blocks are temporarily flashed into attention, their influence on predicted continuations is measured, and useful blocks may be promoted into a resident hot set. Random flashing provides exploration; speculative logit perturbation provides a candidate relevance signal.

The central research question is:

> **Can an LLM use its own change in predicted future as a sufficiently good signal to search a context larger than it can attend to simultaneously?**

That is what the prototype should answer before we attempt to build a production system.
