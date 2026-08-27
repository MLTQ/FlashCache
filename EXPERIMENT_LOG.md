# Flash Cache experiment log

## 2026-08-27 — Qwen3-1.7B Phase 0/1

### Environment

- Model: `Qwen/Qwen3-1.7B`, FP16, eager attention
- Host: `m@192.168.0.202`
- GPU: NVIDIA GeForce RTX 2070 SUPER, UUID `GPU-4e207c93-ed93-c35e-f0f2-e37c8df2b047`
- The RTX 4090 was not used.
- PyTorch: `2.11.0+cu130`; Transformers: `5.3.0`

### Phase 0 — cache validity

Qwen3-1.7B returned a standard `DynamicCache` with 28/28 token-addressable KV layers and no recurrent layers.

- Full forward versus ordinary cached continuation: max logit error `0.02734375`, mean `0.00432635`, same argmax.
- Two branches cloned from the same baseline cache: exact logit identity.
- Three-way cache split and same-order reassembly: exact tensor identity and exact probe-logit identity (`0.0` max error).

This passes the cache-surgery gate for Qwen3-1.7B. By contrast, Qwen3.5-2B exposed only 6 token-addressable layers plus 18 recurrent layers, so arbitrary whole-model token-block concatenation is invalid for that architecture.

### Phase 1 — exhaustive single-block needle probe

Configuration:

- 12 historical blocks, 20 tokens each
- one relevant record: valve X-17 is rated to `413 psi`
- 11 same-template records for other valve identifiers and pressures
- candidate KV conditioned on pinned context only
- baseline recent KV computed without any candidate
- candidate inserted between pinned and recent KV without recomputing recent tokens
- all candidates remapped to the same hot positional slot
- Qwen non-thinking chat template
- four-token fixed speculative trajectory for KL/JS
- 40-token free greedy generation for answer correctness

Seed 7 outcome:

| Condition | Generated answer | Value correct? |
|---|---|---|
| No historical block | `125 psi` | No |
| Relevant block 6 | `413 psi` | Yes |
| Eleven distractor blocks | Other/incorrect values | No (0/11) |

The relevant flash therefore changed the model from an incorrect answer to the uniquely correct answer.

Selection metrics told a different story:

- relevant rank by mean Jensen-Shannon divergence: `12/12`
- relevant rank by ground-truth answer log-probability gain: `1/12`
- relevant answer log-probability gain: `+18.9245` nats

Hot-slot raw-prompt replications at seeds 1, 2, 3, and 7 also ranked the relevant block `12/12` by JS and `1/12` by answer log-probability gain. These seeds vary placement and distractor assignment, but they reuse the same target fact and are not a broad task distribution.

### Interpretation

Supported:

1. Qwen3-1.7B KV blocks can be sliced, restored, and reassembled exactly.
2. Late insertion of a relevant cold KV block can materially improve generation without recomputing recent KV.
3. The relevant fact is strongly accessible after insertion.

Not supported:

1. Maximum raw KL/JS perturbation is not a useful selector on this task; it selected every distractor ahead of the uniquely useful block.
2. This experiment does not yet demonstrate answer-free retrieval because the successful utility ranking used the known synthetic answer only for evaluation.

### Next experiment

Keep raw KL/JS as a negative-control selector and test answer-free, query-conditioned usefulness signals. Start with candidate-greedy answer stability across query paraphrases and model-based consistency between the requested identifier, candidate provenance, and generated answer. Evaluate those selectors against free-generation correctness over a broader randomized task distribution.

## 2026-08-27 — inverse-JS follow-up

The user proposed ranking by `1 - JS`, which is exactly equivalent to ranking candidates by ascending raw JS. The runner now reports both directions explicitly.

Seven non-thinking chat/hot-slot trials used 12 candidate blocks, a four-token fixed trajectory, and 40-token free generation. Four trials reshuffled the X-17/413 task; three introduced new target identifiers and pressures.

| Target | Seed | Raw JS rank | `1 - JS` rank | Baseline correct? | Relevant flash correct? |
|---|---:|---:|---:|---:|---:|
| X-17 / 413 psi | 7 | 12 | 1 | No | Yes |
| X-17 / 413 psi | 4 | 11 | 2 | No | Yes |
| X-17 / 413 psi | 5 | 11 | 2 | No | Yes |
| X-17 / 413 psi | 6 | 11 | 2 | No | Yes |
| R-42 / 188 psi | 11 | 11 | 2 | No | Yes |
| M-88 / 337 psi | 12 | 1 | 12 | No | Yes |
| Q-73 / 612 psi | 13 | 11 | 2 | No | Yes |

Aggregate results:

- no-block baseline accuracy: `0/7`
- relevant-flash accuracy: `7/7`
- exactly one flashed candidate was correct in every trial
- inverse-JS top-1 recall: `1/7` (`14.3%`)
- inverse-JS top-2 recall: `6/7` (`85.7%`)
- inverse-JS MRR: `0.512`, versus raw-JS MRR `0.220`

The inverse ordering is useful as a tentative shortlist heuristic on this small sample, but it is not a valid standalone selector. The M-88/337 task reverses the pattern completely: the uniquely useful block has maximum JS and therefore ranks last by `1 - JS`. Candidate inspection also shows that low JS can favor a distractor whose numeric continuation happens to align more closely with the baseline trajectory. Neither polarity identifies semantic relevance consistently.

The stronger result is independent of ranking direction: across four target facts, late insertion of the answer-bearing block changes an incorrect generation to the uniquely correct answer without recomputing recent KV.

## 2026-08-27 — categorical answer-space follow-up

To test whether the earlier behavior was specific to numbers, three fictional task families were added. Fictional archives prevent the model from answering through pretraining memory:

1. a named person who delivered an address at a historical event;
2. the character who spoke an exact line in a novel;
3. the place where a treaty was signed.

Two distinct questions were run per family. Each trial used 12 same-template candidate blocks, non-thinking chat format, hot-slot placement, 40-token free generation, and a 24-token fixed comparison trajectory long enough to reach the answer region.

| Family | Answer | Raw JS rank | `1 - JS` rank | Baseline correct? | Relevant flash correct? |
|---|---|---:|---:|---:|---:|
| historical person | Elara Voss | 8 | 5 | No | Yes |
| historical person | Tomas Vale | 10 | 3 | No | Yes |
| book quote speaker | Mara Venn | 4 | 9 | No | Yes |
| book quote speaker | Ilan Roake | 8 | 5 | No | Yes |
| treaty location | Stonebridge Hall | 8 | 5 | No | Yes |
| treaty location | Marrow Bay | 1 | 12 | No | Yes |

Aggregate results:

- no-block baseline accuracy: `0/6`
- answer-bearing flash accuracy: `6/6`
- random single-flash accuracy: `6/72 = 8.3%` because every distractor was wrong
- raw-JS top-1 recall: `1/6`; MRR `0.288`
- inverse-JS top-1 recall: `0/6`; MRR `0.188`
- selecting two candidates from each JS tail recalls the relevant block in only `1/6`
- ground-truth answer log-probability gain ranks the relevant block first in `6/6` (evaluation oracle only)

This rules out a number-specific explanation for the core cache effect: relevant late KV insertion also corrects names, quote attribution, and locations. It simultaneously rules out the two-tailed JS shortlist suggested by the numeric tasks. Useful blocks commonly sit near the middle of the divergence distribution on categorical questions.

## 2026-08-27 — iterative flashing and provenance selection

### Sentinel-gated iterative flashing

The proposed page-by-page protocol was implemented with isolated speculative branches. For each candidate page, the model was asked to emit a period while it did not know the answer. A rejected branch was discarded, and only a clean miss transition was committed before testing the next page.

Several variants were tried on the Redhaven Congress question:

- a strict one-token period gate;
- a longer visible-continuation gate;
- negative-phrase and control-token-aware classification;
- raw and chat prompts;
- fresh chat turns and inline `NEXT PAGE:` transitions;
- an oracle control that presented the relevant page first.

The literal sentinel was not a reliable readiness signal. Short gates falsely broke on distractors because the small model emitted formatting or explanations after the period. More permissive gates continued through all pages, including the relevant page. Most decisively, the relevant-first oracle control still emitted a period with probability `0.99962`, despite that same flashed page producing the correct answer under the ordinary answer prompt. The instruction to keep emitting periods dominated the information-dependent behavior that the gate was meant to expose.

This rejects the current literal-sentinel design, not iterative cache search in general. A future gate would need a task whose output changes naturally with page content rather than asking the model to introspect whether it “knows” an answer.

### Answer-free provenance consistency

A content-dependent gate was then tested. With each candidate page flashed in isolation, the model is asked to copy that page's own provenance key—for example, its event title, quoted line, treaty name, or valve identifier. The probe prompt contains neither the question's target key nor its answer. An external controller normalizes case and punctuation, compares the generated page key with the target key already present in the query, and keeps the first matching page. The selected page is then reinserted under the original question for normal answer generation.

The first three trials were exploratory. The third revealed that exact string matching was too brittle: the model correctly generated `"Red Orchard Pact" (1901)` but quotation marks prevented a literal match. After switching to normalized token matching, that case selected the correct page and answered `Vale Abbey`. The matching rule was then frozen.

| Phase | Family | Target | Correct page uniquely selected? | Baseline correct? | Selected-cache answer correct? |
|---|---|---|---:|---:|---:|
| exploratory | historical person | Orchard Gate Convention of 1889 | Yes | No | Yes (`Jonas Pell`) |
| exploratory | book quote | “We counted the bells…” | Yes | No | Yes (`Tomas Grey`) |
| exploratory | treaty location | Red Orchard Pact (1901) | Yes | No | Yes (`Vale Abbey`) |
| post-freeze | historical person | Ashcombe Summit of 1831 | Yes | No | Yes (`Nila Hart`) |
| post-freeze | book quote | “No map admits…” | Yes | No | Yes (`Nella Ward`) |
| post-freeze | treaty location | Greywater Settlement (1838) | Yes | No | Yes (`Dunmere Keep`) |
| post-freeze | valve pressure | R-42 | Yes | No | Yes (`188 psi`) |

Aggregate normalized result: unique correct-page selection `7/7`, no-page baseline accuracy `0/7`, and selected-cache answer accuracy `7/7`. The more defensible held-out result is the four trials run after freezing the normalization rule: `4/4` selection and end-to-end answer accuracy across names, quoted language, a place, and a number.

These are small synthetic samples, so this is evidence of a viable mechanism rather than a mature retrieval result. It is nevertheless the first tested selector here that is answer-free, query-conditioned, and successful end to end. Per-candidate generation confidence, entropy, top-token margin, baseline-prefix agreement, and other diagnostics are retained in the JSONL traces for later winner/loser analysis; none is currently treated as the selector.
