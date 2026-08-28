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

## 2026-08-27 — page-conditioned carrier-state streaming

The iterative proposal was reimplemented after clarifying that cross-page “poisoning” is the intended mechanism. On every wait step, the current response token is now processed with the flashed page present. Only the page span is then removed; the newly appended, page-conditioned token KV remains in the persistent response cache. The clean control observes the same page-conditioned speculative output but recomputes its committed placeholder state without the page.

No relevance labels, known answers, extracted page keys, or known hop counts control the stream. Every physical page is visited in shuffled order. Evaluation labels are used only in the saved telemetry and correctness scoring.

### Tasks and controls

The synthetic personal archive includes:

- direct one-page calibration questions such as “What is Rowan's favorite food?”;
- two-page questions such as “What is my wife's favorite food?”, requiring `wife -> Rowan` and `Rowan -> saffron rice` records;
- same-domain relationship and food distractors.

Each canonical task was tested under four conditions:

1. no archived pages;
2. ordinary full prefill of every shuffled page, with no relevance filtering;
3. simultaneous insertion of every independently cached page;
4. page-by-page poisoned carrier state versus a clean carrier control.

Configuration checks also varied one versus eight carrier tokens per page, hot-slot versus original page positions, one to three corpus passes, four versus twelve pages, a literal sentinel, a forced sweep, and one full relevance-blind warmup pass before answer attempts.

### Results

Across four unique canonical tasks—two direct and two two-page chains:

| Condition | Correct |
|---|---:|
| No pages | `0/4` |
| Normal full prefill of all pages | `4/4` |
| All independently cached pages inserted together | `1/4` |
| Poisoned carrier | `0/4` |
| Clean carrier | `0/4` |

No tested carrier variation produced a correct answer or an accuracy improvement over its clean control. The favorable Rowan/saffron-rice two-hop task was solvable both by normal full prefill and by simultaneous all-page cache insertion, yet an eight-token carrier answered incorrectly after one pass and again after a full warmup plus a second pass.

The cache surgery itself is working. On the fresh direct Rowan/saffron-rice calibration, the answer-bearing page changed the retained token KV relative to the no-page token by max absolute `6.27734` and mean absolute `0.13014`. Despite this material hidden-state difference, poisoned and clean conditions produced the same incorrect final response. Width eight and original logical positions also failed to retain the easier direct fact after its page was removed.

During active flashing, the answer-bearing page could still make speculative text mention the correct food. This occurred in both poisoned and clean conditions because both had the page currently available; neither condition reliably connected that food to the earlier relationship after removal. It is therefore not evidence for persistent multi-hop accumulation.

### Interpretation

The current placeholder-token carrier is a real latent perturbation but not a useful memory channel for Qwen3-1.7B. A cached period token does not preserve even a single explicit fact strongly enough for later generation, so it cannot support unknown-depth chaining as implemented. Repeating pages, widening the carrier, and preserving original positions did not change that outcome.

This negative result does not rule out streaming integration through a richer state update—for example, retaining model-generated semantic tokens, merging page-conditioned branches, or training dedicated memory tokens. It does rule out treating untrained period-token KV residue as sufficient merely because its tensors differ from the clean cache.

## 2026-08-27 — model-generated semantic carrier

The placeholder period was replaced with a fixed-width factual scratchpad fragment generated while each page was flashed. Every page was visited in physical order; relevance labels, answer strings, and hop depth were unavailable to the runtime. Correctness was scored only on the final answer.

The causal control is stricter than the earlier clean carrier: after the page-conditioned run generated its notes, the control re-encoded the exact same visible token IDs at the exact same positions without ever inserting a page. Both paths then received the same final cue and answer horizon. This separates useful visible compression from additional information in page-conditioned token KV.

### Sequential note selection

The first version allowed all earlier retained notes and the current flashed page to jointly choose the next note tokens. It solved the direct Shirly/tacos calibration, but failed both four-page, two-hop tasks. The generated stream entered content attractors:

- With a Dario/lemon-cake distractor first, every later page repeated Dario/lemon cake, including the Shirly relationship and food pages.
- With Rowan/saffron-rice first, later distractors progressively rewrote the stream to lemon cake.

Poisoned and exact-replay accuracy were both `1/3` across the direct calibration and two two-hop trials. This is direct evidence that unconstrained cross-page semantic poisoning is strongly order-sensitive in Qwen3-1.7B.

### Isolated note selection

The next variant used the same model and generic prompt to propose each note from that page alone. The proposed tokens were then forced through the accumulating cache with the page present, retaining page-conditioned KV as before. This does not select pages or consult evaluation metadata; it only prevents earlier notes from changing what the current page writes.

An exploratory width sweep exposed a narrow small-model tradeoff. Sixteen tokens let a relationship-only page invent `Favorite Food: Unknown`; ten tokens truncated `saffron rice` to `saff`; twelve tokens retained the answer while cutting off the unsupported field. Width 12 was then used for subsequent trials.

| Hops | Pages | Answer | No pages | Full prefill | All independent pages | Page-conditioned carrier | Exact clean replay |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2 | 4 | tacos | No | Yes | No | Yes | Yes |
| 2 | 4 | saffron rice | No | Yes | Yes | Yes | Yes |
| 3 | 6 | dumplings | No | Yes | No | No | Yes |
| 3 | 6 | lentil stew | No | Yes | No | Yes | Yes |
| 2 | 12 | mushroom pie | No | Yes | No | Yes | Yes |
| 3 | 12 | sesame noodles | No | Yes | No | No | No |

Aggregate accuracy on these six isolated-selection trials:

- ordinary no-page control: `0/6`;
- semantic no-page control: `0/6`;
- normal full prefill: `6/6`;
- simultaneous insertion of all independently cached pages: `1/6`;
- page-conditioned semantic carrier: `4/6`;
- exact clean replay: `5/6`.

The retained page-conditioned state was materially different from the clean encoding: maximum per-trial page-span deltas were roughly `87` to `135` in the six-page cases. It was nevertheless never uniquely beneficial. On the Maren/Ivo/dumplings three-hop task, exact replay correctly answered `dumplings` while the page-conditioned path answered the distractor `lemon cake`. Thus hidden page residue was uniquely harmful in one primary trial and neutral in the other five.

### Interpretation

This richer carrier demonstrates a useful training-free mechanism, but it is semantic externalization rather than latent KV memory. The model can compress every page into a short visible transcript and later combine facts across pages; this solved a 12-page two-needle haystack that no-page and simultaneous independent-cache controls failed. The exact replay proves that the visible note tokens fully explain the improvement.

The hoped-for net benefit from uncontrolled cross-page poisoning was not observed. Letting earlier poisoned state choose later tokens caused attractors, and retaining page-conditioned state after isolated token selection matched or underperformed clean replay. At the current boundary, twelve isolated tokens per page solve two required facts among twelve pages, while three required facts among twelve pages still collapse to a distractor even though full prefill succeeds.

These trials remain exploratory: the note width was tuned on one task, the task family is fictional personal preferences, and exact-answer substring scoring can credit the right food despite imperfect supporting prose. The complete per-page token traces and control outputs are stored under `outputs/phase4`, with the selected counts in `semantic_carrier_aggregate.json`.

## 2026-08-27 — uninterrupted rotating-page free decode

The original proposal was then tested without the page-summarization detour. The model produced one uninterrupted response while cold pages rotated beneath it on a fixed physical schedule. There were no page notes, visible page markers, resets, gates, relevance decisions, answer oracle, or known-hop control flow.

At each autoregressive step, the scheduled page was inserted, the current response token was processed, and the next token was chosen greedily. The page was then removed while the processed token's page-conditioned KV remained. The chosen token became the next input under the next scheduled page. After a fixed token budget, the final chosen token was committed and both conditions received the same final-answer cue.

The exact clean replay processed the same original probe and every generated token ID at the same positions without inserting pages. It therefore distinguishes effects already externalized in token choices from additional effects in the retained page-conditioned KV.

### Prompt control

An initial carrier-specific prompt described the continuous stream and asked the model to wait for a final signal. As in the earlier sentinel experiment, that behavioral instruction suppressed useful answering. Removing only the waiting rule did not fix it. Seven such exploratory trials all failed and were excluded from the primary comparison.

The final mechanism uses the ordinary unmodified question prompt. The model is not told to summarize, introspect, wait, or follow a carrier format; pages simply rotate while it writes its normal answer.

### Window and order results

Literal per-token rotation on a direct Vera/mushroom-pie task failed. It reacted to both the answer page and distractors but blended `mushroom pie` with `plum tart` into the unsupported answer `plum pudding`.

Eight-token windows allowed a complete short phrase to form:

- When the direct Vera/mushroom-pie page was first, the visible stream immediately produced the correct answer and retained it through later rotations. Poisoned and exact-replay answers were both correct.
- When the direct Shirly/tacos page was last, the response first committed to a distractor trajectory and later misbound `bean stew` to Shirly. Poisoned and replay answers were both wrong.
- A two-hop task with relevant pages in windows two and four failed.
- A favorable two-hop task with `wife = Vera` and `Vera = mushroom pie` in the first two adjacent windows also failed.
- A three-hop task with relevant pages interleaved at physical positions zero, two, and four failed.

A complete one-pass 16-token-window control also failed the two-hop task, so the eight-token failures are not explained solely by truncating the individual source records.

Primary ordinary-prompt, eight-token-window results:

| Condition | Correct |
|---|---:|
| No pages | `0/5` |
| Normal full prefill | `5/5` |
| All independently cached pages inserted simultaneously | `2/5` |
| Uninterrupted page-conditioned decode | `1/5` |
| Exact clean replay | `1/5` |

The sole success was the direct relevant-first ordering. Multi-hop accuracy was `0/3`, including adjacent relevant pages. Hidden page-conditioned state changed final wording in several failures, and the retained one-token KV differed materially from its clean counterpart (maximum deltas `70.1875` to `95.0547` across primary trials), but it never changed an incorrect exact replay into a correct poisoned answer.

### Interpretation

The requested mechanism is now implemented and tested directly. It can preserve a correct answer once that answer-bearing page controls the beginning of a sufficiently wide response window, but it does not reliably redirect an established trajectory and did not combine multiple needles in these trials. Literal per-token rotation is especially destructive because entity and value phrases are assembled under different pages.

The current evidence therefore does not support uncontrolled free decode as an unknown-depth integration mechanism for Qwen3-1.7B. Its failure mode is not absence of page influence—the visible response and KV tensors both change substantially—but unstable binding and strong autoregressive commitment to early distractors. Exact clean replay matching every correctness outcome also provides no evidence that hidden residue adds useful memory beyond the token choices it helped produce.

Complete traces are under `outputs/phase5`; `continuous_carrier_aggregate.json` separates the final ordinary-prompt trials from the discarded carrier-instruction experiments.

## 2026-08-28 — query refresh and latent-capsule controls

Two inference-only attempts tested whether cold pages could update a short hidden query state without exposing page text in the final prompt.

### Query refresh

Each independently cached page was combined with the current question and then removed after recomputing the question KV. The refreshed question cache was retained for answer generation. Four representative tasks covered one-, two-, and three-page chains plus a 12-page haystack.

| Condition | Correct |
|---|---:|
| No page | `0/4` |
| Stale simultaneous independent KV | `0/4` |
| Page-conditioned query refresh | `0/4` |
| Ordinary full prefill | `4/4` |

The page did materially alter the refreshed hidden state, but the update did not preserve a usable binding after the page was removed.

### Query-conditioned latent capsules

A second variant generated fixed-width hidden continuation tokens while each page and the question were active, then removed the page and retained only those token KVs. Widths 4, 8, and 16 all failed the two-hop calibration. A decisive direct one-page control also failed at width 8: full prefill answered `mushroom pie`, while the capsule response said the information was absent.

These negative results narrow the hidden-state route. Neither recomputed question KV nor untrained latent continuation KV was a reliable stand-alone memory carrier for Qwen3-1.7B.

Artifacts: `outputs/phase6` and `outputs/phase7`.

## 2026-08-28 — query-attention shortlisting and exact replay

The next experiments separated two questions:

1. Can the live query identify useful cold pages without an answer oracle?
2. If it can, does independently composed KV work, or must the selected source be replayed exactly?

Every cold page remained independently cached. A query-only forward pass attended over the temporarily assembled pages, and several predeclared page-attention summaries were recorded. The globally selected development metric was total query-to-page attention mass, `all_query_mass`.

On a stratified 24-trial, 12-page ranking suite, the fixed metric placed every required page in top 4 on `5/12` holdout tasks. More importantly, on the six holdout tasks that ordinary full prefill solved within the answer horizon, coverage was `5/6`. The scan averaged `35.16 ms`.

A separate end-to-end holdout suite produced:

| Condition | Strict correct |
|---|---:|
| No page | `0/12` |
| Top-4 independent KV composition | `1/12` |
| Top-4 exact source replay | `4/12` |
| Arbitrary first-4 exact replay | `0/12` |
| Full prefill | `6/12` |

Among the six full-prefill-capable tasks, attention selection covered all required pages on five and exact replay answered four. This is a real answer-free retrieval signal at 12 pages. It also gives another strong causal control: selecting the right independently cached KVs is usually insufficient, while replaying their exact source tokens works.

Teacher-forced answer-choice likelihood was retained as evaluation telemetry but rejected as a controller. Qwen sometimes preferred explanatory text before the canonical answer even when free generation was correct, making the restricted ranking prompt-dependent.

Artifacts: `outputs/phase8`.

## 2026-08-28 — packed KV similarity, hybrid retrieval, and scaling

The query-attention scan rereads all cold pages and therefore scales with archive size. A cheaper proposal index was built from normalized cold-page value vectors. The packed all-layer max-cosine implementation reproduced the slower reference top-K set on `24/24` trials and, after warmup, scanned 12 pages in roughly `6.2 ms`.

Its semantic recall was weaker than query attention. On the 24-trial packed suite, the fixed metric covered all required top-4 pages on `3/12` holdout trials. A hybrid controller used packed similarity to choose eight pages and query attention to rerank four. At 12 pages:

| Selector / condition | All-page coverage or strict answer |
|---|---:|
| Packed top-4 coverage | `3/12` |
| Hybrid top-4 coverage | `6/12` |
| Full-attention top-4 coverage | `5/12` |
| Hybrid exact-replay answer | `5/12` |
| Full-attention exact-replay answer | `5/12` |
| Full prefill answer | `6/12` |

The hybrid scan was slower at 12 pages because it requires a second model forward, but it crossed over as the archive grew:

| Pages | Packed median | Hybrid median | Full attention median |
|---:|---:|---:|---:|
| 12 | `5.96 ms` | `39.82 ms` | `33.60 ms` |
| 32 | `5.90 ms` | `39.46 ms` | `36.27 ms` |
| 64 | `5.97 ms` | `40.32 ms` | `45.07 ms` |
| 128 | `6.30 ms` | `42.05 ms` | `65.11 ms` |

At 128 pages the cold KV occupied about `276 MB`; the packed value index occupied `138 MB`. The packed scan stayed nearly constant because model dimensions and launch overhead dominated this small scale.

However, one-shot coverage collapsed at 32–128 pages. The first query in a multi-hop task does not name the second-hop entity, so no one-shot semantic or KV ranking can reliably retrieve every required page. Full attention also developed late-position and filler-template biases at scale. This motivated an iterative query carrier instead of a single shortlist.

Artifacts: `outputs/phase9`, `outputs/phase10`, and `outputs/phase11`.

## 2026-08-28 — iterative rare-token navigation over 128 pages

### Mechanism

The successful controller uses a deliberately conventional sidecar over the cold pages' retained token IDs:

1. Offline, build an IDF-weighted inverted index from immutable page token IDs.
2. Online, retrieve top 1 for the current question.
3. Replay that one exact source note to Qwen3-1.7B.
4. Ask the model to either return the final value or rewrite the same who/where/what question with one resolved value substituted.
5. Repeat until the model answers or the fixed five-step safety budget is exhausted.

The controller receives no expected answer, relevant page ID, or hop count. A single bounded repair is allowed after an exact repeated lookup or an obvious surface-type mismatch such as a quotation returned to a who-question. The repair sees only the question, selected note, and its prior response. One-edit entity spelling correction is grounded solely in values from the selected source note.

This is not hidden-KV composition. It is model-directed symbolic navigation with exact selected-token replay: the selected note is ordinarily prefilled into a short navigation prompt. The token sidecar can live beside disk/host-memory source pages (and optional KV pages used by other selectors), and only one small source page needs to enter active attention per hop. The final suite runner therefore skips unused cold-page KV encoding entirely.

### Preference-chain regression

The final frozen controller was rerun on 12 shuffled personal-archive tasks spanning one through four source pages, with 128 pages in every archive:

| Condition | Strict correct | Mean online latency |
|---|---:|---:|
| Iterative top-1 navigation | `12/12` | `622.08 ms` |
| No page | `0/12` | `1327.46 ms` |
| Full prefill, 64-token horizon | `4/12` | `1978.53 ms` |

All 30 retrieval steps selected the exact next logical page in order. Accuracy was `3/3` at each depth from one through four. The token index took `2.24 ms` to build offline and `0.136 ms` per online retrieval. No repair was needed.

The low full-prefill strict score is partly a horizon artifact: Qwen often spent the 64-token budget explaining the chain. Phrase presence was `5/12`, so this experiment must not be read as proving that the model could never solve the full prompt with a longer decode. The iterative result does show that navigation reaches concise correct answers sooner under this fixed runtime contract.

### Diverse development and held-out suites

New fictional tasks changed both wording and answer space:

- historical event → closing-address speaker;
- quoted line → literary speaker;
- treaty → signing location.

Each family included one- through four-page chains and template-matched distractors. During development, an audit caught and corrected an earlier compact-filler shortcut: final relation wording now occurs in most distractors, and the evaluator requires selecting each logical page in exact order.

After freezing the controller, a disjoint four-variant-per-family content pool was tested with fresh shuffles:

| Condition | Strict correct | Phrase present | Mean online latency |
|---|---:|---:|---:|
| Iterative top-1 navigation | `10/12` | `10/12` | `751.39 ms` |
| No page | `0/12` | `0/12` | `1344.50 ms` |
| Full prefill, 64-token horizon | `8/12` | `10/12` | `2117.75 ms` |

By answer family, iterative accuracy was history names `4/4`, treaty locations `4/4`, and quotation speakers `2/4`. Ten trials followed every expected page in strict logical order. The two failures were both quotation-carrier failures: Qwen either answered with the intermediate quotation instead of its speaker or rewrote the quotation in a way that revisited the citation page.

The sidecar took `2.29 ms` to build offline and `0.145 ms` per online retrieval. One answer-free repair fired. Overall navigation latency was lower than both controls because successful steps emit short carriers and stop early. At depth four it averaged `1441.66 ms`, slightly above the `1336.71 ms` no-page 64-token failure but below the `2085.31 ms` full-prefill control. These are end-to-end latencies, not a matched tokens-per-second benchmark; the transformer decode kernel itself is unchanged, but multi-hop questions require several short serial generations.

### Interpretation

The original pure hidden-state hypothesis remains unsupported:

- independently composed selected KVs answered only `1/12` in the attention holdout;
- query refresh, latent capsules, placeholder carriers, semantic carriers, and rotating free decode did not produce unique hidden-memory gains;
- packed KV-value similarity was fast but not a robust rare-key retriever.

The broader goal—large cold context with small active attention and meaningfully better answer probability—does have a plausible inference-time path. A tiny token-ID sidecar plus model-written query carriers improved strict held-out accuracy from `0/12` without pages to `10/12` over 128-page mixed-domain archives, while replaying one page at a time. It also matched the full-prefill answer phrase rate at substantially lower measured latency on the 2070S.

The result is synthetic and lexical. It assumes that each next relation exposes a source key that can be carried into another lookup. It does not yet solve implicit semantic links, contradictory records, natural documents, or retrieval where the next page shares no lexical key. The quotation failures show that small carrier wording changes can still derail exact-token retrieval.

The most valuable next tests are therefore:

1. normalize or embed short model-written carriers without using answer labels;
2. scale the token sidecar to thousands of pages while keeping only selected KV/token pages resident;
3. replace synthetic notes with natural multi-document chains and adversarial same-key distractors;
4. benchmark matched generated-token throughput, time to first correct answer, host-to-device page transfer, and page-cache residency separately;
5. test whether training a tiny navigation adapter or memory-token module can recover the latent-KV benefits that inference-only residue did not.

Artifacts: `outputs/phase12` through `outputs/phase15`. The post-freeze primary summaries are `outputs/phase15/navigation_preference_final_seed600_n12_df05/summary.json` and `outputs/phase15/navigation_diverse_postfreeze_seed900_newpool_n12_df1/summary.json`.
