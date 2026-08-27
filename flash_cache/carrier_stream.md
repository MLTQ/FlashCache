# carrier_stream.py

## Purpose

Implements the intended “poisoned period” mechanism: each page is flashed for one response step, removed afterward on a miss, and the token KV created while attending to that page remains in the persistent state. This lets later pages interact with latent residue from earlier pages without selecting or retaining those pages directly.

## Components

### `CarrierStreamResult`
- **Does**: Stores every page-conditioned generation step plus any response that breaks the period stream.

### `make_carrier_stream_task`
- **Does**: Instructs the model to emit one period while accumulated evidence is insufficient and answer when the ongoing state supports it.
- **Rationale**: The prompt describes a continuous evidence stream rather than asking whether one isolated page contains the complete answer.

### `strip_flashed_page`
- **Does**: Removes the inserted page span from an advanced cache while retaining the appended token KV that was computed with the page present.
- **Interacts with**: Dense slicing/concatenation in `dense_cache.py`.
- **Rationale**: Retaining this page-conditioned state is the experimental variable; recomputing the step on the clean cache is the control.

### `classify_carrier_gate`
- **Does**: Maps punctuation-only output and explicit insufficiency prose to a wait while treating other visible content as an answer attempt.
- **Rationale**: Small-model compliance with the one-period format must not stop a trial before it reaches useful pages.

### `visible_tokens_before_control`
- **Does**: Truncates a fixed-horizon speculative response at its first special/control token before gate classification.
- **Rationale**: Greedy test rollouts continue mechanically after end-of-turn; post-termination junk must not become a false answer attempt.

### `run_carrier_stream`
- **Does**: Flashes one page for a configurable number of period-carrier tokens, carries their page-conditioned state across misses, optionally permits a speculative answer to break the stream, and uses a fixed final-answer cue if the flash budget ends first.
- **Interacts with**: `cache_tensor_error` in `dense_cache.py` to record whether each retained token KV actually differs from its no-page counterpart.
- **Interacts with**: `advance_cache`, `flash_candidate`, and `rollout` in `probing.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Carrier-stream runner | Page order may contain repeated full-corpus passes and is never filtered by relevance | Requiring a permutation or using provenance labels for control flow |
| Clean-state control | The same speculative page response is classified, but miss-state KV is recomputed without the page | Carrying candidate KV in both conditions |
| Cache surgery | Active layout is pinned, flashed page, persistent recent/carrier state, appended current token | Moving the flash insertion boundary |
| Correctness evaluation | Ground truth affects telemetry and final scoring only | Consulting relevant IDs during generation |
| End-of-budget comparison | Poisoned and clean states receive the identical answer cue after the same page budget | Varying the final prompt or page count by condition |
| Forced-sweep mode | Each page contributes one state update and a normalized period without interpreting intermediate token content | Stopping or choosing pages from evaluation labels |
| Warmup-then-break mode | One complete physical corpus pass is carried before answer attempts are permitted on the next pass | Enabling based on relevant-page count |
| Carrier width | Poisoned and clean conditions advance the same number of forced period tokens per page | Using wider state only for one condition |

## Notes

- Every classified wait is normalized to a period as the next visible carrier token. Its retained predecessor KV was computed with the current page present.
- A substantive answer attempt is decoded speculatively with the current page retained and ends the stream, whether correct or incorrect.
- Tokens mechanically generated after an end-of-turn control token remain in raw telemetry but do not affect the gate decision.
- The final answer cue is independent of relevance labels and hop depth. It distinguishes failure to break the sentinel from failure to retain useful latent state.
- Forced-sweep mode uses the ordinary question prompt and makes the final token choice only after every scheduled flash; sentinel mode allows an earlier answer attempt.
- Warmup-then-break mode also uses the ordinary question, but permits token-level answer attempts after one relevance-blind corpus pass.
- Width one is the literal page-per-token proposal. Wider settings test whether several page-conditioned placeholder states are needed to preserve relational content across layers.
- Per-step token-KV deltas are diagnostic only; generation never consults them.
