# iterative_search.py

## Purpose

Implements sequential page-by-page Flash Cache search using a sentinel period token. Rejected candidate branches are discarded; only clean no-candidate period steps are committed, preventing rejected pages from leaking into later state.

## Components

### `IterativeSearchResult`
- **Does**: Carries visited-page telemetry, the first page that broke the sentinel pattern, its continuation, and correctness.

### `make_sentinel_search_task`
- **Does**: Wraps the original question in a protocol that requests exactly `.` until the current page contains the answer.
- **Rationale**: Sentinel continuation turns retrieval into an active sequential decision rather than ranking passive distribution changes.

### `make_chat_miss_transition_ids`
- **Does**: Derives the tokenizer-native suffix for a period response, closed assistant turn, fresh next-page user message, and new assistant-generation boundary.
- **Rationale**: Continuing tokens after Qwen's end-of-turn marker traps later pages in an invalid chat state.
- **Rationale**: Qwen's generation prompt includes an empty thinking stub that disappears when a completed assistant message is reconstructed, so the suffix begins at the shared assistant-start boundary rather than the end of the initial rendering.

### `make_inline_miss_transition_ids`
- **Does**: Encodes a compact `.` plus `NEXT PAGE:` controller marker inside the open assistant stream.
- **Rationale**: It refreshes the per-page decision without chat termination or the positional drift of a full follow-up turn.
- **Rationale**: Sentinel and marker are tokenized separately so BPE cannot merge the period with its following newline.

### `classify_gate_tokens`
- **Does**: Treats explicit negative-readiness language as a miss, then skips leading control/formatting tokens until it encounters either an exact or tokenizer-merged period surface or visible alphanumeric content.
- **Rationale**: Small models may emit chat terminators, Markdown, or a prose explanation that the current page lacks the answer instead of following the exact period format.

### `run_iterative_flash_search`
- **Does**: Speculatively tests each page on a private branch, ignores control/punctuation-only continuations, discards misses, commits a clean sentinel transition, and retains the first branch with meaningful visible content.
- **Interacts with**: `advance_cache`, `flash_candidate`, and `rollout` in `probing.py`.
- **Rationale**: Committing the rejected branch's period KV would preserve attention to a page that was supposedly removed.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Iterative experiment | Candidate order is a complete permutation and the first branch with visible alphanumeric content stops search | Continuing past a meaningful break or skipping pages |
| Cache isolation | Misses commit one clean no-candidate step, never the candidate-conditioned branch | Reusing rejected branch state |
| Chat iteration | Each miss transition starts with the sentinel and ends at a fresh assistant-generation probe token | Continuing after an end-of-turn token |
| Correctness evaluation | The retained branch remains resident through answer continuation | Removing the selected candidate before decoding |
| Telemetry | Every visited page records sentinel probability, entropy, margin, and ground-truth labels for offline analysis | Dropping loser rows |

## Notes

- The runner can compare compact inline, full chat-turn, and single-period miss transitions; all are committed on the clean cache.
- The gate treats period-plus-newline token variants as sentinel output before considering later controller words.
- Exact first-token sentinel behavior and probability are still recorded for later threshold analysis.
- A false early break is an end-to-end retrieval failure even if the relevant page appears later.
