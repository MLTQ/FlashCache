# `continuous_carrier.py`

## Purpose

Implements the uninterrupted rotating-page experiment. The model produces one continuous free-form response while cold pages change beneath it on a fixed token schedule; there are no per-page summaries, textual page markers, resets, gates, or relevance-dependent decisions.

## Mechanism

The persistent cache excludes the current input token. For each decode step:

1. Insert the scheduled cold page between pinned and persistent recent/response KV.
2. Process the current response token and choose the next token greedily.
3. Remove only the flashed page while retaining the processed token's page-conditioned KV.
4. Use the selected token as the next input and continue the same response.

The prompt allows the answer to emerge anywhere in the free response; it does not suppress answer tokens until a sentinel. After the fixed carrier budget, the last selected token is committed under the next scheduled page so every visible carrier token has retained KV. A fixed final-answer cue is then appended without a page solely to create a common scoring point.

The exact replay control starts from the same baseline and processes the original probe plus every visible carrier token at the same positions, but never inserts any page. It receives the identical final cue and horizon.

## Components

### `make_continuous_carrier_task`

- **Does**: Requests one free-form reasoning stream, permits the answer whenever supported, and explicitly forbids page notes or page-level structure.
- **Rationale**: Page boundaries must exist only in the KV schedule, never in the visible response.

### `make_rotation_schedule`

- **Does**: Cycles physical page IDs deterministically, changing pages every configured number of processed tokens.
- **Rationale**: Per-token and short-window trials use the same relevance-blind policy.

### `run_continuous_carrier`

- **Does**: Generates a fixed number of tokens under the rotating schedule, retains each page-conditioned input-token KV, and produces a final answer.
- **Interacts with**: `strip_flashed_page` in `carrier_stream.py` and one-token branching in `probing.py`.

### `replay_continuous_carrier`

- **Does**: Re-encodes the exact processed token IDs without pages and asks the same final question.
- **Rationale**: Distinguishes information carried by token identity from extra information in page-conditioned KV.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Continuous experiment runner | Exactly `carrier_token_count` visible tokens and one final commit | Dropping the last selected token from retained KV |
| Exact replay | Probe and visible token IDs, positions, final cue, and horizon are identical | Decode/re-tokenize or inserting textual page boundaries |
| Rotation comparison | Page schedule depends only on block count, token index, and fixed window | Relevance-aware ordering, stopping, or page retention |
| Accuracy analysis | Correctness is evaluated only on the final answer | Scoring facts mentioned in the working response |

## Notes

- The page that selects a token can differ from the page under which that token is processed on the next autoregressive step. This is the natural token-by-token mechanism: token identity carries the previous page's influence, while its retained KV is conditioned on the newly active page.
- Control tokens are masked only to keep the fixed-budget response alive. No textual format is enforced.
