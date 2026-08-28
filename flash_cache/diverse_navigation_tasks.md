# `diverse_navigation_tasks.py`

## Purpose

Builds fictional unknown-depth navigation tasks whose final answers are historical names, literary speakers, or treaty locations. The varied domains check that iterative retrieval is not peculiar to numeric values or favorite-food wording.

## Components

### `DIVERSE_NAVIGATION_FAMILIES`

- **Does**: Declares the history-person, book-quotation, and history-place task families.

### `DiverseNavigationTask`

- **Does**: Extends the shared synthetic-task schema with every relevant physical page ID and the evaluation-only hop depth.

### `make_diverse_navigation_task`

- **Does**: Creates one-to-four-page logical chains, adds compact same-domain fictional distractors up to 128 pages, and deterministically shuffles the archive.
- **Rationale**: Each hop exposes a lexical key for the next retrieval, while no page contains the complete multi-hop question and final answer together.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Iterative navigation suite | `relevant_block_ids` follows logical hop order | Returning physical archive order |
| Token sidecar | Page texts remain immutable after index construction | Query-dependent page mutation |
| Correctness scoring | Expected answer occurs in one final source page only | Reusing answer names or places in distractors |
| Reproducibility | Family, variant, seed, depth, and page count determine the task | Nondeterministic filler generation |

## Notes

- All people, events, quotations, treaties, and places are fictional, preventing pretraining recall from solving the no-page control.
- Compact distractors retain the final relation wording (`closing address`, `line ... spoken by`, or `signed at`) so retrieval cannot shortlist the real page from a unique template phrase.
- Role and shelf labels are deliberately distinctive so each rewritten carrier can identify the next page without answer labels.
- Hop depth and relevant page IDs are evaluation metadata and never enter retrieval or generation prompts.
- Variant indices 0–5 preserve the exploratory/dev content mapping; indices 6 and above rotate through a separate four-entry pool for post-controller validation.
