# `test_diverse_navigation_tasks.py`

## Purpose

Checks that the mixed-answer navigation archives remain deterministic, scale to 128 unique pages, preserve logical provenance, and keep the expected answer out of the question and all distractors.

## Coverage

- Exercises history-person, literary-quotation, and history-place depth-four archives.
- Confirms deterministic regeneration and seed-dependent physical placement.
- Ensures final-relation wording occurs in most distractors rather than uniquely identifying the answer page.
- Rejects unknown families and invalid depths.
- Confirms the post-freeze variant pool has disjoint answers from legacy variants 0–5.
