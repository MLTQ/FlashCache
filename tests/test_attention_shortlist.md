# `test_attention_shortlist.py`

## Purpose

Protects page-span accounting, answer-free attention reductions, deterministic ranking, and selected-cache assembly without requiring a GPU model.

## Coverage

- Variable-length cold pages receive the correct physical spans after pinned KV.
- Subset spans remain keyed by original page IDs and reject duplicates.
- Tail/last-query attention and max-over-query attention remain distinct reductions.
- Ranking is deterministic when scores tie.
- Missing or shape-misaligned attention tensors fail before producing misleading scores.
- Selected cold pages are validated and concatenated in original corpus order rather than score order.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Attention diagnostics | Page IDs map to correct cache spans | Omitting pinned length or assuming equal page sizes |
| Reproducibility | Score ties resolve by ascending page ID | Unstable input-order ranking |
| Shortlist cache | Selection order does not alter corpus order | Concatenating pages in rank order |
