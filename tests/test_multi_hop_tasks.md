# test_multi_hop_tasks.py

## Purpose

Verifies deterministic construction and provenance for questions that require multiple separate pages.

## Components

### `test_multi_hop_task_requires_every_chain_record`
- **Does**: Checks block count, unique chain provenance, answer non-leakage, and the expected endpoints of a four-page chain.

### `test_multi_hop_task_is_reproducible_but_seed_changes_placement`
- **Does**: Confirms a seed fully determines placement while another seed reshuffles the same semantic task.

### `test_depth_one_is_a_direct_carrier_retention_calibration`
- **Does**: Confirms the one-page condition names its subject directly and has one relevant source block.

### `test_large_archive_extends_distractors_without_duplicate_records`

- **Does**: Confirms a 128-page task contains unique deterministic filler notes and still exposes the answer exactly once.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `multi_hop_tasks.py` | Relevant IDs identify logical chain steps inside shuffled physical blocks | Returning unordered provenance |
