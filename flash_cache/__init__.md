# __init__.py

## Purpose

Defines the small public surface of the Flash Cache research package.

## Components

### `inspect_cache`
- **Does**: Re-exports cache-structure inspection for scripts and tests.
- **Interacts with**: `inspect_cache` in `cache_inspection.py`.

### `clone_cache`
- **Does**: Re-exports independent whole-cache cloning for speculative probes.
- **Interacts with**: `clone_cache` in `hybrid_cache.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment scripts | Cache inspection and cloning are importable from `flash_cache` | Removing or renaming either export |

