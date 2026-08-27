# test_carrier_stream.py

## Purpose

Checks the cache surgery that distinguishes poisoned carrier state from the clean-state control and verifies the streaming prompt does not leak the answer.

## Components

### `test_strip_flash_keeps_pinned_recent_and_appended_carrier`
- **Does**: Removes a known middle page span from fake KV tensors and proves the final appended carrier entry survives unchanged.

### `test_carrier_prompt_preserves_blocks_and_hides_answer`
- **Does**: Confirms streaming instructions preserve the corpus and original question without including the evaluation answer.

### `test_gate_normalizes_wait_prose_but_keeps_answer_attempts`
- **Does**: Treats periods and explicit insufficiency prose as waits while allowing a substantive response to break the stream.

### `test_gate_ignores_fixed_rollout_tokens_after_control_token`
- **Does**: Proves tokens generated mechanically after a control/end token are excluded from the gate decision.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `carrier_stream.py` | Flash stripping preserves physical order of pinned, recent, and appended state | Compacting or re-encoding retained KV |
| Stream gate | Response formatting does not decide correctness, but substantive answer attempts remain observable failures | Treating all non-period prose as a hit |
