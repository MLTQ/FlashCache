# `iterative_navigation.py`

## Purpose

Defines the answer-free control language for multi-hop archive navigation. The model sees a current question plus a small set of retrieved exact notes and decides either to rewrite the unresolved question with one relationship resolved or to return the final value.

## Components

### `NAVIGATION_SYSTEM_MESSAGE`

- **Does**: Instructs one-step relationship substitution, preservation of the ultimate who/where/what target, and only `LOOKUP:` or `ANSWER:` actions.
- **Rationale**: One step avoids requiring the controller to know the chain depth while keeping generated navigation short. Fixed examples cover possessive substitution, treaty-location completion, and quotation-to-speaker lookup without leaking any evaluated task entity or answer.

### `make_navigation_user_message`

- **Does**: Renders the current question and selected original page text without summaries or relevance labels.

### `canonicalize_lookup_entities`

- **Does**: Extracts values following `is` in selected source notes and snaps a unique one-edit generated word back to the source spelling.
- **Rationale**: Small models may normalize a fictional name such as `Shirly` to `Shirley`; exact token retrieval needs the carrier to preserve the source key.

### `make_navigation_repair_user_message`

- **Does**: Adds the previous no-progress response and requests one nonrepeating, target-preserving answer or right-hand-value substitution from the same current question and exact notes.
- **Rationale**: A small model occasionally repeats a final question even when its single selected note states the value. One bounded retry preserves model-driven stopping without deterministic answer extraction.

### `navigation_decision_needs_target_repair`

- **Does**: Flags the answer-type mismatch where a who-question receives a quoted line instead of a person.
- **Rationale**: The controller can preserve an explicit target type without knowing the expected answer. The same bounded model repair then decides whether the quotation is an intermediate lookup key.

### `parse_navigation_decision`

- **Does**: Parses labeled actions, accepts a bare rewritten question as a lookup, and treats other nonempty first-line output as an answer.
- **Rationale**: Qwen3-1.7B's format compliance is not the experiment target; the parser remains answer-free and does not inspect correctness.
- **Mislabeled-final fallback**: A `LOOKUP:` declarative favorite-food fact becomes an answer only when it is affirmative and its subject already occurs in the current rewritten carrier. An unresolved “my wife” query therefore cannot stop on an unrelated person's food fact.

### `replace_task_question`

- **Does**: Reuses the exact pinned system/archive while replacing only recent query text for the next KV-similarity scan.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Iterative controller | Decisions depend only on current question and retrieved notes | Passing expected answer, hop depth, or relevance labels into the prompt |
| Cold-page reuse | Rewritten task preserves pinned text and block order exactly | Mutating the archive between steps |
| Format tolerance | Unlabeled full questions become lookups; mislabeled final facts require a subject grounded in the current carrier | Rejecting all nonconforming output or accepting ungrounded final facts |
| Stop rule | Controller stops on model-emitted answer, invalid action, repeat question, or fixed safety budget | Using answer correctness as the online stop signal |
| Repair rule | At most one repair is attempted for a repeated lookup, using the same question and notes | Looping repairs or asserting that a particular answer is present |
| Target-type repair | Only question grammar and generated surface type trigger repair | Comparing against an answer label or candidate list |
| Entity canonicalization | Corrections come only from the exact notes selected in that step and require a unique one-edit match | Consulting the ground-truth chain or globally fuzzy-matching the corpus |

## Notes

- Navigation text is an explicit reasoning carrier. It is short and query-directed, unlike a page summary.
- The current implementation replays selected original token text because independent-KV composition remains unreliable.
