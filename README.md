# FlashCache

Inference-time experiments for treating a large archive as cold memory around a small decoder model. The current test model is `Qwen/Qwen3-1.7B`; all recorded GPU experiments were guarded to run on an NVIDIA RTX 2070 SUPER, not the occupied RTX 4090.

## Current result

Pure hidden-KV accumulation did not work reliably. Query refresh, latent capsules, placeholder carriers, semantic carriers, rotating decode, and independent composition of selected KV pages all failed causal controls or underperformed exact source replay.

The strongest practical controller is now:

1. retain immutable cold page text/token IDs (cold-page KV is optional and unused by this controller);
2. build a tiny offline IDF token-posting sidecar;
3. retrieve one page from the model's current rewritten question;
4. prefill that one exact page in a short navigation prompt and let Qwen emit either the next question or the final answer;
5. repeat without answer labels, known page IDs, or a known hop count.

On 128-page archives, the frozen controller achieved:

| Suite | Iterative | No page | Full prefill, 64-token horizon | Mean retrieval |
|---|---:|---:|---:|---:|
| 12 personal-preference chains, depths 1–4 | `12/12` | `0/12` | `4/12` strict | `0.136 ms/hop` |
| 12 disjoint history/quotation/place chains, depths 1–4 | `10/12` | `0/12` | `8/12` strict | `0.145 ms/hop` |

The diverse full-prefill control contained the answer phrase in `10/12`, equal to iterative navigation; its lower strict score reflects Qwen's long explanations within the fixed horizon. Mean end-to-end latency was `751 ms` for iterative navigation versus `1344 ms` no-page and `2118 ms` full-prefill, but this is not a matched generated-token throughput benchmark.

This is a promising large-cold-context result, not proof that arbitrary hidden KV pages can be composed. The working method is lexical model-directed navigation with exact selected-token replay.

## Reproduce

Local unit tests:

```bash
python -m pytest -q
```

Primary GPU suite shape:

```bash
PYTHONPATH=. python scripts/run_iterative_navigation_suite.py \
  --seed-base 900 \
  --trial-count 12 \
  --task-set diverse \
  --variant-offset 6 \
  --variant-count 100 \
  --blocks 128 \
  --retrieval-k 1 \
  --max-document-fraction 1.0 \
  --max-navigation-steps 5 \
  --navigation-horizon 32 \
  --answer-horizon 64 \
  --expected-gpu "2070 SUPER" \
  --local-files-only \
  --output-dir outputs/reproduction
```

See `EXPERIMENT_LOG.md` for the complete experimental progression, negative controls, metrics, latency scaling, and limitations. Compact JSON/JSONL traces are retained under `outputs/phase0` through `outputs/phase15`.
