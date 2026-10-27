# extraction/ — building steering vectors

One folder per **source domain** (what the vector is extracted from), each with the same shape:

```
extraction/
  src/                                shared extraction code (see src/README.md)
  <domain>_extraction/
    <domain>_keyword/keywords.json    check/switch keyword set for --keywords <domain>
    <domain>_pool/<Dataset>/          the extraction task set (train problems the traces come from)
    <domain>_extraction_1.5B/
      pipeline/                       the scripts that built this arm (DeepSeek-R1-Distill-Qwen-1.5B)
      result/                         traces, hidden_*_0_N/ activations, vector_*/…steervec.pt, packaged .pt + .meta.json
    <domain>_extraction_7B/
      pipeline/  result/              same, for DeepSeek-R1-Distill-Qwen-7B
  general_extraction/  combo_extraction/  control_extraction/
                                      vectors that are not extracted from one domain
                                      (S_general pooled, S_combo weighted sum, R_iso random controls);
                                      same <x>_extraction_<size>/{pipeline,result} shape, no keyword/pool
```

Domains: `math`, `code`, `logic` (the three reasoning domains) and `nonreason` (TriviaQA closed-book recall — the non-reasoning
control domain, ported 2026-08-23 from the `triviaqa-non-reason` branch; not part of S_general / S_combo / R_iso).

| domain | vector | pool | keywords | 1.5B vector (packaged) | 7B vector |
|---|---|---|---|---|---|
| math  | S_math  (`v_math`)  | `math_pool/MATH/train.jsonl` (MATH train, 7,500) | `math_keyword/keywords.json` (upstream SEAL) | `math_extraction_1.5B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt` | `math_extraction_7B/result/vector_500_500/layer_20_transition_reflection_steervec.pt` |
| code  | S_code  (`v_code`)  | `code_pool/APPS/train_code.jsonl` (APPS train, file order) | `code_keyword/keywords.json` | `code_extraction_1.5B/result/apps_v_code.pt` (+ `.meta.json`) | `code_extraction_7B/result/apps_v_code.pt` |
| logic | S_logic (`v_logic`) | `logic_pool/LogiQA2/train.jsonl` (LogiQA 2.0 train, English rows) | `logic_keyword/keywords.json` (derived from traces) | `logic_extraction_1.5B/result/logiqa2_v_logic.pt` (+ `.meta.json`) — built at a 3k cap, **to be regenerated at 10k** | `logic_extraction_7B/result/logiqa2_v_logic.pt` — **STALE** (archived-grader labels), rebuild pending |
| nonreason | v_nonreason (TriviaQA closed-book; the non-reasoning domain) | `nonreason_pool/TriviaQA/train.jsonl` (10,000 rows, seed-42 shuffle of TriviaQA `unfiltered.nocontext` train) | `nonreason_keyword/keywords.json` (derived from traces) | `nonreason_extraction_1.5B/result/triviaqa_v_nonreason.pt` (+ `.meta.json`) — **STALE: 4k cap**, `contains` labels; regenerate at 10k | `nonreason_extraction_7B/result/triviaqa_v_nonreason.pt` (+ `.meta.json`) — **STALE: 4k cap** |
| general | S_general (`v_general`) | pooled boundary activations of the three domains | — | `general_extraction_1.5B/result/S_general_math_apps_logic_phase1.pt` | not built |
| combo | S_combo (`v_combo`) | equal-weight unit-norm sum of the three domain vectors | — | `combo_extraction_1.5B/result/S_combo.pt` | `combo_extraction_7B/result/S_combo.pt` |
| control | R_iso seeds 1–3 | norm-matched isotropic random directions | — | `control_extraction_1.5B/result/R_iso_seed{1,2,3}.pt` | `control_extraction_7B/result/R_iso_seed{1,2,3}.pt` |

Every vector is `mean(check ∪ switch) − mean(other)` at layer 20 and is applied by **adding `coef * S` with
`coef = -1.0`** (see `src/vector_generation.py`). Evaluate any of them with
`scripts/eval_vector.sh <vector.pt> <arm_name>` — results land under `eval/`.

All pipeline scripts `cd` to the repo root and use root-relative paths; run them from anywhere.

## The pipeline (same four steps for every domain)

1. **Generate + grade traces** on the domain's pool with vLLM (`eval/math_eval/pipeline/eval_math_vllm.py --dataset MATH_train`,
   `src/gen_logiqa_vllm.py`, `src/apps/gen_apps_vllm.py`, `eval/nonreason_eval/pipeline/eval_nonreason_vllm.py --dataset TriviaQA_train`)
   → `result/<Pool>/<model>/<tag>/{data.jsonl, math_eval.jsonl}` — always at the **10,000-token cap**.
   The correct/incorrect label comes from that benchmark's own grader (the eval harness under `eval/`).
2. **Hidden states** at `\n\n` step boundaries inside `<think>` for the first N correct and first N incorrect traces:
   `src/hidden_analysis.py --keywords <domain> --keep_layers 20` → `result/…/hidden_{correct,incorrect}_0_N/{prompts.json, hidden.pt}`.
3. **Vector**: `src/vector_generation.py --prefixs correct_0_N incorrect_0_N --layers 20` → `result/…/vector_<tag>/layer_20_transition_reflection_steervec.pt`.
4. (optional) package / rename the `.pt` next to a `.meta.json` in `result/` (`src/package_vector.py` computes the meta from the build dir).

`hidden.pt` files are large and are not committed today (README placeholders mark where they go); `general_extraction`'s pooler reads them from `result/` directly.

The 1.5B pipelines are the validated standard; 7B arms follow the same recipes. Known gaps and the re-run plan: `../`eval/PROVENANCE.md`.

## Adding a domain

Create `extraction/<d>_extraction/{<d>_keyword/keywords.json, <d>_pool/<Dataset>/, <d>_extraction_1.5B/{pipeline,result}}` (10k cap, greedy, same pool rule).
`src/keyword_sets.py` picks the new keyword set up automatically (`--keywords <d>`). Put the trace generator in `src/`
if it is shared across model sizes, and give the benchmark its own harness under `eval/<d>_eval/pipeline/`.
`nonreason_extraction/` is the worked example of this (pool + eval files from one prep script, keyword set derived with
`logic_keyword/discover_*` + `logic_keyword_coverage.py --noise_set <d>`, one pipeline script for both sizes, `src/package_vector.py`).
