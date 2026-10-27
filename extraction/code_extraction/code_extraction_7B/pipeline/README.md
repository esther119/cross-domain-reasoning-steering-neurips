# code_extraction_7B/pipeline

There is no separate 7B script: `../../code_extraction_1.5B/pipeline/scripts/extract_vector.sh` is the APPS recipe for both sizes
(`MODEL` / `RESULT_DIR` / `PACKAGE` / `MAX_TOKENS` / `TARGET` / `KEYWORDS` overridable): APPS train in file order, 10,000-token cap,
traces generated until both pools hold 500 (`apps/gen_apps_vllm.py`, scored by `apps/scoring.py`) → layer-20 hidden states
(`--keywords code`, first 500 correct + first 500 incorrect) → `vector_generation.py` → packaged as `../result/apps_v_code.pt`
+ `.meta.json`.

## The shipped 7B vector is STALE / unverifiable (`../result/STALE.md`)

Only `../result/apps_v_code.pt` (dim 3584, ‖S‖ 78.82, committed 2026-08-18) was delivered — no `baseline_10000/`
record (`data.jsonl`, `math_eval.jsonl`, `gen_config.json`, `selection_*_0_500.json`), no `hidden_{correct,incorrect}_0_500/prompts.json`,
no `apps_v_code.meta.json` — and the 7B `v_code` evals (2026-08-12) predate the committed vector. Neither the problem set, the token
cap, the keyword set nor the 500+500 selection can be checked, so it was not built with the in-repo recipe as committed. The 1.5B standard
(`../../code_extraction_1.5B/result/`) ships `apps_v_code.pt` + `.meta.json` and the `baseline_10000/`
record.

## Rebuild (GPU, one 7B pod)

```bash
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
RESULT_DIR=extraction/code_extraction/code_extraction_7B/result \
PACKAGE=extraction/code_extraction/code_extraction_7B/result/apps_v_code.pt \
bash scripts/extract_vector.sh (DOMAIN=code) 0    # prints the commands; run them in order
```

i.e. `extraction/src/apps/gen_apps_vllm.py --split train --max_tokens 10000 --target 500 --resume`
→ `extraction/src/hidden_analysis.py --keywords code --keep_layers 20 --type {incorrect,correct} --start 0 --sample 500`
→ `extraction/src/vector_generation.py --prefixs correct_0_500 incorrect_0_500 --layers 20 --save_prefix 500_500`
→ `cp …/vector_500_500/layer_20_transition_reflection_steervec.pt extraction/code_extraction/code_extraction_7B/result/apps_v_code.pt`
and write `apps_v_code.meta.json` with the same fields as the 1.5B `apps_v_code.meta.json` (model, dataset/split, layer, keyword lists,
selection rule, generation config, selected problem ids, activation counts, vector_dim / vector_norm, source_vector). Commit the
`baseline_10000/` record and `hidden_*_0_500/prompts.json`. APPS generations are executed during
scoring — run in an isolated environment.

## Depends on this vector (re-run / rebuild afterwards, in this order)

1. every 7B `v_code` arm — `eval/code_eval/code_eval_7B/v_code_on_APPS`, `eval/math_eval/math_eval_7B/v_code_on_MATH500`,
   `eval/logic_eval/logic_eval_7B/v_code_on_LogiQA2`;
2. `S_combo_7b` (`extraction/combo_extraction/combo_extraction_7B/`, uses S_code_7b as a source) and every 7B `v_combo` arm;
3. `R_iso_7b` (`extraction/control_extraction/control_extraction_7B/`, norm-matched to S_combo_7b and screened against S_code_7b) and
   every R_iso_7b arm.
