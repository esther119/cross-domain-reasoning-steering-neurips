# math_extraction_7B/pipeline

`scripts/extract_vector.sh` prints the S_math recipe for `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (the 7B variant of
`../../math_extraction_1.5B/pipeline/scripts/extract_vector.sh`; `MODEL` / `RESULT_DIR` / `MAX_EXAMPLES` / `MAX_TOKENS` / `TARGET` / `KEYWORDS`
are overridable): MATH train, first 3,000 problems in file order, 10,000-token cap, chat template with BOS stripped → layer-20 hidden states
at step boundaries (`--keywords math`, first 500 correct + first 500 incorrect) → `vector_generation.py` →
`../result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt`.

## The shipped 7B vector is STALE / unverifiable (`../result/STALE.md`)

Only the `.pt` (dim 3584, ‖S‖ 82.44, committed 2026-08-18) was delivered — no `math_eval.jsonl` / `predictions.jsonl` / `metrics.json`,
no `hidden_{correct,incorrect}_0_500/prompts.json`, no `.meta.json`. Neither the problem set, the token cap, the keyword set nor the
500+500 selection can be checked, so it was not built with the in-repo recipe as committed. The 1.5B standard
(`../../math_extraction_1.5B/result/baseline_10000/`) ships the full record: 3,000 graded traces,
`hidden_*_0_500/prompts.json` and the vector.

## Rebuild (GPU, one 7B pod)

```bash
bash scripts/extract_vector.sh (DOMAIN=math MODEL_SIZE=7B) 0     # prints the 4 commands; run them in order
```

i.e. `eval/math_eval/pipeline/eval_math_vllm.py --dataset MATH_train --max_examples 3000 --max_tokens 10000 --use_chat_format --remove_bos`
→ `extraction/src/hidden_analysis.py --keywords math --keep_layers 20 --type {incorrect,correct} --start 0 --sample 500`
→ `extraction/src/vector_generation.py --prefixs correct_0_500 incorrect_0_500 --layers 20 --save_prefix 500_500`.
Commit the traces (`math_eval.jsonl`, `predictions.jsonl`, `metrics.json`), `hidden_*_0_500/prompts.json`, and write a `.meta.json` next
to the vector (model, pool, cap, keyword set, selection rule, activation counts, norm) as the 1.5B arms do.

## Depends on this vector (re-run / rebuild afterwards, in this order)

1. every 7B `v_math` arm — `eval/math_eval/math_eval_7B/v_math_on_MATH500`, `eval/logic_eval/logic_eval_7B/v_math_on_LogiQA2`,
   `eval/code_eval/code_eval_7B/v_math_on_APPS`;
2. `S_combo_7b` (`extraction/combo_extraction/combo_extraction_7B/`, uses S_math_7b as a source) and every 7B `v_combo` arm;
3. `R_iso_7b` (`extraction/control_extraction/control_extraction_7B/`, norm-matched to S_combo_7b and screened against S_math_7b) and
   every R_iso_7b arm.
