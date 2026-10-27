# nonreason_extraction_7B/pipeline

`scripts/extract_vector.sh` — a thin wrapper (`MODEL` defaults to the 7B distill) around the one TriviaQA recipe,
`../../nonreason_extraction_1.5B/pipeline/scripts/extract_vector.sh`, which infers `MODEL_SIZE` from `MODEL` and then writes to
`../result/baseline_${MAX_TOKENS}/` and packages `../result/triviaqa_v_nonreason.pt` + `.meta.json`:
`eval/nonreason_eval/pipeline/eval_nonreason_vllm.py --dataset TriviaQA_train` (vLLM, 10k tokens, `--em_mode contains`)
→ `hidden_analysis.py --keywords nonreason --keep_layers 20` (first 500 correct + first 500 incorrect, file order)
→ `vector_generation.py` → `extraction/src/package_vector.py`.

## STALE: the committed 7B vector

`../result/triviaqa_v_nonreason.pt` (dim 3584, ‖S‖ 93.96) and its source
`../result/baseline_4000/vector_trivia_500_500/layer_20_transition_reflection_steervec.pt`
were built on the `triviaqa-non-reason` branch at a **4,000-token cap** (3,000 rows attempted, 659 correct / 2,341 incorrect, 302 unfinished;
`contains` EM labels; keyword set `nonreason`, then called `trivia`). The standard is 10,000 tokens. The record is complete
(`math_eval.jsonl`, `metrics.json`, `hidden_*_0_500/prompts.json`, `.meta.json`), so unlike the 7B S_math / S_code it is auditable — it is
stale only because of the cap. See `../result/STALE.md`.

Rebuild (GPU, one 7B pod; `ATTEMPT=3000` reproduces the shipped row count, the default 10,000 is also fine — pools are first-in-first-fill):

```bash
bash scripts/extract_vector.sh (DOMAIN=nonreason MODEL_SIZE=7B) 0
VECTOR=extraction/nonreason_extraction/nonreason_extraction_7B/result/triviaqa_v_nonreason.pt \
VECTOR_NAME=v_nonreason DATASETS=math,apps,logiqa,triviaqa bash scripts/eval_vector.sh (MODEL_SIZE=7B) 0
```

Depends on it: every 7B `v_nonreason_on_*` arm (`eval/*/*_eval_7B/v_nonreason_on_{MATH500,APPS,LogiQA2,TriviaQA}`). Nothing else:
v_nonreason is **not** part of S_general / S_combo / R_iso (those stay math / code / logic).
