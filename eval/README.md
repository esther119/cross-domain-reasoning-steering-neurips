# eval/ — evaluating steering vectors

One folder per **benchmark domain**, each with the same shape:

```
eval/
  src/modeling_utils/            the shared steering model (Qwen2 with the layer-20 hook); imported by every *_steering.py
  <domain>_eval/
    <domain>_eval_task/          the eval data that lives in-repo (the rest is loaded from the HF Hub)
    pipeline/                    this benchmark family's own harness: eval_<domain>_vllm.py (unsteered, fast) and
                                 eval_<domain>_steering.py (HF generate + steering), graders/utils, recipes
    <domain>_eval_1.5B/          results for DeepSeek-R1-Distill-Qwen-1.5B, one folder per arm
    <domain>_eval_7B/            results for DeepSeek-R1-Distill-Qwen-7B
```

| domain | benchmark | harness | grader |
|---|---|---|---|
| `math_eval`  | MATH500 (`HuggingFaceH4/MATH-500`); `--dataset MATH_train` generates the extraction traces | `eval_math_vllm.py` / `eval_math_steering.py` | Omni-MATH rule grader (`get_math_results.py` + `eval_math_rule/`) |
| `code_eval`  | APPS test, seeded 500 (`codeparrot/apps` parquet branch) | `eval_code_vllm.py` / `eval_code_steering.py` | tests executed (`apps_utils.py`, `code_evaluation/testing_util`) |
| `logic_eval` | LogiQA 2.0 = `LogiQA2` (English test rows; pinned clean 500 in-repo) | `eval_logic_vllm.py` / `eval_logic_steering.py` | `logic_utils.extract_choice` (`</think>`-anchored, no fallback) |
| `nonreason_eval` | TriviaQA closed-book = `TriviaQA` (validation split, in-repo `test.jsonl`, seeded 500; the NON-REASONING domain); `--dataset TriviaQA_train` generates the extraction traces | `eval_nonreason_vllm.py` / `eval_nonreason_steering.py` | `triviaqa_utils.triviaqa_eval_main` (normalized alias-EM of the post-`</think>` `Answer: X`, unfinished = wrong; `--em_mode aliases` = official, `contains` / `value` diagnostics) |

Retired (moved to the retired-benchmarks folder (removed from the repo on 2026-08-23 (team decision; in git history))): GSM8K, MBPP, MMLU (`knowledge_eval`), LiveCodeBench smokes.

The text harnesses were forked from one file (`eval_MATH_steering.py`, commit 8f38c8f) and trimmed to their own
dataset; the generation / checkpoint / sharding core is identical in each, and the steering vector is just a parameter
(`--steering --steering_vector <pt> --steering_layer 20 --steering_coef -1.0`). Every steering fork also takes
`--gen_mode hf|pruned --prune_chunk N --model_dtype fp32|bf16` (defaults hf / 256 / fp32 = the standard; all three are in the
resume fingerprint; `pruned` drops finished rows between chunks via `eval/src/steer_generate.py` — faster, but a different batch
composition, so arms run with it or with bf16 do not pair exactly with standard arms).

## Protocol (the 1.5B standard — 7B follows it)

* 10,000-token budget, greedy, chat template, `--remove_bos`, layer 20, coef −1.0, seed 42.
* MATH500: all 500. APPS: `--apps_split test --random_sample --sample_seed 42 --max_examples 500`. LogiQA2:
  `--logiqa_english_only --logiqa_eval_selection eval/logic_eval/logic_eval_task/LogiQA2/eval_rand42_500_clean.json`
  (the harness refuses a selection without the English filter — the selection indexes the English-only pool).
  TriviaQA: `--random_sample --sample_seed 42 --max_examples 500 --em_mode aliases` over the 11,313-row validation file.
* One batch size per benchmark per model size, for every arm: 1.5B 25 / 25 / 1 / 25 for MATH500 / LogiQA2 / APPS / TriviaQA
  (`scripts/eval_vector.sh`), 7B 16 / 8 / 2 / 16 (`scripts/eval_vector.sh (MODEL_SIZE=7B)`); plain HF generate in fp32.
  Batch composition is part of an example's numerics (left padding), so arms that are McNemar-paired must share it.
* Every run writes `run_config.json`; resumes refuse a changed fingerprint.

## Result folders: `<vector>_on_<BENCH>/` — flat, one folder per arm

```
eval/math_eval/math_eval_1.5B/
  baseline_on_MATH500/        predictions.jsonl  math_eval.jsonl  metrics.json  example_prompt.txt  run_config.json  RUN_NOTE.md
  v_math_on_MATH500/          same files + figures/ (baseline-vs-steered plots)
  v_code_on_MATH500/  v_logic_on_MATH500/  v_nonreason_on_MATH500/  v_general_on_MATH500/  v_combo_on_MATH500/  R_iso_seed1_on_MATH500/
  baseline_on_MATH500_32k/  v_general_on_MATH500_32k/   the labelled 32k budget-sweep pair (32k tokens, first 250 of the selection; PRs #85/#86) — the one deliberate exception to the 10k budget
  v_combo_on_MATH500_smoke20/ a 20-problem smoke that once shared the v_combo folder (split out when the tree was flattened)
```

* Arm names: `baseline`, `v_math` (S_math), `v_code` (S_code), `v_logic` (S_logic), `v_nonreason` (TriviaQA vector), `v_general` (S_general),
  `v_combo` (S_combo), `R_iso_seedN` (random control). Benchmarks: `MATH500`, `APPS`, `LogiQA2`, `TriviaQA`. Model size is the parent folder
  (no `_7b` suffix on arms or vector files).
* The harness writes straight into `--save_dir`; everything that used to be encoded as nested folders (steering vector, coef, `remove_bos`,
  selection) is in `run_config.json`, and a `RUN_NOTE.md` records the original nesting for runs that were flattened on 2026-08-23. Resuming into
  a folder written under different settings is refused by the `run_config.json` fingerprint, so one folder = one run; use a different arm name
  (`<arm>_on_<BENCH>_<variant>`) for smokes and variants.
* When the same arm was run twice on the same benchmark (7B only), the later job carries a suffix (`_run2`, `_run3`) — `PROVENANCE.md`
  lists the jobs, dates and which steered arms each baseline pairs with.

## Running

```bash
# one vector on MATH500 + APPS + LogiQA2 (+ TriviaQA with DATASETS=math,apps,logiqa,triviaqa; baselines too); arm name -> folder name
DATASETS=math,apps,logiqa bash scripts/eval_vector.sh \
    extraction/math_extraction/math_extraction_1.5B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt \
    v_math 0
# 7B (pinned batch sizes):  VECTOR=<pt> VECTOR_NAME=v_logic DATASETS=math,apps,logiqa bash scripts/eval_vector.sh (MODEL_SIZE=7B) 0
# multi-GPU sharded arm, resumable:  DATASET=math|logiqa|apps|triviaqa scripts/eval_vector.sh (GPUS=…) <vector.pt> <arm>
# random controls:                   scripts/eval_vector.sh (ARM=R_iso_seedN, one call per seed)
# non-reasoning domain end to end:   SIZE=1.5B|7B bash scripts/extract_vector.sh + scripts/eval_vector.sh (DOMAIN/ARM=nonreason)   (prep -> baseline -> S_math arm -> build -> v_nonreason arms)
```

Result folders are tracked by default (the traces are the record): commit a finished run with a plain `git add <arm folder>`. Only `hidden.pt` activation dumps are gitignored (they live on the HF Hub).
Known-stale arms and the re-run list: `../`eval/PROVENANCE.md` (the imported `v_nonreason` arms carry `STALE.md` files: 4k-cap vector,
`--gen_mode pruned`, non-standard batches — `nonreason_eval/README.md`, `PROVENANCE.md`).
