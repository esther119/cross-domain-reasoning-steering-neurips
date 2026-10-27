# math_eval — MATH500

* `math_eval_task/` — `MATH/test.jsonl` (full MATH test, 5,000; not used by the harness — MATH-500 is loaded from `HuggingFaceH4/MATH-500`).
* `pipeline/` — `eval_math_vllm.py` (unsteered baseline / trace generation, vLLM; `--dataset MATH500`, or `--dataset MATH_train` to generate
  the S_math extraction pool from `extraction/math_extraction/math_pool/MATH/train.jsonl`), `eval_math_steering.py` (steered, HF generate,
  batch checkpoints + `--num_shards` sharding), `get_math_results.py` + `eval_math_rule/` (vendored Omni-MATH rule grader; `setup_runpod.sh`
  installs its `latex2sympy`), `baseline.sh` / `steering.sh` (MATH500 recipes).
* `math_eval_1.5B/`, `math_eval_7B/` — `<vector>_on_MATH500/` (see `../README.md` for naming).

`DATASETS=math bash scripts/eval_vector.sh <vector.pt> <arm>` is the driver (baseline + steered, 500 problems);
`DATASET=math scripts/eval_vector.sh (GPUS=…)` for the sharded, resumable version.
