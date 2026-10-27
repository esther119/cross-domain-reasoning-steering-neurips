# code_eval — APPS

* `code_eval_task/APPS/` — the evaluation set: `test_rand42_500.jsonl`, the seeded 500 APPS test problems every arm is scored on (pinned by id through
  `--apps_task_file`; hidden tests reload from the `codeparrot/apps` parquet branch at run time; `.with_tests.jsonl.gz` is the reading copy).
* `pipeline/` — `eval_code_vllm.py` (unsteered), `eval_code_steering.py` (steered; per-example checkpoints, sharding, `--finalize_only`),
  `apps_utils.py` (APPS loader/prompt/scoring with difficulty breakdowns; `--timeout 10` per test, matching the extraction scorer),
  `code_evaluation/` (vendored execution harness + `testing_util`; also backs the harness's `--benchmark livecodebench` path, which is kept
  in the code but is not part of the experiment plan). Generated programs are **executed** during grading — run only in an isolated environment.
* `code_eval_1.5B/`, `code_eval_7B/` — `<vector>_on_APPS/`.

`VECTOR=<vector.pt> ARM=<arm> MODEL_SIZE=<size> DATASETS=apps bash scripts/eval_vector.sh 0` is the driver (`RUN_BASELINE=1` adds the baseline;
500 problems, `APPS_TIMEOUT=10`); add `GPUS=0,1,2,3` for the sharded, resumable version.
