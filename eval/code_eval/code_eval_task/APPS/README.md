# APPS (test) — the evaluation set

`test_rand42_500.jsonl` — the 500 APPS test problems every `*_on_APPS` arm is scored on. Derivation (what the harness did on every
run before this file existed): `load_apps(split="test", max_examples=500, random_sample=True, sample_seed=42)` over the usable rows
(rows with parsable tests) of the `codeparrot/apps` parquet branch (`all/test/0000.parquet`, `0001.parquet`), `random.Random(42).sample`,
sorted by `problem_id`. 314 interview / 95 introductory / 91 competition. Materialised on 2026-08-23 and checked against the
`predictions.jsonl` of every APPS arm at both sizes (baseline, v_code, v_nonreason, R_iso_seed1, … at 1.5B and 7B): same 500 ids, same
order. Fields: `problem_id`, `difficulty`, `question`, `starter_code`, `url` (1 MB).

`test_rand42_500.with_tests.jsonl.gz` — the same 500 rows with `input_output` (the hidden tests: 107 MB raw, 34 MB gzipped; the
reference `solutions` are omitted). A reading copy: the harness never reads it — tests are always reloaded from the parquet branch.

How the harness uses the set: `scripts/eval_vector.sh` passes `--apps_task_file eval/code_eval/code_eval_task/APPS/test_rand42_500.jsonl`
(`APPS_TASK_FILE=`), so the 500 are pinned by id — `apps_utils.load_apps` takes the ids from the file and reloads the tests from APPS —
instead of being re-derived from the parquet row order on every run (`--random_sample --sample_seed 42 --max_examples 500` are still
passed and are a no-op on a 500-row list; the result is the same 500 in the same order, verified above). The one visible difference: new
`run_config.json` files record `apps_task_file` (older ones record `null`).

Regenerate: `python -c "import sys; sys.path.insert(0,'eval/code_eval/pipeline'); from apps_utils import load_apps; ..."` with the call above.
