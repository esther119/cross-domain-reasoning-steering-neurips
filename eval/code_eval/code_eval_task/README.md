# code_eval_task

`APPS/` — the evaluation set: the seeded 500 APPS *test* problems every `*_on_APPS` arm is scored on (`test_rand42_500.jsonl`, ids +
statements; `test_rand42_500.with_tests.jsonl.gz`, the same rows with the hidden tests). See `APPS/README.md` for the derivation and the
`--apps_task_file` pin. Hidden tests are reloaded from the `codeparrot/apps` parquet branch at run time by `pipeline/apps_utils.py`.
The APPS *train* split used for extraction is `extraction/code_extraction/code_pool/APPS/train_code.jsonl`.
