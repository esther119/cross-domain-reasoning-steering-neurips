# logic_eval_task/LogiQA2

| file | what |
|---|---|
| `test.jsonl` | LogiQA 2.0 test, English MRC rows (1,572 of 3,166); order = `load_logiqa(english_only=True)` order, which `pool_idx` indexes |
| `eval_rand42_500_clean.json` | **use this**: 500 items, zero overlap with anything generated for extraction |
| `eval_rand42_500.json` | original seeded 500 (16 items also appear in the attempted train rows) — kept because the first runs scored it |
| `eval_patch16.json` | the 16 backfills used to restate old runs (`pipeline/splice_logiqa_clean.py`) |

Pass a selection with `--logiqa_english_only --logiqa_eval_selection <file>`; the leaf result folder is named after it (`eval_rand42_500_clean/`).
The selection files index the English-only pool, so the harness refuses `--logiqa_eval_selection` without `--logiqa_english_only`
(applied to the unfiltered 3,166-row pool the same indices pick a half-Chinese set — that is what happened to the four archived 7B
the five deleted mixed-language runs).
