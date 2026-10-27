# logic_eval — LogiQA 2.0 (`LogiQA2`)

* `logic_eval_task/LogiQA2/` — `test.jsonl` (English test rows, 1,572 — the pool `pool_idx` indexes), `eval_rand42_500_clean.json`
  (**the 500 to use**: seed-42 sample with the 16 train-overlapping items replaced), `eval_rand42_500.json` (original, 16 overlaps; only the
  first baseline/steered runs scored it), `eval_patch16.json` (the 16 backfills).
* `pipeline/` — `eval_logic_vllm.py`, `eval_logic_steering.py` (`--dataset LogiQA2`; pass `--logiqa_english_only --logiqa_eval_selection <json>`
  — the harness refuses a selection without the English filter, because the selection files index the English-only pool; `--dataset LogiQA`
  is accepted as a deprecated alias), `logic_utils.py` (loader, prompt, `is_cjk_heavy`, the `</think>`-anchored `extract_choice` grader — also
  imported by the extraction generator), `splice_logiqa_clean.py` (restate a 500-row run on the clean set from a 16-row patch run).
* `logic_eval_1.5B/`, `logic_eval_7B/` — `<vector>_on_LogiQA2/…/eval_rand42_500_clean/` (clean) or `…/rand42_500` (legacy seeded sample).
  1.5B arms run through `scripts/eval_vector.sh` (batch 25/25/1); **every 7B arm runs through `scripts/eval_vector.sh (MODEL_SIZE=7B)`**
  (pinned 16/8/2). Five 7B runs scored on a mixed-language set (no `--logiqa_english_only`) were deleted from the repo on 2026-08-23 (in git history); see `logic_eval_7B/README.md`.
