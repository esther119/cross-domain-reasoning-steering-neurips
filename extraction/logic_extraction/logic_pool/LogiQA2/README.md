# LogiQA 2.0 extraction pool (English rows)

`train.jsonl` — 12,567 English MRC rows of the LogiQA 2.0 train split (the untranslated Chinese mirror is dropped with
`is_cjk_heavy`, threshold 0.1, in `eval/logic_eval/pipeline/logic_utils.py` — the same filter the eval side uses).
The generator (`extraction/src/gen_logiqa_vllm.py --split train --filter_cjk`) streams `datatune/LogiQA2.0` with the same filter;
this file is the snapshot of that stream in order.

The eval side (test split, the pinned 500-problem selections) lives in `eval/logic_eval/logic_eval_task/LogiQA2/`.
Known overlap: LogiQA 2.0's train and test splits are not disjoint — 16 of the original 500 eval items also appear in the
2,000 attempted train rows, which is why `eval_rand42_500_clean.json` exists.
