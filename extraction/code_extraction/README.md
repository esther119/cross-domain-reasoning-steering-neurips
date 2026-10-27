# code_extraction — S_code (`v_code`)

* `code_keyword/keywords.json` — code-adapted check/switch lists (`--keywords code`; contains-matching, + hmm / what if / edge case …).
* `code_pool/APPS/train_code.jsonl` — APPS train in file order (5,000 rows; parquet branch of `codeparrot/apps`).
* `code_extraction_1.5B/`
  * `result/apps_v_code.pt` + `.meta.json` — **the S_code used everywhere** (500 correct + 500 incorrect APPS traces, layer 20).
    Built in the companion code-vector repo with the same pipeline; the meta records the full recipe.
  * `result/baseline_10000/` — the shipped traces that built it (`data.jsonl`, `math_eval.jsonl.gz`, `gen_config.json`,
    `selection_{correct,incorrect}_0_500.json`). `general_extraction` decompresses `math_eval.jsonl.gz` in place on first use and writes
    `hidden_{correct,incorrect}_0_500/hidden.pt` next to it (`--keywords code`).
  * `pipeline/scripts/extract_vector.sh` — the recipe (prints the commands). The earlier MBPP-based code vector (120+120 traces) is retired:
    the retired-benchmarks folder (removed from the repo on 2026-08-23 (team decision; in git history)).
* `code_extraction_7B/result/apps_v_code.pt` — 7B S_code (dim 3584). **Unverifiable / STALE**: no traces, hidden prompts or meta were shipped and the 7B evals predate the committed file; rebuild with the recipe above on the 7B model (see `code_extraction_7B/pipeline/README.md`).
