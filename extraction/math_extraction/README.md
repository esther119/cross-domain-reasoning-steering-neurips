# math_extraction — S_math (`v_math`)

* `math_keyword/keywords.json` — upstream SEAL's check/switch lists (`--keywords math`).
* `math_pool/MATH/train.jsonl` — MATH train (7,500 problems); traces are generated in file order.
* `math_extraction_1.5B/` — `pipeline/scripts/extract_vector.sh` is the upstream recipe (prints the four commands);
  run the printed commands on a GPU pod. `result/baseline_10000/` holds the
  graded traces (`math_eval.jsonl`, `predictions.jsonl`, `metrics.json`), `hidden_{correct,incorrect}_0_500/prompts.json`
  (hidden.pt is not committed) and the vector `vector_500_500/layer_20_transition_reflection_steervec.pt` — the S_math used everywhere.
* `math_extraction_7B/` — `pipeline/scripts/extract_vector.sh` (same recipe, 7B model). `result/baseline_10000/vector_500_500/…steervec.pt`
  is the 7B S_math (only the vector was shipped — no traces, hidden prompts or meta, so it cannot be audited against the 1.5B standard; see `math_extraction_7B/pipeline/README.md` for the rebuild).
