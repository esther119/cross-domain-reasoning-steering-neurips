# general_extraction — S_general (`v_general`)

S_general pools the **raw boundary activations** of all three domains (not the finished vectors) into one contrast
`mean(check ∪ switch) − mean(other)`. Phase 1 (default) is count-weighted; `BALANCE=1` gives the equal-domain Phase 2.

* `general_extraction_1.5B/pipeline/build_general_vector.sh` — one GPU command: (re)generates LogiQA traces if needed, extracts
  layer-20 hidden states for MATH (`--keywords math`), APPS (`code`) and LogiQA (`logic`) next to their traces under the domain
  `result/` folders, then pools with `extraction/src/build_general_vector.py`.
  Inputs: `math_extraction_1.5B/result/baseline_10000`, `code_extraction_1.5B/result/baseline_10000`,
  `logic_extraction_1.5B/result/baseline_${LOGIQA_MAX_TOKENS}_regraded` (now `baseline_10000_regraded`; the shipped
  vector pooled the superseded 3k-cap `baseline_3000_regraded`, so S_general is rebuilt once the 10k LogiQA 2.0 pool exists).
* `general_extraction_1.5B/result/S_general_math_apps_logic_phase1.pt` + `.meta.json` — the vector (dim 1536). Apply with coef −1.0 at layer 20.
* `general_extraction_7B/` — not built yet (placeholder).

Compare with S_combo (`../combo_extraction`), the weighted-sum alternative: `docs/s_general_vs_s_combo.html`.
