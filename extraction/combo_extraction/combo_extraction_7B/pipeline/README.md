# combo_extraction_7B/pipeline

`../result/S_combo.pt` was built with `../../combo_extraction_1.5B/pipeline/build_combo_vector.py` pointed at the three 7B domain vectors (see `S_combo.meta.json` for the exact inputs and norm).

**Pending rebuild:** one of those inputs, `extraction/logic_extraction/logic_extraction_7B/result/logiqa2_v_logic.pt`, is STALE — it was built from
traces labelled by the archived grader (73 "correct" traces never closed `</think>`; see `extraction/logic_extraction/logic_extraction_7B/pipeline/README.md`).
Rebuild S_logic_7b from `…/baseline_10000_regraded` first, then rerun the builder here and re-run the 7B `v_combo` arms through
`scripts/eval_vector.sh (MODEL_SIZE=7B)`. Until then the 7B S_combo numbers in `docs/steering_transfer_results_7b_combo.html` are provisional.
