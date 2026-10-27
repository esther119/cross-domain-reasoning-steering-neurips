# combo_extraction — S_combo (`v_combo`)

`S_combo = normalize_scale(Ŝ_math + Ŝ_code + Ŝ_logic)`: the three finished domain vectors, unit-normalised, summed with equal
weights, rescaled to the mean source norm so `coef = -1.0` is a like-for-like intervention. CPU only, no hidden.pt.

* `combo_extraction_1.5B/pipeline/build_combo_vector.py` — the builder (imports helpers from `extraction/src/`); `--help` for options.
  ```bash
  python extraction/combo_extraction/combo_extraction_1.5B/pipeline/build_combo_vector.py \
      --out extraction/combo_extraction/combo_extraction_1.5B/result/S_combo.pt
  ```
* `combo_extraction_1.5B/result/S_combo.pt` + `.meta.json` — dim 1536.
* `combo_extraction_7B/result/S_combo.pt` + `.meta.json` — dim 3584, same builder over the 7B S_math / apps_v_code_7b / logiqa2_v_logic
  (`--norm mean-source`, ‖S‖ ≈ 85.04). **STALE pending the S_logic_7b rebuild** (regraded pool; its norm will change): rebuild S_combo_7b
  from the new `logiqa2_v_logic.pt`, then `R_iso_7b` (`MODEL_SIZE=7B bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh`,
  norm-matched to S_combo_7b) and re-run every 7B `v_combo` / R_iso_7b arm.

Why equal weights / why rescale: `docs/s_combo_construction.html`; geometry numbers: `python extraction/src/vector_geometry.py`.
