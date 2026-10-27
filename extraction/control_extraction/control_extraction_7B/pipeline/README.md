# control_extraction_7B/pipeline

There is no separate 7B builder: `../../control_extraction_1.5B/pipeline/build_random_vectors.sh` builds both sizes. With `MODEL_SIZE=7B`
it writes `../result/R_iso_seed{1,2,3}.pt` + `.meta.json` (dim 3584) under the 1.5B protocol — target norm = ‖S_combo_7b‖ via `--like`
(the mean of the three 7B domain-vector norms), each seed screened (|cos| ≤ 0.08) against every packaged 7B vector (S_math_7b, S_code_7b,
S_logic_7b = `logiqa2_v_logic.pt`, S_combo_7b; S_general_7b once it exists) and against the earlier seeds. It refuses to run if any of the
four is missing.

**The current draws are STALE** (`../result/STALE.md`). `R_iso_7b_seed{1,2,3}` (2026-08-18) were drawn with an explicit `--norm 80.628`
= mean(S_math_7b 82.44, S_code_7b 78.82) — S_logic_7b (93.86) did not exist yet — and screened against those two only. The mean of all
three is 85.04 = ‖S_combo_7b‖. `verify_random_vectors.py --dir extraction/control_extraction/control_extraction_7B/result --prefix R_iso`
fails both protocol checks on them (norm not matched to S_combo_7b; not screened against S_logic_7b / S_combo_7b).

Rebuild (CPU, seconds) — only **after** the 7B S_logic rebuild (regraded pool) and the S_combo_7b rebuild that follows it:

```bash
MODEL_SIZE=7B bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh
python extraction/control_extraction/control_extraction_1.5B/pipeline/verify_random_vectors.py \
    --dir extraction/control_extraction/control_extraction_7B/result --prefix R_iso
```

Then re-run every R_iso_7b eval arm — MATH500 / LogiQA2 / APPS (`eval/*/*_eval_7B/R_iso_seedN_on_<BENCH>/`); results taken with the old
draws are not comparable with the SEAL arms and must not be reported.
