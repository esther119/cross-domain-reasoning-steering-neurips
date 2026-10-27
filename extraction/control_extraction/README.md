# control_extraction — R_iso random-direction controls

`R_iso = target_norm · g/‖g‖, g ~ N(0, I_d)` from a pinned seed (seeds 1–3), evaluated exactly like the SEAL arms (layer 20, coef −1.0).
Rationale: `docs/random_vector_build_plan.html`.

## Protocol (validated at 1.5B; 7B must follow it)

* **Norm** — built with `--like S_combo_<size>`: the target norm is ‖S_combo‖ of that size, which is the mean of the three domain-vector
  norms (S_combo is scaled `mean-source`). 1.5B: 57.06 (S_math 59.67 / S_code 56.22 / S_logic 55.29). 7B: 85.04 (S_math 82.44 /
  S_code 78.82 / S_logic 93.86 with the current S_logic_7b). Never a typed number.
* **Screen** — each seed must have |cos| ≤ 0.08 against **every packaged vector of that size** that exists on disk (S_math, S_code,
  S_logic, S_general where built, S_combo) and against every earlier seed. The builder refuses to run if S_math, S_code, S_logic or
  S_combo of that size is missing; a missing S_general (not built at 7B) is skipped with a warning.
* **Record** — a `.meta.json` next to each `.pt` (seed, attempts, target_norm, references, torch version). `verify_random_vectors.py`
  checks both protocol points against the packaged vectors of the size it infers from `--dir` / `--prefix` (or `--model-size`).

## Files

* `control_extraction_1.5B/pipeline/build_random_vectors.sh` — one script for both sizes (`MODEL_SIZE=1.5B|7B`, default 1.5B), calls
  `build_random_vector.py` per seed; `verify_random_vectors.py` is the CPU checklist `scripts/eval_vector.sh (ARM=R_iso_seedN, one call per seed)` runs before GPU time.
* `control_extraction_1.5B/result/R_iso_seed{1,2,3}.pt` + `.meta.json` — dim 1536, norm 57.06, screened against all five 1.5B vectors. **Valid.**
* `control_extraction_7B/result/R_iso_seed{1,2,3}.pt` + `.meta.json` — dim 3584. **STALE** (`control_extraction_7B/result/STALE.md`):
  drawn on 2026-08-18 with an explicit `--norm 80.628` = mean of only S_math_7b and S_code_7b (S_logic_7b did not exist yet) and screened
  against those two only. The protocol norm is ‖S_combo_7b‖ = 85.04 and the screen must include S_logic_7b and S_combo_7b; the verify
  command below fails both protocol checks on them.

## Build / verify

```bash
bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh                 # 1.5B: reproduces the committed R_iso_seed{1,2,3}
MODEL_SIZE=7B bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh   # 7B:   R_iso_7b_seed{1,2,3} with the same protocol
python extraction/control_extraction/control_extraction_1.5B/pipeline/verify_random_vectors.py                                                       # 1.5B
python extraction/control_extraction/control_extraction_1.5B/pipeline/verify_random_vectors.py --dir extraction/control_extraction/control_extraction_7B/result --prefix R_iso
```

The 7B rebuild is CPU-only (seconds) but must run **after** the 7B S_logic rebuild (regraded pool) **and** the S_combo_7b rebuild that
follows it — S_combo_7b sets the norm and is one of the screens. Order: S_logic_7b → S_combo_7b → `MODEL_SIZE=7B … build_random_vectors.sh`
→ verify → re-run **every** R_iso_7b eval arm (MATH500 / LogiQA2 / APPS; `eval/*/*_eval_7B/R_iso_seedN_on_<BENCH>/`). Running it earlier
would match the draws to the old S_logic_7b norm and make them stale again.

* Evaluate: `scripts/eval_vector.sh (ARM=R_iso_seedN, one call per seed)` (1.5B; `VEC_DIR=… PREFIX=R_iso_7b` for 7B) or
  `scripts/eval_vector.sh <pt> R_iso_seedN` → `eval/*/*_eval_<size>/R_iso_seedN_on_<BENCH>/`.
