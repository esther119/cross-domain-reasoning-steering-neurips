# logic_extraction_7B/pipeline

`scripts/extract_vector.sh` — the 7B S_logic build, mirroring the canonical 1.5B recipe
(`../../logic_extraction_1.5B/pipeline/scripts/extract_vector.sh` + `scripts/extract_vector.sh`):
`extraction/src/gen_logiqa_vllm.py` (vLLM, 10k tokens) → `hidden_analysis.py --keywords logic` → `vector_generation.py`.

* `SAVE_TAG` (or `BUILD_DIR`) picks the build dir under `result/`; default
  `baseline_10000_regraded`, i.e. traces are reused and only hidden states + vector are (re)built.
* It runs the same **strict-label guard** as the 1.5B pipeline: any reused trace labelled CORRECT while never
  closing `</think>` aborts the build with the regrade command to run.

## STALE: the committed 7B vectors

`../result/logiqa2_v_logic.pt` and `../result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt`
were built from labels produced by the **archived** pipeline (the archived `logic/gen_logiqa2_vllm.py` pipeline (in git history; the folder was removed on 2026-08-23)), whose grader falls back to a
bare letter when the generation never closed `</think>`: 73 of its 724 "correct" traces stated no answer. The canonical grader is
`eval/logic_eval/pipeline/logic_utils.extract_choice` (no fallback).

Regraded on 2026-08-23 into `../result/baseline_10000_regraded/`
(1,250 traces kept, **650 correct / 600 incorrect**, 185 unfinished, **74 labels flipped correct→incorrect**, every flip an unfinished trace;
no trace is `all_eval: true` without a closing `</think>`). Both classes still fill 500, so the 500+500 build stands.

Rebuild, then re-package and re-derive everything downstream:

```bash
bash scripts/extract_vector.sh (DOMAIN=logic MODEL_SIZE=7B) 0     # SAVE_TAG=baseline_10000_regraded
cp .../baseline_10000_regraded/vector_logic_500_500/layer_20_transition_reflection_steervec.pt \
   extraction/logic_extraction/logic_extraction_7B/result/logiqa2_v_logic.pt     # + refresh the .meta.json
```

**S_combo_7b** (`extraction/combo_extraction/combo_extraction_7B/result/S_combo.pt`, an equal-weight unit-norm sum that includes this vector)
and **R_iso_7b** (`extraction/control_extraction/control_extraction_7B/result/`, norm-matched to it) depend on it and must be rebuilt after it;
then re-run the 7B `v_logic`, `v_combo` and `R_iso_seed1` arms through `scripts/eval_vector.sh (MODEL_SIZE=7B)` (pinned batch sizes 16 / 8 / 2).
