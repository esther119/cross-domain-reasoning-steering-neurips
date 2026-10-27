# logic_extraction — S_logic (`v_logic`), from LogiQA 2.0

* `logic_keyword/` — `keywords.json` (`--keywords logic`), plus the tooling that derived it from real traces:
  `discover_logic_keywords.sh` (probe batch + coverage) and `logic_keyword_coverage.py` (which phrases the `math` set misses, ranked).
* `logic_pool/LogiQA2/train.jsonl` — LogiQA 2.0 train, English rows only (12,567 of 25,318; `is_cjk_heavy` filter).
* `logic_extraction_1.5B/`
  * `pipeline/scripts/extract_vector.sh` — traces (`src/gen_logiqa_vllm.py`, 2,000 rows, **10k tokens**) → hidden states → vector.
  * `pipeline/relabel_logiqa_build.py` + `pipeline/scripts/extract_vector.sh` — the re-grade: generations that never closed `</think>` are
    INCORRECT (the legacy extractor fabricated a letter); traces are re-labelled, not dropped, and the vector rebuilt.
    `scripts/extract_vector.sh` builds from `BUILD_TAG` (default `baseline_10000_regraded`).
  * `result/baseline_3000/` — original traces + legacy labels (`gen_config.json`, `selection_*.json`,
    `data.jsonl`, `math_eval.jsonl`); `baseline_3000_regraded/` — the strictly-graded set, `hidden_*_0_500/prompts.json`, and
    `vector_logic_500_500/…steervec.pt`.
  * `result/logiqa2_v_logic.pt` + `.meta.json` — the packaged S_logic used everywhere today.

    > **Built at a 3,000-token cap; superseded by the 10k protocol.** Every eval and the MATH/APPS extraction pools use 10,000 tokens; this
    > pool alone was generated at 3,000, so its traces are truncated relative to the rest of the study. It is the current best vector and stays
    > in place until the 10k pool exists. Regenerate on a GPU with
    > `bash scripts/extract_vector.sh (DOMAIN=logic) 0` (now 10k → `baseline_10000/`) →
    > `python extraction/logic_extraction/logic_extraction_1.5B/pipeline/relabel_logiqa_build.py --build_dir …/baseline_10000 --out_dir …/baseline_10000_regraded` →
    > `bash scripts/extract_vector.sh (DOMAIN=logic SKIP_GEN=1 TAG=<regraded tag>) 0`, then re-package as `logiqa2_v_logic.pt`.
    > **S_general, S_combo and R_iso depend on it** (S_general pools this pool's boundary activations; S_combo sums this vector; R_iso is
    > norm-matched to it), so all three are rebuilt after it, and every `v_logic` / `v_general` / `v_combo` / `R_iso` arm is re-run.
  * `result/logs/` — run logs of the rebuild.
* `logic_extraction_7B/`
  * `pipeline/scripts/extract_vector.sh` — same pipeline on the 7B model (10k budget); `SAVE_TAG`/`BUILD_DIR` select the build dir and it
    runs the same strict-label guard as the 1.5B recipe. See `logic_extraction_7B/pipeline/README.md`.
  * `result/baseline_10000/` — the 1,250 traces (10k cap) with the **archived pipeline's** labels
    (the archived `logic/gen_logiqa2_vllm.py` pipeline (in git history; the folder was removed on 2026-08-23), whose grader has a bare-letter fallback: 73 of its 724 "correct" traces never
    closed `</think>`), plus `vector_500_500/…steervec.pt` built from those labels.
  * `result/baseline_10000_regraded/` — **the canonical set**: the same 1,250 traces re-graded on 2026-08-23 with
    `eval/logic_eval/pipeline/logic_utils.extract_choice` via the 1.5B `relabel_logiqa_build.py` (650 correct / 600 incorrect, 185 unfinished,
    74 labels flipped correct→incorrect, all of them unfinished).
  * `result/logiqa2_v_logic.pt` — **STALE**: built from the archived-grader labels. Rebuild from `baseline_10000_regraded`
    (`bash scripts/extract_vector.sh (DOMAIN=logic MODEL_SIZE=7B) 0`), then rebuild **S_combo_7b** and
    **R_iso_7b** (both derived from it) and re-run the 7B `v_logic` / `v_combo` / `R_iso` arms through `scripts/eval_vector.sh (MODEL_SIZE=7B)`.
