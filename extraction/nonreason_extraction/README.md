# nonreason_extraction — v_nonreason (TriviaQA closed-book, the NON-REASONING domain)

The fourth source domain, ported on 2026-08-23 from the `triviaqa-non-reason` branch (there: vector `v_non_rea`,
keyword set `trivia`, `vectors/triviaqa_v_non_rea[_7b].pt`, `results/results_for_{,7b_}trivia_vectors/`). Arm name **`v_nonreason`**,
keyword set **`nonreason`**, dataset **`TriviaQA`** (`mandarjoshi/trivia_qa`, `unfiltered.nocontext`, closed-book — no evidence documents).

**Why this domain.** Closed-book factual recall is CoT-insensitive: Sprague et al. (ICLR 2025, arXiv:2409.12183) measure average
chain-of-thought gains of +12.3 % on math and +14.2 % on symbolic reasoning vs +0.7 % on everything else. Reflection / transition
thoughts on TriviaQA are therefore pure waste, and SEAL steering should cut tokens with accuracy flat — the pre-registered
in-domain prediction. Cite that paper for the "non-reasoning" claim rather than arguing from the benchmark's name. v_nonreason is
also the cross-domain probe: a vector extracted where reasoning does not help, applied to the three reasoning benchmarks.

* `nonreason_keyword/` — `keywords.json` (`--keywords nonreason`; 52 check / 16 switch phrases, all contains-matching), `README.md`
  with the design rationale (on a recall task "I remember / recall / think that X" is the *execution* step, not reflection; the set
  targets epistemic uncertainty about a retrieved fact, being stuck, self-correction), and `discover_nonreason_keywords.sh` (probe
  batch + coverage pass via `../logic_extraction/logic_keyword/logic_keyword_coverage.py --noise_set nonreason`). Probe artefacts:
  `nonreason_extraction_1.5B/result/keyword_probe/` (300 traces, 4k cap, `aliases` EM).
* `nonreason_pool/TriviaQA/train.jsonl` — 10,000 rows = the first 10k of the seed-42 shuffle of the deduplicated 87,542-row train split
  (80 rows overlapping validation by normalized text dropped first). Extraction reads the first `ATTEMPT` rows in file order.
  `nonreason_pool/prepare_triviaqa.py` regenerates it and the eval files (`eval/nonreason_eval/nonreason_eval_task/TriviaQA/{test.jsonl,
  eval_rand42_500.json}`) byte-identically; `nonreason_pool/TriviaQA/README.md` is the data contract (config choice, shuffle / disjointness,
  prompt, grading modes, alias noise).
* `nonreason_extraction_1.5B/`
  * `pipeline/scripts/extract_vector.sh` — **the recipe for both sizes** (`MODEL` selects the distill; `MODEL_SIZE` is inferred):
    `eval/nonreason_eval/pipeline/eval_nonreason_vllm.py --dataset TriviaQA_train` (vLLM, greedy, chat template + BOS stripped,
    **`MAX_TOKENS=10000`**, `ATTEMPT=10000` rows, `EM_MODE=contains` labels) → yield + label-quality check → `src/hidden_analysis.py
    --keywords nonreason --keep_layers 20` (first 500 correct + first 500 incorrect, file order) → `src/vector_generation.py` →
    `src/package_vector.py` → `result/triviaqa_v_nonreason.pt` + `.meta.json`. `KEYWORDS=math` is the control build (packaged with a
    `_mathkw` suffix). Why `contains`: strict EM rejects verbose-but-correct recall and would not fill 500 correct from 10,000 rows at 1.5B
    (`acc_official` 0.049 vs `acc_contains` 0.0765); the alias noise it admits is 0.9 % of rows.
  * `result/baseline_4000/` — the shipped build: all 10,000 graded traces (`math_eval.jsonl`,
    `metrics.json`: 765 correct / 9,235 incorrect, 1,850 unfinished, `em_mode: contains`), `hidden_{correct,incorrect}_0_500/prompts.json`
    (hidden.pt not committed), `vector_trivia_500_500/layer_20_transition_reflection_steervec.pt`.
  * `result/triviaqa_v_nonreason.pt` + `.meta.json` — **the packaged v_nonreason used everywhere today** (dim 1536, ‖S‖ 57.38; 9,690
    boundary activations, 76 % reflection/transition). The meta is the branch's build record with only `name` renamed.

    > **STALE — built at a 4,000-token cap (standard: 10,000), `contains` labels.** `result/STALE.md`. It is the current best v_nonreason and
    > stays until the 10k pool exists. Regenerate on a GPU with `bash scripts/extract_vector.sh (DOMAIN=nonreason) 0`
    > (→ `baseline_10000/`, repackages `triviaqa_v_nonreason.pt`), then re-run every 1.5B `v_nonreason_on_*` arm.
* `nonreason_extraction_7B/`
  * `pipeline/scripts/extract_vector.sh` — thin wrapper (`MODEL` = the 7B distill) around the 1.5B script; `pipeline/README.md`.
  * `result/baseline_4000/` — the shipped 7B build: 3,000 traces (659 correct / 2,341 incorrect,
    302 unfinished), `hidden_*_0_500/prompts.json`, `vector_trivia_500_500/…steervec.pt`.
  * `result/triviaqa_v_nonreason.pt` + `.meta.json` — the 7B v_nonreason (dim 3584, ‖S‖ 93.96). **STALE** for the same reason (4k cap;
    `result/STALE.md`); rebuild with the wrapper, then re-run the 7B `v_nonreason_on_*` arms through `scripts/eval_vector.sh (MODEL_SIZE=7B)`.

**Not part of S_general / S_combo / R_iso.** Those stay 3-domain (math / code / logic): `general_extraction` pools only the MATH / APPS /
LogiQA2 activations, `combo_extraction` sums the three domain vectors, `control_extraction` norm-matches and screens against them.
v_nonreason is evaluated as its own arm only.

## Results (branch numbers, 4k-cap vectors) — details in `eval/nonreason_eval/`

1.5B (500-example evals, 10k-token cap, greedy): in-domain TriviaQA acc 0.052 → 0.052 at **−28 % tokens** (answer rate 77.8 → 84.0 %);
MATH500 0.644 → **0.780**; LogiQA2 clean-500 0.258 → **0.304**; APPS 0.112 → 0.106. 7B: TriviaQA 0.166 → **0.194** (−38 % tokens),
MATH500 0.844 → **0.892**, APPS 0.284 → **0.324**, LogiQA (mixed-language selection, not the clean 500) 0.514 → 0.516. The mechanism is the
same everywhere: the vector never improves a finished answer, it makes 6–18 pts more sequences stop re-checking and commit; net accuracy is
(rescued loopers) × (how often the committed answer is right). Geometry at 1.5B: cos to S_math 0.63, S_logic 0.77, S_code 0.62 — the most
off-axis domain vector. Per-arm tables, provenance and the `v_nonreason_on_*` folders: `eval/nonreason_eval/README.md` and the other
`eval/<d>_eval/<d>_eval_<size>/v_nonreason_on_<BENCH>/` dirs.
