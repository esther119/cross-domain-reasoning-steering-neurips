# nonreason_eval — TriviaQA closed-book, the NON-REASONING domain (`nonreason`, arm `v_nonreason`)

Closed-book factual recall is where chain-of-thought buys essentially nothing (Sprague et al., ICLR 2025: +12–14 % on
math/symbolic vs +0.7 % elsewhere), so it is the control domain for "SEAL cuts tokens with accuracy flat". Ported
2026-08-23 from branch `triviaqa-non-reason` (there: dataset `TriviaQA`, vector `v_non_rea[_7b]`).

```
nonreason_eval/
  nonreason_eval_task/TriviaQA/   test.jsonl (11,313 validation rows, native order) + eval_rand42_500.json (the seeded 500) + README
  pipeline/                       eval_nonreason_vllm.py (unsteered / extraction traces), eval_nonreason_steering.py (HF generate + steering),
                                  triviaqa_utils.py (loader, prompt, alias-EM grader)
  nonreason_eval_1.5B/            baseline_on_TriviaQA/  v_nonreason_on_TriviaQA/   (+ STALE.md)
  nonreason_eval_7B/              baseline_on_TriviaQA/  v_nonreason_on_TriviaQA/   (+ STALE.md)
  RESULTS_v_nonreason_branch.md   the branch's write-up of the imported arms (provisional numbers; read its header)
```

* **Task data.** `mandarjoshi/trivia_qa` config `unfiltered.nocontext` (one row per question; the `test` split ships without
  answers, so validation is the eval pool). The extraction pool (`train.jsonl`, 10k-row seed-42 shuffled prefix, disjoint
  from validation) lives in `extraction/nonreason_extraction/nonreason_pool/TriviaQA/`; the prep script is
  `extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py`.
* **Harness.** Forks of the math text harness (same checkpoint / shard / fingerprint core): `--dataset TriviaQA` (eval) |
  `TriviaQA_train` (traces). Prompt = `triviaqa_utils.TRIVIAQA_INSTRUCTION` ("Think step-by-step, then end your response
  with 'Answer: X'"; the wording is kept across domains so a token delta reflects the domain, not the instruction), single
  user turn, chat template, BOS stripped. Grader = normalized exact match of the post-`</think>` `Answer: X` against `gt`
  + `aliases` (`--em_mode aliases` = TriviaQA's official metric; `contains` / `value` are diagnostics also written to
  `metrics.json`: `acc`, `answer_rate`, `acc_answered`, `acc_official`, `acc_contains`, `acc_value_only`, `n_alias_only_match`).
  A generation that never closes `</think>` is unfinished = wrong. `--em_mode` is grading-only (not in the resume
  fingerprint): re-score with `the harness's --finalize_only (scripts/eval_vector.sh does this after a sharded run)`.
  `eval_nonreason_vllm.py` adds `--tensor_parallel_size` (TP=1 for the 7B on >4-GPU boxes). Like the other steering forks,
  `eval_nonreason_steering.py` has `--gen_mode hf|pruned`, `--prune_chunk`, `--model_dtype fp32|bf16` (defaults = the standard).
* **Protocol (standard).** 10,000-token cap, greedy, `--random_sample --sample_seed 42 --max_examples 500` over the validation
  file (selection `rand42_500`), layer 20, coef −1.0, plain HF generate in fp32, batch **25 (1.5B)** via
  `scripts/eval_vector.sh` (`DATASETS=...,triviaqa`) / **16 (7B)** via `scripts/eval_vector.sh (MODEL_SIZE=7B)`; multi-GPU
  `DATASET=triviaqa scripts/eval_vector.sh (GPUS=…) <vector.pt> <arm>`; end to end `SIZE=1.5B|7B bash scripts/extract_vector.sh + scripts/eval_vector.sh (DOMAIN/ARM=nonreason)`.
* **Arms.** `baseline_on_TriviaQA` (HF harness, `--steering` off, so it pairs batch-for-batch with the steered arms),
  `v_nonreason_on_TriviaQA` (in-domain), `v_math_on_TriviaQA` (S_math transfer INTO the domain — not run yet), and the
  transfer arms OUT of the domain under `math_eval / code_eval / logic_eval` (`v_nonreason_on_{MATH500,APPS,LogiQA2}`).
  Folders are flat (predictions.jsonl, math_eval.jsonl, metrics.json, run_config.json, RUN_NOTE.md directly in the arm folder).
  `baseline_4000_vector_trivia_500_500_layer_20_transition_reflection_steervec/…`; new runs of the packaged vector:
  `nonreason_extraction_<size>_result_triviaqa_v_nonreason[_7b]/…`), `figures/` for the dashboards.
* **Status: STALE.** Every imported arm (both sizes) comes from a vector built at a **4k cap** and from evals run with
  `--gen_mode pruned` at non-standard batch sizes (1.5B 100 / 100 / 64 / 100, 7B 32 / 32 / 16 / 32 for TriviaQA / MATH500 /
  APPS / LogiQA; 7B APPS in bf16) — see `nonreason_eval_{1.5B,7B}/STALE.md`, the per-arm `STALE.md` files, `../PROVENANCE.md`
  and `eval/PROVENANCE.md`. The 7B LogiQA arm was on the mixed-language 500 and was deleted from the repo on 2026-08-23 (in git history). Numbers: `RESULTS_v_nonreason_branch.md`;
  paired stats: `report_stats.py --benchmark triviaqa (the stats tool was removed with analysis/ on 2026-08-23; in git history)` / `triviaqa_7b` (+ the `v_nonreason` rows of
  `math500`, `apps`, `logiqa`, `math500_7b`, `apps_7b`).
