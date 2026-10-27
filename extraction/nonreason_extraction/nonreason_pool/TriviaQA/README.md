# TriviaQA (closed-book) — the `nonreason` extraction pool. NON-REASONING domain.

* **Vector built from it:** `v_nonreason` = `../../nonreason_extraction_1.5B/result/triviaqa_v_nonreason.pt` (+ `.meta.json`);
  7B: `../../nonreason_extraction_7B/result/triviaqa_v_nonreason.pt`. Source vectors:
  `…/result/baseline_4000/vector_trivia_500_500/` (built + packaged on the `triviaqa-non-reason`
  branch by what is now `../../nonreason_extraction_1.5B/pipeline/scripts/extract_vector.sh`).
* **Eval pool** (same dataset, the validation split): `eval/nonreason_eval/nonreason_eval_task/TriviaQA/{test.jsonl, eval_rand42_500.json}`.
* **Prep:** `python extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py` regenerates `train.jsonl` here and the two
  eval files over there, byte-identical (it writes both because the disjointness check needs both splits at once).

## Why this domain

Closed-book factual recall is CoT-insensitive: Sprague et al. (ICLR 2025, arXiv:2409.12183) measure average CoT gains of
+12.3 % on math and +14.2 % on symbolic vs +0.7 % on everything else. So reflection/transition thoughts on TriviaQA are pure
waste and SEAL steering should cut tokens with accuracy flat. Cite that paper for the "non-reasoning" claim rather than
arguing from the benchmark's name.

## Config — `mandarjoshi/trivia_qa`, `unfiltered.nocontext`

| split | rows | use |
|---|---|---|
| train | 87,622 | extraction source (80 dropped, see disjointness) |
| validation | 11,313 | eval source |

Do NOT use `rc.nocontext`: its train split is 138,384 rows because each question repeats once per evidence document, which
would seed the extraction pool with near-duplicate questions. Do NOT use `rc` (7.2 GB) or `unfiltered` (13.3 GB) — those carry
the evidence documents this closed-book setup must not see. The `test` split ships without answers, so validation is the eval pool.

## Files (committed, like the MATH and LogiQA2 pools — pin the files, not just the recipe)

* `train.jsonl` (here) — 10,000 rows, the FIRST 10k of the seed-42 shuffle of the 87,542-row deduplicated train pool.
  Extraction reads the first `ATTEMPT` rows of this file in order (`ATTEMPT=10000` default — the 1.5B build needed all of them;
  the 7B build used the first 3,000). The shuffle happens BEFORE the truncation, so this is a strict prefix of
  `prepare_triviaqa.py --full`: any `ATTEMPT <= 10,000` generates identical traces from either file, and the build script refuses
  `ATTEMPT >` rows-in-file with a pointer to `--full`.
* `eval/nonreason_eval/nonreason_eval_task/TriviaQA/test.jsonl` — 11,313 rows, native validation order. Eval scripts subsample it
  with `--random_sample --sample_seed 42 --max_examples 500` -> results land in `rand42_500/`, the same convention as MATH500.
* `eval/nonreason_eval/nonreason_eval_task/TriviaQA/eval_rand42_500.json` — the exact 500 pool indices + question_ids that rule
  selects (shuffle `range(n)` with `random.Random(42)`, keep the first 500, restore file order). Verification/documentation only —
  the eval scripts recompute the selection, they do not read this file. Zero overlap with `train.jsonl`.

**Train is shuffled with seed 42 at prep time, on purpose.** The build generates in file order (no `--random_sample`) because
`hidden_analysis.load_evaluated_traces` zips `train.jsonl` positionally against `math_eval.jsonl` and asserts the problems match.
Randomizing in the file instead of at sample time keeps that alignment while still making the first-N prefix a seeded random sample.

**Disjointness** is enforced at prep time by `question_id` AND normalized question text: 80 train rows overlapped validation by
text and were dropped BEFORE the shuffle (so TriviaQA's splits are NOT perfectly disjoint — the check is not paranoia). Motivated by
LogiQA 2.0, where 16 of 500 eval items leaked into the attempted train rows and 4 reached the shipped vector
(`extraction/logic_extraction/logic_pool/LogiQA2/`).

## Prompt (`triviaqa_utils.TRIVIAQA_INSTRUCTION`, in `eval/nonreason_eval/pipeline/`)

"Answer the following trivia question. Think step-by-step, then end your response with 'Answer: X' where X is the answer."
The think-step-by-step wording is kept deliberately even though the task needs no reasoning: it holds the prompt shape constant
across domains, so a token delta between TriviaQA and MATH reflects the domain, not the instruction. The DeepSeek chat template
appends `<think>\n`, so generation starts inside the thinking block and the SEAL trigger (steer on `\n\n` boundaries while inside
`<think>`) fires as usual.

## Grading (`triviaqa_utils.triviaqa_eval_main`)

The answer region is what follows `</think>`; a generation that never closes it is UNFINISHED (no answer exists to read; counted
as incorrect, tallied separately as `n_unfinished`). Within the region the LAST `Answer: X` / `Answer - X` / `answer is X` wins
(the separator is required — the bare word "answer" must not match, or "no answer here" becomes an answered-wrong row and inflates
`answer_rate`); `\boxed{}` is unwrapped. Verdicts are normalized EM (lowercase, strip punctuation/articles/whitespace) in three
flavours, all written to `math_eval.jsonl`; `--em_mode` picks which one lands in `all_eval` / `mv_eval`:

| `--em_mode` | meaning |
|---|---|
| `aliases` (harness default; the eval metric) | official TriviaQA metric: EM vs any dataset alias |
| `value` | EM vs the canonical value only (strict) |
| `contains` (**what both shipped vectors' labels used**) | an alias appears as a whole-token run inside a verbose answer line |

`metrics.json` reports `acc` plus `answer_rate`, `acc_answered`, `acc_official`, `acc_value_only`, `acc_contains`,
`n_alias_only_match`. `em_mode` is grading-only: it is NOT in the resume fingerprint, so a finished arm is re-scored with
`the harness's --finalize_only --em_mode`.

**Extraction labels use `contains`** (the pipeline default): strict EM rejects verbose-but-correct recall ("The answer is the city
of Paris"), which halves the correct-class yield on a 1.5B distill (probe: `acc_official` 0.040 vs `acc_contains` 0.080; build:
0.049 vs 0.0765), while the alias-noise it admits is small (alias-only matches 89 / 10,000 = 0.9 % at 1.5B, 76 / 3,000 at 7B).

## Alias noise

The alias lists come from Wikipedia entity linking and sometimes carry the wrong sense of a name. Real case (`tc_49`): "Who had a
70s No 1 hit with Kiss You All Over?" has value "Exile" and aliases including "Internal exile" and "Voluntary exile", because the
band shares a name with the political concept. EM accepts any alias, so a wrong-sense answer scores correct. No clean filter exists
(requiring the alias to contain the value still admits "internal exile"; requiring equality would reject the legitimate "Sunset
Blvd" for "Sunset Boulevard"), so the official metric is kept and the gap is instrumented: if `n_alias_only_match` exceeds ~2 % of
rows, rebuild the extraction labels with `EM_MODE=value`.

## Row schema

`problem`, `question_id`, `level` ("trivia"), `gt`, `aliases`. `problem` and `level` are the names
`hidden_analysis.load_evaluated_traces` reads.
