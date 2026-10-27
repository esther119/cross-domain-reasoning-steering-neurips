# nonreason_keyword — check/switch keyword set for the `nonreason` domain (TriviaQA, closed-book)

`keywords.json` is the keyword set `extraction/src/hidden_analysis.py --keywords nonreason` uses to
tag each `\n\n`-delimited reasoning step inside `<think>` as **check** (reflection),
**switch** (transition) or **other** (execution). It is loaded at import time by
`extraction/src/keyword_sets.py`, which scans `extraction/*_extraction/*_keyword/keywords.json`;
the folder name before `_keyword` is the `--keywords` name. (On the `triviaqa-non-reason` branch this
was `KEYWORD_SETS["trivia"]` in `hidden_analysis.py`; the lists are byte-for-byte the ones that built
`triviaqa_v_nonreason.pt` / `triviaqa_v_nonreason.pt` — see the `keyword_lists` block of their `.meta.json`.)

Fields: `check_words` / `switch_words` are case-insensitive substring matches; `check_prefix` /
`switch_prefix` are case-insensitive step-prefix matches. Check is tested before switch.
Both `*_prefix` lists are empty here: everything is contains-matching.

## How it was derived (not invented up front)

The same procedure as the `logic` set: `discover_nonreason_keywords.sh` generates a probe batch of
TriviaQA-train traces (300 traces, DeepSeek-R1-Distill-Qwen-1.5B; the committed probe was run at a
**4,000-token cap, `aliases` EM** — the script now defaults to 10,000), measures how much of it the
`math` set leaves untagged, and `../../logic_extraction/logic_keyword/logic_keyword_coverage.py
--noise_set nonreason` ranks candidate phrases by how many distinct untagged steps each would recover.
Every candidate was judged against `review_samples_math.jsonl` before it went in. Probe artefacts:
`../nonreason_extraction_1.5B/result/keyword_probe/`
(`math_eval.jsonl`, `coverage_math.json`, `candidate_phrases_math.json`, `review_samples_math.jsonl`).

**The key design call, and it is task-specific:** on a recall task the discriminating signal is
*epistemic uncertainty about a retrieved fact*, plus being stuck / self-correcting — not retrieval
itself. "I remember that X" / "I recall that X" / "I know that X" are the **execution** step here (the
model doing the task) even though raw n-gram frequency ranks them high (remember 18 %, recall 12 % of
untagged steps). Tagging them as check would put execution behaviour on both sides of the
difference-in-means and gut the vector. Same reason bare "i think" (56 % of untagged steps) and bare
"maybe" (39 %) are excluded: on this task they are the model's default assertion voice ("I think he
was a champion in the 1910s", "Maybe it was X") — i.e. the retrieval attempt — not a hedge about one.
Only the self-referential forms ("i think i'm …", "maybe i should …", "maybe i'm …") are reflection.

* **Kept from `math`:** verify / make sure / hold on / think again / 's correct / 's incorrect /
  let me check / seems right, and the switch vocabulary.
* **Promoted to contains-matching:** "wait" and "alternatively" both occur mid-step in real traces
  ("But wait, I'm not entirely…", "…but I'm not sure. Alternatively, I think …"), so upstream's
  startswith-only prefix matching misses them.
* **Added from the probe's untagged steps** (each validated on `review_samples_math.jsonl`):
  not-sure / not-certain / not-entirely / not-100% / unsure / not confident; stuck / not getting
  anywhere / not making progress; confusing-it / mixing-up / overcomplicating / getting-this-wrong /
  misunderstanding; double-check; "i think i'm", "maybe i should", "maybe i'm", "hmm".
* **Switch additions:** "another thought / possibility / option / candidate / angle", "let me try"
  (a retry: "Let me try to think of other countries"), "on second thought".
* **Tried and dropped:** "or maybe", "or perhaps" — on the probe they fired on content ("a fantasy or
  maybe a sci-fi book") more than on candidate switching, and each covered <= 12 of 4,873 steps.
  The branch's rationale also lists "instead" as dropped for the same reason ("used instead of"), but
  `instead` **is** in the shipped `switch_words` list; it is kept here because that is the list the two
  committed vectors were built with. Revisit it at the 10k rebuild.

**Probe result** (300 traces, 4,873 steps): `math` set -> check 63.3 % / switch 6.6 % / other 30.1 %;
this set -> check 77.6 % / switch 4.2 % / other 18.2 %. The remaining "other" is retrieval / assertion /
arithmetic ("I remember that", "I think the", "So the", "M = 1000"), i.e. execution — as intended.

Re-measure after any edit (tokenizer only, no GPU):

```bash
python extraction/logic_extraction/logic_keyword/logic_keyword_coverage.py \
  --data_dir extraction/nonreason_extraction/nonreason_extraction_1.5B/result/keyword_probe \
  --data_path extraction/nonreason_extraction/nonreason_pool/TriviaQA/train.jsonl \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --keywords nonreason --noise_set nonreason
```

Changing this file changes which boundary activations enter the contrast vector, so bump the
`--save_prefix` (or regenerate the vector) rather than silently rebuilding over an old one.
