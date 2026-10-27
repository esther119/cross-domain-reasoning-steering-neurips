"""TriviaQA (closed-book) support for the nonreason eval harness.

Used by `eval_nonreason_vllm.py` (unsteered / extraction traces) and
`eval_nonreason_steering.py` (steered) -- the nonreason forks of the text
harness, same generation + steering core as the MATH500 / LogiQA2 forks under
eval/<domain>_eval/pipeline/. TriviaQA is the NON-REASONING domain: closed-book factual
recall, where chain-of-thought buys essentially nothing (Sprague et al., ICLR
2025, arXiv:2409.12183 measure +0.7% average CoT gain outside math/symbolic/logic
vs +12.3% on math). So reflection/transition thoughts here are pure waste, and
SEAL steering should cut tokens with accuracy flat.

  - config:  unfiltered.nocontext -- 87,622 train / 11,313 validation, ONE ROW
             PER QUESTION. Do NOT use rc.nocontext (138,384 train rows because
             each question repeats once per evidence document, which would seed
             the extraction pool with near-duplicates), and do not use the
             context-bearing `rc` / `unfiltered` configs (7.2 GB / 13.3 GB).
             The `test` split ships without answers, so validation is the eval
             pool -- same choice CommonsenseQA/open-domain QA work makes.
  - prompt:  a chat instruction that elicits a <think> block + a final
             'Answer: X' (so the SEAL steering trigger on `\\n\\n` inside <think>
             still fires), identical in shape to the LogiQA/MMLU prompts. The
             "think step-by-step" wording is kept deliberately even though the
             task needs no reasoning: it holds the prompt shape constant across
             domains, so a token-count delta between TriviaQA and MATH reflects
             the domain and not the instruction.
  - grade:   normalized exact match against the dataset's own answer aliases,
             which is TriviaQA's official metric. `triviaqa_eval_main` mirrors
             get_math_results.main's output schema (all_pred / all_eval /
             mv_pred / mv_eval / mv_index + math_eval.jsonl + metrics.json), so
             extraction/src/hidden_analysis.py
             (load_evaluated_traces) work unchanged.

ALIAS NOISE -- read before trusting a number. TriviaQA's alias lists come from
Wikipedia entity linking and sometimes carry the wrong sense of a name. Real
example from validation (question_id tc_49):

    Q: Who had a 70s No 1 hit with Kiss You All Over?
    value: "Exile"
    aliases: ["Internal exile", "Exiles", "Transported for life",
              "Exile (politics and government)", "Voluntary exile", ...]

The band is named Exile, so every alias for *political* exile came along. Because
EM accepts any alias, a model answering "voluntary exile" scores correct. There
is no clean filter for this -- requiring the alias to contain the value still
admits "internal exile", and requiring equality with the value would reject the
legitimate "Sunset Blvd" for "Sunset Boulevard". So this module does what the
official metric does (accept any alias) and INSTRUMENTS the gap instead: every
row also gets a value-only verdict, and metrics.json reports
`n_alias_only_match` / `acc_value_only`. If alias-only matches are more than a
couple of percent, rebuild the extraction labels with --em_mode value.
"""
import json
import os
import re
import string
from collections import Counter


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_triviaqa_hf(split="validation", config="unfiltered.nocontext", max_rows=None):
    """Stream TriviaQA from the Hub into the flat dict shape this module uses.

    Used by the prep script (extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py)
    to materialize eval/nonreason_eval/nonreason_eval_task/TriviaQA/test.jsonl (eval
    pool) and extraction/nonreason_extraction/nonreason_pool/TriviaQA/train.jsonl
    (extraction pool). The eval scripts read the local JSONL instead, so a build and an eval always
    agree on row order (hidden_analysis.load_evaluated_traces zips
    data_path against math_eval.jsonl positionally and asserts the problems
    match, so order drift is a hard failure, not silent corruption).
    """
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", config, split=split)
    out = []
    for r in ds:
        ans = r["answer"]
        value = (ans.get("value") or "").strip()
        if not value:
            continue  # the `test` splits ship unanswered; skip defensively
        aliases = [value] + [a for a in (ans.get("aliases") or []) if a and a.strip()]
        seen, deduped = set(), []
        for a in aliases:
            k = normalize_answer(a)
            if k and k not in seen:
                seen.add(k)
                deduped.append(a.strip())
        out.append({
            "problem": r["question"].strip(),
            "question_id": r["question_id"],
            "level": "trivia",   # hidden_analysis.load_evaluated_traces reads d["level"]
            "gt": value,
            "aliases": deduped,
        })
        if max_rows and len(out) >= max_rows:
            break
    return out


def load_triviaqa_jsonl(path):
    """Read a prepared TriviaQA JSONL: eval/nonreason_eval/nonreason_eval_task/TriviaQA/test.jsonl
    or extraction/nonreason_extraction/nonreason_pool/TriviaQA/train.jsonl (rows: problem,
    question_id, level, gt, aliases)."""
    with open(path) as fin:
        return [json.loads(line) for line in fin if line.strip()]


TRIVIAQA_INSTRUCTION = (
    "Answer the following trivia question. "
    "Think step-by-step, then end your response with 'Answer: X' where X is the answer.\n\n"
    "Question: {question}"
)


def build_triviaqa_prompt(ex):
    return TRIVIAQA_INSTRUCTION.format(question=ex["question"].strip())


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
# Take everything after the last "Answer:" / "Answer -" / "answer is" on that
# line. `.` does not match a newline, so this cannot swallow following lines.
# The separator is REQUIRED: an earlier draft made `[:\-]` optional, which let
# the bare word "answer" match ("no answer line here" -> "line here"), so a
# generation that never stated an answer was graded answered-wrong instead of
# unanswered, inflating answer_rate. Two alternatives, so "The answer is Paris"
# (no colon) still parses via the second branch.
_ANSWER_RE = re.compile(
    r"answer(?:\s+is)?\s*[:\-]\s*(.+)"    # Answer: X | Answer - X | answer is: X
    r"|answer\s+is\s+(.+)",                 # the answer is X
    re.IGNORECASE,
)


def normalize_answer(s):
    """TriviaQA/SQuAD normalization: lowercase, drop punctuation, drop articles,
    collapse whitespace. Applied to both sides of every comparison."""
    s = (s or "").lower()
    s = "".join(" " if ch in _PUNCT else ch for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _extract_box(pred_str):
    """\\boxed{...} fallback -- models trained on MATH sometimes box a trivia answer
    even though this prompt never asks for it."""
    if "boxed" not in pred_str:
        return None
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack, a = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
            a += c
        return a
    return ans.split("$")[0].strip()


def extract_answer(gen):
    """Pull the short answer string out of a reasoning-model generation.

    Same contract as logic_utils.extract_choice, adapted from multiple choice to
    free-form short answer:
      * the answer region is what follows `</think>`, not the start of the text
      * a generation that never closed `</think>` gets None -- it ran out of
        budget mid-reasoning and never reached an answer. No guessing: inventing
        a span from truncated reasoning is what made the LogiQA legacy extractor
        unusable (see logic_utils.extract_choice_legacy).
      * within the answer region take the LAST 'Answer:' match, since the prompt
        asks the model to *end* with it.

    Returns the normalized answer string, or None when the model never answered.
    """
    if "</think>" not in gen:
        return None
    tail = gen.split("</think>", 1)[1]

    matches = [m.group(1) or m.group(2) for m in _ANSWER_RE.finditer(tail)]
    cand = matches[-1] if matches else tail

    # \boxed{} has to be unwrapped BEFORE normalization, and it has to be checked
    # on the candidate rather than only as an "no Answer: found" fallback: a model
    # that writes "Answer: \boxed{Paris}" matches _ANSWER_RE, so a fallback-only
    # check would never fire and the braces would normalize into the answer text.
    if "boxed" in cand:
        boxed = _extract_box(cand)
        if boxed:
            cand = boxed
    elif not matches:
        return None

    # Strip the wrappers models put around a short answer.
    cand = cand.split("\n")[0]
    cand = cand.strip().strip("*").strip()
    norm = normalize_answer(cand)
    return norm or None


def _contains_alias(pred_norm, aliases, gt):
    """True if any alias occurs as a whole-token run inside the prediction.

    Needed because a reasoning model is verbose in its answer line. Measured on
    the 10-trace probe: the model answered "Nikkei 225" for gt "Nikkei" and
    restated the question ("the target of the failed Bomb Plot of 1944 was ...")
    -- both score 0 under strict EM despite the first being plainly correct.
    Token-run matching (not raw substring) so "exile" does not match "exiled".

    FALSE-POSITIVE RISK: an answer line that enumerates candidates ("it could be
    X or Y") matches on the wrong one. metrics.json reports strict and contains
    side by side so the gap is always visible.
    """
    if pred_norm is None:
        return False
    toks = pred_norm.split()
    for a in [gt] + list(aliases or []):
        ct = normalize_answer(a).split()
        if not ct:
            continue
        for i in range(len(toks) - len(ct) + 1):
            if toks[i:i + len(ct)] == ct:
                return True
    return False


def em_verdicts(pred_norm, aliases, gt):
    """(official, value_only, contains) verdicts for one prediction.

    official   -- normalized EM against any alias (TriviaQA's own metric)
    value_only -- normalized EM against the canonical `value` alone; the
                  diagnostic that exposes wrong-sense alias hits
    contains   -- any alias appears as a token run in the answer; tolerant of a
                  verbose reasoning model (see _contains_alias)
    """
    if pred_norm is None:
        return False, False, False
    value_only = pred_norm == normalize_answer(gt)
    official = value_only or any(pred_norm == normalize_answer(a) for a in (aliases or []))
    contains = official or _contains_alias(pred_norm, aliases, gt)
    return official, value_only, contains


def _aliases_from_row(example):
    """Recover the alias list for a prediction row.

    eval_nonreason_vllm.py / eval_nonreason_steering.py write predictions.jsonl as
    {"answer": example["gt"], "solution": example["answer"], ...}. The TriviaQA
    branches put the alias LIST in example["answer"], so it lands in `solution`.
    Falls back to the canonical value alone for rows written before that, so an
    older predictions.jsonl still grades (just strictly).
    """
    sol = example.get("solution")
    if isinstance(sol, list):
        return sol
    if isinstance(sol, str) and sol.strip().startswith("["):
        try:
            parsed = json.loads(sol)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass
    return [example.get("answer", "")]


def triviaqa_eval_main(res_path, save=False, output_dir=None, grader="reasoning",
                       em_mode="aliases"):
    """Grade TriviaQA predictions, keeping get_math_results.main's output schema so
    hidden_analysis.py works unchanged.

    grader="reasoning" treats a generation that never closed </think> as
    UNFINISHED: it ran out of budget still inside the reasoning block, so no
    answer exists to read. Unfinished scores as incorrect -- matching TriviaQA's
    exact_match semantics, where a non-answer is simply wrong -- but is counted
    separately, because "recalled the wrong fact" and "never stopped thinking"
    are different failures and only the first belongs in a clean incorrect class.

    em_mode="aliases" (default) writes the official alias-EM verdict into
    all_eval, which is what hidden_analysis reads to split correct/incorrect.
    em_mode="value" writes the stricter value-only verdict there instead; use it
    to rebuild extraction labels if n_alias_only_match comes back high.
    """
    with open(res_path) as f:
        data = [json.loads(line) for line in f]

    n_alias_only = 0
    for example in data:
        gens = example.get("model_generation") or [example.get("model_output", "")]
        gt = example["answer"]
        aliases = _aliases_from_row(example)

        all_pred = [extract_answer(g) for g in gens]
        verdicts = [em_verdicts(p, aliases, gt) for p in all_pred]
        all_official = [v[0] for v in verdicts]
        all_value_only = [v[1] for v in verdicts]
        all_contains = [v[2] for v in verdicts]
        n_alias_only += sum(1 for o, v, _ in verdicts if o and not v)

        all_eval = {"value": all_value_only,
                    "contains": all_contains}.get(em_mode, all_official)
        all_unfinished = ["</think>" not in g for g in gens]

        valid = [p for p in all_pred if p is not None]
        if valid:
            pred = Counter(valid).most_common(1)[0][0]
            index = all_pred.index(pred)
        else:
            pred, index = None, 0

        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["all_eval_official"] = all_official
        example["all_eval_value_only"] = all_value_only
        example["all_eval_contains"] = all_contains
        example["mv_pred"] = pred
        example["mv_eval"] = bool(all_eval[index]) if all_eval else False
        example["mv_eval_value_only"] = bool(all_value_only[index]) if all_value_only else False
        example["mv_eval_official"] = bool(all_official[index]) if all_official else False
        example["mv_eval_contains"] = bool(all_contains[index]) if all_contains else False
        example["mv_index"] = index
        # Unfinished only when NO sampled generation terminated; with n=1 that is
        # just "this generation never closed </think>".
        example["unfinished"] = all(all_unfinished)
        example["answered"] = pred is not None

    n = len(data)
    acc = sum(e["mv_eval"] for e in data) / n if n else 0.0
    acc_value_only = sum(e["mv_eval_value_only"] for e in data) / n if n else 0.0
    acc_official = sum(e["mv_eval_official"] for e in data) / n if n else 0.0
    acc_contains = sum(e["mv_eval_contains"] for e in data) / n if n else 0.0
    n_unfinished = sum(e["unfinished"] for e in data)
    n_answered = sum(e["answered"] for e in data)
    acc_answered = (sum(e["mv_eval"] for e in data) / n_answered) if n_answered else 0.0

    metrics = {
        "acc": acc,                                   # unfinished counts as incorrect
        "n": n,
        "n_answered": n_answered,
        "n_unfinished": n_unfinished,
        "answer_rate": n_answered / n if n else 0.0,  # did the model terminate and answer
        "acc_answered": acc_answered,                 # accuracy among those that answered
        "acc_value_only": acc_value_only,             # strict: canonical value only
        "acc_official": acc_official,                 # strict EM vs alias list (TriviaQA's metric)
        "acc_contains": acc_contains,                 # alias appears in a verbose answer line
        "n_alias_only_match": n_alias_only,           # alias-noise exposure (see module docstring)
        "grader": grader,
        "em_mode": em_mode,
    }
    print(f"Accuracy: {acc:.3f}  "
          f"(answered {n_answered}/{n} = {metrics['answer_rate']:.3f}, "
          f"unfinished {n_unfinished}, acc|answered {acc_answered:.3f})")
    print(f"[em] strict {acc_official:.3f} | contains {acc_contains:.3f} | "
          f"value-only {acc_value_only:.3f}"
          + (f"  <-- strict is costing {acc_contains - acc_official:.3f}; "
             f"a verbose answer line is being penalized, prefer --em_mode contains"
             if acc_contains - acc_official > 0.02 else ""))
    print(f"[alias] {n_alias_only}/{n} rows matched only via a non-canonical alias"
          + ("  <-- >2%, consider --em_mode value for extraction labels"
             if n and n_alias_only / n > 0.02 else ""))

    if save:
        with open(os.path.join(output_dir, "math_eval.jsonl"), "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

    return acc
