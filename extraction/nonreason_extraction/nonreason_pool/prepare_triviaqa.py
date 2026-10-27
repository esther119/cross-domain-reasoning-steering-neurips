#!/usr/bin/env python
"""Materialize the TriviaQA extraction pool and eval files for the `nonreason` domain.

    python extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py            # committed layout: 10k train prefix + full test
    python extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py --full     # all 87k train rows (same order; a superset)
    python extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py --smoke    # 300 train / 200 test, for a pilot

Writes (repo-root-relative; the script cd's to the repo root itself):
    extraction/nonreason_extraction/nonreason_pool/TriviaQA/train.jsonl          the extraction pool
    eval/nonreason_eval/nonreason_eval_task/TriviaQA/test.jsonl                   the eval pool (validation split)
    eval/nonreason_eval/nonreason_eval_task/TriviaQA/eval_rand42_500.json         the 500 eval rows the harness' seeded
                                                                                  sampling rule selects (documentation only)
Both trees are written by one script because the disjointness check needs both splits at once.

Config is unfiltered.nocontext (87,622 train / 11,313 validation, one row per
question). See eval/nonreason_eval/pipeline/triviaqa_utils.py for why the other
configs are wrong.

WHAT IS COMMITTED, AND WHY. test.jsonl (all 11,313 eval rows, native order) and
train.jsonl (the first 10,000 rows of the seed-42 shuffle) are checked in, like the
MATH and LogiQA2 pools. Pinning the files -- not just the recipe -- is what makes the
eval pool and the extraction order reproducible if the Hub copy ever changes, and it
is what let the LogiQA leak be diagnosed after the fact. Because the shuffle happens
BEFORE truncation, the committed 10k file is a strict prefix of the --full file: any
ATTEMPT <= 10,000 generates identical traces from either.

TRAIN IS SHUFFLED WITH SEED 42 AT PREP TIME, ON PURPOSE.
`eval/nonreason_eval/pipeline/eval_nonreason_vllm.py --dataset TriviaQA_train
--max_examples N` takes the FIRST N rows in file order, and
hidden_analysis.load_evaluated_traces then zips train.jsonl positionally against
math_eval.jsonl and asserts the problems match. So the build cannot use
--random_sample (that would reorder the generated traces away from the file) -- the
randomization has to live in the file itself. Shuffling here makes the first-N prefix
a seeded random sample while keeping the positional alignment the extraction step
depends on.

TEST IS LEFT IN NATIVE ORDER. The eval scripts do their own seeded sampling via
--random_sample --sample_seed 42 --max_examples 500, which writes results into a
`rand42_500/` subdir -- the same convention as the MATH500 and LogiQA2 evals.
eval_rand42_500.json records that selection (shuffle range(n) with
random.Random(42), keep the first 500, restore file order) for verification; the
harness recomputes it and never reads the file.
"""
import argparse
import json
import os
import random
import sys

# extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py -> repo root is four levels up.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# triviaqa_utils (HF loader, normalizer, prompt, grader) lives with the nonreason eval harness.
sys.path.insert(0, os.path.join(ROOT, "eval", "nonreason_eval", "pipeline"))
from triviaqa_utils import load_triviaqa_hf, normalize_answer  # noqa: E402

POOL_DIR = "extraction/nonreason_extraction/nonreason_pool/TriviaQA"
EVAL_DIR = "eval/nonreason_eval/nonreason_eval_task/TriviaQA"
SEED = 42
EVAL_K = 500


def write_jsonl(path, rows):
    with open(path, "w") as fout:
        for r in rows:
            fout.write(json.dumps(r) + "\n")
    print(f"[prep] wrote {len(rows):>6} rows -> {path}")


def eval_selection(test_rows, k, seed, test_path):
    """The harness' --random_sample rule: shuffle range(n) with random.Random(seed),
    keep the first k, restore file order."""
    idx = list(range(len(test_rows)))
    random.Random(seed).shuffle(idx)
    keep = sorted(idx[:k])
    return {
        "note": (f"Pool indices into {test_path} selected by the eval scripts' rule: shuffle range(n) "
                 f"with random.Random({seed}), keep the first {k}, restore file order "
                 f"(--random_sample --sample_seed {seed} --max_examples {k} -> results land in "
                 f"rand{seed}_{k}/). Documentation/verification only -- the eval scripts recompute it, "
                 f"they do not read this file."),
        "n_pool": len(test_rows),
        "seed": seed,
        "k": k,
        "indices": keep,
        "question_ids": [test_rows[i]["question_id"] for i in keep],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="unfiltered.nocontext")
    ap.add_argument("--pool_dir", default=POOL_DIR, help="where train.jsonl goes (root-relative)")
    ap.add_argument("--eval_dir", default=EVAL_DIR, help="where test.jsonl + eval_rand42_500.json go (root-relative)")
    ap.add_argument("--smoke", action="store_true",
                    help="300 train / 200 test, to calibrate yield and runtime cheaply.")
    ap.add_argument("--max_train", type=int, default=10000,
                    help="rows of the shuffled train pool to write (default 10000, the "
                         "committed prefix; the build reads the first ATTEMPT of these).")
    ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--full", action="store_true",
                    help="write every train row (superset of the default prefix, same order).")
    args = ap.parse_args()

    os.chdir(ROOT)   # every path below is repo-root-relative

    max_train = 300 if args.smoke else (None if args.full else args.max_train)
    max_test = 200 if args.smoke else args.max_test

    os.makedirs(args.pool_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

    print(f"[prep] loading TriviaQA {args.config} ...")
    train = load_triviaqa_hf(split="train", config=args.config)
    test = load_triviaqa_hf(split="validation", config=args.config)
    print(f"[prep] loaded train={len(train)} validation={len(test)}")

    # Disjointness. TriviaQA's splits are already disjoint by construction, but
    # LogiQA 2.0's were not (16 of 500 eval items leaked into the attempted train
    # rows and 4 reached the shipped vector), so this is checked rather than assumed.
    # Question text is the key as well as the id: ids are per-split and would not
    # catch a duplicated question.
    test_ids = {r["question_id"] for r in test}
    test_qs = {normalize_answer(r["problem"]) for r in test}
    before = len(train)
    train = [r for r in train
             if r["question_id"] not in test_ids
             and normalize_answer(r["problem"]) not in test_qs]
    dropped = before - len(train)
    print(f"[prep] disjointness: dropped {dropped} train rows overlapping validation "
          f"({len(train)} remain)")

    random.Random(SEED).shuffle(train)
    print(f"[prep] train shuffled with seed {SEED} (first-N prefix is a random sample)")

    if max_train:
        train = train[:max_train]
    if max_test:
        test = test[:max_test]

    write_jsonl(os.path.join(args.pool_dir, "train.jsonl"), train)
    test_path = os.path.join(args.eval_dir, "test.jsonl")
    write_jsonl(test_path, test)

    if len(test) >= EVAL_K:
        sel = eval_selection(test, EVAL_K, SEED, test_path)
        sel_path = os.path.join(args.eval_dir, f"eval_rand{SEED}_{EVAL_K}.json")
        with open(sel_path, "w") as f:
            json.dump(sel, f, indent=0)   # no trailing newline: byte-identical to the committed file
        print(f"[prep] wrote the rand{SEED}_{EVAL_K} selection ({EVAL_K} indices) -> {sel_path}")
    else:
        print(f"[prep] test has {len(test)} rows (< {EVAL_K}); no eval_rand{SEED}_{EVAL_K}.json written")

    alias_counts = [len(r["aliases"]) for r in test]
    print(f"[prep] aliases per question: mean {sum(alias_counts)/len(alias_counts):.1f}, "
          f"max {max(alias_counts)}")
    print(f"[prep] sample row: {json.dumps(train[0], ensure_ascii=False)[:220]}")


if __name__ == "__main__":
    main()
