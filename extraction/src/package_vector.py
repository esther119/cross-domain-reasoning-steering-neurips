#!/usr/bin/env python
"""Package a built SEAL steering vector into <out_dir>/<name>.pt + <name>.meta.json.

Same provenance fields as the other packaged vectors' .meta.json (apps_v_code,
logiqa2_v_logic), computed from the build directory rather than
typed by hand: generation stats from math_eval.jsonl, pool/boundary counts from
hidden_{correct,incorrect}_0_<N>/hidden.pt, keyword lists from KEYWORD_SETS (the
extraction/<d>_extraction/<d>_keyword/keywords.json files, via hidden_analysis), and
the vector's own dim / norm.

    python extraction/src/package_vector.py \
      --source extraction/logic_extraction/logic_extraction_1.5B/result/baseline_3000_regraded/vector_logic_500_500/layer_20_transition_reflection_steervec.pt \
      --build_dir extraction/logic_extraction/logic_extraction_1.5B/result/baseline_3000_regraded \
      --name logiqa2_v_logic --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --dataset "LogiQA 2.0 train (English)" --keywords logic --layer 20 --sample 500 \
      --max_tokens 3000
      # -> extraction/logic_extraction/logic_extraction_1.5B/result/logiqa2_v_logic.pt (+ .meta.json)

--out_dir defaults to the nearest `result/` ancestor of --build_dir, i.e. the
`extraction/<d>_extraction/<d>_extraction_<size>/result/` folder the packaged vectors
live in; pass it explicitly for any other layout.

The packaged copy is for sharing / S_general-style pooling / the eval drivers. The
source .pt stays where it was built (its path spells out the build settings).
"""
import argparse
import datetime
import json
import os
import sys

import torch

# Sibling import: this file lives in extraction/src next to hidden_analysis.py / keyword_sets.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hidden_analysis import KEYWORD_SETS  # noqa: E402


def pool_counts(build_dir, kind, sample, layer):
    path = os.path.join(build_dir, f"hidden_{kind}_0_{sample}", "hidden.pt")
    if not os.path.exists(path):
        return None
    data = torch.load(path, weights_only=False)[layer]
    c = s = o = 0
    for k in data:
        h = data[k]
        ci, si = len(h["check_index"]), len(h["switch_index"])
        c += ci
        s += si
        o += h["step"].shape[0] - ci - si
    return {"traces": len(data), "check": c, "switch": s, "other": o, "boundaries": c + s + o}


def generation_stats(build_dir):
    path = os.path.join(build_dir, "math_eval.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    correct = sum(bool(r["all_eval"][0]) for r in rows)
    unfinished = sum("</think>" not in r["model_generation"][0] for r in rows)
    answered = sum(1 for r in rows if r.get("all_pred", [None])[0] is not None) if rows and "all_pred" in rows[0] else None
    stats = {
        "temperature": 0, "n": 1, "use_chat_format": True, "remove_bos": True,
        "attempted": n, "correct": correct, "incorrect": n - correct,
        "accuracy_all_attempts": round(correct / n, 4) if n else None,
        "unfinished_kept_as_incorrect": unfinished,
        "finished": n - unfinished,
        "accuracy_among_finished": round(correct / (n - unfinished), 4) if n - unfinished else None,
    }
    if answered is not None:
        stats["answered"] = answered
        stats["answer_rate"] = round(answered / n, 4) if n else None
    # Alias instrumentation, if the grader wrote it.
    if rows and "all_eval_official" in rows[0]:
        stats["alias_only_correct"] = sum(1 for r in rows
                                          if r["all_eval_official"][0] and not r["all_eval_value_only"][0])
        stats["correct_value_only"] = sum(bool(r["all_eval_value_only"][0]) for r in rows)
        stats["correct_contains"] = sum(bool(r["all_eval_contains"][0]) for r in rows)
    # Grading mode the labels on disk used (if the harness wrote it to metrics.json).
    mpath = os.path.join(build_dir, "metrics.json")
    if os.path.exists(mpath):
        try:
            em_mode = json.load(open(mpath)).get("em_mode")
        except (ValueError, AttributeError):
            em_mode = None
        if em_mode:
            stats["em_mode"] = em_mode
    return stats


def default_out_dir(build_dir):
    """Nearest ancestor of build_dir named `result` (the <d>_extraction_<size>/result/ folder)."""
    d = os.path.abspath(build_dir)
    while True:
        if os.path.basename(d) == "result":
            return os.path.relpath(d) if not os.path.isabs(build_dir) else d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="built layer_<L>_transition_reflection_steervec.pt")
    ap.add_argument("--build_dir", required=True, help="dir holding math_eval.jsonl and hidden_*_0_<N>/")
    ap.add_argument("--name", required=True, help="package name, e.g. logiqa2_v_logic -> <out_dir>/<name>.pt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--keywords", default="math", choices=sorted(KEYWORD_SETS))
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=None, help="generation budget used for the traces")
    ap.add_argument("--out_dir", default=None,
                    help="where <name>.pt / <name>.meta.json go (default: the nearest `result/` ancestor of --build_dir)")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or default_out_dir(args.build_dir)
    if not out_dir:
        raise SystemExit(f"--build_dir {args.build_dir} has no `result/` ancestor; pass --out_dir explicitly")

    vec = torch.load(args.source, weights_only=True)
    if not torch.isfinite(vec).all():
        raise SystemExit(f"non-finite values in {args.source}")
    os.makedirs(out_dir, exist_ok=True)
    out_pt = os.path.join(out_dir, f"{args.name}.pt")
    torch.save(vec.clone(), out_pt)

    kw = KEYWORD_SETS[args.keywords]
    pools = {k: pool_counts(args.build_dir, k, args.sample, args.layer) for k in ("correct", "incorrect")}
    present = [p for p in pools.values() if p]
    activations = ({"check": sum(p["check"] for p in present),
                    "switch": sum(p["switch"] for p in present),
                    "other": sum(p["other"] for p in present)} if present else None)

    meta = {
        "name": args.name,
        "created": datetime.date.today().isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "layer": args.layer,
        "extraction_dtype": str(vec.dtype).replace("torch.", ""),
        "sign_convention": "vector = mean(H_reflection ∪ H_transition) − mean(H_execution); "
                           "apply by ADDING coef * vector with coef = -1.0 (SEAL code convention)",
        "apply_coef": -1.0,
        "boundary_token": "any vocab token containing 'ĊĊ' (\\n\\n), inside <think> only",
        "keyword_lists": {
            "set": args.keywords,
            "reflection": kw["check_words"], "reflection_prefix": kw["check_prefix"],
            "transition": kw["switch_words"], "transition_prefix": kw["switch_prefix"],
            "priority": "reflection checked before transition",
        },
        "selection_rule": f"{args.sample} correct + {args.sample} incorrect, first-in first-fill in file order "
                          f"(--start 0 --sample {args.sample}). NO trace is excluded: a generation that never "
                          f"closed </think> stated no answer, so it is a FAILED attempt, filed as INCORRECT and "
                          f"kept -- same rule as v_math / v_code / v_logic.",
        "generation": {"engine": "vLLM", "max_tokens": args.max_tokens, **(generation_stats(args.build_dir) or {})},
        "pools": pools,
        "activations": activations,
        "vector_dim": list(vec.shape),
        "vector_norm": float(vec.float().norm()),
        "source_vector": args.source,
        "build_dir": args.build_dir,
    }
    if args.note:
        meta["note"] = args.note
    out_meta = os.path.join(out_dir, f"{args.name}.meta.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[package] {out_pt}  dim {list(vec.shape)}  norm {meta['vector_norm']:.3f}")
    print(f"[package] {out_meta}")
    if activations:
        T = sum(activations.values())
        print(f"[package] boundaries {T} | check {activations['check']} ({activations['check']/T*100:.1f}%) "
              f"switch {activations['switch']} other {activations['other']}")


if __name__ == "__main__":
    main()
