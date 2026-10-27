import argparse
import os
import re
import json
import random
import sys
import gc
import glob

# eval/<domain>_eval/pipeline/<this file>  ->  repo root is three levels up.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "eval", "src"))  # modeling_utils: the shared steering model

import torch
import evaluate
from transformers import  AutoTokenizer
from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:  # vllm unused on the HF path
    LLM = SamplingParams = LoRARequest = None
from collections import Counter
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from tqdm import tqdm

from logic_utils import load_logiqa, build_logiqa_prompt, logic_eval_main
from steer_generate import generate_pruned, decode_generated  # eval/src: --gen_mode pruned
os.environ["TOKENIZERS_PARALLELISM"] = "false"

exact_match = evaluate.load("exact_match")


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output



# Batch-level checkpoint, so a crash 10 hours into a run costs one batch rather
# than the whole arm. Written alongside the final predictions.jsonl and kept
# after a successful run, so re-grading never means re-generating.
CHECKPOINT_GLOB = "predictions.partial*.jsonl"

# Settings a resume has to agree on. Ones already encoded in save_dir (vector
# name, coef) are listed too so the check stands alone; the ones that are not --
# steering_layer, max_tokens, batch_size, model -- are the whole point, since
# nothing else would catch two half-arms merged into one file.
FINGERPRINT_KEYS = [
    "model_name_or_path", "dataset", "sample_seed",
    "random_sample", "max_examples", "start", "max_tokens", "batch_size",
    "use_chat_format", "remove_bos", "steering", "steering_vector",
    "steering_layer", "steering_coef",
    "logiqa_english_only", "logiqa_eval_selection",
    "gen_mode", "prune_chunk", "model_dtype",
]


def checkpoint_path(save_dir, shard_rank, num_shards):
    if num_shards == 1:
        return os.path.join(save_dir, "predictions.partial.jsonl")
    return os.path.join(save_dir, f"predictions.partial.{shard_rank}of{num_shards}.jsonl")


def _read_checkpoint(path):
    """Parse one checkpoint file, dropping a torn final line from a killed run."""
    records = []
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in rec:
                records.append(rec)
    return records


def load_completed(save_dir):
    """Example index -> record, across every checkpoint file in save_dir.

    Reading *all* files rather than only this process's own is what lets a run
    resume on a different GPU count: the batch-to-shard assignment changes, but
    a batch any earlier process finished is still skipped.
    """
    completed = {}
    for path in sorted(glob.glob(os.path.join(save_dir, CHECKPOINT_GLOB))):
        for rec in _read_checkpoint(path):
            completed[rec["idx"]] = rec
    return completed


def check_or_write_fingerprint(args, n_examples):
    """Refuse to resume a checkpoint written under different settings."""
    path = os.path.join(args.save_dir, "run_config.json")
    current = {k: getattr(args, k) for k in FINGERPRINT_KEYS}
    current["n_examples"] = n_examples

    if os.path.exists(path):
        with open(path) as fin:
            previous = json.load(fin)
        drift = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
        if drift:
            raise SystemExit(
                f"Refusing to resume: {path} was written under a different config.\n"
                + "\n".join(f"  {k}: checkpoint={p!r} now={n!r}" for k, (p, n) in sorted(drift.items()))
                + "\nDelete the directory to start clean, or fix the arguments."
            )
        return

    # Atomic, so a concurrent shard never reads a half-written file.
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fout:
        json.dump(current, fout, indent=2, sort_keys=True)
    os.replace(tmp, path)


def finalize(save_dir, dataset):
    """Merge the checkpoints into predictions.jsonl, then score."""
    completed = load_completed(save_dir)
    with open(os.path.join(save_dir, "run_config.json")) as fin:
        n = json.load(fin)["n_examples"]

    missing = [i for i in range(n) if i not in completed]
    if missing:
        # A partial arm is worse than no arm: the accuracy still looks plausible.
        raise SystemExit(
            f"Incomplete: {len(missing)}/{n} predictions missing (first: {missing[:5]}). "
            "Re-run the same command -- finished batches are skipped."
        )

    out = os.path.join(save_dir, "predictions.jsonl")
    with open(out, "w") as fout:
        for i in range(n):
            rec = dict(completed[i])
            rec.pop("idx")  # keep the artifact identical to pre-checkpoint runs
            fout.write(json.dumps(rec) + "\n")
    print(f"Merged {n} predictions -> {out}")

    logic_eval_main(out, save=True, output_dir=save_dir)


def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    if args.dataset == "LogiQA2":
        for ex in load_logiqa(english_only=args.logiqa_english_only):
            test_data.append({
                "question": ex["question"],
                "passage": ex["passage"],
                "options": ex["options"],
                "answer": ex["gt"],
                "gt": ex["gt"],
            })
    else:
        raise ValueError("Dataset not supported")

    if args.logiqa_eval_selection:
        # Explicit pool indices override the seeded sample. Indices are positions
        # in load_logiqa(english_only=True) order, i.e. rows of eval/logic_eval/logic_eval_task/LogiQA2/test.jsonl.
        # Used for eval_rand42_500_clean.json (contamination-free) and for scoring
        # just the replacement items.
        with open(args.logiqa_eval_selection) as fin:
            sel = json.load(fin)
        idxs = [item["pool_idx"] for item in sel["items"]]
        bad = [i for i in idxs if i >= len(test_data)]
        if bad:
            raise ValueError(f"selection has out-of-range pool_idx (pool size {len(test_data)}): {bad[:5]}")
        test_data = [test_data[i] for i in idxs]
        print(f"[eval] explicit selection: {len(test_data)} items from {args.logiqa_eval_selection}")
    elif args.random_sample:
        if args.max_examples and len(test_data) > args.max_examples:
            # Seeded random sample over the full split instead of the first N in
            # file order. Same scheme as the companion code-vector repository's eval/benchmarks.py (shuffle
            # indices, keep sorted first N) so both harnesses score identical tasks.
            idx = list(range(len(test_data)))
            random.Random(args.sample_seed).shuffle(idx)
            test_data = [test_data[i] for i in sorted(idx[:args.max_examples])]
    else:
        if args.start:
            test_data = test_data[args.start:]
        if args.max_examples and len(test_data) > args.max_examples:
            test_data = test_data[:args.max_examples]

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for i, example in enumerate(test_data):
        content = build_logiqa_prompt(example)
        prompt = content + "\nAnswer: "
        if args.use_chat_format:
            # Multiple-choice prompts carry their own instruction: always a single user turn.
            messages = [{"role": "user", "content": content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    if args.shard_rank == 0:
        with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
            fout.write(prompts[0])

    check_or_write_fingerprint(args, len(prompts))


    # The batch is the unit of both work and recovery. That is not just because
    # generate() produces one at a time: batches are left-padded to their
    # longest member, so an example's neighbours are part of its numerics.
    # Keeping batch boundaries at fixed global offsets -- across shards, and
    # across a resume -- means every example is scored in the same company it
    # would have had in a single uninterrupted run. The paired McNemar test
    # depends on that, the same way it depends on identical problem ordering.
    batch_starts = list(range(0, len(prompts), args.batch_size))
    my_starts = [s for j, s in enumerate(batch_starts)
                 if j % args.num_shards == args.shard_rank]

    completed = load_completed(args.save_dir)
    todo = [s for s in my_starts
            if not all(i in completed for i in range(s, min(s + args.batch_size, len(prompts))))]

    tag = "" if args.num_shards == 1 else f"[shard {args.shard_rank}/{args.num_shards}] "
    print(f"{tag}{len(my_starts)} batches assigned, "
          f"{len(my_starts) - len(todo)} already checkpointed, {len(todo)} to run")

    checkpoint = checkpoint_path(args.save_dir, args.shard_rank, args.num_shards)
    # Rewrite our own file from its parsed records, so a torn trailing line from
    # a killed run cannot corrupt the lines about to be appended.
    if os.path.exists(checkpoint):
        mine = _read_checkpoint(checkpoint)
        with open(checkpoint, "w") as fout:
            for rec in mine:
                fout.write(json.dumps(rec) + "\n")

    if not todo:
        print(f"{tag}nothing to generate")
        return

    if "qwen" in args.model_name_or_path.lower():
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=(torch.bfloat16 if args.model_dtype == "bf16" else torch.float32))
    else:
        raise ValueError("Model not supported")

    if args.steering:
        steer_vec = torch.load(args.steering_vector, weights_only=True)
        steer_vec = steer_vec.to(model.device)
        model.set_steering_flag(steering_flag=True, steering_layer=args.steering_layer, steer_vec=steer_vec,  steer_coef=args.steering_coef, tokenizer=tokenizer)

    with open(checkpoint, "a") as fout:
        for s in tqdm(todo, desc=tag.strip() or None):
            if args.steering:
                model.start_new_round()
            batch = prompts[s:s+args.batch_size]
            if args.gen_mode == "pruned":
                gen_ids = generate_pruned(model, tokenizer, batch, args.max_tokens,
                                          chunk=args.prune_chunk, steering=args.steering)
                output = decode_generated(tokenizer, gen_ids)
            else:
                tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
                tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
                with torch.no_grad():
                    output = model.generate(**tokenized_batch, do_sample=False, max_new_tokens=args.max_tokens,use_cache=True)
                prompt_len = tokenized_batch["input_ids"].shape[1]
                output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]

            for offset, text in enumerate(output):
                i = s + offset
                fout.write(json.dumps({
                    "idx": i,
                    "prompt": prompts[i],
                    "problem": test_data[i]["question"],
                    "answer": test_data[i]["gt"],
                    "solution": test_data[i]["answer"],
                    "model_generation": [trim_output(text)],
                }) + "\n")
            # A batch lands all-or-nothing, and hits the disk before the next
            # generate() call -- which is ~30 min at this decode rate. So a
            # crash, an OOM or a spot reclaim costs one batch, not the run.
            fout.flush()
            os.fsync(fout.fileno())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LogiQA2",
        choices=["LogiQA2", "LogiQA"],
        help="LogiQA2 (datatune/LogiQA2.0 test split, MRC rows). 'LogiQA' is a "
             "deprecated alias for LogiQA2 kept so old command lines keep working.",
    )
    parser.add_argument(
        "--logiqa_english_only",
        action="store_true",
        default=False,
        help="Drop CJK-heavy rows. LogiQA 2.0 is ~half "
             "untranslated Chinese; this applies the same is_cjk_heavy filter "
             "gen_logiqa_vllm.py uses for the build set. Off by default so "
             "existing mixed-language LogiQA results stay reproducible.",
    )
    parser.add_argument(
        "--logiqa_eval_selection",
        type=str,
        default=None,
        help="JSON with {'items': [{'pool_idx': int}, ...]} naming exactly which "
             "rows to score, overriding --random_sample/--max_examples. Indices "
             "index load_logiqa(english_only=True) order. Use "
             "eval/logic_eval/logic_eval_task/LogiQA2/eval_rand42_500_clean.json for the contamination-free set.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10000,   # the universal cap for extraction and eval; never lower it for a paper arm
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--steering",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--steering_vector",
        type=str,
        default=None
    )
    parser.add_argument(
        "--steering_layer",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--steering_coef",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the batches across this many independent processes, one per "
             "GPU. Each writes its own checkpoint; the arm is only scored once "
             "every example is present. The shards never communicate -- there "
             "are no gradients here, so do not reach for torchrun.",
    )
    parser.add_argument(
        "--shard_rank",
        type=int,
        default=0,
        help="Which shard this process is, in [0, num_shards).",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Weights/compute dtype for the HF steering model. fp32 (default) is the "
             "standard -- the convention every committed standard arm used; bf16 halves memory "
             "(7B fits a 40 GB card at batch 16) and ~doubles speed, at the cost of slightly "
             "different greedy trajectories vs an fp32 baseline -- a numerics confound to note "
             "wherever the two are compared. In the resume fingerprint.",
    )
    parser.add_argument(
        "--gen_mode",
        type=str,
        default="hf",
        choices=["hf", "pruned"],
        help="How a batch is decoded. hf (default, the standard: every committed standard arm) is model.generate(): the batch runs until its LAST row stops, and transformers' DynamicCache re-copies the whole KV every step, so with a 10k cap and ~20%% looping rows every batch costs batch x 10k steps at a per-step cost that grows with batch x length (~7 h per 500-example arm on an A100 at any batch size). pruned calls generate() in --prune_chunk-token chunks and drops finished rows between chunks (eval/src/steer_generate.py), so cost tracks the rows still running. Same greedy computation per row; only batch composition (fp32 noise) differs. Recorded in the resume fingerprint.",
    )
    parser.add_argument(
        "--prune_chunk",
        type=int,
        default=256,
        help="Tokens generated between pruning passes when --gen_mode pruned. Smaller = finished rows leave sooner, more generate() re-entries.",
    )
    parser.add_argument(
        "--finalize_only",
        action="store_true",
        help="Do not generate: stitch the shard checkpoints already in --save_dir into predictions.jsonl "
             "and grade them (used by scripts/eval_vector.sh after a sharded run; pass --num_shards 1).",
    )
    parser.add_argument(
        "--print_save_dir",
        action="store_true",
        help="Print the derived save_dir and exit, so a launcher can address "
             "the same directory without reimplementing the path logic.",
    )

    args = parser.parse_args()

    if args.dataset == "LogiQA":
        # Deprecated alias: the benchmark is LogiQA 2.0 and result folders are named *_on_LogiQA2.
        print("[eval] --dataset LogiQA is a deprecated alias for LogiQA2; using LogiQA2")
        args.dataset = "LogiQA2"

    if not 0 <= args.shard_rank < args.num_shards:
        parser.error(f"--shard_rank must be in [0, {args.num_shards}), got {args.shard_rank}")

    if args.logiqa_eval_selection and not args.logiqa_english_only:
        # The pinned selection files (eval_rand42_500*.json) index load_logiqa(english_only=True)
        # order. Applied to the unfiltered pool they pick different rows -- half of them Chinese.
        # That is exactly how the archived 7B legacy-500 runs went wrong, so refuse instead of scoring garbage.
        parser.error("--logiqa_eval_selection requires --logiqa_english_only: the selection files "
                     "index the English-only pool (eval/logic_eval/logic_eval_task/LogiQA2/test.jsonl order)")
    if args.logiqa_eval_selection and (args.max_examples is not None or args.start is not None):
        print("[eval] warning: --max_examples/--start are ignored when --logiqa_eval_selection is given "
              "(the selection file decides exactly which rows are scored)")

    # Results are written directly into --save_dir (one flat folder per arm, e.g.
    # eval/math_eval/math_eval_1.5B/v_math_on_MATH500/). The settings that older runs encoded as
    # nested folders (steering vector, coef, remove_bos, selection) are recorded in run_config.json,
    # and resuming into a folder written under different settings is refused by the fingerprint check.
    if args.print_save_dir:
        print(args.save_dir)
        sys.exit(0)

    print(args.save_dir)
    if args.finalize_only:
        finalize(args.save_dir, args.dataset)
        sys.exit(0)
    main(args)

    if args.num_shards == 1:
        finalize(args.save_dir, args.dataset)
    else:
        # Scoring waits for every shard; one finishing says nothing about the rest.
        print(f"[shard {args.shard_rank}/{args.num_shards}] generation done. Finalize once every shard is done:\n"
              f"  python <this harness> <same arguments> --num_shards 1 --finalize_only")


        
