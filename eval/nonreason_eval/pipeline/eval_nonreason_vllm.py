
import argparse
import os
import re
import json
import random
import sys
import gc

# eval/<domain>_eval/pipeline/<this file>  ->  repo root is three levels up.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial

from triviaqa_utils import load_triviaqa_jsonl, build_triviaqa_prompt, triviaqa_eval_main
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TriviaQA (closed-book) is the NON-REASONING domain. "TriviaQA" evaluates the
# held-out pool (nonreason_eval_task/TriviaQA/test.jsonl: the 11,313 validation
# rows in native order); "TriviaQA_train" is the v_nonreason extraction pool
# (extraction/nonreason_extraction/nonreason_pool/TriviaQA/train.jsonl) and only
# makes sense for baseline trace generation.
TRIVIAQA_DATASETS = ("TriviaQA", "TriviaQA_train")
TRIVIAQA_TEST_PATH = os.path.join(_HERE, "..", "nonreason_eval_task", "TriviaQA", "test.jsonl")
TRIVIAQA_TRAIN_PATH = os.path.join(REPO_ROOT, "extraction", "nonreason_extraction", "nonreason_pool", "TriviaQA", "train.jsonl")

exact_match = evaluate.load("exact_match")


def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits



def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output






def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    if args.dataset in TRIVIAQA_DATASETS:
        # Read the prepared JSONL rather than the Hub so the row order the build
        # generates in is byte-identical to the file hidden_analysis.py zips
        # against. Run the prep script under extraction/nonreason_extraction/ first.
        data_path = TRIVIAQA_TRAIN_PATH if args.dataset == "TriviaQA_train" else TRIVIAQA_TEST_PATH
        for ex in load_triviaqa_jsonl(data_path):
            test_data.append({
                "question": ex["problem"],
                # The alias list rides in "answer" so it lands in predictions.jsonl
                # as "solution", where triviaqa_eval_main reads it back. gt stays
                # the canonical value.
                "answer": ex["aliases"],
                "gt": ex["gt"],
            })
    else:
        raise ValueError("Dataset not supported")
    if args.max_examples and len(test_data) > args.max_examples:
        if args.random_sample:
            # Seeded random sample over the full split instead of the first N in
            # file order. Same scheme as the companion code-vector repository's eval/benchmarks.py (shuffle
            # indices, keep sorted first N) so both harnesses score identical tasks.
            idx = list(range(len(test_data)))
            random.Random(args.sample_seed).shuffle(idx)
            test_data = [test_data[i] for i in sorted(idx[:args.max_examples])]
        else:
            test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for i, example in enumerate(test_data):
        content = build_triviaqa_prompt(example)
        prompt = content + "\nAnswer: "
        if args.use_chat_format:
            # The TriviaQA instruction carries its own 'Answer: X' contract: always a single user turn.
            messages = [{"role": "user", "content": content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    model = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, swap_space=16, gpu_memory_utilization=0.98, enable_lora=args.peft is not None, tensor_parallel_size=(args.tensor_parallel_size or torch.cuda.device_count()), max_lora_rank=128, max_model_len=args.max_tokens+2000)

    if not args.logit_adjustment:

        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens)
    else:
        vocab = tokenizer.get_vocab()
        logit_adjustment_tokens = torch.LongTensor([vocab[token] for token in vocab.keys() if any([x in token for x in args.logit_adjustment_tokens])]).to("cuda")
        logit_adjustment_process = partial(logit_adjustment, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens,
                                        logits_processors=[logit_adjustment_process]
                                        )
    if args.peft is not None:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
    else:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            attempts.append(ith_output.text)
        result.append(attempts)
    

    outputs = [[trim_output(o) for o in output] for output in result]

    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "answer": example["gt"],
        "solution":  example["answer"],
        "model_generation": output,
    } for example, output, prompt in zip(test_data, outputs, prompts)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_examples",
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
        default="results/triviaqa"
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
        "--peft",
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
        default="TriviaQA",
        choices=["TriviaQA", "TriviaQA_train"],
        help="TriviaQA (the held-out validation pool, nonreason_eval_task/TriviaQA/test.jsonl) / "
             "TriviaQA_train (the v_nonreason extraction pool, extraction/nonreason_extraction/nonreason_pool/TriviaQA/train.jsonl)",
    )
    parser.add_argument(
        "--em_mode",
        type=str,
        default="aliases",
        choices=["aliases", "contains", "value"],
        help="TriviaQA grading: 'aliases' (default) is the official normalized-EM "
             "metric over the dataset's alias list; 'contains' also accepts an alias "
             "appearing as a token run inside a verbose answer line; 'value' scores "
             "only the canonical answer. Use 'value' to rebuild extraction labels if "
             "metrics.json reports a high n_alias_only_match.",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10000,   # the universal cap for extraction and eval; never lower it for a paper arm
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=0,
        help="vLLM tensor parallelism. 0 (default) = every visible GPU, the "
             "historical behaviour. Set explicitly for DeepSeek-R1-Distill-Qwen-7B "
             "on a >4-GPU box: it has 28 attention / 4 KV heads, so TP=8 crashes; "
             "TP=1 fits the fp16 weights (~15 GB) on any 24 GB+ card.",
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        type=str,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )


    args = parser.parse_args()



    # Record the run settings next to the outputs (the steering harness does the same via its fingerprint).
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as fout:
        json.dump({k: (v if isinstance(v, (str, int, float, bool, list, type(None))) else str(v)) for k, v in vars(args).items()},
                  fout, indent=2, sort_keys=True)

    main(args)
    triviaqa_eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True,
                       output_dir=args.save_dir, em_mode=args.em_mode)
    



