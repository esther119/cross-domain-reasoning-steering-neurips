import os
import json
from tqdm import tqdm, trange
from eval_math_rule.evaluation.grader import math_equal
from eval_math_rule.evaluation.parser import extract_answer, parse_ground_truth, strip_string
from collections import Counter

import multiprocessing
import queue

def math_equal_with_timeout(pred, gt_ans, timeout):
    def target(result_queue):
        try:
            result_queue.put(math_equal(pred, gt_ans))
        except Exception as e:
            result_queue.put(e)


    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(result_queue,))
    process.start()

    process.join(timeout)

    if process.is_alive():
        print(f"Timeout occurred for prediction: {pred}")
        process.terminate()
        process.join()    
        return False

   
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        print("Result queue timed out")
        return False

    if isinstance(result, Exception):
        print(f"Error occurred: {result}")
        return False

    return result



def parallel_math_equal(all_pred, gt_ans, timeout=20):
    results = []
    for pred in all_pred:
        results.append(math_equal_with_timeout(pred, gt_ans, timeout))
    return results



def main(res_path, save=False, k=None, output_dir=None):
    # args = parse_args()
    with open(res_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    for example in tqdm(data):
        # gt_cot, gt = parse_ground_truth(example, data_name="omni-math")
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None:
            example["model_generation"] = example["model_generation"][:k]
        gt_cot = example["answer"]
        gt_ans = extract_answer(gt_cot, data_name="omni-math")
        gt_cot = str(gt_cot).strip()
        gt_ans = strip_string(gt_ans, skip_unit=False)
        # A generation that never closed </think> ran out of budget still inside the
        # reasoning block: it never reached an answer region, so a \boxed{} found in
        # mid-reasoning is a tentative value, not a stated answer. Unfinished scores as
        # incorrect -- the same rule as logic_utils.extract_choice --
        # and is counted separately so "reasoned and got it wrong" stays distinguishable
        # from "never stopped reasoning".
        all_unfinished = ["</think>" not in g for g in example["model_generation"]]
        all_pred = [
            "" if unf else extract_answer(p, data_name="omni-math")
            for p, unf in zip(example["model_generation"], all_unfinished)
        ]
        all_pred = [strip_string(p, skip_unit=False) for p in all_pred]
        # all_eval = [math_equal(p, gt_ans) for p in all_pred]
        all_eval = parallel_math_equal(all_pred, gt_ans, timeout=5)
        effective_pred = [p for p, o in zip(all_pred, example["model_generation"]) if "boxed" in o]
        if len(effective_pred) == 0:
            effective_pred = all_pred
        counter = Counter(effective_pred)
        pred = counter.most_common(1)[0][0]
        index = all_pred.index(pred)
        eval = all_eval[index]
        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"] = pred
        example["mv_eval"] = eval
        example["mv_index"] = index
        example["unfinished"] = all(all_unfinished)
        example["answered"] = not all(all_unfinished)

    n = len(data)
    acc = sum([example["mv_eval"] for example in data]) / n
    n_unfinished = sum(e["unfinished"] for e in data)
    n_answered = n - n_unfinished
    metrics = {
        "acc": acc,
        "n": n,
        "n_answered": n_answered,
        "n_unfinished": n_unfinished,
        "answer_rate": n_answered / n if n else 0.0,
        "acc_answered": (sum(e["mv_eval"] for e in data) / n_answered) if n_answered else 0.0,
        "grader": "reasoning",
    }
    print(f"Accuracy: {acc:.3f}  (answered {n_answered}/{n}, unfinished {n_unfinished})")
    
    if save:
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        
        metric_file= os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump(metrics, f)

    