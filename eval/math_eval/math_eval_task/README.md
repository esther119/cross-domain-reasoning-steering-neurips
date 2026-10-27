# math_eval_task

| file | what | used by |
|---|---|---|
| `MATH/test.jsonl` | full MATH test (5,000) | nothing yet — MATH-500 is loaded from the Hub (`HuggingFaceH4/MATH-500`) |

The MATH *train* split is the extraction pool: `extraction/math_extraction/math_pool/MATH/train.jsonl`
(`pipeline/eval_math_vllm.py --dataset MATH_train`).
