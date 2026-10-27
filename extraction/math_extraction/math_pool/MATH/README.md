# MATH train pool (7,500 problems)

Source for S_math extraction. Consumed by `eval/math_eval/pipeline/eval_math_vllm.py --dataset MATH_train`
(trace generation + Omni-MATH rule grading) and by `extraction/src/hidden_analysis.py --data_path` (problem metadata / level).
The MATH *test* split lives in `eval/math_eval/math_eval_task/MATH/` (MATH-500 itself is loaded from `HuggingFaceH4/MATH-500`).
