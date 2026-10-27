#!/usr/bin/env bash
# S_math steered on MATH500 (100-example smoke). Run from the repo root.
# The LiveCodeBench half of the original recipe was retired on 2026-08-23: the retired-benchmarks folder (removed from the repo on 2026-08-23 (team decision; in git history))
gpu=${1:-0}
CUDA_VISIBLE_DEVICES=$gpu python eval/math_eval/pipeline/eval_math_steering.py \
    --model_name_or_path  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_dir eval/math_eval/math_eval_1.5B/v_math_on_MATH500/steer-from-MATH-10000-use-original \
    --max_tokens 10000 \
    --use_chat_format \
    --batch_size  25 \
    --dataset MATH500 \
    --remove_bos \
    --steering \
    --steering_vector extraction/math_extraction/math_extraction_1.5B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt \
    --steering_layer 20 \
    --steering_coef -1.0 \
    --start 0 \
    --max_examples 100

