#!/usr/bin/env bash
# Recipe (prints the command): unsteered MATH500 baseline with the vLLM harness. Run from the repo root.
gpu=${1:-0}
echo CUDA_VISIBLE_DEVICES=$gpu python eval/math_eval/pipeline/eval_math_vllm.py \
    --model_name_or_path  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --save_dir eval/math_eval/math_eval_1.5B/baseline_on_MATH500/vllm_baseline_10000 \
    --max_tokens 10000 \
    --use_chat_format \
    --dataset "MATH500" \
    --remove_bos