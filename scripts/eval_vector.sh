#!/usr/bin/env bash
# Evaluate one steering vector (or the unsteered baseline) on the active benchmarks, each through its own
# harness, at the pinned protocol for the model size. One driver covers every arm: the vector, the arm name,
# the model size, the benchmarks and the GPUs are parameters. Run from anywhere (cd's to the repo root).
#
#   VECTOR=<path.pt> ARM=v_math    MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0                 # one GPU
#   VECTOR=<path.pt> ARM=v_logic   MODEL_SIZE=7B   GPUS=0,1,2,3 DATASETS=math bash scripts/eval_vector.sh
#                                                                  # sharded over 4 GPUs, checkpointed, resumable
#   VECTOR=baseline  MODEL_SIZE=7B bash scripts/eval_vector.sh 0                                  # baselines only
#   for s in 1 2 3; do VECTOR=extraction/control_extraction/control_extraction_1.5B/result/R_iso_seed$s.pt \
#       ARM=R_iso_seed$s MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0; done                       # random controls
#
# Results (flat, one folder per arm):  eval/<domain>_eval/<domain>_eval_<MODEL_SIZE>/<ARM>_on_<BENCH>/
#   math -> MATH500 (all 500)   code -> APPS (test, seeded 500)   logic -> LogiQA2 (clean 500, English-only)
# Each folder holds predictions.jsonl, math_eval.jsonl / code_eval.jsonl, metrics.json, example_prompt.txt and
# run_config.json (the fingerprint: resuming into a folder written under different settings is refused).
#
# Placeholders (env):
#   VECTOR       path to the .pt (required; "baseline" evaluates only the unsteered model)
#   ARM          result name: v_math | v_code | v_logic | v_general | v_combo | R_iso_seedN | ...
#   MODEL_SIZE   1.5B | 7B  (-> MODEL; pins the batch sizes: 1.5B 25/1/25, 7B 16/2/8 for MATH500/APPS/LogiQA2)
#   DATASETS     comma list of math,apps,logiqa (default all three; livecodebench accepted, retired)
#   GPUS         comma list of GPU indices (default: the first positional arg, else 0). More than one GPU = data-parallel
#                shards, one process per GPU, every batch checkpointed; a killed shard costs one batch and a re-run
#                resumes. Batch boundaries are the same as a single-GPU run, so numerics are unchanged.
#   RUN_BASELINE 1 also runs baseline_on_<BENCH> (default 0 when VECTOR is a file, 1 when VECTOR=baseline)
#   MAX_ATTEMPTS 3 (retries of a failed shard set)   MAX_TOKENS 10000 (universal)   LAYER 20   COEF -1.0   SAMPLE_SEED 42
#   GEN_MODE hf|pruned  MODEL_DTYPE fp32|bf16  (the standard is hf + fp32; pruned/bf16 are exploratory only)
#   EVAL_ROOT eval, *_BATCH_SIZE overrides, LOGIQA_SELECTION, APPS_TASK_FILE (the seeded 500, pinned by id),
#   APPS_SPLIT (test), EVAL_WORKERS, APPS_TIMEOUT
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }

: "${VECTOR:?set VECTOR=<path.pt> or VECTOR=baseline}"
: "${MODEL_SIZE:=1.5B}"
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-${MODEL_SIZE}}"
if [[ "$VECTOR" == "baseline" ]]; then ARM=baseline; RUN_BASELINE=1; else : "${ARM:?set ARM=<arm name, e.g. v_math>}"; : "${RUN_BASELINE:=0}"; fi
[[ "$ARM" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ARM '$ARM' must be letters/digits/._-" >&2; exit 2; }
[[ "$VECTOR" == "baseline" || -f "$VECTOR" ]] || { echo "vector not found: $VECTOR" >&2; exit 2; }
: "${GPUS:=${1:-0}}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"; NUM_GPUS=${#GPU_ARR[@]}
: "${DATASETS:=math,apps,logiqa}"
: "${EVAL_ROOT:=eval}"
: "${MAX_TOKENS:=10000}"; : "${LAYER:=20}"; : "${COEF:=-1.0}"; : "${SAMPLE_SEED:=42}"; : "${MAX_ATTEMPTS:=3}"
: "${GEN_MODE:=hf}"; : "${PRUNE_CHUNK:=256}"; : "${MODEL_DTYPE:=fp32}"
case "$MODEL_SIZE" in
  1.5B) : "${MATH_BATCH_SIZE:=25}"; : "${APPS_BATCH_SIZE:=1}"; : "${LOGIQA_BATCH_SIZE:=25}"; : "${LCB_BATCH_SIZE:=25}" ;;
  7B)   : "${MATH_BATCH_SIZE:=16}"; : "${APPS_BATCH_SIZE:=2}"; : "${LOGIQA_BATCH_SIZE:=8}";  : "${LCB_BATCH_SIZE:=16}" ;;
  *) echo "MODEL_SIZE must be 1.5B or 7B (got '$MODEL_SIZE'); set the *_BATCH_SIZE pins yourself for other models" >&2; exit 2 ;;
esac
: "${MATH_MAX_EXAMPLES:=500}"; : "${APPS_MAX_EXAMPLES:=500}"; : "${LOGIQA_MAX_EXAMPLES:=500}"; : "${LCB_MAX_EXAMPLES:=400}"
: "${APPS_SPLIT:=test}"; : "${LCB_RELEASE:=release_v1}"; : "${EVAL_WORKERS:=12}"; : "${APPS_TIMEOUT:=10}"
: "${LOGIQA_SELECTION:=eval/logic_eval/logic_eval_task/LogiQA2/eval_rand42_500_clean.json}"
: "${APPS_TASK_FILE:=eval/code_eval/code_eval_task/APPS/test_rand42_500.jsonl}"   # the seeded 500 by id; tests reload from APPS
# The growing-KV pattern of HF decoding fragments the caching allocator; expandable_segments avoids that
# (measured: no effect on speed or numerics).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MATH_ROOT="$EVAL_ROOT/math_eval/math_eval_$MODEL_SIZE";                MATH_PIPE=eval/math_eval/pipeline
CODE_ROOT="$EVAL_ROOT/code_eval/code_eval_$MODEL_SIZE";                CODE_PIPE=eval/code_eval/pipeline
LOGIC_ROOT="$EVAL_ROOT/logic_eval/logic_eval_$MODEL_SIZE";             LOGIC_PIPE=eval/logic_eval/pipeline

IFS=',' read -r -a SELECTED <<< "$DATASETS"
for d in "${SELECTED[@]}"; do case "$d" in math|apps|logiqa|livecodebench) ;; *) echo "unknown dataset '$d' (math,apps,logiqa)" >&2; exit 2 ;; esac; done
enabled() { case ",$DATASETS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
if enabled logiqa && [[ ! -f "$LOGIQA_SELECTION" ]]; then echo "LOGIQA_SELECTION not found: $LOGIQA_SELECTION" >&2; exit 2; fi
if enabled apps && [[ ! -f "$APPS_TASK_FILE" ]]; then echo "APPS_TASK_FILE not found: $APPS_TASK_FILE" >&2; exit 2; fi

COMMON=(--model_name_or_path "$MODEL" --max_tokens "$MAX_TOKENS" --use_chat_format --remove_bos
        --gen_mode "$GEN_MODE" --prune_chunk "$PRUNE_CHUNK" --model_dtype "$MODEL_DTYPE" --num_shards "$NUM_GPUS")
STEER=(--steering --steering_vector "$VECTOR" --steering_layer "$LAYER" --steering_coef "$COEF")
MATH_ARGS=(  --dataset MATH500 --batch_size "$MATH_BATCH_SIZE" --random_sample --sample_seed "$SAMPLE_SEED" --max_examples "$MATH_MAX_EXAMPLES")
APPS_ARGS=(  --benchmark apps --apps_split "$APPS_SPLIT" --apps_task_file "$APPS_TASK_FILE" --batch_size "$APPS_BATCH_SIZE" --random_sample --sample_seed "$SAMPLE_SEED"
             --max_examples "$APPS_MAX_EXAMPLES" --eval_workers "$EVAL_WORKERS" --timeout "$APPS_TIMEOUT")
LOGIC_ARGS=( --dataset LogiQA2 --batch_size "$LOGIQA_BATCH_SIZE" --logiqa_english_only --logiqa_eval_selection "$LOGIQA_SELECTION")
LCB_ARGS=(   --benchmark livecodebench --release "$LCB_RELEASE" --batch_size "$LCB_BATCH_SIZE" --max_examples "$LCB_MAX_EXAMPLES")

warm_caches() {  # N shards racing to download the same weights / parquet files is how caches get corrupted: do it once, serially.
  [[ "$NUM_GPUS" -gt 1 ]] || return 0
  python - "$MODEL" "$DATASETS" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])
ds = sys.argv[2].split(",")
if "math" in ds:
    from datasets import load_dataset; load_dataset("HuggingFaceH4/MATH-500", split="test")
if "apps" in ds:
    sys.path.insert(0, "eval/code_eval/pipeline"); from apps_utils import load_apps; load_apps(split="test", max_examples=1)
print("caches ready")
PY
}

run() {  # run <harness.py> <save_dir> <args...>   — single process, or NUM_GPUS shards + finalize
  local harness=$1 save=$2; shift 2
  echo ">>> $save  (gpus: $GPUS)"
  if [[ "$NUM_GPUS" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python "$harness" "${COMMON[@]}" "$@" --save_dir "$save"
    return
  fi
  mkdir -p "$save/logs"
  local attempt rank pids failed
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    pids=(); failed=0
    for rank in $(seq 0 $((NUM_GPUS - 1))); do
      CUDA_VISIBLE_DEVICES="${GPU_ARR[$rank]}" python "$harness" "${COMMON[@]}" "$@" --save_dir "$save" \
        --shard_rank "$rank" >"$save/logs/rank${rank}.log" 2>&1 &
      pids+=($!)
    done
    for rank in "${!pids[@]}"; do wait "${pids[$rank]}" || { echo "!!! shard $rank failed -- see $save/logs/rank${rank}.log" >&2; failed=1; }; done
    [[ "$failed" -eq 0 ]] && break
    [[ "$attempt" -lt "$MAX_ATTEMPTS" ]] && { echo ">>> retrying (finished batches are skipped)"; sleep 10; } \
      || { echo "!!! giving up after $MAX_ATTEMPTS attempts; re-running resumes from the checkpoints" >&2; exit 1; }
  done
  # Stitch the shard checkpoints into predictions.jsonl and grade, once (for APPS this executes the generated code).
  python "$harness" "${COMMON[@]}" "$@" --save_dir "$save" --num_shards 1 --finalize_only
}

echo "=== eval_vector: ARM=$ARM MODEL=$MODEL DATASETS=$DATASETS GPUS=$GPUS RUN_BASELINE=$RUN_BASELINE VECTOR=$VECTOR"
warm_caches
if [[ "$RUN_BASELINE" == "1" ]]; then
  enabled math     && run $MATH_PIPE/eval_math_steering.py           "$MATH_ROOT/baseline_on_MATH500"       "${MATH_ARGS[@]}"
  enabled apps     && run $CODE_PIPE/eval_code_steering.py           "$CODE_ROOT/baseline_on_APPS"          "${APPS_ARGS[@]}"
  enabled logiqa   && run $LOGIC_PIPE/eval_logic_steering.py         "$LOGIC_ROOT/baseline_on_LogiQA2"      "${LOGIC_ARGS[@]}"
  enabled livecodebench && run $CODE_PIPE/eval_code_steering.py      "$CODE_ROOT/baseline_on_LiveCodeBench" "${LCB_ARGS[@]}"
fi
if [[ "$VECTOR" != "baseline" ]]; then
  enabled math     && run $MATH_PIPE/eval_math_steering.py           "$MATH_ROOT/${ARM}_on_MATH500"       "${MATH_ARGS[@]}"   "${STEER[@]}"
  enabled apps     && run $CODE_PIPE/eval_code_steering.py           "$CODE_ROOT/${ARM}_on_APPS"          "${APPS_ARGS[@]}"   "${STEER[@]}"
  enabled logiqa   && run $LOGIC_PIPE/eval_logic_steering.py         "$LOGIC_ROOT/${ARM}_on_LogiQA2"      "${LOGIC_ARGS[@]}"  "${STEER[@]}"
  enabled livecodebench && run $CODE_PIPE/eval_code_steering.py      "$CODE_ROOT/${ARM}_on_LiveCodeBench" "${LCB_ARGS[@]}"    "${STEER[@]}"
fi
echo "Finished. Results under $EVAL_ROOT/<domain>_eval/<domain>_eval_$MODEL_SIZE/${ARM}_on_<BENCH>/"
