#!/usr/bin/env bash
# Build Phase 1 S_general from MATH + APPS + LogiQA (unbalanced pool of all
# boundary vectors).
#
# In-repo layout (everything root-relative; this script cd's to the repo root):
#   extraction/math_extraction/math_extraction_1.5B/result/baseline_10000/        MATH traces + hidden_*_0_500/hidden.pt
#   extraction/code_extraction/code_extraction_1.5B/result/baseline_10000/        shipped APPS traces (math_eval.jsonl.gz, decompressed on first use)
#   extraction/logic_extraction/logic_extraction_1.5B/result/baseline_${LOGIQA_MAX_TOKENS}_regraded/
#                                                      canonical, strictly graded LogiQA traces (10k cap = baseline_10000_regraded;
#                                                      the shipped S_general used the superseded 3k pool, baseline_3000_regraded)
#   extraction/code_extraction/code_extraction_1.5B/result/apps_v_code.pt                            shipped code-domain home vector (cosine smoke)
#   extraction/logic_extraction/logic_extraction_1.5B/result/logiqa2_v_logic.pt                        shipped logic-domain home vector (cosine smoke)
#   extraction/src/apps/                               APPS generation + scoring
#   extraction/src/gen_logiqa_vllm.py + eval/logic_eval/pipeline/logic_utils.py
#                                                      canonical LogiQA generation + scoring
#   extraction/src/hidden_analysis.py                  ONE extractor — --keywords math|code|logic
#   hidden.pt files are written next to their traces (hidden_<type>_0_N/hidden.pt) and are the
#   inputs to the pooler below; hidden.pt is gitignored (multi-GB) — park it on the HF Hub, as the 7B records do.
#
# Usage:
#   bash extraction/general_extraction/general_extraction_1.5B/pipeline/build_general_vector.sh            # full Phase 1 (GPU)
#   SKIP_HIDDEN=1 bash .../build_general_vector.sh       # pool only
#   SKIP_LOGIQA_GEN=1 bash .../build_general_vector.sh   # skip LogiQA gen if traces exist
#   BALANCE=1 bash .../build_general_vector.sh           # Phase 2 equal-domain subsample
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"   # repo root
cd "$REPO_DIR"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

: "${GPU:=0}"
: "${HF_HOME:=${HF_HOME:-/workspace/hf_cache}}"
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${STEER_LAYER:=20}"
: "${VEC_SAMPLES:=500}"
: "${SKIP_HIDDEN:=0}"
: "${SKIP_LOGIQA_GEN:=0}"
: "${BALANCE:=0}"
: "${BALANCE_SEED:=42}"
: "${MAX_CJK:=0.10}"
: "${LOGIQA_MAX_TOKENS:=10000}"     # universal 10k cap; LOGIQA_DIR is derived from it (baseline_10000_regraded).
                                    # LOGIQA_MAX_TOKENS=3000 reproduces the superseded pool the shipped vector used.
: "${LOGIQA_MAX_EXAMPLES:=2000}"

export HF_HOME CUDA_VISIBLE_DEVICES="$GPU"
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL_TAG="$(basename "$MODEL")"
MATH_DIR="extraction/math_extraction/math_extraction_1.5B/result/baseline_10000"
MATH_DATA="extraction/math_extraction/math_pool/MATH/train.jsonl"
APPS_DIR="extraction/code_extraction/code_extraction_1.5B/result/baseline_10000"   # shipped traces live here (math_eval.jsonl.gz)
APPS_DATA="${APPS_DIR}/data.jsonl"
LOGIQA_SOURCE="extraction/logic_extraction/logic_pool/LogiQA2/train.jsonl"
LOGIQA_DIR="extraction/logic_extraction/logic_extraction_1.5B/result/baseline_${LOGIQA_MAX_TOKENS}_regraded"
LOGIQA_DATA="${LOGIQA_DIR}/data.jsonl"
LOGIQA_EVAL="${LOGIQA_DIR}/math_eval.jsonl"

# hidden.pt written by hidden_analysis.py, next to the traces
MATH_HIDDEN_C="${MATH_DIR}/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
MATH_HIDDEN_I="${MATH_DIR}/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"
APPS_HIDDEN_C="${APPS_DIR}/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
APPS_HIDDEN_I="${APPS_DIR}/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"
LOGIQA_HIDDEN_C="${LOGIQA_DIR}/hidden_correct_0_${VEC_SAMPLES}/hidden.pt"
LOGIQA_HIDDEN_I="${LOGIQA_DIR}/hidden_incorrect_0_${VEC_SAMPLES}/hidden.pt"

OUT_DIR="extraction/general_extraction/general_extraction_1.5B/result"
if [[ "$BALANCE" == "1" ]]; then
    OUT_VEC="${OUT_DIR}/S_general_math_apps_logic_phase2.pt"
else
    OUT_VEC="${OUT_DIR}/S_general_math_apps_logic_phase1.pt"
fi
MATH_VEC="${MATH_DIR}/vector_${VEC_SAMPLES}_${VEC_SAMPLES}/layer_${STEER_LAYER}_transition_reflection_steervec.pt"
APPS_VEC="extraction/code_extraction/code_extraction_1.5B/result/apps_v_code.pt"
LOGIQA_VEC="extraction/logic_extraction/logic_extraction_1.5B/result/logiqa2_v_logic.pt"

echo "=================================================================="
echo " S_general · MATH + APPS + LogiQA"
echo " MODEL=$MODEL  GPU=$GPU  LAYER=$STEER_LAYER  SAMPLES=$VEC_SAMPLES"
echo " BALANCE=$BALANCE  SKIP_HIDDEN=$SKIP_HIDDEN  SKIP_LOGIQA_GEN=$SKIP_LOGIQA_GEN"
echo " MATH_DIR=$MATH_DIR"
echo " APPS_DIR=$APPS_DIR"
echo " LOGIQA_DIR=$LOGIQA_DIR"
echo " OUT=$OUT_VEC"
echo "=================================================================="

# ------------------------------------------------------------------ #
# Sanity: required inputs
# ------------------------------------------------------------------ #
[[ -f "$MATH_DIR/math_eval.jsonl" ]] || { echo "ERROR: missing $MATH_DIR/math_eval.jsonl"; exit 1; }
[[ -f "$MATH_DATA" ]] || { echo "ERROR: missing $MATH_DATA"; exit 1; }
[[ ( -f "$APPS_DIR/math_eval.jsonl.gz" || -f "$APPS_DIR/math_eval.jsonl" ) && -f "$APPS_DIR/data.jsonl" ]] || {
    echo "ERROR: missing shipped APPS traces under $APPS_DIR"
    echo "  Expected data.jsonl + math_eval.jsonl(.gz) (from the companion code-vector repository's baseline_10000)."
    exit 1
}
[[ -f "$LOGIQA_SOURCE" ]] || {
    echo "ERROR: missing canonical English LogiQA source $LOGIQA_SOURCE"
    exit 1
}
[[ -f "$APPS_VEC" ]] || echo "WARN: missing $APPS_VEC — cosine compare vs apps will be skipped."

# ------------------------------------------------------------------ #
# Decompress the shipped APPS traces once (math_eval.jsonl.gz -> math_eval.jsonl)
# ------------------------------------------------------------------ #
if [[ ! -f "$APPS_DIR/math_eval.jsonl" ]]; then
    echo "[apps] Decompressing shipped traces in $APPS_DIR"
    gunzip -c "$APPS_DIR/math_eval.jsonl.gz" > "$APPS_DIR/math_eval.jsonl"
fi
[[ -f "$APPS_DATA" && -f "$APPS_DIR/math_eval.jsonl" ]] || {
    echo "ERROR: missing APPS traces in $APPS_DIR"; exit 1;
}

# ------------------------------------------------------------------ #
# 0) LogiQA traces (canonical strict grader; English-only pool filter)
# ------------------------------------------------------------------ #
mkdir -p "$LOGIQA_DIR"
if [[ "$SKIP_HIDDEN" == "1" ]]; then
    echo "[logiqa] SKIP_HIDDEN=1 — not generating traces (need durable hidden.pt)."
elif [[ -f "$LOGIQA_HIDDEN_C" && -f "$LOGIQA_HIDDEN_I" ]]; then
    echo "[logiqa] working hidden.pt already present — skipping generation."
elif [[ "$SKIP_LOGIQA_GEN" == "1" ]]; then
    [[ -f "$LOGIQA_EVAL" && -f "$LOGIQA_DATA" ]] || {
        echo "ERROR: SKIP_LOGIQA_GEN=1 but missing $LOGIQA_EVAL"
        exit 1
    }
    echo "[logiqa] SKIP_LOGIQA_GEN=1 — using existing traces."
elif [[ -f "$LOGIQA_EVAL" && -f "$LOGIQA_DATA" ]]; then
    echo "[logiqa] canonical traces already exist — reusing them."
else
    echo "[logiqa] Generating $LOGIQA_MAX_EXAMPLES canonical English traces..."
    python -u extraction/src/gen_logiqa_vllm.py \
        --model_name_or_path "$MODEL" \
        --save_dir "$LOGIQA_DIR" \
        --max_tokens "$LOGIQA_MAX_TOKENS" \
        --max_examples "$LOGIQA_MAX_EXAMPLES" \
        --split train \
        --filter_cjk \
        --cjk_threshold "$MAX_CJK" \
        --use_chat_format \
        --remove_bos
fi

if [[ "$SKIP_HIDDEN" != "1" && ! ( -f "$LOGIQA_HIDDEN_C" && -f "$LOGIQA_HIDDEN_I" ) ]]; then
    python - "$LOGIQA_EVAL" "$VEC_SAMPLES" <<'PY'
import json
import sys

path, target = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path)]
fabricated = sum(
    "</think>" not in row["model_generation"][0] and bool(row["all_eval"][0])
    for row in rows
)
correct = sum(bool(row["all_eval"][0]) for row in rows)
incorrect = len(rows) - correct
if fabricated:
    raise SystemExit(
        f"ERROR: {fabricated} unfinished traces are labelled correct in {path}; "
        "regrade with extraction/logic_extraction/logic_extraction_1.5B/pipeline/relabel_logiqa_build.py before extraction."
    )
if correct < target or incorrect < target:
    raise SystemExit(
        f"ERROR: LogiQA pools under-filled: correct={correct}, incorrect={incorrect}, "
        f"target={target}. Increase LOGIQA_MAX_EXAMPLES and regenerate."
    )
print(
    f"[logiqa] strict-label guard passed: correct={correct}, "
    f"incorrect={incorrect}, target={target}"
)
PY
fi

# ------------------------------------------------------------------ #
# 1) MATH hidden states (layer STEER_LAYER, keywords=math)
# ------------------------------------------------------------------ #
if [[ "$SKIP_HIDDEN" == "1" ]]; then
    echo "[math] SKIP_HIDDEN=1 — not regenerating."
else
    if [[ -f "$MATH_HIDDEN_I" ]]; then
        echo "[math] incorrect hidden.pt exists, skipping."
    else
        echo "[math] Extracting incorrect hidden states (layer $STEER_LAYER, keywords=math)..."
        python extraction/src/hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$MATH_DATA" \
            --data_dir "$MATH_DIR" \
            --type incorrect --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords math
    fi
    if [[ -f "$MATH_HIDDEN_C" ]]; then
        echo "[math] correct hidden.pt exists, skipping."
    else
        echo "[math] Extracting correct hidden states (layer $STEER_LAYER, keywords=math)..."
        python extraction/src/hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$MATH_DATA" \
            --data_dir "$MATH_DIR" \
            --type correct --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords math
    fi
fi

# ------------------------------------------------------------------ #
# 2) APPS hidden states (SAME hidden_analysis.py, keywords=code)
# ------------------------------------------------------------------ #
if [[ "$SKIP_HIDDEN" == "1" ]]; then
    echo "[apps] SKIP_HIDDEN=1 — not regenerating."
else
    if [[ -f "$APPS_HIDDEN_I" ]]; then
        echo "[apps] incorrect hidden.pt exists, skipping."
    else
        echo "[apps] Extracting incorrect hidden states (layer $STEER_LAYER, keywords=code)..."
        python extraction/src/hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$APPS_DATA" \
            --data_dir "$APPS_DIR" \
            --type incorrect --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords code
    fi
    if [[ -f "$APPS_HIDDEN_C" ]]; then
        echo "[apps] correct hidden.pt exists, skipping."
    else
        echo "[apps] Extracting correct hidden states (layer $STEER_LAYER, keywords=code)..."
        python extraction/src/hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$APPS_DATA" \
            --data_dir "$APPS_DIR" \
            --type correct --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords code
    fi
fi

# ------------------------------------------------------------------ #
# 3) LogiQA hidden states (SAME hidden_analysis.py, keywords=logic)
# ------------------------------------------------------------------ #
if [[ "$SKIP_HIDDEN" == "1" ]]; then
    echo "[logiqa] SKIP_HIDDEN=1 — not regenerating."
else
    [[ -f "$LOGIQA_EVAL" && -f "$LOGIQA_DATA" ]] || {
        echo "ERROR: missing LogiQA traces under $LOGIQA_DIR"
        exit 1
    }
    if [[ -f "$LOGIQA_HIDDEN_I" ]]; then
        echo "[logiqa] incorrect hidden.pt exists, skipping."
    else
        echo "[logiqa] Extracting incorrect hidden states (layer $STEER_LAYER, keywords=logic)..."
        python extraction/src/hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$LOGIQA_DATA" \
            --eval_path "$LOGIQA_EVAL" \
            --data_dir "$LOGIQA_DIR" \
            --type incorrect --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords logic
    fi
    if [[ -f "$LOGIQA_HIDDEN_C" ]]; then
        echo "[logiqa] correct hidden.pt exists, skipping."
    else
        echo "[logiqa] Extracting correct hidden states (layer $STEER_LAYER, keywords=logic)..."
        python extraction/src/hidden_analysis.py \
            --model_path "$MODEL" \
            --data_path "$LOGIQA_DATA" \
            --eval_path "$LOGIQA_EVAL" \
            --data_dir "$LOGIQA_DIR" \
            --type correct --start 0 --sample "$VEC_SAMPLES" \
            --keep_layers "$STEER_LAYER" \
            --keywords logic
    fi
fi

for f in \
    "$MATH_HIDDEN_C" "$MATH_HIDDEN_I" \
    "$APPS_HIDDEN_C" "$APPS_HIDDEN_I" \
    "$LOGIQA_HIDDEN_C" "$LOGIQA_HIDDEN_I"
do
    [[ -f "$f" ]] || { echo "ERROR: missing $f — run without SKIP_HIDDEN on a GPU pod."; exit 1; }
done

# ------------------------------------------------------------------ #
# 4) Pool ALL boundary vectors → S_general Phase 1 (or Phase 2 if BALANCE=1)
# ------------------------------------------------------------------ #
mkdir -p "$OUT_DIR"
COMPARE_FLAGS=()
[[ -f "$MATH_VEC" ]] && COMPARE_FLAGS+=(--compare "math=${MATH_VEC}")
[[ -f "$APPS_VEC" ]] && COMPARE_FLAGS+=(--compare "apps=${APPS_VEC}")
[[ -f "$LOGIQA_VEC" ]] && COMPARE_FLAGS+=(--compare "logic=${LOGIQA_VEC}")

BALANCE_FLAGS=()
if [[ "$BALANCE" == "1" ]]; then
    BALANCE_FLAGS=(--balance --seed "$BALANCE_SEED")
    echo "[pool] Building Phase 2 S_general (equal-domain balance, seed=$BALANCE_SEED)..."
else
    echo "[pool] Building Phase 1 S_general (no balancing)..."
fi

python extraction/src/build_general_vector.py \
    --domain "math=${MATH_HIDDEN_C},${MATH_HIDDEN_I}" \
    --domain "apps=${APPS_HIDDEN_C},${APPS_HIDDEN_I}" \
    --domain "logic=${LOGIQA_HIDDEN_C},${LOGIQA_HIDDEN_I}" \
    --layer "$STEER_LAYER" \
    --out "$OUT_VEC" \
    "${BALANCE_FLAGS[@]}" \
    "${COMPARE_FLAGS[@]}"

echo "=================================================================="
echo " DONE."
echo "   vector : $OUT_VEC"
echo "   meta   : ${OUT_VEC%.pt}.meta.json"
echo "   hidden : $MATH_HIDDEN_C"
echo "            $MATH_HIDDEN_I"
echo "            $APPS_HIDDEN_C"
echo "            $APPS_HIDDEN_I"
echo "            $LOGIQA_HIDDEN_C"
echo "            $LOGIQA_HIDDEN_I"
echo "   apply  : coef -1.0 at layer $STEER_LAYER"
echo "   next   : git add $OUT_VEC ${OUT_VEC%.pt}.meta.json  (hidden.pt stays out of git: multi-GB; park it on the HF Hub)"
echo "=================================================================="
