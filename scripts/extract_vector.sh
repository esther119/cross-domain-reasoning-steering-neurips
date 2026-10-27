#!/usr/bin/env bash
# Build one domain steering vector with the SEAL pipeline — the same four steps for every domain and
# model size; only the parameters change.  Run from anywhere (cd's to the repo root).
#
#   DOMAIN=math      MODEL_SIZE=1.5B bash scripts/extract_vector.sh 0      # S_math  1.5B
#   DOMAIN=code      MODEL_SIZE=7B   bash scripts/extract_vector.sh 0      # S_code  7B
#   DOMAIN=logic     MODEL_SIZE=1.5B bash scripts/extract_vector.sh 0      # S_logic 1.5B
#   DOMAIN=logic MODEL_SIZE=7B TAG=baseline_10000_regraded SKIP_GEN=1 bash scripts/extract_vector.sh 0
#                                                                           # rebuild from an existing (regraded) pool
#
# Steps  1) generate + grade traces on the domain's extraction pool (vLLM, greedy, chat template, BOS
#           stripped, MAX_TOKENS cap)            -> RESULT/<TAG>/{data.jsonl, math_eval.jsonl, ...}
#        2) label guard: both classes hold >= SAMPLE traces; no trace graded correct without a closing
#           </think> (fails for logic where the label can only come after </think>; warns for
#           math/code whose graders can legitimately find an answer inside the trace)
#        3) layer-20 boundary activations for the first SAMPLE correct and first SAMPLE incorrect traces
#           (file order, nothing excluded; unfinished traces are incorrect and kept)
#                                                 -> RESULT/<TAG>/hidden_{correct,incorrect}_0_SAMPLE/
#        4) S = mean(check u switch) - mean(other) -> RESULT/<TAG>/vector_<KEYWORDS>_<SAMPLE>_<SAMPLE>/
#           layer_20_transition_reflection_steervec.pt, packaged with a .meta.json as RESULT/<PKG>.pt
#
# Placeholders (env):
#   DOMAIN      math | code | logic                                   (required)
#   MODEL_SIZE  1.5B | 7B  -> MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-<size>   (or set MODEL directly)
#   MAX_TOKENS  generation cap, default 10000 (the universal cap; do not lower it for extraction)
#   ATTEMPT     rows of the pool to generate on: math 3000, code (stop at SAMPLE+SAMPLE), logic 2000
#   SAMPLE      traces per class, default 500
#   KEYWORDS    check/switch set, default = DOMAIN (extraction/<DOMAIN>_extraction/<DOMAIN>_keyword/keywords.json)
#   TAG         result sub-folder, default baseline_<MAX_TOKENS>;  SKIP_GEN=1 reuses RESULT/<TAG> traces
#   PKG         packaged vector name, default math_v_math | apps_v_code | logiqa2_v_logic
#   LAYER       steering layer, default 20;  GPU index is the first positional arg (default 0)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }

gpu=${1:-0}
: "${DOMAIN:?set DOMAIN=math|code|logic}"
: "${MODEL_SIZE:=1.5B}"
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-${MODEL_SIZE}}"
: "${MAX_TOKENS:=10000}"
: "${SAMPLE:=500}"
: "${KEYWORDS:=$DOMAIN}"
: "${LAYER:=20}"
: "${TAG:=baseline_${MAX_TOKENS}}"
: "${SKIP_GEN:=0}"
: "${TP:=1}"                         # vLLM tensor parallelism for trace generation (7B has 4 KV heads: keep 1)
SRC=extraction/src
RESULT="extraction/${DOMAIN}_extraction/${DOMAIN}_extraction_${MODEL_SIZE}/result"
DIR="$RESULT/$TAG"
export CUDA_VISIBLE_DEVICES="$gpu"

case "$DOMAIN" in
  math)
    : "${ATTEMPT:=3000}"; : "${PKG:=math_v_math}"
    DATA_PATH=extraction/math_extraction/math_pool/MATH/train.jsonl
    POOL_DESC="MATH train (extraction/math_extraction/math_pool/MATH/train.jsonl), first ${ATTEMPT} rows"
    GEN=(python -u eval/math_eval/pipeline/eval_math_vllm.py --model_name_or_path "$MODEL" --save_dir "$DIR"
         --dataset MATH_train --max_examples "$ATTEMPT" --max_tokens "$MAX_TOKENS" --use_chat_format --remove_bos)
    STRICT=1 ;;
  code)
    : "${PKG:=apps_v_code}"
    DATA_PATH="$DIR/data.jsonl"          # gen_apps_vllm writes the attempted rows here
    POOL_DESC="APPS train in file order (codeparrot/apps parquet), generated until both pools hold ${SAMPLE}"
    GEN=(python -u $SRC/apps/gen_apps_vllm.py --model_name_or_path "$MODEL" --save_dir "$DIR" --split train
         --max_tokens "$MAX_TOKENS" --target "$SAMPLE" --resume)
    STRICT=1 ;;
  logic)
    : "${ATTEMPT:=2000}"; : "${PKG:=logiqa2_v_logic}"
    DATA_PATH="$DIR/data.jsonl"
    POOL_DESC="LogiQA 2.0 train, English rows (extraction/logic_extraction/logic_pool/LogiQA2/train.jsonl), first ${ATTEMPT}"
    GEN=(python -u $SRC/gen_logiqa_vllm.py --model_name_or_path "$MODEL" --save_dir "$DIR" --split train --filter_cjk
         --max_examples "$ATTEMPT" --max_tokens "$MAX_TOKENS" --use_chat_format --remove_bos)
    STRICT=1 ;;
  *) echo "DOMAIN must be math|code|logic (got '$DOMAIN')" >&2; exit 2 ;;
esac

echo "=================================================================="
echo " extract_vector: DOMAIN=$DOMAIN MODEL=$MODEL MAX_TOKENS=$MAX_TOKENS SAMPLE=$SAMPLE KEYWORDS=$KEYWORDS LAYER=$LAYER"
echo " traces -> $DIR     packaged -> $RESULT/$PKG.pt"
echo "=================================================================="
PYTHONPATH=$SRC python -c "from keyword_sets import KEYWORD_SETS; assert '$KEYWORDS' in KEYWORD_SETS, 'no keyword set $KEYWORDS under extraction/*_extraction/*_keyword/keywords.json'"
mkdir -p "$DIR"

# ---- 1) traces ---------------------------------------------------------------
if [[ "$SKIP_GEN" == "1" || -f "$DIR/math_eval.jsonl" ]]; then
  echo "[1/4] traces present in $DIR (SKIP_GEN=$SKIP_GEN) -- not regenerating"
  [[ -f "$DIR/math_eval.jsonl" ]] || { echo "missing $DIR/math_eval.jsonl" >&2; exit 2; }
else
  echo "[1/4] generating + grading traces ..."
  "${GEN[@]}"
fi
[[ -f "$DATA_PATH" ]] || { echo "missing pool/data file $DATA_PATH" >&2; exit 2; }

# ---- 2) label guard ----------------------------------------------------------
echo "[2/4] label guard ..."
python - "$DIR/math_eval.jsonl" "$SAMPLE" "$STRICT" <<'PY'
import json, sys
path, need, strict = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
rows = [json.loads(l) for l in open(path) if l.strip()]
ok = [bool(r["all_eval"][0]) for r in rows]
unf = ["</think>" not in r["model_generation"][0] for r in rows]
bad = sum(u and o for u, o in zip(unf, ok))
c, i = sum(ok), len(ok) - sum(ok)
pool_i = [k for k, o in enumerate(ok) if not o][:need]
print(f"[guard] traces {len(rows)} | correct {c} | incorrect {i} | unfinished {sum(unf)} | correct-without-</think> {bad}")
print(f"[guard] first-{need} incorrect pool: {sum(unf[k] for k in pool_i)} unfinished, {sum(not unf[k] for k in pool_i)} answered-wrong")
if bad and strict:
    raise SystemExit(f"[guard] FAIL: {bad} traces are labelled correct without closing </think> -- regrade first "
                     "(logic: extraction/logic_extraction/logic_extraction_1.5B/pipeline/relabel_logiqa_build.py)")
if bad: print(f"[guard] WARNING: {bad} correct-without-</think> traces (grader found an answer inside the trace); kept.")
if c < need or i < need:
    raise SystemExit(f"[guard] FAIL: cannot fill {need}+{need}: correct {c}, incorrect {i}. Raise ATTEMPT and regenerate.")
print(f"[guard] OK to build {need}+{need}")
PY

# ---- 3) hidden states --------------------------------------------------------
for kind in incorrect correct; do
  if [[ -f "$DIR/hidden_${kind}_0_${SAMPLE}/hidden.pt" ]]; then
    echo "[3/4] hidden_${kind}_0_${SAMPLE}/hidden.pt exists -- skipping"
  else
    echo "[3/4] hidden states -- $kind (layer $LAYER, keywords=$KEYWORDS) ..."
    python -u $SRC/hidden_analysis.py --model_path "$MODEL" --data_path "$DATA_PATH" --data_dir "$DIR" \
      --type "$kind" --start 0 --sample "$SAMPLE" --keep_layers "$LAYER" --keywords "$KEYWORDS"
  fi
done

# ---- 4) vector + package -----------------------------------------------------
echo "[4/4] vector (layer $LAYER) + package ..."
python -u $SRC/vector_generation.py --data_dir "$DIR" --prefixs "correct_0_${SAMPLE}" "incorrect_0_${SAMPLE}" \
  --layers "$LAYER" --save_prefix "${KEYWORDS}_${SAMPLE}_${SAMPLE}" --overwrite
VEC="$DIR/vector_${KEYWORDS}_${SAMPLE}_${SAMPLE}/layer_${LAYER}_transition_reflection_steervec.pt"
python -u $SRC/package_vector.py --source "$VEC" --build_dir "$DIR" --name "$PKG" --model "$MODEL" \
  --dataset "$POOL_DESC" --split train --keywords "$KEYWORDS" --layer "$LAYER" --sample "$SAMPLE" --max_tokens "$MAX_TOKENS" \
  --out_dir "$RESULT" --note "Built by scripts/extract_vector.sh (DOMAIN=$DOMAIN MODEL_SIZE=$MODEL_SIZE TAG=$TAG)."
echo "== done: $RESULT/$PKG.pt (+ .meta.json); raw vector $VEC"
echo "   apply with coef -1.0 at layer $LAYER:  VECTOR=$RESULT/$PKG.pt ARM=v_$DOMAIN MODEL_SIZE=$MODEL_SIZE bash scripts/eval_vector.sh"
