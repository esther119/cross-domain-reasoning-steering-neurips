#!/usr/bin/env bash
# Phase 1 for v_nonreason: derive the `nonreason` keyword set
# (extraction/nonreason_extraction/nonreason_keyword/keywords.json) from real TriviaQA
# reasoning traces instead of guessing -- the same procedure that produced the `logic` set
# (extraction/logic_extraction/logic_keyword/discover_logic_keywords.sh). Generates a probe
# batch of TriviaQA-train traces, measures how much of it the "math" set leaves untagged,
# and ranks candidate phrases by how many distinct untagged steps each would recover.
#
# Usage:
#   bash extraction/nonreason_extraction/nonreason_keyword/discover_nonreason_keywords.sh [gpu_index]
#   PROBE_EXAMPLES=300 MAX_TOKENS=10000 bash extraction/nonreason_extraction/nonreason_keyword/discover_nonreason_keywords.sh 0
#
# The committed probe (…/nonreason_extraction_1.5B/result/keyword_probe/)
# was generated on the triviaqa-non-reason branch at MAX_TOKENS=4000, EM_MODE=aliases; the
# default here is the universal 10,000-token cap. Generation is skipped while that dir holds a
# math_eval.jsonl -- delete it (or set PROBE_TAG) to re-probe at 10k.
#
# The probe doubles as the BUILD YIELD CALIBRATION: its printed accuracy / answer_rate is what
# sets ATTEMPT for nonreason_extraction_1.5B/pipeline/scripts/extract_vector.sh, since you
# need SAMPLE correct AND SAMPLE incorrect, and on closed-book trivia the correct class is the
# binding one (a 1.5B distill is weak here). The script prints the implied ATTEMPT at the end.
#
# vLLM generation needs a GPU; the coverage/candidate pass is tokenizer-only and can be re-run
# freely while iterating on keywords.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."      # repo root (paths below are root-relative)

gpu=${1:-0}
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${MAX_TOKENS:=10000}"          # universal 10k cap (same as the extraction pools and evals)
: "${PROBE_EXAMPLES:=300}"
: "${MIN_STEP_COUNT:=3}"
: "${TOP_K:=40}"
: "${SEED:=42}"
: "${SAMPLE:=500}"                  # traces per class the real build wants, for the ATTEMPT math
: "${EM_MODE:=contains}"            # label mode the build uses (see scripts/extract_vector.sh)
: "${TP:=1}"
: "${PROBE_TAG:=keyword_probe}"

if [[ -z "${MODEL_SIZE:-}" ]]; then
    case "$MODEL" in
        *1.5B*) MODEL_SIZE=1.5B ;;
        *7B*)   MODEL_SIZE=7B ;;
        *) echo "Cannot infer MODEL_SIZE from MODEL=$MODEL; set MODEL_SIZE=1.5B|7B" >&2; exit 2 ;;
    esac
fi
TAG=$(basename "$MODEL")
POOL=extraction/nonreason_extraction/nonreason_pool/TriviaQA/train.jsonl
HARNESS=eval/nonreason_eval/pipeline/eval_nonreason_vllm.py
COVERAGE=extraction/logic_extraction/logic_keyword/logic_keyword_coverage.py
DIR="extraction/nonreason_extraction/nonreason_extraction_${MODEL_SIZE}/result/${PROBE_TAG}"

if [[ ! -f "$POOL" ]]; then
    echo "$POOL missing; run: python extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py" >&2
    exit 2
fi

if [[ -f "$DIR/math_eval.jsonl" ]]; then
    echo "[1/2] $DIR/math_eval.jsonl exists -- skipping generation (delete to re-probe)"
else
    echo "[1/2] generate ${PROBE_EXAMPLES} TriviaQA train traces (max_tokens=${MAX_TOKENS}) ..."
    # File order, no --random_sample: train.jsonl is already seed-42 shuffled, and the
    # coverage pass zips it positionally against math_eval.jsonl.
    CUDA_VISIBLE_DEVICES=$gpu python -u "$HARNESS" \
        --model_name_or_path "$MODEL" --save_dir "$DIR" \
        --dataset TriviaQA_train --max_examples "$PROBE_EXAMPLES" \
        --max_tokens "$MAX_TOKENS" --em_mode "$EM_MODE" \
        --tensor_parallel_size "$TP" --use_chat_format --remove_bos
fi

echo "[2/2] 'math' baseline coverage + ranked candidate phrases ..."
python -u "$COVERAGE" \
    --data_dir "$DIR" --data_path "$POOL" \
    --model_path "$MODEL" --keywords math --noise_set nonreason \
    --min_step_count "$MIN_STEP_COUNT" --top_k "$TOP_K" --seed "$SEED"

python - "$DIR/math_eval.jsonl" "$SAMPLE" "$POOL" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
sample = int(sys.argv[2])
pool = sys.argv[3]
n = len(rows)
correct = sum(bool(r["all_eval"][0]) for r in rows)
unfinished = sum("</think>" not in r["model_generation"][0] for r in rows)
answered = sum(1 for r in rows if r.get("all_pred", [None])[0] is not None)
acc = correct / n if n else 0.0
print(f"\n[yield] probe {n} rows | correct {correct} ({acc*100:.1f}%) | incorrect {n-correct} "
      f"| unfinished {unfinished} ({unfinished/n*100:.1f}%) | answered {answered}")
if correct == 0:
    print(f"[yield] NO correct traces at this max_tokens. The build needs {sample} of them, so it")
    print( "[yield] CANNOT proceed on this setting. Options, in order of preference:")
    print( "[yield]   * raise MAX_TOKENS (a truncated trace never states an answer -> never correct)")
    print( "[yield]   * grade with EM_MODE=contains (the model answers in a full sentence, which")
    print( "[yield]     strict EM rejects -- check acc_contains in the printout above)")
    print( "[yield]   * lower SAMPLE for the build (a 250+250 vector is still a vector)")
else:
    need = int(sample / acc * 1.15)   # 15% headroom
    print(f"[yield] implied ATTEMPT for {sample}+{sample}: ~{need} rows "
          f"({sample}/{acc:.3f} + 15% headroom)")
    if need > 10000:
        print(f"[yield] NOTE: > the 10,000-row committed prefix in {pool}. Re-prep with")
        print(f"[yield]       python extraction/nonreason_extraction/nonreason_pool/prepare_triviaqa.py --full   (same order, superset)")
PY

cat <<EOT

Probe complete. Review, in order:
  $DIR/candidate_phrases_math.json   -- phrases ranked by how many 'other' steps they'd recover
  $DIR/review_samples_math.jsonl     -- real step text (other / check / switch), to confirm a
                                        candidate is genuine reflection/transition language

THE JUDGMENT CALL FOR THIS DOMAIN: on closed-book recall, "I remember that X" /
"I recall that X" rank high by frequency but are the EXECUTION step -- the model doing
the task. Tagging them as check would put execution behaviour on both sides of the
difference-in-means and gut the vector. What discriminates here is EPISTEMIC
UNCERTAINTY about a retrieved fact ("not sure", "I think it might be", "double-check").
Judge every candidate against review_samples_math.jsonl on that basis.

Then edit extraction/nonreason_extraction/nonreason_keyword/keywords.json and re-measure:

  python $COVERAGE \\
    --data_dir "$DIR" --data_path "$POOL" \\
    --model_path "$MODEL" --keywords nonreason --noise_set nonreason --seed "$SEED"

Iterate until new candidates only recover a small marginal fraction of the remaining
'other' steps. Only then build with KEYWORDS=nonreason
(bash scripts/extract_vector.sh (DOMAIN=nonreason)).
EOT
