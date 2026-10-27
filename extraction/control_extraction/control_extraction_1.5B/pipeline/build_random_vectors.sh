#!/usr/bin/env bash
# Build the R_iso control draws (Control A). CPU only, a few seconds.
#
# One script for both model sizes; the protocol is the one validated at 1.5B:
#   * target norm  = ‖S_combo_<size>‖ (passed as --like), i.e. the mean of the three
#                    domain-vector norms — 57.06 at 1.5B, 85.04 at 7B. Never a typed number.
#   * screening    = |cos| <= 0.08 against EVERY packaged vector of that size
#                    (S_math, S_code, S_logic, S_general when it exists, S_combo)
#                    AND against every earlier seed, so the draws are verified mutually
#                    near-orthogonal as they are built.
#   * refuses to run if S_math, S_code, S_logic or S_combo of that size is missing
#     (a control screened against fewer vectors is exactly the defect this fixes);
#     a missing S_general is skipped with a warning (not built at 7B).
#
#   bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh               # 1.5B: R_iso_seed{1,2,3}
#   MODEL_SIZE=7B bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh # 7B:   R_iso_seed{1,2,3}
#
# The 7B build must run AFTER the 7B S_logic rebuild and the S_combo_7b rebuild that
# follows it (S_combo_7b sets the norm and is one of the screens); every R_iso eval
# arm is then re-run. Rationale: docs/random_vector_build_plan.html
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."   # repo root

MODEL_SIZE=${MODEL_SIZE:-1.5B}
SEEDS=${SEEDS:-"1 2 3"}

case "$MODEL_SIZE" in
  1.5B)
    default_prefix=R_iso
    S_MATH=${S_MATH:-extraction/math_extraction/math_extraction_1.5B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt}
    S_CODE=${S_CODE:-extraction/code_extraction/code_extraction_1.5B/result/apps_v_code.pt}
    S_LOGIC=${S_LOGIC:-extraction/logic_extraction/logic_extraction_1.5B/result/logiqa2_v_logic.pt}
    S_GENERAL=${S_GENERAL:-extraction/general_extraction/general_extraction_1.5B/result/S_general_math_apps_logic_phase1.pt}
    S_COMBO=${S_COMBO:-extraction/combo_extraction/combo_extraction_1.5B/result/S_combo.pt}
    ;;
  7B)
    default_prefix=R_iso
    S_MATH=${S_MATH:-extraction/math_extraction/math_extraction_7B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt}
    S_CODE=${S_CODE:-extraction/code_extraction/code_extraction_7B/result/apps_v_code.pt}
    S_LOGIC=${S_LOGIC:-extraction/logic_extraction/logic_extraction_7B/result/logiqa2_v_logic.pt}
    S_GENERAL=${S_GENERAL:-extraction/general_extraction/general_extraction_7B/result/S_general_7b.pt}   # not built yet -> skipped with a warning
    S_COMBO=${S_COMBO:-extraction/combo_extraction/combo_extraction_7B/result/S_combo.pt}
    ;;
  *)
    echo "ERROR: MODEL_SIZE must be 1.5B or 7B (got '$MODEL_SIZE')" >&2
    exit 2
    ;;
esac

# Transitional: the packaged 1.5B logic vector is being renamed logiqa_v_logic.pt -> logiqa2_v_logic.pt
# (same bytes, LogiQA 2.0 identity). Fall back to the old name until the rename lands.
if [[ ! -f "$S_LOGIC" && -f "${S_LOGIC/logiqa2_v_logic/logiqa_v_logic}" ]]; then
  S_LOGIC="${S_LOGIC/logiqa2_v_logic/logiqa_v_logic}"
  echo "note: logiqa2_v_logic.pt not found, using the legacy name $S_LOGIC" >&2
fi

PREFIX=${PREFIX:-$default_prefix}
OUT_DIR=${OUT_DIR:-extraction/control_extraction/control_extraction_${MODEL_SIZE}/result}
LIKE=${LIKE:-$S_COMBO}    # norm AND dim come from S_combo_<size> = mean of the three domain norms

# Hard requirement: every packaged domain vector of this size plus S_combo must exist.
missing=0
for spec in "S_math=$S_MATH" "S_code=$S_CODE" "S_logic=$S_LOGIC" "S_combo=$S_COMBO"; do
  if [[ ! -f "${spec#*=}" ]]; then
    echo "ERROR: required ${spec%%=*} (${MODEL_SIZE}) is missing: ${spec#*=}" >&2
    missing=1
  fi
done
if [[ "$missing" == 1 ]]; then
  echo "ERROR: refusing to build ${PREFIX} — R_iso must be norm-matched to S_combo_${MODEL_SIZE} and screened" >&2
  echo "       against every packaged ${MODEL_SIZE} vector. Build the missing vector(s) first." >&2
  exit 1
fi
if [[ ! -f "$LIKE" ]]; then
  echo "ERROR: --like vector is missing: $LIKE" >&2
  exit 1
fi

# Same order as the committed 1.5B builds: S_math, S_code, S_logic, S_general, S_combo.
REFS=()
add_ref() {  # add_ref NAME PATH — skip (with a warning) only the optional S_general
  if [[ -f "$2" ]]; then
    REFS+=(--check-against "$1=$2")
  else
    echo "warning: $1 not found at $2 — not screening against it" >&2
  fi
}
add_ref S_math    "$S_MATH"
add_ref S_code    "$S_CODE"
add_ref S_logic   "$S_LOGIC"
add_ref S_general "$S_GENERAL"
add_ref S_combo   "$S_COMBO"

echo "model size : $MODEL_SIZE"
echo "prefix     : $PREFIX  ->  $OUT_DIR/${PREFIX}_seedN.pt"
echo "--like     : $LIKE   (target norm = ‖S_combo_${MODEL_SIZE}‖ = mean of the three domain-vector norms)"
echo "screens    : $(( ${#REFS[@]} / 2 )) packaged ${MODEL_SIZE} vector(s) + earlier seeds, |cos| <= 0.08"
echo

for seed in $SEEDS; do
  out="$OUT_DIR/${PREFIX}_seed${seed}.pt"
  prior=()
  for earlier in $SEEDS; do
    [[ "$earlier" == "$seed" ]] && break
    prior+=(--check-against "${PREFIX}_seed${earlier}=$OUT_DIR/${PREFIX}_seed${earlier}.pt")
  done

  python extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vector.py \
    --seed "$seed" \
    --like "$LIKE" \
    "${REFS[@]}" \
    ${prior[@]+"${prior[@]}"} \
    --out "$out"
  echo
done

echo "Built: $OUT_DIR/${PREFIX}_seed{$(echo $SEEDS | tr ' ' ',')}.pt"
echo "Next : python extraction/control_extraction/control_extraction_1.5B/pipeline/verify_random_vectors.py --dir $OUT_DIR --prefix $PREFIX"
