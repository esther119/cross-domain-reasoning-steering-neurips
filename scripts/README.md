# scripts/ — two parameterised drivers, and a sample line for every arm

Everything an arm needs is a parameter of one of two scripts. Run from anywhere (each `cd`s to the repo root);
`MODEL_SIZE` (1.5B | 7B) picks the model and pins the batch sizes; the GPU index is the first positional argument
(or `GPUS=` for several).

| script | what |
|---|---|
| `extract_vector.sh` | build a domain vector: traces → label guard → layer-20 hidden states → vector → packaged `.pt` + `.meta.json`. Placeholders: `DOMAIN=math\|code\|logic`, `MODEL_SIZE`, `MAX_TOKENS` (10000), `ATTEMPT`, `SAMPLE` (500), `KEYWORDS` (= DOMAIN), `TAG`, `SKIP_GEN=1` (rebuild from an existing pool) |
| `eval_vector.sh` | evaluate one vector (`VECTOR=`, `ARM=`) or the baselines (`VECTOR=baseline`) on `DATASETS=math,apps,logiqa`, each through its own harness, into flat folders `eval/<domain>_eval/<domain>_eval_<size>/<ARM>_on_<BENCH>/`. `GPUS=0,1,2,3` shards the run over several GPUs (checkpointed, resumable, finalised with the harness's `--finalize_only`); `RUN_BASELINE=1` adds the baselines |

Builders that are not parameter variants of each other stay with their vector family:
`extraction/general_extraction/general_extraction_1.5B/pipeline/build_general_vector.sh` (S_general pooler),
`extraction/combo_extraction/combo_extraction_1.5B/pipeline/build_combo_vector.py` (S_combo),
`extraction/control_extraction/control_extraction_1.5B/pipeline/{build_random_vectors.sh,verify_random_vectors.py}` (R_iso, `MODEL_SIZE=`),
`extraction/logic_extraction/logic_extraction_1.5B/pipeline/relabel_logiqa_build.py` (regrade a LogiQA2 pool),
`extraction/<d>_extraction/<d>_keyword/discover_*_keywords.sh` (derive a keyword set from traces).

## Samples — extraction (one line per arm; change the placeholders, nothing else)

```bash
# domain vectors, 1.5B (the validated standard: 10k cap, 500 + 500, layer 20)
DOMAIN=math      MODEL_SIZE=1.5B bash scripts/extract_vector.sh 0          # -> extraction/math_extraction/math_extraction_1.5B/result/math_v_math.pt
DOMAIN=code      MODEL_SIZE=1.5B bash scripts/extract_vector.sh 0          # -> .../code_extraction_1.5B/result/apps_v_code.pt
DOMAIN=logic     MODEL_SIZE=1.5B bash scripts/extract_vector.sh 0          # -> .../logic_extraction_1.5B/result/logiqa2_v_logic.pt
# same three at 7B: MODEL_SIZE=7B          (e.g. DOMAIN=logic MODEL_SIZE=7B bash scripts/extract_vector.sh 0)
# rebuild a vector from an already-generated (e.g. regraded) pool without regenerating traces
DOMAIN=logic MODEL_SIZE=7B TAG=baseline_10000_regraded SKIP_GEN=1 bash scripts/extract_vector.sh 0
# regrade a LogiQA2 pool with the canonical grader first (CPU)
# (already applied for 7B, whose pool is baseline_10000_regraded; substitute your own <tag> for a new pool)
python extraction/logic_extraction/logic_extraction_1.5B/pipeline/relabel_logiqa_build.py \
  --build_dir extraction/logic_extraction/logic_extraction_<size>/result/<tag> \
  --out_dir   extraction/logic_extraction/logic_extraction_<size>/result/<tag>_regraded

# pooled / combined / control vectors (math + code + logic)
bash extraction/general_extraction/general_extraction_1.5B/pipeline/build_general_vector.sh            # S_general (GPU: hidden states, then pool)
python extraction/combo_extraction/combo_extraction_1.5B/pipeline/build_combo_vector.py \
  --out extraction/combo_extraction/combo_extraction_1.5B/result/S_combo.pt                            # S_combo (CPU)
MODEL_SIZE=1.5B bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh   # R_iso seeds 1-3 (CPU)
MODEL_SIZE=7B   bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh
```

## Samples — evaluation (one line per arm × size; `DATASETS` defaults to math,apps,logiqa)

```bash
R=extraction   # shorthand below
# baselines (once per size)
VECTOR=baseline MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0
VECTOR=baseline MODEL_SIZE=7B   bash scripts/eval_vector.sh 0
# domain vectors
VECTOR=$R/math_extraction/math_extraction_1.5B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt ARM=v_math MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0
VECTOR=$R/code_extraction/code_extraction_1.5B/result/apps_v_code.pt                    ARM=v_code      MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0
VECTOR=$R/logic_extraction/logic_extraction_1.5B/result/logiqa2_v_logic.pt              ARM=v_logic     MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0
# pooled / combined
VECTOR=$R/general_extraction/general_extraction_1.5B/result/S_general_math_apps_logic_phase1.pt ARM=v_general MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0
VECTOR=$R/combo_extraction/combo_extraction_1.5B/result/S_combo.pt                      ARM=v_combo     MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0
# random controls: verify the draws once (CPU), then one call per seed — "random" is just the VECTOR/ARM pair
python $R/control_extraction/control_extraction_1.5B/pipeline/verify_random_vectors.py --dir $R/control_extraction/control_extraction_1.5B/result --prefix R_iso
for s in 1 2 3; do VECTOR=$R/control_extraction/control_extraction_1.5B/result/R_iso_seed$s.pt ARM=R_iso_seed$s MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0; done
# 7B: same lines with MODEL_SIZE=7B and the 7B vector paths, e.g.
VECTOR=$R/math_extraction/math_extraction_7B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt ARM=v_math MODEL_SIZE=7B bash scripts/eval_vector.sh 0
VECTOR=$R/logic_extraction/logic_extraction_7B/result/logiqa2_v_logic.pt                ARM=v_logic     MODEL_SIZE=7B bash scripts/eval_vector.sh 0
# one benchmark only, no baseline, sharded over four GPUs (checkpointed + resumable; re-run the same line to resume)
VECTOR=$R/logic_extraction/logic_extraction_7B/result/logiqa2_v_logic.pt ARM=v_logic MODEL_SIZE=7B DATASETS=logiqa GPUS=0,1,2,3 bash scripts/eval_vector.sh
```

Exploratory knobs (not the standard): `GEN_MODE=pruned` (drop finished rows between chunks, ~4× faster on
long-tail batches), `MODEL_DTYPE=bf16` (7B on 40 GB cards). Arms meant for the paired tables must keep
`hf` + `fp32` and the pinned batch sizes.
