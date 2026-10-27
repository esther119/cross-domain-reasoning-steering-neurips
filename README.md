# Three Domains, One Direction: Reasoning-Calibration Steering Transfers Across Domains

Esther Yang\*, Wenyu Wang\*, Justin Tandon, Akhilesh Bitla, Madhur Panwar, Kevin Zhu, Anusha Mujumdar

\*Equal contribution

Code for extracting reasoning-calibration steering vectors from DeepSeek-R1-Distill-Qwen (1.5B and 7B) on math
(MATH), code (APPS) and logic (LogiQA 2.0), pooling them into a single general direction, and evaluating every
vector on every domain. Built on [SEAL](https://github.com/VITA-Group/SEAL).

## Setup

Python 3.10+ with a CUDA GPU. Install the main dependencies:

```bash
pip install torch transformers vllm datasets evaluate peft numpy sympy regex tqdm timeout-decorator
pip install -r eval/math_eval/pipeline/eval_math_rule/evaluation/requirements.txt
```

## Running the pipeline

Everything goes through two drivers in `scripts/`. `MODEL_SIZE` (`1.5B` or `7B`) picks the model, and the first
argument is the GPU index.

**1. Extract a domain vector.** This generates traces, labels them, collects layer-20 hidden states and writes a
packaged `.pt` vector.

```bash
DOMAIN=math  MODEL_SIZE=1.5B bash scripts/extract_vector.sh 0
DOMAIN=code  MODEL_SIZE=1.5B bash scripts/extract_vector.sh 0
DOMAIN=logic MODEL_SIZE=1.5B bash scripts/extract_vector.sh 0
```

**2. Build the general (pooled) vector** from the three domain vectors:

```bash
bash extraction/general_extraction/general_extraction_1.5B/pipeline/build_general_vector.sh
```

**3. Evaluate.** Run the unsteered baseline once per size, then one call per vector:

```bash
VECTOR=baseline MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0
VECTOR=extraction/general_extraction/general_extraction_1.5B/result/S_general_math_apps_logic_phase1.pt \
  ARM=v_general MODEL_SIZE=1.5B bash scripts/eval_vector.sh 0
```

`DATASETS=math` (or `apps`, `logiqa`) limits the benchmarks, and `GPUS=0,1,2,3` shards a run over several GPUs (resumable).
Results land in `eval/<domain>_eval/<domain>_eval_<size>/<ARM>_on_<BENCH>/`.

[`scripts/README.md`](scripts/README.md) has a sample command for every arm, including the combined and random
control vectors and the 7B runs. `extraction/README.md` and `eval/README.md` describe each stage in detail.

## License

MIT (see [`LICENSE`](LICENSE)). Files derived from LogiQA 2.0 keep that dataset's CC BY-NC-SA 4.0 license.
