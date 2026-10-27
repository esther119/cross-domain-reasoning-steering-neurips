# eval/src — shared eval code

`modeling_utils/modeling_qwen2.py` — the Qwen2 causal LM with the SEAL steering hook (`set_steering_flag(...)`, `start_new_round()`):
adds `coef * S` to the residual stream at the chosen layer at every `\n\n` step boundary inside `<think>`. Every
`eval/<domain>_eval/pipeline/eval_<domain>_steering.py` puts this directory on `sys.path` and imports it; it is the one piece of the eval
harness that is genuinely shared. Requires `transformers >= 4.46` internals (pinned in `requirements.txt`).
