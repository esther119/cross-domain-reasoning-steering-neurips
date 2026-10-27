"""Greedy generation for the HF steering harness that drops finished rows.

`model.generate()` runs a batch until its LAST member stops, feeding pad tokens to
the rows that already emitted EOS. With a 10k-token cap and reasoning models that
loop on ~20% of prompts, that means every batch pays batch x 10k steps -- and
because transformers' DynamicCache torch.cat()s the whole cache every step, the
per-step cost itself grows with (rows x length). Measured on an A100 for the 1.5B
distill: 27 ms/step at short lengths -> ~250 ms/step averaged over a 10k run, i.e.
~7 h per 500-example arm no matter the batch size.

This function generates in chunks of `chunk` tokens and, between chunks, removes
the rows that finished from the batch: their tokens are stored, the KV cache and
the model's per-row steering state are sliced down to the survivors, and
generation continues for those alone. Cost then tracks the number of rows still
running rather than the batch size. On a short-answer probe (79% of traces finish
by 1k tokens, 20% run to the cap) the cost model says ~4x per batch.

Semantics: each row sees exactly the computation model.generate() would give it
-- same prompt, same left padding, same positions (HF derives position_ids from
attention_mask, so a shrinking batch changes nothing per row), same greedy
argmax, same EOS stop, same cap. What changes is only which other rows share the
batch, i.e. the same fp32 batch-composition noise that any batch-size choice
already carries. The steering hooks live inside the model forward and are keyed on
each row's own tokens, so they are unaffected; the model's per-row
`steering_think_flag` is sliced alongside the cache.

Continuation between chunks relies on documented generate() behaviour: pass the
full sequences so far plus the returned DynamicCache, and generate() feeds only
the uncached (last) token -- exactly one per row, which is what the steering
forward asserts when `new_round` is False.
"""
import torch
from transformers.cache_utils import DynamicCache


def _eos_ids(model, tokenizer):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    return set(int(e) for e in eos)


@torch.no_grad()
def generate_pruned(model, tokenizer, prompts, max_new_tokens, chunk=256, steering=False):
    """Return one generated-token-id list per prompt (prompt tokens excluded).

    Equivalent to `model.generate(**tok(prompts), do_sample=False,
    max_new_tokens=max_new_tokens, use_cache=True)[:, prompt_len:]` row by row,
    minus the trailing pads generate() would emit after EOS.
    """
    device = model.device
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]
    B = input_ids.shape[0]
    eos_ids = _eos_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id

    if steering:
        model.start_new_round()

    active = list(range(B))            # original row index of each row still in the batch
    done = [None] * B                  # generated ids per original row
    cur_ids, cur_attn, past = input_ids, attn, None
    produced = 0                       # tokens generated so far for every active row

    while active and produced < max_new_tokens:
        n = min(chunk, max_new_tokens - produced)
        out = model.generate(
            input_ids=cur_ids, attention_mask=cur_attn, past_key_values=past,
            do_sample=False, max_new_tokens=n, use_cache=True,
            return_dict_in_generate=True, output_scores=False,
            pad_token_id=pad_id,
            # generate() hands back a legacy tuple-of-tuples when the caller passed no
            # cache in (i.e. on the FIRST chunk only); ask for the Cache object, and
            # convert defensively -- a tuple has no batch_select_indices.
            return_legacy_cache=False,
        )
        seqs = out.sequences                       # [b, cur_len + <=n]
        past = out.past_key_values
        if isinstance(past, tuple):
            past = DynamicCache.from_legacy_cache(past)
        new = seqs[:, cur_ids.shape[1]:]           # this chunk's tokens
        produced += new.shape[1]

        keep = []
        for r in range(seqs.shape[0]):
            gen_all = seqs[r, prompt_len:].tolist()
            hit = [i for i, t in enumerate(gen_all) if t in eos_ids]
            finished = bool(hit) or produced >= max_new_tokens
            if finished:
                # keep through the first EOS (generate() itself stops the row there
                # and pads afterwards; decode(skip_special_tokens) drops the EOS)
                done[active[r]] = gen_all[:hit[0] + 1] if hit else gen_all
            else:
                keep.append(r)

        if not keep:
            break
        if len(keep) < seqs.shape[0]:
            idx = torch.tensor(keep, device=device)
            seqs = seqs[idx]
            cur_attn = cur_attn[idx]
            past.batch_select_indices(idx)
            if getattr(model, "steering_think_flag", None) is not None:
                model.steering_think_flag = model.steering_think_flag[idx]
            active = [active[r] for r in keep]
        # Every surviving row is unfinished, so all of its new positions are real
        # tokens: extend its mask with ones. (generate() only pads AFTER an EOS, and
        # rows with an EOS were just dropped.)
        cur_attn = torch.cat(
            [cur_attn, torch.ones((seqs.shape[0], seqs.shape[1] - cur_attn.shape[1]),
                                  dtype=cur_attn.dtype, device=device)], dim=1)
        cur_ids = seqs

    # rows that never finished within max_new_tokens (loop above exits with them active)
    for r, orig in enumerate(active):
        if done[orig] is None:
            done[orig] = cur_ids[r, prompt_len:].tolist()
    return done


def decode_generated(tokenizer, gen_ids):
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen_ids]
