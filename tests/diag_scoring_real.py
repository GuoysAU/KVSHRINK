"""Compare option scores, factored vs dense, on real samples of a real model.

Mirrors experiment.py's shared_context scoring loop exactly, but scores every
option twice - once against a DynamicCache of the dense compressed tensors and
once against a FactoredCache - and prints both score vectors plus the argmax.
A divergence here, with the layer-wise factor errors already known to be small,
points at the plumbing (mask, cache length, dispatch) rather than the algebra.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402
from core.config import Config                                   # noqa: E402
from core.factored_cache import FactoredCache, register, use_factored_cache  # noqa: E402
from strategies.adaptive import AdaptiveStrategy                 # noqa: E402
from tasks.task_loader import create_task_from_config            # noqa: E402
from utils.tau_loader import load_adaptive_tau                   # noqa: E402


def score_options(model, tok, cache_builder, context_ids, options, device):
    """Sum of option-token log-probs, exactly as the scoring loop does."""
    out = []
    prompt_length = len(context_ids)
    for opt in options:
        opt_ids = tok.encode(opt, add_special_tokens=False)
        cache = cache_builder()
        opt_t = torch.tensor([opt_ids], device=device)
        cache_position = torch.arange(prompt_length, prompt_length + len(opt_ids),
                                      device=device, dtype=torch.long)
        attention_mask = torch.ones((1, prompt_length + len(opt_ids)),
                                    device=device, dtype=torch.long)
        with torch.no_grad(), use_factored_cache(cache):
            o = model(opt_t, past_key_values=cache, use_cache=True,
                      cache_position=cache_position,
                      attention_mask=attention_mask, return_dict=True)
        logp = torch.log_softmax(o.logits.float(), dim=-1)[0, :-1, :]
        tgt = torch.tensor(opt_ids[1:], device=device)
        out.append(float(logp.gather(-1, tgt[:, None]).sum()) if len(opt_ids) > 1 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="hellaswag")
    ap.add_argument("--adaptive-tau", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--preserve-last", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    register(model)
    dev = next(model.parameters()).device

    cfg = Config(task_name=args.task, max_samples=args.samples, model_path=args.model)
    task = create_task_from_config(cfg)

    agree = 0
    for si in range(args.samples):
        ex = task[si]
        context, options, label = task.get_scoring_inputs(ex)
        ids = tok.encode(context, add_special_tokens=True)
        ctx = torch.tensor([ids], device=dev)

        strategy_kwargs = dict(
            preserve_last_n=args.preserve_last,
            preserve_first_n=0,
            compression_method="shared_basis",
            value_residual_bits=1,
            qjl_seed=0,
            decomposition_backend="batched_eigh",
            eigh_compute_device="cpu",
        )
        dense_strategy = AdaptiveStrategy(cache_backend="dense", **strategy_kwargs)
        factored_strategy = AdaptiveStrategy(cache_backend="factored", **strategy_kwargs)
        tau_map = load_adaptive_tau(args.adaptive_tau, sample_idx=si)
        dense_strategy.update_tau(tau_map)
        factored_strategy.update_tau(tau_map)

        with torch.no_grad():
            pre = model(ctx, use_cache=True, return_dict=True)
        unc = pre.past_key_values

        def compress(strategy):
            dense, facs = [], []
            for li in range(len(unc)):
                lay = unc.layers[li]
                result = strategy.compress_layer_kv((lay.keys, lay.values), li)
                dense.append(result.dense_kv)
                facs.append(result.factors)
            return dense, facs

        dense_layers, _ = compress(dense_strategy)
        _, facs = compress(factored_strategy)
        have = all(f is not None for f in facs)

        def dense_builder():
            model.config._attn_implementation = "sdpa"
            c = DynamicCache()
            for i, kv in enumerate(dense_layers):
                c.update(kv[0], kv[1], i)
            return c

        def fac_builder():
            model.config._attn_implementation = "kvshrink_factored"
            c = FactoredCache()
            for i, f in enumerate(facs):
                c.add_layer(i, f)
            return c

        sd = score_options(model, tok, dense_builder, ids, options, dev)
        sf = score_options(model, tok, fac_builder, ids, options, dev) if have else None

        ad = int(max(range(len(sd)), key=lambda i: sd[i]))
        af = int(max(range(len(sf)), key=lambda i: sf[i])) if sf else -1
        agree += int(ad == af)
        print(f"sample {si}  ctx={len(ids):4d}  factors={have}  label={label}")
        print(f"   dense    {[round(x, 3) for x in sd]}  -> {ad}")
        if sf:
            print(f"   factored {[round(x, 3) for x in sf]}  -> {af}"
                  f"   {'OK' if ad == af else '*** DIVERGES ***'}")

    print(f"\nargmax agreement: {agree}/{args.samples}")


if __name__ == "__main__":
    main()
