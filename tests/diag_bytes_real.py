"""Accounted bytes (Eq. 60) vs bytes the factored cache actually holds.

Eq. 60 charges one bit per residual sign and the per-head rank r_h. This script
measures, on a real sample, what the factored cache actually holds and compares
it with that accounting. The uncompressed window is reported separately: Eq. 60
leaves it out of both the numerator and the denominator.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
from core.config import Config                                  # noqa: E402
from strategies.adaptive import AdaptiveStrategy                # noqa: E402
from tasks.task_loader import create_task_from_config           # noqa: E402
from utils.tau_loader import load_adaptive_tau                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="boolq")
    ap.add_argument("--adaptive-tau", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--preserve-last", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    cfg = Config(task_name=args.task, max_samples=args.sample + 1, model_path=args.model)
    task = create_task_from_config(cfg)
    ctx = task.get_scoring_inputs(task[args.sample])[0]
    ids = tok.encode(ctx, add_special_tokens=True)
    dev = next(model.parameters()).device
    with torch.no_grad():
        pre = model(torch.tensor([ids], device=dev), use_cache=True, return_dict=True)
    unc = pre.past_key_values

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
    tau_map = load_adaptive_tau(args.adaptive_tau, sample_idx=args.sample)
    dense_strategy.update_tau(tau_map)
    factored_strategy.update_tau(tau_map)

    acct = orig = factors_b = signs8 = signs1 = tail_b = ragged_b = 0
    for li in range(len(unc)):
        lay = unc.layers[li]
        dense_result = dense_strategy.compress_layer_kv((lay.keys, lay.values), li)
        hs = dense_result.stats["heads"]
        acct += sum(h["K"]["comp_bytes"] + h["V"]["comp_bytes"] for h in hs)
        orig += sum(h["K"]["orig_bytes"] * 2 for h in hs)

        factored_result = factored_strategy.compress_layer_kv((lay.keys, lay.values), li)
        f = factored_result.factors
        for n in ("U_r", "sigma_psi_t", "C_V"):
            for t in f[n]:
                factors_b += t.numel() * t.element_size()
        # what per-head ragged storage would cost instead of padding to r_max
        M, D, es = f["mid_seq"], f["head_dim"], f["U_r"][0].element_size()
        ragged_b += sum((M * r + r * D + r * D) * es for r in f["rank"])
        s, nm = f["sketch"]
        signs8 += s.numel() * s.element_size()      # packed code, as stored
        signs1 += s.numel() * s.element_size()
        factors_b += nm.numel() * nm.element_size()
        tail_b += sum(f[k].numel() * f[k].element_size() for k in ("k_post", "v_post"))

    kb = 1024.0
    actual = factors_b + signs8 + tail_b
    fix2 = factors_b + signs1 + tail_b
    fix23 = ragged_b + signs1 + tail_b + (factors_b - sum([0]))  # scales already in factors_b
    fix23 = ragged_b + signs1 + tail_b
    print(f"\ncontext tokens             : {len(ids)}")
    print(f"uncompressed KV (mid only) : {orig/kb:9.1f} KB")
    print(f"Eq.(60) accounted          : {acct/kb:9.1f} KB   CR {orig/acct:6.2f}x  <- reported")
    print(f"actually resident          : {actual/kb:9.1f} KB   CR {orig/actual:6.2f}x")
    print(f"   factors + scales        : {factors_b/kb:9.1f} KB  (per-head rank)          ")
    print(f"   packed 1-bit signs      : {signs8/kb:9.1f} KB")
    print(f"   uncompressed window     : {tail_b/kb:9.1f} KB  (not charged by Eq. 60)")
    print(f"resident minus window      : {(actual-tail_b)/kb:9.1f} KB   "
          f"CR {orig/(actual-tail_b):6.2f}x  <- compare with accounted")



if __name__ == "__main__":
    main()
