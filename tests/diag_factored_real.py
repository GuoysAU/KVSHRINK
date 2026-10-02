"""Localise where the factored path diverges from the dense path, on a real model.

Runs one sample end to end and reports, per layer:
  * factor reconstruction error   : || U_r (Sigma_r Psi_r^T) - K_hat_dense ||
  * value reconstruction error    : || U_r C_V + sketch - V_hat_dense ||
  * attention output error        : factored_attention vs dense attention
and finally the option-score difference for the whole model.

Usage (from the repo root, with the model on GPU):

    python tests/diag_factored_real.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --task winogrande --sample 0
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402
from core.config import Config                                  # noqa: E402
from core.compressor import SVDCompressor                       # noqa: E402
from core.factored_attention import factored_attention          # noqa: E402
from tasks.task_loader import create_task_from_config           # noqa: E402
from utils.tau_loader import load_adaptive_tau                  # noqa: E402


def dense_attn(q, k, v, scaling):
    p = torch.softmax(((q @ k.transpose(-1, -2)) * scaling).float(), dim=-1).to(q.dtype)
    return p @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="winogrande")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tau", type=float, default=None,
                    help="uniform tau; default uses the adaptive tau file")
    ap.add_argument("--adaptive-tau", default=None)
    ap.add_argument("--preserve-last", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()

    cfg = Config(task_name=args.task, max_samples=args.sample + 1, model_path=args.model)
    task = create_task_from_config(cfg)
    ex = task[args.sample]
    if task.supports_scoring():
        got = task.get_scoring_inputs(ex)
        context = got[0]
    else:
        context = ex.prompt

    ids = tok.encode(context, add_special_tokens=True)
    dev = next(model.parameters()).device
    ctx = torch.tensor([ids], device=dev)
    print(f"context tokens: {len(ids)}   preserve_last_n: {args.preserve_last}")

    with torch.no_grad():
        out = model(ctx, use_cache=True, return_dict=True)
    cache = out.past_key_values

    tau_map = None
    if args.tau is None and args.adaptive_tau:
        tau_map = load_adaptive_tau(args.adaptive_tau, sample_idx=args.sample)

    n_layers = len(cache)
    print(f"\n{'layer':>5} {'mid':>5} {'rank(min/max)':>14} "
          f"{'K err':>10} {'V err':>10} {'attn err':>10}")
    worst = (0.0, -1)
    for li in range(n_layers):
        layer = cache.layers[li]
        k, v = layer.keys, layer.values
        H = k.shape[1]
        if tau_map is not None:
            taus = [tau_map[(li, h)] for h in range(H)]
        else:
            taus = [args.tau if args.tau is not None else 0.9] * H

        dk, dv, _, _, _ = SVDCompressor.compress_layer_shared_basis_eigh(
            k, v, taus, preserve_last_n=args.preserve_last, preserve_first_n=0,
            value_residual_bits=1, qjl_seed=0, return_factors=False,
            eigh_compute_device="cpu")
        _, _, _, _, f = SVDCompressor.compress_layer_shared_basis_eigh(
            k, v, taus, preserve_last_n=args.preserve_last, preserve_first_n=0,
            value_residual_bits=1, qjl_seed=0, return_factors=True,
            eigh_compute_device="cpu")
        if f is None:
            print(f"{li:5d}  (skipped: context shorter than preserve window)")
            continue

        mid = f["mid_seq"]
        from core.factored_attention import _pad_ragged
        U, SP, CV = _pad_ragged(f, torch.float32, k.device)
        kref = dk[0, :, :mid, :].float()
        vref = dv[0, :, :mid, :].float()
        kerr = float((U @ SP - kref).norm() / kref.norm().clamp_min(1e-9))

        vhat = U @ CV
        sk = f.get("sketch")
        if sk is not None:
            signs, norms = sk
            proj = SVDCompressor._get_qjl_projection(
                dimension=f["head_dim"], device=U.device, seed=f["qjl_seed"]).float()
            D = f["head_dim"]
            s = SVDCompressor.unpack_signs(signs, D, torch.float32).reshape(H, mid, D)
            nm = norms.reshape(H, mid, 1).float()
            vhat = vhat + (math.sqrt(math.pi / 2.0) / D) * (nm * s) @ proj
        verr = float((vhat - vref).norm() / vref.norm().clamp_min(1e-9))

        d_head = f["head_dim"]
        hq = model.config.num_attention_heads
        q = torch.randn(1, hq, 3, d_head, device=U.device, dtype=k.dtype)
        g = hq // H
        scaling = 1.0 / math.sqrt(d_head)
        ref = dense_attn(q, dk.repeat_interleave(g, 1), dv.repeat_interleave(g, 1), scaling)
        got = factored_attention(q, f, scaling=scaling,
                                 tail_k=f["k_post"], tail_v=f["v_post"])
        aerr = float((got - ref).norm() / ref.norm().clamp_min(1e-9))
        if aerr > worst[0]:
            worst = (aerr, li)

        print(f"{li:5d} {mid:5d} {min(f['rank']):6d}/{max(f['rank']):<7d} "
              f"{kerr:10.2e} {verr:10.2e} {aerr:10.2e}")

    print(f"\nworst attention error: {worst[0]:.3e} at layer {worst[1]}")
    print("K/V err ~1e-3 is fp16 noise; anything >1e-2 localises the bug.")


if __name__ == "__main__":
    main()
