"""Does the longest XSum article fit on this GPU?

Runs the single longest dev-set prompt through the same path the pipeline uses
(prefill -> compress -> generate), and reports peak GPU memory. The risk is at
prefill, where the full uncompressed KV cache exists before any compression.

    python tests/probe_longest_xsum.py --model meta-llama/Llama-2-13b-hf
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from core.model_wrapper import GenerationRunner                # noqa: E402
from strategies.uniform import UniformStrategy                 # noqa: E402
from tasks.task_loader import create_task                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=1,
                    help="1 = the longest prompt, 2 = second longest, ...")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:            # experiment.py setup() 里也是这么做的
        tok.pad_token = tok.eos_token
    task = create_task("xsum", "data/xsum/dev.jsonl", max_samples=None)
    task.load_data()

    lengths = [(len(tok.encode(e.prompt, add_special_tokens=True)), i)
               for i, e in enumerate(task)]
    lengths.sort(reverse=True)
    n_tok, idx = lengths[args.rank - 1]
    print("第 %d 长的样本: index=%d, %d tokens" % (args.rank, idx, n_tok))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    after_load = torch.cuda.memory_allocated() / 1e9
    print("  模型加载后已分配: %.2f GB" % after_load)

    strategy = UniformStrategy(
        0.9, preserve_last_n=8, compression_method="shared_basis",
        value_residual_bits=1, qjl_seed=0,
        decomposition_backend="batched_eigh", cache_backend="factored",
    )
    runner = GenerationRunner(model, tok, strategy)

    torch.cuda.reset_peak_memory_stats()
    text, stats = runner.generate_compressed(
        task[idx].prompt, max_new_tokens=args.max_new_tokens, do_sample=False
    )
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print()
    print("  跑通了。峰值已分配 %.2f GB / 显卡 %.2f GB (余量 %.2f GB)"
          % (peak, total, total - peak))
    print("  prefill %.2fs, svd %.2fs, decode %.2fs, CR %.2fx"
          % (stats["prefill_time"], stats["svd_time"], stats["decode_time"],
             stats.get("overall_compression_ratio", float("nan"))))
    print("  生成开头: %s" % text[:80].replace("\n", " "))


if __name__ == "__main__":
    main()
