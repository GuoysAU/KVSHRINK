"""
生成每个样本的自适应 tau 值
通过捕获模型 attention 的熵值来计算每层每个 head 的 tau 参数。
"""
import torch
import math
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tasks.task_loader import create_task_from_config
from core.config import Config






def entropy_from_probs(attn_probs: torch.Tensor) -> torch.Tensor:
    """
    计算 attention 概率分布的熵（行熵：每个 query 的注意力分散程度）

    Args:
        attn_probs: Attention 权重 [batch, heads, seq, seq]

    Returns:
        熵值 [batch, heads, seq]
    """
    probs = attn_probs.to(dtype=torch.float32).clamp_min(1e-20)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy


def received_attention_entropy(attn_probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    计算 key-importance 分布的熵（received：哪些 key 被反复关注）

    通过除以每个 key 被关注的次数来消除 causal mask 导致的位置偏差。

    Args:
        attn_probs: Attention 权重 [batch, heads, seq, seq]
        eps: 数值稳定性常数

    Returns:
        熵值 [batch, heads]，范围 [0, log(seq_len)]
    """
    seq_len = attn_probs.shape[-1]
    device = attn_probs.device

    # 列求和：每个 key 被所有 query 关注的总和
    # attn_probs[..., i, j] = query i 对 key j 的注意力
    # received[j] = sum_i attn_probs[i, j] = key j 被关注的总量
    received_raw = attn_probs.sum(dim=-2)  # [batch, heads, seq]

    # 除以每个 key 被关注的次数，消除 causal mask 导致的位置偏差
    # key 0 被 N 个 query 看到，key N-1 只被 1 个 query 看到
    # num_queries[j] = N - j，即 [N, N-1, ..., 1]
    num_queries = torch.arange(seq_len, 0, -1, device=device, dtype=torch.float32)
    received = received_raw / num_queries  # 平均每次被关注的强度

    # 归一化为概率分布
    received_prob = received / received.sum(dim=-1, keepdim=True)
    received_prob = received_prob.clamp_min(eps)

    # 计算实际熵
    entropy = -(received_prob * received_prob.log()).sum(dim=-1)  # [batch, heads]

    # 计算 causal mask 下均匀 attention 的基线熵（结构归一化）
    # 均匀 attention 下，received[j] = sum_{i=j}^{n-1} 1/(i+1)
    positions = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32)
    # 利用反向 cumsum 计算：baseline_received[j] = 1/(j+1) + 1/(j+2) + ... + 1/n
    baseline_received = (1.0 / positions).flip(0).cumsum(0).flip(0)
    baseline_prob = baseline_received / baseline_received.sum()
    baseline_prob = baseline_prob.clamp_min(eps)
    baseline_entropy = -(baseline_prob * baseline_prob.log()).sum()

    # 归一化：实际熵 / 基线熵
    # 如果 attention 和均匀一样分散 → 归一化熵 ≈ 1
    # 如果 attention 更集中（如 sink）→ 归一化熵 < 1
    entropy_norm = entropy / baseline_entropy.clamp_min(eps)
    # entropy_norm = entropy 

    return entropy_norm


def taus_from_entropy(
    entropy: torch.Tensor,
    tau_base: float = 1.0,
    # Defaults match the command-line defaults below, which are the settings
    # used for all reported results.
    mode: str = "weighted_mean",
    tau_min: float = 0.0,
    eps: float = 1e-5,
    layer_level: str = None,
    alpha: float = 1 / math.e,
) -> torch.Tensor:
    """
    根据熵值计算 tau 参数

    Args:
        entropy: 熵值
            - mode in ["mean", "weighted_mean", "var"]: [batch, heads, seq]
            - mode == "received": [batch, heads]
        tau_base: tau 基础值
        mode: 统计模式 ("mean", "weighted_mean", "var" 或 "received")
        tau_min: tau 最小值
        eps: 数值稳定性常数
        layer_level: 如果设置，则该layer所有head使用同一tau值
                     "mean": 取所有head的tau均值
                     "median": 取所有head的tau中位数
                     "max": 取所有head的tau最大值（最保守，压缩最少）
                     None: 每个head使用独立的tau（默认）
        alpha: 幂映射指数，tau = metric**alpha（默认 1/e）

    Returns:
        tau 值 [batch, heads]
    """
    if mode == "received":
        # received 模式：entropy 已经归一化到 [0, 1]
        metric = entropy
    else:
        # 行熵模式：entropy 是 [batch, heads, seq]
        # 先对每行归一化：消除 causal mask 导致的熵上界差异
        # 第 q 个 token 只能看 q+1 个 token，最大熵 = log(q+1)
        # positions: [1, 2, ..., seq_len]
        seq_len = entropy.shape[-1]
        positions = torch.arange(1, seq_len + 1, device=entropy.device, dtype=entropy.dtype)
        max_entropy = positions.log().clamp_min(eps)  # [seq], 避免 log(1)=0 导致除零
        entropy_norm = entropy / max_entropy          # 归一化到 [0, 1]        # entropy: [batch, heads, seq], max_entropy: [seq] -> 广播

        if mode == "mean":
            metric = entropy_norm.mean(dim=-1)  # 对序列长度求平均
        elif mode == "weighted_mean":
            # 位置加权平均：后面的 token 权重更大，第一个 token 权重为 0
            # 因为 token 0 只能看到自己，entropy 必为 0，提供不了信息
            # i 从 0 到 n-1，sum(i) = n*(n-1)/2，所以 weight[i] = 2*i / [n*(n-1)]
            positions_0 = torch.arange(seq_len, device=entropy.device, dtype=entropy.dtype)  # [0, 1, ..., n-1]
            weights = 2 * positions_0 / (seq_len * (seq_len - 1))  # [seq]，归一化后和为 1
            metric = (entropy_norm * weights).sum(dim=-1)  # [batch, heads]
        elif mode == "var":
            metric = entropy_norm.var(dim=-1)   # 对序列长度求方差
        else:
            raise ValueError(f"Unsupported tau mode: {mode}")

    tau = torch.pow(metric, alpha) * tau_base

    # 应用最小值约束
    if tau_min > 0:
        tau = tau_min + (1 - tau_min) * tau
    tau = torch.clamp(tau, max=0.95, min=0.1)


    # 如果指定layer_level，则该layer所有head使用同一tau值
    if layer_level is not None:
        if layer_level == "mean":
            layer_tau = tau.mean(dim=-1, keepdim=True)  # [batch, 1]
        elif layer_level == "median":
            layer_tau = tau.median(dim=-1, keepdim=True).values  # [batch, 1]
        elif layer_level == "max":
            layer_tau = tau.max(dim=-1, keepdim=True).values  # [batch, 1]
        else:
            raise ValueError(f"Unsupported layer_level mode: {layer_level}")
        tau = layer_tau.expand_as(tau)        # 广播到所有head

    return tau


def save_tau_map(
    task_name: str,
    tau_records: Dict[int, List],
    output_dir: Path,
    layer_level: bool = False
):
    """
    保存 tau 映射到 JSON 文件

    Args:
        task_name: 任务名称（用于生成文件名）
        tau_records: tau 记录
            - per-head模式: {sample_idx: [[layer0_heads], [layer1_heads], ...]}
            - layer-level模式: {sample_idx: [layer0_tau, layer1_tau, ...]}
        output_dir: 输出目录
        layer_level: 是否为layer-level模式
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_name}_adaptive_tau.json"
    output_data = {
        "format": "layer_level" if layer_level else "per_head",
        "samples": tau_records
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Tau map saved to: {output_path}")
    print(f"  Format: {'layer-level' if layer_level else 'per-head'}")
    print(f"  Total samples: {len(tau_records)}")
    if tau_records:
        first_sample = tau_records[0]
        print(f"  Layers: {len(first_sample)}")
        if not layer_level:
            print(f"  Heads per layer: {len(first_sample[0])}")


def main(
    task_name: str = "xsum",
    samples: int = None,
    model_path: str = None,
    output_dir: str = "data/adaptiveTau",
    mode: str = "weighted_mean",
    tau_base: float = 1.0,
    tau_min: float = 0.0,
    layer_level: str = None,
    alpha: float = 1 / math.e,
):
    """
    主函数：生成自适应 tau 值

    Args:
        task_name: 任务名称
        samples: 处理的样本数量
        model_path: 模型路径（None 则使用默认）
        output_dir: 输出目录
        mode: 统计模式 ("mean", "weighted_mean", "var" 或 "received")
        tau_base: tau 基础值
        tau_min: tau 最小值
        layer_level: layer级别tau聚合方式 ("mean", "median", "max", None)
    """
    print("=" * 60)
    print("Adaptive Tau Generation")
    print("=" * 60)

    if not model_path:
        raise ValueError("model_path is required (pass --model /path/to/model)")
    config = Config(
        task_name=task_name,
        max_samples=samples,
        do_sample=False,
        model_path=model_path,
    )

    print(f"\nConfiguration:")
    print(f"  Task: {task_name}")
    print(f"  Samples: {samples}")
    print(f"  Model: {config.model_path}")
    print(f"  Mode: {mode}")
    print(f"  Tau base: {tau_base}")
    print(f"  Tau min: {tau_min}")
    print(f"  Alpha: {alpha}")
    print(f"  Layer level: {layer_level or 'per-head (disabled)'}")
    print(f"  Output: {output_dir}")

    # 加载 tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（必须使用 eager attention 才能输出 attentions）
    print(f"Loading model with attention output enabled...")
    dtype = torch.float16 if config.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # 必须用 eager 才能输出 attention
        trust_remote_code=True,
    )

    # 加载任务数据
    print(f"Loading task data...")
    task = create_task_from_config(config)
    print(f"  Total examples loaded: {len(task)}")

    # 处理每个样本
    tau_records: Dict[int, List[List[float]]] = {}

    with torch.no_grad():
        limit = len(task) if samples is None else min(samples, len(task))
        for idx in tqdm(range(limit), desc="Generating tau", ncols=80):
            example = task[idx]

            # 对于支持 scoring 的 task，使用 scoring context（与评估时一致）
            if task.supports_scoring():
                context, _, _ = task.get_scoring_inputs(example)
                input_text = context
            else:
                input_text = example.prompt

            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            # Forward pass with attention output
            # Note: we disable KV cache here to reduce memory usage,
            # since tau generation only needs attentions, not past_key_values.
            outputs = model(
                **inputs,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )

            attentions = outputs.attentions
            if attentions is None:
                raise RuntimeError(
                    "Model did not return attentions. "
                    "Ensure attn_implementation='eager' is set."
                )

            # 为每一层计算 tau
            sample_taus = []
            for layer_attn in attentions:
                # layer_attn: [batch, heads, seq, seq]
                if mode == "received":
                    # key-importance 方法：计算 received 分布的熵
                    entropy = received_attention_entropy(layer_attn)  # [batch, heads]
                else:
                    # 行熵方法
                    entropy = entropy_from_probs(layer_attn)  # [batch, heads, seq]

                tau = taus_from_entropy(
                    entropy,
                    tau_base=tau_base,
                    mode=mode,
                    tau_min=tau_min,
                    layer_level=layer_level,
                    alpha=alpha,
                ).mean(dim=0)  # [heads]

                if layer_level:
                    # layer-level模式：所有head相同，只存一个值
                    sample_taus.append(round(tau[0].item(), 4))
                else:
                    # per-head模式：存所有head的值
                    tau_list = [round(val.item(), 4) for val in tau]
                    sample_taus.append(tau_list)

            tau_records[idx] = sample_taus

    save_tau_map(task_name, tau_records, Path(output_dir), layer_level=layer_level is not None)
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-sample adaptive tau values based on attention entropy"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="xsum",
        help="Task name (e.g., xsum, boolq, openbookqa)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/adaptiveTau",
        help="Output directory (file name will be auto-generated)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="weighted_mean",
        choices=["mean", "weighted_mean", "received", "var"],
        help="Statistic used to derive tau from entropy. "
             "'mean': simple average of normalized row entropy. "
             "'weighted_mean': position-weighted average (later tokens have higher weight). "
             "'var': variance of normalized row entropy. "
             "'received': key-importance method (entropy of received attention distribution)."
    )
    parser.add_argument(
        "--tau-base",
        type=float,
        default=1.0,
        help="Scale factor applied after sigmoid when deriving tau (default: 1.0)"
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=0.0,
        help="Lower bound for tau values"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1 / math.e,
        help="Exponent of the power mapping tau = metric**alpha (default: 1/e)"
    )
    parser.add_argument(
        "--layer-level",
        type=str,
        default=None,
        choices=["mean", "median", "max"],
        help="Use layer-level tau (all heads in a layer share same tau). "
             "Options: mean (average), median, max (most conservative). "
             "Default: None (per-head tau)"
    )

    args = parser.parse_args()

    main(
        task_name=args.task,
        samples=args.samples,
        model_path=args.model,
        output_dir=args.output,
        mode=args.mode,
        tau_base=args.tau_base,
        tau_min=args.tau_min,
        layer_level=args.layer_level,
        alpha=args.alpha,
    )
