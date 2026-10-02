import json
from typing import Dict, Tuple


# 缓存已解析的 tau 文件，避免每个样本都重新读取/解析整份 JSON。
# 大任务的 tau 文件可达数百 MB（如 hellaswag 261MB），逐样本重解析会让 GPU 空转、
# 单核 CPU 被打满（实测一次运行累计重复读取上百 GB）。整个进程只解析一次。
_TAU_FILE_CACHE: Dict[str, object] = {}


def _load_tau_data(path: str):
    """读取并解析 tau 文件，按路径缓存；同一进程内只解析一次。"""
    data = _TAU_FILE_CACHE.get(path)
    if data is None:
        with open(path, 'r') as f:
            data = json.load(f)
        _TAU_FILE_CACHE[path] = data
    return data


def load_adaptive_tau(path: str, sample_idx: int = 0) -> Dict[Tuple[int, int], float]:
    """
    Load adaptive tau for a specific sample.

    Args:
        path: path to adaptive_tau.json
        sample_idx: which sample (default: 0)

    Returns:
        {(layer_idx, head_idx): tau}
        对于layer-level格式，所有head_idx都映射到同一个tau值
    """
    data = _load_tau_data(path)

    # 检测新格式（带format字段）
    if isinstance(data, dict) and "format" in data:
        fmt = data["format"]
        samples = data["samples"]
        sample_data = samples.get(str(sample_idx), samples.get(sample_idx))
        if sample_data is None:
            raise ValueError(f"Sample {sample_idx} not found in {path}")

        tau_map = {}
        if fmt == "layer_level":
            # layer-level格式: [tau0, tau1, tau2, ...]
            # 需要知道head数量，这里假设40（可以从其他地方获取）
            # 但为了兼容性，我们返回特殊格式让AdaptiveStrategy处理
            for layer_idx, tau in enumerate(sample_data):
                # 用特殊key (layer_idx, -1) 表示layer-level tau
                tau_map[(layer_idx, -1)] = float(tau)
        else:
            # per-head格式: [[head0, head1, ...], ...]
            for layer_idx, layer_heads in enumerate(sample_data):
                for head_idx, tau in enumerate(layer_heads):
                    tau_map[(layer_idx, head_idx)] = float(tau)
        return tau_map

    # 兼容旧格式
    # 1. {"0": [[...], [...]], "1": [[...], [...]]}  -> per-sample dict
    # 2. [[...], [...], ...]  -> single sample list
    if isinstance(data, dict) and str(sample_idx) in data:
        sample_data = data[str(sample_idx)]
    elif isinstance(data, list):
        sample_data = data
    else:
        raise ValueError(f"Unknown format in {path}")

    tau_map = {}
    for layer_idx, layer_heads in enumerate(sample_data):
        for head_idx, tau in enumerate(layer_heads):
            tau_map[(layer_idx, head_idx)] = float(tau)

    return tau_map
