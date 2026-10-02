"""
KV Cache 提取和分析工具

提供从模型中提取KV cache的通用功能，用于可视化分析、
相似性计算等下游任务。
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_KVcache(
    model_path: str,
    task_name: str,
    data_path: str = None,
    sample_idx: int = 0,
    max_samples: int = 10,
    device: str = "auto"
) -> Tuple:
    """
    从指定任务加载模型并提取KV cache

    这是一个端到端的便捷函数，用于快速获取KV cache用于分析。

    Args:
        model_path: HuggingFace模型路径
        task_name: 任务名称 (xsum, openbookqa, boolq, gsm8k)
        data_path: 数据集路径（可选，有默认值）
        sample_idx: 使用第几个样本
        max_samples: 加载多少个样本
        device: 设备 ("auto", "cuda", "cpu")

    Returns:
        (model, tokenizer, prompt, kv_data)
        - model: 已加载的模型
        - tokenizer: 对应的tokenizer
        - prompt: 选中样本的prompt文本
        - kv_data: 提取的KV cache字典

    Example:
        >>> from utils.kv_utils import extract_KVcache
        >>> model, tokenizer, prompt, kv_data = extract_KVcache(
        ...     model_path="/path/to/llama-2-13b",
        ...     task_name="xsum",
        ...     sample_idx=0
        ... )
        >>> print(f"KV cache: {len(kv_data['layers'])} layers")
    """
    # 添加项目路径
    project_path = Path(__file__).parent.parent
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))

    from tasks.task_loader import create_task

    # 1. 加载模型
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded")

    # 2. 加载数据集
    print(f"Loading {task_name} task...")
    if data_path is None:
        data_path = str(project_path / f"data/{task_name}/dev.jsonl")

    task = create_task(
        task_name=task_name,
        data_path=data_path,
        max_samples=max_samples
    )
    task.load_data()

    prompt = task.examples[sample_idx].prompt

    print(f"✓ Loaded {len(task.examples)} samples")
    print(f"  Using sample {sample_idx}")
    print(f"  Prompt: {len(prompt)} chars")

    # 3. 提取KV cache
    print("\nExtracting KV cache...")
    kv_data = extract_kv_from_model(model, tokenizer, prompt, device)

    print("✓ KV cache extracted")
    print(f"  Tokens: {kv_data['seq_len']}")
    print(f"  Layers: {len(kv_data['layers'])}")
    print(f"  Heads: {kv_data['layers'][0]['num_heads']}")

    return model, tokenizer, prompt, kv_data


def extract_kv_from_model(
    model,
    tokenizer,
    prompt: str,
    device: str = "auto"
) -> Dict:
    """
    从模型中提取指定prompt的KV cache

    Args:
        model: HuggingFace模型实例
        tokenizer: 对应的tokenizer
        prompt: 输入文本
        device: 设备

    Returns:
        包含所有层所有head的K和V矩阵的字典
        {
            'prompt': str,
            'seq_len': int,
            'layers': [
                {
                    'layer': int,
                    'num_heads': int,
                    'seq_len': int,
                    'head_dim': int,
                    'heads': [
                        {
                            'head': int,
                            'K': ndarray [seq_len, head_dim],
                            'V': ndarray [seq_len, head_dim]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    if device != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True, output_attentions=False)

    past_key_values = outputs.past_key_values

    # 提取所有层的KV
    kv_data = []
    for layer_idx, layer_cache in enumerate(past_key_values):
        key_states = layer_cache[0]  # [batch, num_heads, seq_len, head_dim]
        value_states = layer_cache[1]

        batch_size, num_heads, seq_len, head_dim = key_states.shape

        layer_data = {
            "layer": layer_idx,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "heads": []
        }

        for head_idx in range(num_heads):
            K = key_states[0, head_idx, :, :].cpu().numpy()
            V = value_states[0, head_idx, :, :].cpu().numpy()

            layer_data["heads"].append({
                "head": head_idx,
                "K": K,
                "V": V
            })

        kv_data.append(layer_data)

    return {
        "prompt": prompt,
        "seq_len": seq_len,
        "layers": kv_data
    }
