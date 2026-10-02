"""
实验结果保存和打印工具

提供保存实验结果到 JSON 文件和打印指标的功能。
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def print_metrics(title: str, metrics: Dict[str, Any]):
    """打印评估指标"""
    print(f"\n{title}:")
    if "accuracy" in metrics:
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
    if "rougeL" in metrics:
        print(f"  ROUGE-L: {metrics['rougeL']['f1']:.4f}")


def print_comparison(baseline_metrics: Dict[str, Any], compressed_metrics: Dict[str, Any]):
    """打印 baseline 和 compressed 的性能对比"""
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)

    if "accuracy" in baseline_metrics and "accuracy" in compressed_metrics:
        acc_drop = baseline_metrics["accuracy"] - compressed_metrics["accuracy"]
        print(f"  Accuracy Drop: {acc_drop:.4f}")

    if "rougeL" in baseline_metrics and "rougeL" in compressed_metrics:
        rouge_drop = baseline_metrics["rougeL"]["f1"] - compressed_metrics["rougeL"]["f1"]
        print(f"  ROUGE-L Drop: {rouge_drop:.4f}")


def save_experiment_results(
    results: Dict[str, Any],
    config,
    output_dir: str,
    experiment_name: str,
    save_details: bool = True,
    quiet: bool = False
):
    """
    保存实验结果到两个分离的 JSON 文件:
    1. 主文件: 包含 metadata, config, overall_results
    2. 详细文件: 包含样本级别的详细数据（可选）

    Args:
        results: 实验结果字典，包含 "baseline" 和/或 "compressed" 的结果
        config: Config 对象
        output_dir: 输出目录
        experiment_name: 实验名称
        save_details: 是否保存详细的 layer-level 数据
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_path = output_dir / f"{experiment_name}.json"
    details_path = output_dir / f"{experiment_name}_details.json"

    # ===== 构建主文件 =====
    main_data = {
        "metadata": {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "config": {
            "mode": config.mode,
            "tau": config.tau,
            "adaptive_tau_path": config.adaptive_tau_path,
            "task_name": config.task_name,
            "max_samples": config.max_samples,
            "model_path": config.model_path,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "dtype": config.dtype,
            "compression_mode": "post_prefill",
            "svd_method": config.svd_method,
            "decomposition_backend": config.decomposition_backend,
            "cache_backend": config.cache_backend,
            "eigh_compute_device": config.eigh_compute_device,
            "preserve_last_tokens": config.preserve_last_tokens,
            "value_residual_bits": config.value_residual_bits,
            "qjl_seed": config.qjl_seed,
            "eval_method": config.eval_method,
            "run_baseline": config.run_baseline,
            "run_compressed": config.run_compressed,
        },
        "overall_results": {}
    }

    # Baseline 结果
    if "baseline" in results:
        baseline = results["baseline"]
        main_data["overall_results"]["baseline"] = {
            "metrics": baseline["metrics"],
            "avg_total_time": baseline["avg_total_time"],
            "avg_compression_ratio": 1.0,  # Baseline 没有压缩
        }

    # Compressed 结果
    if "compressed" in results:
        compressed = results["compressed"]
        main_data["overall_results"]["compressed"] = {
            "metrics": compressed["metrics"],
            "avg_prefill_time": compressed["avg_prefill_time"],
            "avg_svd_time": compressed["avg_svd_time"],
            "avg_decode_time": compressed["avg_decode_time"],
            "avg_total_time": compressed["avg_total_time"],
            "avg_compression_ratio": compressed["avg_compression_ratio"],
        }

    # 性能对比
    if "baseline" in results and "compressed" in results:
        baseline_metrics = results["baseline"]["metrics"]
        compressed_metrics = results["compressed"]["metrics"]

        # 根据任务类型选择主要指标
        if "rougeL" in baseline_metrics:
            # ROUGE任务：使用F1作为主要指标
            metric_drop = baseline_metrics["rougeL"]["f1"] - compressed_metrics["rougeL"]["f1"]
            metric_name = "rouge_l_drop"
        elif "accuracy" in baseline_metrics:
            # 分类任务：使用accuracy作为主要指标
            metric_drop = baseline_metrics["accuracy"] - compressed_metrics["accuracy"]
            metric_name = "accuracy_drop"
        else:
            # 未知指标
            metric_drop = 0.0
            metric_name = "metric_drop"

        main_data["performance_comparison"] = {
            metric_name: metric_drop,
            "total_time_overhead": compressed["avg_total_time"] - results["baseline"]["avg_total_time"],
            "compression_ratio": compressed["avg_compression_ratio"],
        }

    # 保存主文件
    with open(main_path, 'w', encoding='utf-8') as f:
        json.dump(main_data, f, indent=2, ensure_ascii=False)

    if not quiet:
        print(f"\n✓ Main results saved to: {main_path}")

    # ===== 构建详细文件 =====
    if save_details:
        details_data = {}

        if "baseline" in results:
            # Baseline 样本详情（不包含 layer 信息）
            baseline_details = []
            for sample in results["baseline"]["sample_details"]:
                baseline_details.append({
                    "sample_id": sample["sample_id"],
                    "prompt_length": sample["prompt_length"],
                    "generated_length": sample["generated_length"],
                    "prediction": sample["prediction"],
                    "label": sample["label"],
                    "total_time": sample["total_time"],
                })
            details_data["baseline"] = baseline_details

        if "compressed" in results:
            # Compressed 样本详情（分离样本信息和压缩详情，提高可读性）
            compressed_samples = []
            compression_details = []

            for sample in results["compressed"]["sample_details"]:
                # 样本基本信息（不含layer_compression_details）
                sample_info = {
                    "sample_id": sample["sample_id"],
                    "prompt_length": sample["prompt_length"],
                    "generated_length": sample["generated_length"],
                    "prediction": sample["prediction"],
                    "label": sample["label"],
                    "prefill_time": sample["prefill_time"],
                    "svd_time": sample["svd_time"],
                    "decode_time": sample["decode_time"],
                    "total_time": sample["total_time"],
                    "overall_compression_ratio": sample["overall_compression_ratio"],
                }
                compressed_samples.append(sample_info)

                # 压缩详情（如果存在）
                if "layer_compression_details" in sample:
                    compression_details.append({
                        "sample_id": sample["sample_id"],
                        "layers": sample["layer_compression_details"]
                    })

            details_data["compressed"] = compressed_samples

            # 只有在有压缩详情时才添加
            if compression_details:
                details_data["compression_details"] = compression_details

        # 保存详细文件
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details_data, f, indent=2, ensure_ascii=False)

        if not quiet:
            print(f"✓ Detailed results saved to: {details_path}")
            print(f"\nSummary:")
            if "baseline" in results:
                print(f"  Baseline samples: {len(results['baseline']['sample_details'])}")
            if "compressed" in results:
                print(f"  Compressed samples: {len(results['compressed']['sample_details'])}")
                # 检查是否有 compression_details
                if compression_details:
                    total_layers = len(compression_details[0]['layers'])
                    total_heads = len(compression_details[0]['layers'][0]['heads'])
                    print(f"  Compression details: {len(compression_details)} samples")
                    print(f"  Layers: {total_layers}, Heads per layer: {total_heads}")
    else:
        if not quiet:
            print(f"✓ Detailed file skipped (save_details=False)")
            print(f"\nSummary:")
            if "baseline" in results:
                print(f"  Baseline samples: {len(results['baseline']['sample_details'])}")
            if "compressed" in results:
                print(f"  Compressed samples: {len(results['compressed']['sample_details'])}")
