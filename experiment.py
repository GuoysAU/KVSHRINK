import torch
import time
import argparse
from tqdm import tqdm
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import Config
from core.model_wrapper import GenerationRunner
from strategies.uniform import UniformStrategy
from strategies.adaptive import AdaptiveStrategy
from utils.tau_loader import load_adaptive_tau
from utils.result_saver import save_experiment_results, print_metrics, print_comparison
from tasks.task_loader import create_task_from_config
from evaluation.metrics import compute_compression_stats, evaluate_predictions
from evaluation.scoring import ScoringRunner
from core.factored_cache import register as register_factored

class Experiment:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.task = None
        self.strategy = None
        self.generation_runner = None
        self.scoring_runner = None
        self._baseline_result = None  # stashed so periodic checkpoints can include it

    def run(self) -> Dict[str, Any]:
        self.setup()
        if not self.config.run_baseline and not self.config.run_compressed:
            raise ValueError("At least one of baseline/compressed must be enabled.")

        results = {}
        baseline_result = None
        compressed_result = None

        if self.config.run_baseline:
            print("="*60)
            print("Running Baseline")
            print("="*60)

            baseline_result = self.run_baseline()
            results["baseline"] = baseline_result
            self._baseline_result = baseline_result

            print_metrics("Baseline", baseline_result["metrics"])
            print(f"  Avg Time: {baseline_result['avg_total_time']:.3f}s")

        if self.config.run_compressed:
            print("\n" + "="*60)
            print(f"Running Compressed ({self.config.mode.upper()})")
            print("="*60)

            compressed_result = self.run_compressed()
            results["compressed"] = compressed_result

            print_metrics("Compressed", compressed_result["metrics"])
            print(f"  Avg Prefill Time: {compressed_result['avg_prefill_time']:.3f}s")
            print(f"  Avg SVD Time: {compressed_result['avg_svd_time']:.3f}s")
            print(f"  Avg Decode Time: {compressed_result['avg_decode_time']:.3f}s")
            print(f"  Avg Total Time: {compressed_result['avg_total_time']:.3f}s")
            print(f"  Avg Compression Ratio: {compressed_result['avg_compression_ratio']:.2f}x")

        if baseline_result is not None and compressed_result is not None:
            print_comparison(baseline_result["metrics"], compressed_result["metrics"])
            print(f"  SVD Overhead: {compressed_result['avg_svd_time']:.3f}s")
        save_experiment_results(
            results=results,
            config=self.config,
            output_dir=self.config.output_dir,
            experiment_name=self.config.experiment_name,
            save_details=self.config.save_details
        )

        return results

    def setup(self):
        print()
        print("="*60)
        print("System Setup")
        print("="*60)

        if (
            self.config.run_compressed
            and self.config.decomposition_backend == "batched_eigh"
            and self.config.eigh_compute_device == "cpu"
        ):
            # Four threads reduce scheduling overhead for these small EIGH matrices.
            cpu_eigh_threads = 4
            torch.set_num_threads(min(cpu_eigh_threads, torch.get_num_threads()))

        ## 1. 加载HF模型
        print(f"Loading model from {self.config.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            dtype = torch.float16 if self.config.dtype == "float16" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        ## 2. 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        ## 3.加载task
        print(f"Loading task {self.config.task_name}...")
        self.task = create_task_from_config(self.config)
        
        ## 4.创建压缩策略strategy
        print(f"Creating {self.config.mode} strategy...")
        self.create_strategy()

        ## 5.创建 generation/scoring runners
        self.generation_runner = GenerationRunner(self.model, self.tokenizer, self.strategy)
        self.scoring_runner = ScoringRunner(
            self.model, self.tokenizer, self.task, self.strategy
        )
        print("Setup complete.\n")
    
    def create_strategy(self):
        strategy_kwargs = {
            "preserve_last_n": self.config.preserve_last_tokens,
            "preserve_first_n": 0,
            "compression_method": self.config.svd_method,
            "value_residual_bits": self.config.value_residual_bits,
            "qjl_seed": self.config.qjl_seed,
            "decomposition_backend": self.config.decomposition_backend,
            "cache_backend": self.config.cache_backend,
            "eigh_compute_device": self.config.eigh_compute_device,
        }

        if self.config.mode == "adaptive":
            self.strategy = AdaptiveStrategy(**strategy_kwargs)
        elif self.config.mode == "uniform":
            self.strategy = UniformStrategy(tau=self.config.tau, **strategy_kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def run_baseline(self) -> Dict[str, Any]:
        predictions, labels, sample_details = [], [], []
        total_samples = len(self.task)
        use_scoring = self._is_scoring_task()
        generation_kwargs = None if use_scoring else self._build_generation_kwargs()

        for i, example in enumerate(tqdm(self.task, total=total_samples, desc="Baseline", ncols=80)):
            if use_scoring:
                start_time = time.perf_counter()
                prediction, label, prompt_length = self.scoring_runner.evaluate_baseline(example)
                generated_length = 0
                total_time = time.perf_counter() - start_time
            else:
                output, stats = self.generation_runner.generate_baseline(
                    example.prompt,
                    **generation_kwargs,
                )
                prediction = self.task.parse_output(output)
                label = example.label
                prompt_length = stats["prompt_length"]
                generated_length = stats["generated_length"]
                total_time = stats["total_time"]

            predictions.append(prediction)
            labels.append(label)
            sample_detail = {
                "sample_id": i,
                "prompt_length": prompt_length,
                "generated_length": generated_length,
                "prediction": prediction,
                "label": label,
                "total_time": total_time,
            }
            sample_details.append(sample_detail)

            if self.config.verbose:
                print(f"Time: {total_time:.2f}s, Pred: {sample_detail['prediction']}, Label: {sample_detail['label']}")

        print()
        metrics = self._compute_metrics(predictions, labels, use_scoring)
        avg_total_time = self._safe_average([s["total_time"] for s in sample_details])

        return {
            "predictions": predictions,
            "labels": labels,
            "metrics": metrics,
            "sample_details": sample_details,
            "avg_total_time": avg_total_time,
        }

    def run_compressed(self) -> Dict[str, Any]:
        if self.config.cache_backend == "factored": # 注册并选择自定义 (compressed representation) attention backend for HF, After baseline
            register_factored(self.model)
            print("  Cache backend: factored")

        predictions, labels, sample_details = [], [], []
        total_samples = len(self.task)
        use_scoring = self._is_scoring_task()
        generation_kwargs = None if use_scoring else self._build_generation_kwargs()

        for i, example in enumerate(tqdm(self.task, total=total_samples, desc="Compressed", ncols=80)):
            self._update_adaptive_tau_if_needed(sample_idx=i)
            if use_scoring:
                prediction, label, stats = self.scoring_runner.evaluate_compressed(example)
            else:
                output, stats = self.generation_runner.generate_compressed(
                    example.prompt,
                    **generation_kwargs,
                )
                prediction = self.task.parse_output(output)
                label = example.label
                ratio, orig_bytes, comp_bytes = compute_compression_stats(
                    stats["layer_compression_details"]
                )
                stats["overall_compression_ratio"] = ratio
                stats["total_orig_bytes"] = orig_bytes
                stats["total_comp_bytes"] = comp_bytes

            predictions.append(prediction)
            labels.append(label)
            sample_detail = {
                "sample_id": i,
                "prompt_length": stats["prompt_length"],
                "generated_length": stats["generated_length"],
                "prediction": prediction,
                "label": label,
                "prefill_time": stats["prefill_time"],
                "svd_time": stats["svd_time"],
                "decode_time": stats["decode_time"],
                "total_time": stats["total_time"],
                "overall_compression_ratio": stats["overall_compression_ratio"],
                "total_orig_bytes": stats["total_orig_bytes"],
                "total_comp_bytes": stats["total_comp_bytes"],
            }
            if self.config.save_details:
                sample_detail["layer_compression_details"] = stats["layer_compression_details"]
            sample_details.append(sample_detail)

            if self.config.verbose:
                print(f"SVD: {sample_detail['svd_time']:.2f}s, Total: {sample_detail['total_time']:.2f}s, Pred: {sample_detail['prediction']}")

            if (i + 1) % 500 == 0: # checkpoint：长跑中途崩溃不至于丢掉已算的样本
                self._save_compressed_checkpoint(
                    predictions, labels, sample_details, use_scoring
                )
        print()
        return self._summarize_compressed(predictions, labels, sample_details, use_scoring)

    
    def _is_scoring_task(self) -> bool:
        """判断当前任务是否使用 scoring 评估"""
        return self.config.eval_method == "scoring" and self.task.supports_scoring()

    @staticmethod
    def _safe_average(values) -> float:
        return sum(values) / len(values) if values else 0.0

    def _build_generation_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature if self.config.do_sample else None,
            "top_p": self.config.top_p if self.config.do_sample else None,
        }
        stops = self.task.stop_strings()
        if stops:
            kwargs["stop_strings"] = stops
        return kwargs

    def _compute_metrics(self, predictions, labels, use_scoring: bool) -> Dict[str, Any]:
        if use_scoring:
            correct = sum(p == l for p, l in zip(predictions, labels))
            accuracy = correct / len(predictions) if predictions else 0.0
            return {"accuracy": accuracy}
        return evaluate_predictions(predictions, labels, self.task)

    def _update_adaptive_tau_if_needed(self, sample_idx: int) -> None:
        if self.config.mode == "adaptive":
            tau_map = load_adaptive_tau(self.config.adaptive_tau_path, sample_idx=sample_idx)
            self.strategy.update_tau(tau_map)

    def _summarize_compressed(self, predictions, labels, sample_details, use_scoring) -> Dict[str, Any]:
        """从已累积的样本汇总 compressed 的总体指标（accuracy/CR/耗时）。
        run_compressed 的最终返回和周期性 checkpoint 共用此函数，保证格式完全一致。"""
        metrics = self._compute_metrics(predictions, labels, use_scoring)
        avg_prefill_time = self._safe_average([s["prefill_time"] for s in sample_details])
        avg_svd_time = self._safe_average([s["svd_time"] for s in sample_details])
        avg_decode_time = self._safe_average([s["decode_time"] for s in sample_details])
        avg_total_time = self._safe_average([s["total_time"] for s in sample_details])

        # 计算真实的总体压缩比：所有样本的总原始大小 / 总压缩大小
        total_all_orig_bytes = sum(s["total_orig_bytes"] for s in sample_details)
        total_all_comp_bytes = sum(s["total_comp_bytes"] for s in sample_details)
        avg_compression_ratio = total_all_orig_bytes / total_all_comp_bytes if total_all_comp_bytes > 0 else 1.0

        return {
            "predictions": predictions,
            "labels": labels,
            "metrics": metrics,
            "sample_details": sample_details,
            "avg_prefill_time": avg_prefill_time,
            "avg_svd_time": avg_svd_time,
            "avg_decode_time": avg_decode_time,
            "avg_total_time": avg_total_time,
            "avg_compression_ratio": avg_compression_ratio,
        }

    def _save_compressed_checkpoint(self, predictions, labels, sample_details, use_scoring):
        """周期性持久化 compressed 的部分结果，直接写入正式结果文件。
        跑完时 run() 末尾的 save_experiment_results 会再覆盖一次（带完整数据），
        所以不需要额外的 checkpoint 文件。"""
        partial = self._summarize_compressed(predictions, labels, sample_details, use_scoring)
        results = {"compressed": partial}
        if self._baseline_result is not None:
            results["baseline"] = self._baseline_result
        save_experiment_results(
            results=results,
            config=self.config,
            output_dir=self.config.output_dir,
            experiment_name=self.config.experiment_name,
            save_details=False,
            quiet=True,
        )
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="adaptive", choices=["uniform", "adaptive"],
                        help="adaptive: attention-guided per-head thresholds (the "
                             "method evaluated in the paper). uniform: a single "
                             "fixed threshold, used for the ablation in Figure 3.")
    parser.add_argument("--uniform_tau", type=float, default=0.9)
    parser.add_argument("--adaptive-tau", type=str, default=None)
    parser.add_argument("--task", type=str, default="boolq")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-hf",
                        help="Local path or Hugging Face model id")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--no-compressed", action="store_true", help="Skip compressed evaluation")
    parser.add_argument("--no-details", action="store_true", help="Skip saving detailed layer-level data (faster)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument("--name", type=str, default=None, help="Custom experiment name")
    parser.add_argument(
        "--eval-method",
        type=str,
        default="scoring",
        choices=["generation", "scoring"],
        help="Evaluation method: generation (parse model output) or scoring (likelihood-based, recommended for multiple-choice)",
    )
    parser.add_argument(
        "--svd-method",
        type=str,
        default="shared_basis",
        choices=["shared_basis", "independent"],
        help="SVD compression method: shared_basis (use the Key basis for V, "
             "default) or independent (K and V decomposed separately). "
             "independent has no factored representation, so it also needs "
             "--cache-backend dense",
    )
    parser.add_argument(
        "--value-residual-bits",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use a 1-bit QJL correction for the shared-basis V residual (default: 1)",
    )
    parser.add_argument(
        "--qjl-seed",
        type=int,
        default=0,
        help="Seed for the fixed Gaussian QJL projection (default: 0)",
    )
    parser.add_argument(
        "--decomposition-backend",
        choices=["batched_eigh", "reference_svd"],
        default="batched_eigh",
        help="Numerical decomposition implementation (default: batched_eigh, "
             "the path the reported numbers come from; reference_svd is the "
             "equivalent but slower per-head SVD fallback)",
    )
    parser.add_argument(
        "--cache-backend",
        choices=["factored", "dense"],
        default="factored",
        help="Cache representation used during decoding (default: factored, "
             "the path used for the reported results; dense reconstructs K/V instead)",
    )
    parser.add_argument(
        "--eigh-compute-device",
        choices=["cpu", "input"],
        default="cpu",
        help="Where batched EIGH runs: CPU or the Gram tensor's input device",
    )
    args = parser.parse_args()
    if args.no_baseline and args.no_compressed:
        parser.error("At least one of baseline/compressed must be enabled.")
    if args.mode == "adaptive" and not args.adaptive_tau and not args.no_compressed:
        parser.error(
            "--mode adaptive needs a per-head threshold file: pass "
            "--adaptive-tau <path>, generated by utils/generate_adaptive_tau.py. "
            "Use --mode uniform --uniform_tau <value> for the fixed-threshold "
            "ablation instead."
        )
    config = Config(
        mode=args.mode,
        tau=args.uniform_tau,
        adaptive_tau_path=args.adaptive_tau,
        task_name=args.task,
        max_samples=args.max_samples,
        model_path=args.model,
        output_dir=args.output_dir,
        run_baseline=not args.no_baseline,
        run_compressed=not args.no_compressed,
        verbose=args.verbose,
        save_details=not args.no_details,
        experiment_name=args.name or f"{args.mode}_{args.task}",
        eval_method=args.eval_method,
        svd_method=args.svd_method,
        decomposition_backend=args.decomposition_backend,
        cache_backend=args.cache_backend,
        eigh_compute_device=args.eigh_compute_device,
        value_residual_bits=args.value_residual_bits,
        qjl_seed=args.qjl_seed,
    )
    
    exp = Experiment(config)
    exp.run()


if __name__ == "__main__":
    main()
