from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    # Local path or Hugging Face model id.  Overridden by --model on the
    # command line and by run_experiment.sh, which sweeps all three models.
    model_path: str = "meta-llama/Llama-2-13b-hf"
    dtype: str = "float16"
    
    task_name: str = "openbookqa"
    data_path: Optional[str] = None  # 添加
    max_samples: Optional[int] = 100
    shuffle: bool = False  # 添加
    
    max_new_tokens: int = 128
    # 默认使用确定性的贪心解码；显式开启时才采样。
    do_sample: bool = False
    temperature: float = 0.8
    top_p: float = 0.95
    
    mode: str = "uniform"
    tau: float = 0.9
    adaptive_tau_path: Optional[str] = None
    preserve_last_tokens: int = 8  # 保留最后 N 个 token 不压缩

    # SVD压缩方法：shared_basis（用K的基压缩V）或 independent（K和V独立SVD）
    svd_method: str = "shared_basis"  # 可选: "shared_basis" 或 "independent"
    value_residual_bits: int = 1  # Main method: 1-bit QJL correction for shared-basis V residual
    qjl_seed: int = 0

    # Execution backends are explicit so a result can be reproduced from its JSON.
    # batched_eigh is the default everywhere because it produced the reported
    # numbers; reference_svd is the per-head SVD fallback, equivalent but slower.
    decomposition_backend: str = "batched_eigh"  # batched_eigh or reference_svd
    cache_backend: str = "factored"  # factored or dense; 与 run_experiment.sh 默认一致
    eigh_compute_device: str = "cpu"  # cpu or input

    run_baseline: bool = True
    run_compressed: bool = True
    output_dir: str = "results"
    experiment_name: str = "default"
    verbose: bool = False
    save_details: bool = True  # 是否保存详细的 layer-level 数据

    # 评估方法：generation（生成式）或 scoring（似然度打分）
    # scoring 适用于多选题任务（hellaswag, openbookqa, boolq），不依赖模型按格式输出
    eval_method: str = "scoring"  # 可选: "generation" 或 "scoring"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if self.mode == "adaptive" and self.run_compressed and not self.adaptive_tau_path:
            raise ValueError("adaptive compressed runs require adaptive_tau_path")
        if self.value_residual_bits not in (0, 1):
            raise ValueError("value_residual_bits must be 0 or 1")
        if self.value_residual_bits and self.svd_method != "shared_basis":
            raise ValueError("value residual correction requires svd_method='shared_basis'")
        if self.svd_method not in ("shared_basis", "independent"):
            raise ValueError("svd_method must be 'shared_basis' or 'independent'")
        if self.decomposition_backend not in ("reference_svd", "batched_eigh"):
            raise ValueError(
                "decomposition_backend must be 'reference_svd' or 'batched_eigh'"
            )
        if self.cache_backend not in ("dense", "factored"):
            raise ValueError("cache_backend must be 'dense' or 'factored'")
        if self.eigh_compute_device not in ("cpu", "input"):
            raise ValueError("eigh_compute_device must be 'cpu' or 'input'")
        if self.cache_backend == "factored":
            if self.svd_method != "shared_basis":
                raise ValueError("factored cache requires svd_method='shared_basis'")
            if self.decomposition_backend != "batched_eigh":
                raise ValueError(
                    "factored cache requires decomposition_backend='batched_eigh'"
                )
