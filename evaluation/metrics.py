"""Evaluation Metrics"""

from typing import List, Dict, Any
from rouge_score import rouge_scorer


def calculate_accuracy(predictions: List[str], labels: List[str]) -> float:
    if len(predictions) != len(labels):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, labels={len(labels)}")

    if len(predictions) == 0:
        return 0.0

    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def calculate_gsm8k_accuracy(predictions: List[str], labels: List[str]) -> float:
    """
    Calculate accuracy for GSM8K using float comparison.

    This is more lenient than string comparison:
    - "20.00" == "20" (both are 20.0)
    - Handles comma-separated numbers like "2,125"

    与 PyramidKV 的判定逻辑保持一致。
    """
    if len(predictions) != len(labels):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, labels={len(labels)}")

    if len(predictions) == 0:
        return 0.0

    correct = 0
    for pred, label in zip(predictions, labels):
        try:
            # Remove commas and convert to float
            pred_val = float(pred.replace(',', '').strip())
            label_val = float(label.replace(',', '').strip())
            if abs(pred_val - label_val) < 1e-6:
                correct += 1
        except (ValueError, AttributeError):
            # Fallback to string comparison if conversion fails
            if pred.strip() == label.strip():
                correct += 1

    return correct / len(predictions)


def calculate_rouge(predictions: List[str], references: List[str], use_stemmer: bool = True) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-L using official rouge_score package.

    use_stemmer=True  matches xsum / xsum_unitxt (original behavior).
    use_stemmer=False matches lm_eval / unitxt metrics.rouge default (xsum_lmeval).
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)

    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)["rougeL"]
        precisions.append(score.precision)
        recalls.append(score.recall)
        f1s.append(score.fmeasure)

    return {
        "rougeL": {
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }
    }


def evaluate_predictions(predictions: List[str], labels: List[str], task) -> Dict[str, Any]:
    task_name = task.task_name if hasattr(task, 'task_name') else "unknown"
    task_name = task_name.lower()

    if task_name in {"xsum", "xsum_unitxt"}:
        return calculate_rouge(predictions, labels, use_stemmer=True)
    elif task_name == "xsum_lmeval":
        # Aligned with Palu lm_eval: use_stemmer=False, prompt ends with '.'
        return calculate_rouge(predictions, labels, use_stemmer=False)
    elif task_name == "gsm8k":
        # GSM8K uses float comparison (more lenient)
        # "20.00" == "20", handles comma-separated numbers
        return {"accuracy": calculate_gsm8k_accuracy(predictions, labels)}
    else:
        return {"accuracy": calculate_accuracy(predictions, labels)}


def compute_compression_stats(layer_compression_details):
    """从 layer-level 压缩详情汇总总字节数和压缩比。

    Returns: (compression_ratio, total_orig_bytes, total_comp_bytes)
    """
    total_orig_bytes = 0
    total_comp_bytes = 0

    for layer_detail in layer_compression_details:
        for head_detail in layer_detail["heads"]:
            total_orig_bytes += head_detail["K"]["orig_bytes"]
            total_orig_bytes += head_detail["V"]["orig_bytes"]
            total_comp_bytes += head_detail["K"]["comp_bytes"]
            total_comp_bytes += head_detail["V"]["comp_bytes"]

    ratio = total_orig_bytes / total_comp_bytes if total_comp_bytes > 0 else 1.0
    return ratio, total_orig_bytes, total_comp_bytes
