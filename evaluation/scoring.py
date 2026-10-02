"""
Scoring-based evaluation for multiple-choice tasks.

使用似然度打分而不是生成式评估，这是 lm_eval 等标准评估框架使用的方法。
对于每个选项，计算 P(continuation | context)，选择概率最高的选项。

自由函数用于未压缩 baseline；ScoringRunner 同时负责 baseline 与压缩后的打分。
"""

import time
import torch
from typing import List, Tuple, Optional
from transformers import DynamicCache
from core.factored_cache import FactoredCache, use_factored_cache
from core.model_wrapper import compress_cache_layers, iter_cache_layers
from core.timing import PhaseTimer, sync_cuda
from evaluation.metrics import compute_compression_stats


def _compute_option_logprob_from_token_ids(
    model,
    context_ids: List[int],
    option_ids: List[int],
    device: torch.device,
) -> float:
    """
    基于已 tokenized 的 context/option 计算 option 的平均对数似然。
    """
    full_ids = context_ids + option_ids
    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    start_idx = len(context_ids) - 1
    end_idx = len(context_ids) + len(option_ids) - 1

    option_logits = logits[0, start_idx:end_idx, :]
    option_targets = torch.tensor(option_ids, device=device)
    log_probs = torch.nn.functional.log_softmax(option_logits, dim=-1)
    token_log_probs = log_probs[torch.arange(len(option_ids), device=device), option_targets]
    return token_log_probs.mean().item()


def score_multiple_choice(
    model,
    tokenizer,
    context: str,
    options: List[str],
    device: torch.device = None,
    context_ids: Optional[List[int]] = None,
) -> Tuple[int, List[float]]:
    """
    对多选题进行打分，返回最佳选项的索引。

    Args:
        model: HuggingFace 模型
        tokenizer: 对应的 tokenizer
        context: 上下文/问题
        options: 选项列表
        device: 计算设备

    Returns:
        (最佳选项索引, 所有选项的分数列表)
    """
    if device is None:
        device = next(model.parameters()).device
    if context_ids is None:
        context_ids = tokenizer.encode(context, add_special_tokens=True)

    scores = []
    for option in options:
        option_ids = tokenizer.encode(option, add_special_tokens=False)
        score = _compute_option_logprob_from_token_ids(model, context_ids, option_ids, device)
        scores.append(score)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, scores

def score_per_option_context(
    model,
    tokenizer,
    contexts: List[str],
    continuations: List[str],
    device: torch.device = None,
) -> Tuple[int, List[float]]:
    """
    对每个选项有独立 context 的情况进行打分，返回最佳选项的索引。
    
    Args:
        model: HuggingFace 模型
        tokenizer: 对应的 tokenizer
        contexts: 每个选项的 context 列表
        continuations: 每个选项的 continuation 列表
        device: 计算设备
        
    Returns:
        (最佳选项索引, 所有选项的分数列表)
    """
    if device is None:
        device = next(model.parameters()).device

    scores = []
    for ctx, cont in zip(contexts, continuations):
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=True)
        cont_ids = tokenizer.encode(cont, add_special_tokens=False)
        score = _compute_option_logprob_from_token_ids(model, ctx_ids, cont_ids, device)
        scores.append(score)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, scores


class ScoringRunner:
    """在原始模型和压缩 cache 上给选项打分。

    与 core.model_wrapper.GenerationRunner 对称：那个跑生成，这个跑打分。
    """

    def __init__(self, model, tokenizer, task, strategy):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.strategy = strategy

    def evaluate_baseline(self, example):
        """使用 scoring 方法评估单个样本（baseline）"""
        device = next(self.model.parameters()).device
        scoring_mode = self.task.scoring_mode()

        if scoring_mode == "shared_context":
            context, options, label_idx = self.task.get_scoring_inputs(example)
            context_ids = self.tokenizer.encode(context, add_special_tokens=True)
            best_idx, _ = score_multiple_choice(
                self.model,
                self.tokenizer,
                context,
                options,
                device=device,
                context_ids=context_ids,
            )
            prompt_length = len(context_ids)
        elif scoring_mode == "per_option_context":
            contexts, continuations, label_idx = self.task.get_scoring_inputs(example)
            best_idx, _ = score_per_option_context(
                self.model,
                self.tokenizer,
                contexts,
                continuations,
                device=device,
            )
            prompt_length = len(self.tokenizer.encode(contexts[0], add_special_tokens=True))
        else:
            raise ValueError(f"Unknown scoring mode: {scoring_mode}")

        return best_idx, label_idx, prompt_length

    def evaluate_compressed(self, example):
        """按 scoring mode 分派到对应实现。"""
        scoring_mode = self.task.scoring_mode()

        if scoring_mode == "shared_context":
            return self._evaluate_shared_context_compressed(example)
        if scoring_mode == "per_option_context":
            return self._evaluate_per_option_context_compressed(example)
        raise ValueError(f"Unknown scoring mode: {scoring_mode}")

    def _evaluate_shared_context_compressed(self, example):
        """Context 只 prefill 并压缩一次，压缩后的 cache 给所有选项打分。"""
        device = next(self.model.parameters()).device
        context, options, label_idx = self.task.get_scoring_inputs(example)
        context_ids = self.tokenizer.encode(context, add_special_tokens=True)
        prompt_length = len(context_ids)

        sync_cuda()
        start_total = time.perf_counter()
        context_tensor = torch.tensor([context_ids], device=device)
        with PhaseTimer() as t, torch.no_grad():
            outputs = self.model(context_tensor, use_cache=True, return_dict=True)
        prefill_time = t.elapsed

        context_last_logits = outputs.logits[:, -1, :].clone()
        uncompressed_cache = outputs.past_key_values
        cache_layers = iter_cache_layers(uncompressed_cache)
        del outputs, uncompressed_cache

        with PhaseTimer() as t:
            layer_results = compress_cache_layers(self.strategy, cache_layers)
        svd_time = t.elapsed

        layer_compression_details = [result.stats for result in layer_results]

        sync_cuda()
        start_decode = time.perf_counter()
        scores = []
        for option in options:
            option_ids = self.tokenizer.encode(option, add_special_tokens=False)
            scores.append(self._score_continuation(
                layer_results, option_ids, prompt_length, context_last_logits, device
            ))

        sync_cuda()
        decode_time = time.perf_counter() - start_decode
        total_time = time.perf_counter() - start_total
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        overall_compression_ratio, total_orig_bytes, total_comp_bytes = compute_compression_stats(
            layer_compression_details
        )

        stats = {
            "prompt_length": prompt_length,
            "generated_length": 0,
            "prefill_time": prefill_time,
            "svd_time": svd_time,
            "decode_time": decode_time,
            "total_time": total_time,
            "overall_compression_ratio": overall_compression_ratio,
            "total_orig_bytes": total_orig_bytes,
            "total_comp_bytes": total_comp_bytes,
            "layer_compression_details": layer_compression_details,
        }
        return best_idx, label_idx, stats

    def _evaluate_per_option_context_compressed(self, example):
        """每个选项有各自的 context，逐个 prefill 并压缩。"""
        device = next(self.model.parameters()).device
        contexts, continuations, label_idx = self.task.get_scoring_inputs(example)
        scores = []
        stats_accumulator = {
            "prefill_time": 0.0,
            "svd_time": 0.0,
            "decode_time": 0.0,
            "total_orig_bytes": 0,
            "total_comp_bytes": 0
        }
        layer_compression_details = []

        sync_cuda()
        start_total = time.perf_counter()
        for ctx, cont in zip(contexts, continuations):
            ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=True)
            cont_ids = self.tokenizer.encode(cont, add_special_tokens=False)
            prompt_length = len(ctx_ids)

            context_tensor = torch.tensor([ctx_ids], device=device)
            with PhaseTimer() as t, torch.no_grad():
                outputs = self.model(context_tensor, use_cache=True, return_dict=True)
            stats_accumulator["prefill_time"] += t.elapsed

            context_last_logits = outputs.logits[:, -1, :].clone()
            uncompressed_cache = outputs.past_key_values
            cache_layers = iter_cache_layers(uncompressed_cache)
            del outputs, uncompressed_cache

            with PhaseTimer() as t:
                layer_results = compress_cache_layers(self.strategy, cache_layers)
            stats_accumulator["svd_time"] += t.elapsed

            current_layer_compression_details = [
                result.stats for result in layer_results
            ]

            sync_cuda()
            start_decode = time.perf_counter()
            scores.append(self._score_continuation(
                layer_results, cont_ids, prompt_length, context_last_logits, device
            ))
            sync_cuda()
            stats_accumulator["decode_time"] += time.perf_counter() - start_decode

            _, total_orig_bytes, total_comp_bytes = compute_compression_stats(
                current_layer_compression_details
            )
            stats_accumulator["total_orig_bytes"] += total_orig_bytes
            stats_accumulator["total_comp_bytes"] += total_comp_bytes
            layer_compression_details = current_layer_compression_details

        sync_cuda()
        total_time = time.perf_counter() - start_total

        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        prompt_length = len(self.tokenizer.encode(contexts[0], add_special_tokens=True))
        overall_compression_ratio = stats_accumulator["total_orig_bytes"] / stats_accumulator["total_comp_bytes"] if stats_accumulator["total_comp_bytes"] > 0 else 1.0

        stats = {
            "prompt_length": prompt_length,
            "generated_length": 0,
            "prefill_time": stats_accumulator["prefill_time"],
            "svd_time": stats_accumulator["svd_time"],
            "decode_time": stats_accumulator["decode_time"],
            "total_time": total_time,
            "overall_compression_ratio": overall_compression_ratio,
            "total_orig_bytes": stats_accumulator["total_orig_bytes"],
            "total_comp_bytes": stats_accumulator["total_comp_bytes"],
            "layer_compression_details": layer_compression_details,
        }
        return best_idx, label_idx, stats

    def _score_continuation(self, layer_results, token_ids, prompt_length,
                            context_last_logits, device):
        """在压缩后的 cache 上前向 token_ids，返回平均 token log-prob。

        首 token 的 log-prob 来自 context 的最后一个位置（压缩前算好的
        context_last_logits），其余来自这次前向。
        """
        token_tensor = torch.tensor([token_ids], device=device)
        option_cache = self._build_option_cache(layer_results)

        cache_position = torch.arange(
            prompt_length, prompt_length + len(token_ids),
            device=device, dtype=torch.long
        )
        attention_mask = torch.ones(
            (1, prompt_length + len(token_ids)),
            device=device, dtype=torch.long
        )

        with torch.no_grad(), use_factored_cache(option_cache):
            option_outputs = self.model(
                token_tensor,
                past_key_values=option_cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            option_logits = option_outputs.logits

        first_log_prob = torch.nn.functional.log_softmax(context_last_logits, dim=-1)[0, token_ids[0]]
        if len(token_ids) == 1:
            return first_log_prob.item()

        rest_logits = option_logits[0, :-1, :]
        rest_targets = torch.tensor(token_ids[1:], device=device)
        rest_log_probs = torch.nn.functional.log_softmax(rest_logits, dim=-1)
        rest_token_log_probs = rest_log_probs[torch.arange(len(rest_targets)), rest_targets]
        all_log_probs = torch.cat([first_log_prob.unsqueeze(0), rest_token_log_probs])
        return all_log_probs.mean().item()

    @staticmethod
    def _build_option_cache(layer_results):
        """Build a fresh cache for one scoring pass."""
        if layer_results and all(result.factors is not None for result in layer_results):
            return FactoredCache.from_factors(
                result.factors for result in layer_results
            )
        if not layer_results or any(result.dense_kv is None for result in layer_results):
            raise RuntimeError("compression produced a mixture of dense and factored layers")
        cache = DynamicCache()
        for layer_idx, result in enumerate(layer_results):
            key, value = result.dense_kv
            cache.update(key, value, layer_idx)
        return cache
