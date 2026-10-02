import time
from contextlib import closing
import torch
from typing import Tuple, Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from core.factored_cache import FactoredCache, use_factored_cache
from strategies.base import CompressionStrategy
from core.timing import PhaseTimer, sync_cuda


def _make_stop_criteria(tokenizer, stop_strings: List[str]) -> Optional[StoppingCriteriaList]:
    """Build StoppingCriteriaList from stop strings using text-level matching.

    Matches lm_eval's until=[...] semantics: stop when the decoded text of the
    last generated token contains any stop string.  Token-ID set membership is
    intentionally avoided because standalone tokenization of a stop string
    (e.g. "\n") can produce artifact token IDs that appear in unrelated tokens.
    """
    if not stop_strings:
        return None

    class _StopOnString(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            decoded = tokenizer.decode(
                [input_ids[0, -1].item()], skip_special_tokens=False
            )
            return any(s in decoded for s in stop_strings)

    return StoppingCriteriaList([_StopOnString()])


def iter_cache_layers(past_key_values):
    """Yield cache layers and release each source reference after consumption."""
    if hasattr(past_key_values, "layers"):
        for layer_idx, layer in enumerate(past_key_values.layers):
            try:
                yield layer_idx, (layer.keys, layer.values)
            finally:
                layer.keys = None
                layer.values = None
        return

    if hasattr(past_key_values, "key_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            try:
                yield layer_idx, (
                    past_key_values.key_cache[layer_idx],
                    past_key_values.value_cache[layer_idx],
                )
            finally:
                past_key_values.key_cache[layer_idx] = None
                past_key_values.value_cache[layer_idx] = None
        return

    # Tuple/list cache APIs do not expose mutable layer slots. The temporary
    # list still drops its references one layer at a time; the original cache
    # object is released by the caller after this generator is exhausted.
    cache_layers = list(past_key_values)
    del past_key_values
    for layer_idx, layer_kv in enumerate(cache_layers):
        try:
            yield layer_idx, layer_kv
        finally:
            cache_layers[layer_idx] = None


def compress_cache_layers(strategy, cache_layers):
    """Compress an owned cache iterator and release each source layer promptly."""
    results = []
    with closing(cache_layers):
        for layer_idx, layer_kv in cache_layers:
            results.append(strategy.compress_layer_kv(layer_kv, layer_idx))
    return results


class GenerationRunner:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, strategy: CompressionStrategy):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.device = next(model.parameters()).device

    @staticmethod
    def _get_cache_seq_length(past_key_values) -> int:
        """Return the number of tokens represented in the cache."""
        if past_key_values is None:
            return 0
        # 新版 Cache 接口
        if hasattr(past_key_values, "get_seq_length"):
            try:
                return int(past_key_values.get_seq_length())
            except TypeError:
                return int(past_key_values.get_seq_length(layer_idx=0))
        # 旧版 tuple/list 接口
        if isinstance(past_key_values, (list, tuple)) and past_key_values:
            layer0 = past_key_values[0]
            key_tensor = layer0[0] if isinstance(layer0, (list, tuple)) else layer0
            if isinstance(key_tensor, torch.Tensor) and key_tensor.ndim >= 3:
                return int(key_tensor.shape[-2])   # [B, H, T, D] → T
        return 0
    
        
    # ---------------- 外部接口：压缩版生成 ----------------
    def generate_compressed(self, prompt: str, max_new_tokens: int, **gen_kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Post-Prefill 压缩模式（当前实现）

        流程：
        1. Prefill 阶段获取 logits 和未压缩的 KV cache
        2. 从 prefill 的 logits 采样第一个 token
        3. 批量压缩所有层的 KV cache
        4. 使用压缩后的 cache 生成剩余 tokens

        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成 token 数
            **gen_kwargs: 生成参数（temperature, top_p 等）

        Returns:
            (生成的文本, 统计信息字典)
        """

        stop_criteria = _make_stop_criteria(self.tokenizer, gen_kwargs.pop("stop_strings", []))

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            # truncation=True,
        ).to(self.device)

        prompt_length = inputs["input_ids"].shape[1]

        # Step 1: Prefill 阶段 - 获取 logits 和 cache
        # 端到端独立计时：阶段之间还有第一个 token 的选取、empty_cache()、
        # cache 构造等准备工作，三段相加会漏掉它们。
        sync_cuda()
        start_total = time.perf_counter()
        with PhaseTimer() as t, torch.no_grad():
            outputs = self.model(**inputs, use_cache=True, return_dict=True)
        prefill_time = t.elapsed

        # 获取 prefill 的输出
        logits = outputs.logits[:, -1, :]  # 最后一个位置的 logits (用于生成第一个 token)
        uncompressed_cache = getattr(outputs, "past_key_values", None)
        if uncompressed_cache is None:
            raise ValueError("Model forward did not return past_key_values")

        # Step 2: 从 prefill 的 logits 采样第一个 token
        # 注意: 这个 logits 是在 prefill 时就计算好的,不受压缩影响
        do_sample = gen_kwargs.get("do_sample", False)
        if do_sample:
            temperature = gen_kwargs.get("temperature", 1.0)
            temperature = max(temperature, 1e-5)
            logits = logits / temperature

            top_p = gen_kwargs.get("top_p", None)
            if top_p is not None and top_p < 1.0:
                logits = self._top_p_filtering(logits, top_p)

            probs = torch.softmax(logits, dim=-1)
            first_token = torch.multinomial(probs, num_samples=1)
        else:
            first_token = torch.argmax(logits, dim=-1, keepdim=True)

        # Step 3: 压缩 KV cache（逐层释放原始cache以节省显存）
        # 计时只包压缩本身：empty_cache() 与 cache 构造是准备开销，计在区间外，
        # 否则报出来的不是压缩成本。
        cache_layers = iter_cache_layers(uncompressed_cache)
        del outputs, uncompressed_cache, logits
        with PhaseTimer() as t:
            layer_results = compress_cache_layers(self.strategy, cache_layers)
        svd_time = t.elapsed

        layer_compression_details = [result.stats for result in layer_results]
        torch.cuda.empty_cache()

        # 使用 tuple 格式的压缩 cache (匹配旧代码)
        if layer_results and all(result.factors is not None for result in layer_results):
            compressed_cache = FactoredCache.from_factors(
                result.factors for result in layer_results
            )
        elif layer_results and all(result.dense_kv is not None for result in layer_results):
            compressed_cache = tuple(result.dense_kv for result in layer_results)
        else:
            raise RuntimeError(
                "compression produced a mixture of dense and factored layers"
            )
        del layer_results

        # Step 4: 生成剩余 tokens (使用压缩后的 cache)
        if max_new_tokens > 1:
            # 将第一个 token 加入到输入序列
            generated = torch.cat([inputs["input_ids"], first_token], dim=-1)

            # 更新 attention_mask
            if inputs.get("attention_mask") is not None:
                attention_mask = torch.cat([
                    inputs["attention_mask"],
                    torch.ones((1, 1), device=self.device, dtype=inputs["attention_mask"].dtype)
                ], dim=-1)
            else:
                attention_mask = None

            # 使用 model.generate() 生成剩余 tokens (匹配旧代码行为)
            remaining_tokens = max_new_tokens - 1
            if remaining_tokens > 0:
                cache_seq_len = self._get_cache_seq_length(compressed_cache)
                next_input = generated[:, -1:]  # 最后一个 token (即 first_token)

                generate_kwargs = {
                    "past_key_values": compressed_cache,
                    "use_cache": True,
                    "return_dict_in_generate": True,
                    "cache_position": torch.arange(
                        cache_seq_len,
                        cache_seq_len + next_input.shape[1],
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "pad_token_id": self.tokenizer.pad_token_id,
                }

                if attention_mask is not None:
                    generate_kwargs["attention_mask"] = attention_mask
                if stop_criteria:
                    generate_kwargs["stopping_criteria"] = stop_criteria

                with PhaseTimer() as t, torch.no_grad(), use_factored_cache(compressed_cache):
                    outputs = self.model.generate(
                        input_ids=next_input,
                        max_new_tokens=remaining_tokens,
                        do_sample=do_sample,
                        temperature=gen_kwargs.get("temperature", 1.0),
                        top_p=gen_kwargs.get("top_p", 0.95),
                        **generate_kwargs,
                    )
                decode_time = t.elapsed

                # outputs.sequences: [batch, 1 + remaining_tokens]
                # 提取新生成的 tokens (去掉输入的 next_input)
                if outputs.sequences.shape[1] > 1:
                    new_tokens_from_generate = outputs.sequences[:, 1:]  # 去掉第一个 (next_input)
                    generated = torch.cat([generated, new_tokens_from_generate], dim=-1)

                # 提取所有新生成的 tokens (去掉 prompt)
                all_new_tokens = generated[0, prompt_length:]
            else:
                decode_time = 0.0
                all_new_tokens = generated[0, prompt_length:]
        else:
            # 只需要生成 1 个 token
            decode_time = 0.0
            all_new_tokens = first_token[0]

        # 解码输出
        output_text = self.tokenizer.decode(all_new_tokens, skip_special_tokens=True)

        sync_cuda()
        total_time = time.perf_counter() - start_total

        # 计算总体统计
        stats = {
            "prompt_length": prompt_length,
            "generated_length": len(all_new_tokens),
            "prefill_time": prefill_time,
            "svd_time": svd_time,
            "decode_time": decode_time,
            "total_time": total_time,
            "layer_compression_details": layer_compression_details,
        }

        return output_text, stats

    @staticmethod
    def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Create mask and apply filtering
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        filtered_logits = logits.masked_fill(mask, float('-inf'))
        return filtered_logits
    
    # ---------------- Baseline：原始 generate ----------------
    def generate_baseline(self, prompt: str, max_new_tokens: int, **gen_kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        不使用压缩的标准生成（作为对照基准）

        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成 token 数
            **gen_kwargs: 生成参数（temperature, top_p 等）

        Returns:
            (生成的文本, 统计信息字典)

            统计信息包含:
                - prompt_length: prompt token 数
                - generated_length: 实际生成的 token 数
                - total_time: 总时间（prefill + decode）
        """
        stop_criteria = _make_stop_criteria(self.tokenizer, gen_kwargs.pop("stop_strings", []))

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            # truncation=True,  # 匹配旧代码: baseline 不使用 truncation
        ).to(self.device)

        prompt_length = inputs["input_ids"].shape[1]

        extra = {"stopping_criteria": stop_criteria} if stop_criteria else {}
        sync_cuda()
        start_time = time.perf_counter()
        with torch.no_grad():
            full_output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
                **extra,
            )
        # 提取新生成的部分（去掉原始 prompt）
        # model.generate 返回: [prompt_tokens..., new_token_1, new_token_2, ...]
        new_tokens_only = full_output_ids[0, prompt_length:]
        output_text = self.tokenizer.decode(new_tokens_only, skip_special_tokens=True)

        # 与 generate_compressed 同口径：都在 decode 之后停表并显式同步。
        sync_cuda()
        total_time = time.perf_counter() - start_time

        stats = {
            "prompt_length": prompt_length,
            "generated_length": len(new_tokens_only),
            "total_time": total_time,
        }

        return output_text, stats
