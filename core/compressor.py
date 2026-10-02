import torch
import time
import math
from typing import Tuple, Dict, Any


class SVDCompressor:
    _qjl_projection_cache = {}

    # -- preserve window --------------------------------------------------
    # Every entry point below works on the same three-way view of the
    # sequence: a preserved prefix, a compressed middle, and a preserved
    # suffix. These helpers are the single definition of that geometry.
    # ``seq_dim`` is 1 for per-head [batch, seq, head_dim] tensors and 2 for
    # whole-layer [batch, heads, seq, head_dim] tensors.

    @staticmethod
    def _split_preserved(t: torch.Tensor, preserve_first_n: int,
                         preserve_last_n: int, seq_dim: int):
        """Split into (first, middle, last); an end is None when not preserved."""
        lead = (slice(None),) * seq_dim
        first = t[lead + (slice(None, preserve_first_n),)] if preserve_first_n > 0 else None
        last = t[lead + (slice(-preserve_last_n, None),)] if preserve_last_n > 0 else None
        middle = t[lead + (slice(preserve_first_n or None, -preserve_last_n or None),)]
        return first, middle, last

    @staticmethod
    def _join_preserved(first, middle, last, seq_dim: int) -> torch.Tensor:
        """Inverse of _split_preserved: concatenate whichever parts exist."""
        parts = [p for p in (first, middle, last) if p is not None]
        return torch.cat(parts, dim=seq_dim) if len(parts) > 1 else parts[0]

    @staticmethod
    def _skipped_stats(seq_len: int, head_dim: int, orig_bytes: int):
        """K/V stats for a sequence too short to leave anything to compress."""
        base = {
            "rank": seq_len, "max_rank": seq_len, "relative_error": 0.0,
            "energy_retained": 1.0, "seq_len": seq_len, "head_dim": head_dim,
            "orig_bytes": orig_bytes, "comp_bytes": orig_bytes,
            "compression_ratio": 1.0, "svd_time": 0.0, "status": "skipped",
        }
        return dict(base, tensor="K"), dict(base, tensor="V")

    @staticmethod
    def _eigh_batched(gram: torch.Tensor, compute_device: str = "cpu"):
        """Batched symmetric eigendecomposition of [..., D, D] Gram matrices.

        For the small D=head_dim matrices used here, CPU LAPACK is ~2.5x faster
        than GPU cuSOLVER (which handles many small batched eigh poorly). Compute
        on CPU and move the factors back when compute_device is "cpu"; use
        "input" to keep the computation on the Gram tensor's device.
        """
        if compute_device not in ("cpu", "input"):
            raise ValueError("compute_device must be 'cpu' or 'input'")
        if gram.is_cuda and compute_device == "cpu":
            w, v = torch.linalg.eigh(gram.cpu())
            return w.to(gram.device), v.to(gram.device)
        return torch.linalg.eigh(gram)

    @staticmethod
    def _get_qjl_projection(
        dimension: int,
        device: torch.device,
        seed: int,
    ) -> torch.Tensor:
        """Return a deterministic Gaussian QJL projection shared across rows."""
        cache_key = (dimension, str(device), seed)
        projection = SVDCompressor._qjl_projection_cache.get(cache_key)
        if projection is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            projection = torch.randn(
                dimension,
                dimension,
                generator=generator,
                dtype=torch.float32,
            ).to(device)
            SVDCompressor._qjl_projection_cache[cache_key] = projection
        return projection

    @staticmethod
    def pack_signs(signs: torch.Tensor) -> torch.Tensor:
        """Pack a +-1 matrix into one bit per entry: [N, D] -> [N, D/8] uint8."""
        n, d = signs.shape
        if d % 8 != 0:
            raise ValueError(f"head_dim {d} must be a multiple of 8 to bit-pack")
        bits = (signs > 0).to(torch.uint8).reshape(n, d // 8, 8)
        weights = (1 << torch.arange(8, device=signs.device, dtype=torch.uint8))
        return (bits * weights).sum(-1).to(torch.uint8)

    @staticmethod
    def unpack_signs(packed: torch.Tensor, head_dim: int,
                     dtype: torch.dtype) -> torch.Tensor:
        """Inverse of pack_signs: [N, D/8] uint8 -> [N, D] in {-1, +1}."""
        n = packed.shape[0]
        shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
        bits = (packed.unsqueeze(-1) >> shifts) & 1
        return bits.reshape(n, head_dim).to(dtype) * 2 - 1

    @staticmethod
    def _reconstruct_qjl_residual(
        residual: torch.Tensor,
        storage_dtype: torch.dtype,
        seed: int,
        return_code: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize and reconstruct row vectors with the 1-bit QJL estimator.

        With ``return_code=True`` the sign code and row scales are attached to the
        stats dict under ``_code`` so the factored path can store them instead of
        the reconstruction. The entry is popped by the caller and never reaches
        the JSON stats dump.
        """
        num_rows, dimension = residual.shape
        projection = SVDCompressor._get_qjl_projection(
            dimension=dimension,
            device=residual.device,
            seed=seed,
        )

        row_norms = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        stored_norms = row_norms.to(storage_dtype).float()
        projected = residual @ projection.T
        signs = torch.where(projected >= 0, 1.0, -1.0)
        scale = math.sqrt(math.pi / 2.0) / dimension
        reconstructed = scale * stored_norms * (signs @ projection)

        packed_sign_bytes = (num_rows * dimension + 7) // 8
        norm_bytes_per_row = 2 if storage_dtype == torch.float16 else 4
        norm_bytes = num_rows * norm_bytes_per_row
        meta = {
            "enabled": True,
            "bits": 1,
            "seed": seed,
            "code_bytes": packed_sign_bytes,
            "norm_bytes": norm_bytes,
            "comp_bytes": packed_sign_bytes + norm_bytes,
        }
        if return_code:
            # Packed so the stored code really occupies the nd bits that
            # Eq. (60) charges; one int8 per sign would be 8x that.
            meta["_code"] = (SVDCompressor.pack_signs(signs),
                             stored_norms.to(storage_dtype))
        return reconstructed, meta

    @staticmethod
    def compress_tensor(
        tensor: torch.Tensor,
        tau: float,
        tensor_name: str = "tensor"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        使用 SVD 压缩张量，并返回详细的压缩统计信息

        Args:
            tensor: 输入张量 [batch, seq_len, head_dim] 或其他形状
            tau: 能量保留阈值
            tensor_name: 张量名称（"K" 或 "V"）

        Returns:
            (压缩后的张量, 统计信息字典)
        """
        original_shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype

        # Reshape 为矩阵进行 SVD
        matrix = tensor.reshape(-1, original_shape[-1]).float()

        start = time.perf_counter()
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

        # 计算能量和选择 rank
        total_energy = (S ** 2).sum()
        cumsum_energy = torch.cumsum(S ** 2, dim=0)
        k = torch.searchsorted(cumsum_energy, tau * total_energy).item() + 1
        k = max(1, min(k, len(S)))

        # 压缩
        compressed = (U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :])
        svd_time = time.perf_counter() - start

        compressed = compressed.reshape(original_shape).to(dtype).to(device)

        # 计算统计信息
        max_rank = len(S)
        energy_retained = (cumsum_energy[k-1] / total_energy).item()

        # 计算相对误差
        error = torch.norm(matrix - compressed.reshape(matrix.shape).float())
        original_norm = torch.norm(matrix)
        relative_error = (error / original_norm).item() if original_norm > 0 else 0.0

        # 计算内存占用（以字节为单位，假设 float16）
        bytes_per_element = 2 if dtype == torch.float16 else 4
        orig_bytes = tensor.numel() * bytes_per_element
        # 压缩后存储 U[:, :k], S[:k], Vh[:k, :]
        comp_bytes = (U[:, :k].numel() + S[:k].numel() + Vh[:k, :].numel()) * bytes_per_element
        compression_ratio = orig_bytes / comp_bytes if comp_bytes > 0 else 1.0

        stats = {
            "tensor": tensor_name,
            "rank": k,
            "max_rank": max_rank,
            "relative_error": relative_error,
            "energy_retained": energy_retained,
            "seq_len": original_shape[-2] if len(original_shape) >= 2 else original_shape[0],
            "head_dim": original_shape[-1],
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "compression_ratio": compression_ratio,
            "svd_time": svd_time,
            "status": "success"
        }

        return compressed, stats

    @staticmethod
    def compress_head(
        key: torch.Tensor,
        value: torch.Tensor,
        tau: float,
        preserve_last_n: int = 0,
        preserve_first_n: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        压缩一个 attention head 的 K 和 V

        Args:
            key: Key 张量 [batch, seq_len, head_dim]
            value: Value 张量 [batch, seq_len, head_dim]
            tau: 能量保留阈值
            preserve_last_n: 保留最后N个token不压缩（保护RoPE几何）
            preserve_first_n: 保留前N个token不压缩（保护few-shot示例）

        Returns:
            (压缩后的 key, 压缩后的 value, 统计信息字典)
        """
        seq_len = key.shape[1]
        total_preserve = preserve_first_n + preserve_last_n

        # 如果需要保留的 token 数量大于等于序列长度，则不压缩
        if total_preserve >= seq_len:
            # 返回原始数据，压缩比为 1.0
            bytes_per_element = 2 if key.dtype == torch.float16 else 4
            stats_k, stats_v = SVDCompressor._skipped_stats(
                seq_len, key.shape[-1], key.numel() * bytes_per_element)
            stats = {
                "K": stats_k, "V": stats_v,
                "total_svd_time": 0.0, "avg_compression_ratio": 1.0,
            }
            return key, value, stats

        # 分割序列：[前N个保留] + [压缩区域] + [后N个保留]
        k_first, k_middle, k_last = SVDCompressor._split_preserved(
            key, preserve_first_n, preserve_last_n, seq_dim=1)
        v_first, v_middle, v_last = SVDCompressor._split_preserved(
            value, preserve_first_n, preserve_last_n, seq_dim=1)

        # 对压缩区域进行 SVD 压缩
        compressed_k_middle, stats_k = SVDCompressor.compress_tensor(k_middle, tau, tensor_name="K")
        compressed_v_middle, stats_v = SVDCompressor.compress_tensor(v_middle, tau, tensor_name="V")

        # 拼接：[前N个保留] + [压缩后的中间部分] + [后N个保留]
        compressed_k = SVDCompressor._join_preserved(k_first, compressed_k_middle, k_last, seq_dim=1)
        compressed_v = SVDCompressor._join_preserved(v_first, compressed_v_middle, v_last, seq_dim=1)

        # 合并统计信息
        # 总体压缩比 = 总原始大小 / 总压缩大小
        total_orig_bytes = stats_k["orig_bytes"] + stats_v["orig_bytes"]
        total_comp_bytes = stats_k["comp_bytes"] + stats_v["comp_bytes"]
        total_compression_ratio = total_orig_bytes / total_comp_bytes if total_comp_bytes > 0 else 1.0

        stats = {
            "K": stats_k,
            "V": stats_v,
            "total_svd_time": stats_k["svd_time"] + stats_v["svd_time"],
            "avg_compression_ratio": total_compression_ratio,  # 使用真实总体压缩比
        }

        return compressed_k, compressed_v, stats

    @staticmethod
    def compress_head_shared_basis(
        key: torch.Tensor,
        value: torch.Tensor,
        tau: float,
        preserve_last_n: int = 0,
        preserve_first_n: int = 0,
        value_residual_bits: int = 0,
        qjl_seed: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        使用K的SVD基来压缩V (Attention-aware compression)

        原理：
        - K = U * Σ * Vh，选择前r个成分
        - K' = U_r * Σ_r * Vh_r
        - V' = U_r * U_r^T * V  (将V投影到K的主成分空间)

        这样保证了V的压缩是基于K的结构，被attention选中的token方向会被保留

        Args:
            key: Key 张量 [batch, seq_len, head_dim]
            value: Value 张量 [batch, seq_len, head_dim]
            tau: 能量保留阈值 (用于K)
            preserve_last_n: 保留最后N个token不压缩（保护RoPE几何）
            preserve_first_n: 保留前N个token不压缩（保护few-shot示例）
            value_residual_bits: 0 disables residual correction; 1 enables QJL
            qjl_seed: random seed for the fixed Gaussian QJL projection

        Returns:
            (压缩后的 key, 压缩后的 value, 统计信息字典)
        """
        original_shape = key.shape
        batch, seq_len, head_dim = original_shape
        device = key.device
        dtype = key.dtype
        total_preserve = preserve_first_n + preserve_last_n

        # 如果需要保留的 token 数量大于等于序列长度，则不压缩
        if total_preserve >= seq_len:
            bytes_per_element = 2 if dtype == torch.float16 else 4
            stats_k, stats_v = SVDCompressor._skipped_stats(
                seq_len, head_dim, key.numel() * bytes_per_element)
            stats = {
                "K": stats_k, "V": stats_v,
                "total_svd_time": 0.0, "avg_compression_ratio": 1.0,
            }
            return key, value, stats

        # 分割序列：[前N个保留] + [压缩区域] + [后N个保留]
        k_first, k_middle, k_last = SVDCompressor._split_preserved(
            key, preserve_first_n, preserve_last_n, seq_dim=1)
        v_first, v_middle, v_last = SVDCompressor._split_preserved(
            value, preserve_first_n, preserve_last_n, seq_dim=1)

        # 对压缩区域进行共享基压缩
        comp_k_middle, comp_v_middle, stats = SVDCompressor._compress_shared_basis_core(
            k_middle,
            v_middle,
            tau,
            value_residual_bits=value_residual_bits,
            qjl_seed=qjl_seed,
        )

        # 拼接：[前N个保留] + [压缩后的中间部分] + [后N个保留]
        compressed_k = SVDCompressor._join_preserved(k_first, comp_k_middle, k_last, seq_dim=1)
        compressed_v = SVDCompressor._join_preserved(v_first, comp_v_middle, v_last, seq_dim=1)

        return compressed_k, compressed_v, stats

    @staticmethod
    def _compress_shared_basis_core(
        key: torch.Tensor,
        value: torch.Tensor,
        tau: float,
        value_residual_bits: int = 0,
        qjl_seed: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """共享基压缩的核心实现"""
        if value_residual_bits not in (0, 1):
            raise ValueError("value_residual_bits must be 0 or 1")

        original_shape = key.shape
        device = key.device
        dtype = key.dtype

        # Reshape为矩阵: [batch * seq_len, head_dim]
        k_matrix = key.reshape(-1, original_shape[-1]).float()
        v_matrix = value.reshape(-1, original_shape[-1]).float()

        start = time.perf_counter()

        # 只对K做SVD
        U, S, Vh = torch.linalg.svd(k_matrix, full_matrices=False)

        # 基于K的能量选择rank
        total_energy = (S ** 2).sum()
        cumsum_energy = torch.cumsum(S ** 2, dim=0)
        r = torch.searchsorted(cumsum_energy, tau * total_energy).item() + 1
        r = max(1, min(r, len(S)))

        # 压缩K: K' = U_r * Σ_r * Vh_r
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        k_compressed = U_r @ torch.diag(S_r) @ Vh_r

        # 用K的基压缩V: V' = U_r * U_r^T * V
        # 这相当于把V投影到K的前r个主成分空间
        v_projected = U_r @ (U_r.T @ v_matrix)

        svd_time = time.perf_counter() - start

        projection_residual = v_matrix - v_projected
        residual_stats = {
            "enabled": False,
            "bits": 0,
            "seed": qjl_seed,
            "code_bytes": 0,
            "norm_bytes": 0,
            "comp_bytes": 0,
        }
        residual_time = 0.0
        v_reconstructed = v_projected
        if value_residual_bits == 1:
            residual_start = time.perf_counter()
            residual_reconstructed, residual_stats = (
                SVDCompressor._reconstruct_qjl_residual(
                    projection_residual,
                    storage_dtype=dtype,
                    seed=qjl_seed,
                )
            )
            v_reconstructed = v_projected + residual_reconstructed
            residual_time = time.perf_counter() - residual_start

        # Reshape回原始形状
        compressed_k = k_compressed.reshape(original_shape).to(dtype).to(device)
        compressed_v = v_reconstructed.reshape(original_shape).to(dtype).to(device)

        # 统计信息
        max_rank = len(S)
        energy_retained_k = (cumsum_energy[r-1] / total_energy).item()

        # K的相对误差
        k_error = torch.norm(k_matrix - k_compressed)
        k_norm = torch.norm(k_matrix)
        k_relative_error = (k_error / k_norm).item() if k_norm > 0 else 0.0

        # V的投影误差与最终重构误差
        v_projection_error = torch.norm(projection_residual)
        v_error = torch.norm(v_matrix - v_reconstructed)
        v_norm = torch.norm(v_matrix)
        v_projection_relative_error = (
            (v_projection_error / v_norm).item() if v_norm > 0 else 0.0
        )
        v_relative_error = (v_error / v_norm).item() if v_norm > 0 else 0.0

        # 计算V在K的主成分空间中保留的能量比例
        v_projected_energy = torch.norm(v_projected) ** 2
        v_total_energy = torch.norm(v_matrix) ** 2
        v_energy_retained = (v_projected_energy / v_total_energy).item() if v_total_energy > 0 else 1.0

        # 内存计算
        bytes_per_element = 2 if dtype == torch.float16 else 4
        orig_bytes_k = key.numel() * bytes_per_element
        orig_bytes_v = value.numel() * bytes_per_element
        # 压缩存储: U_r, S_r, Vh_r (K用), U_r 和 V_latent (V用，V_latent = U_r^T @ V)
        # K: U_r [m, r] + S_r [r] + Vh_r [r, n]
        # V: 共用U_r + V_latent [r, n]
        m, n = k_matrix.shape
        comp_bytes_k = (m * r + r + r * n) * bytes_per_element
        # V stores its shared-basis latent plus the optional packed residual code.
        comp_bytes_v = (r * n) * bytes_per_element + residual_stats["comp_bytes"]

        stats_k = {
            "tensor": "K",
            "rank": r,
            "max_rank": max_rank,
            "relative_error": k_relative_error,
            "energy_retained": energy_retained_k,
            "seq_len": original_shape[-2],
            "head_dim": original_shape[-1],
            "orig_bytes": orig_bytes_k,
            "comp_bytes": comp_bytes_k,
            "compression_ratio": orig_bytes_k / comp_bytes_k if comp_bytes_k > 0 else 1.0,
            "svd_time": svd_time,
            "status": "success"
        }

        stats_v = {
            "tensor": "V",
            "rank": r,  # 使用K的rank
            "max_rank": max_rank,
            "relative_error": v_relative_error,
            "projection_relative_error": v_projection_relative_error,
            "energy_retained": v_energy_retained,  # V在K空间中保留的能量
            "seq_len": original_shape[-2],
            "head_dim": original_shape[-1],
            "orig_bytes": orig_bytes_v,
            "comp_bytes": comp_bytes_v,
            "compression_ratio": orig_bytes_v / comp_bytes_v if comp_bytes_v > 0 else 1.0,
            "svd_time": 0.0,  # V不需要单独做SVD
            "residual_time": residual_time,
            "residual_correction": residual_stats,
            "status": "success"
        }

        # 总体压缩比 = 总原始大小 / 总压缩大小
        total_orig_bytes = orig_bytes_k + orig_bytes_v
        total_comp_bytes = comp_bytes_k + comp_bytes_v
        total_compression_ratio = total_orig_bytes / total_comp_bytes if total_comp_bytes > 0 else 1.0

        stats = {
            "K": stats_k,
            "V": stats_v,
            "total_svd_time": svd_time + residual_time,
            "residual_time": residual_time,
            "avg_compression_ratio": total_compression_ratio,  # 使用真实总体压缩比
        }

        return compressed_k, compressed_v, stats

    @staticmethod
    def compress_layer_shared_basis_eigh(
        key: torch.Tensor,
        value: torch.Tensor,
        taus,
        preserve_last_n: int = 0,
        preserve_first_n: int = 0,
        value_residual_bits: int = 0,
        qjl_seed: int = 0,
        return_factors: bool = False,
        eigh_compute_device: str = "cpu",
    ):
        """Batched-eigh fast path for shared-basis compression (whole layer at once).

        Mathematically equivalent to looping ``compress_head_shared_basis`` over the
        heads, but replaces ``num_heads`` per-head SVDs of ``[seq, head_dim]`` with a
        single batched eigendecomposition of the ``[num_heads, head_dim, head_dim]``
        Gram matrices ``K^T K``.  For ``K = U S V^T`` we have ``K^T K = V S^2 V^T``,
        so the eigenvectors equal the right singular vectors and the eigenvalues equal
        the squared singular values:

            rank r        : same energy rule on the eigenvalue spectrum
            K'            = K (V_r V_r^T)                       == U_r S_r V_r^T
            V_proj        = K V_r diag(1/lambda_r) V_r^T (K^T V) == U_r U_r^T V

        The 1-bit residual sketch is unchanged (shared Gaussian projection per seed),
        applied row-wise to all heads in one call.  Returns the same per-head stats
        structure as the looped path.

        key, value: [batch, num_heads, seq_len, head_dim]
        taus:       per-head tau, length num_heads
        return_factors: return the compact factor bundle instead of dense K/V
        eigh_compute_device: "cpu" or the Gram tensor's input device
        Returns: (compressed_key, compressed_value, heads_stats, layer_svd_time, factors)
        """
        if value_residual_bits not in (0, 1):
            raise ValueError("value_residual_bits must be 0 or 1")

        batch, num_heads, seq_len, head_dim = key.shape
        device = key.device
        dtype = key.dtype
        bytes_per_element = 2 if dtype == torch.float16 else 4
        total_preserve = preserve_first_n + preserve_last_n

        # Nothing to compress: mirror the looped path's "skipped" stats.
        if total_preserve >= seq_len:
            heads_stats = []
            ob = batch * seq_len * head_dim * bytes_per_element
            for h in range(num_heads):
                sk, sv = SVDCompressor._skipped_stats(seq_len, head_dim, ob)
                heads_stats.append({"head": h, "tau": float(taus[h]), "K": sk, "V": sv})
            return key, value, heads_stats, 0.0, None

        # Split [first | middle | last]; only the middle is compressed.
        k_first, k_mid, k_last = SVDCompressor._split_preserved(
            key, preserve_first_n, preserve_last_n, seq_dim=2)
        v_first, v_mid, v_last = SVDCompressor._split_preserved(
            value, preserve_first_n, preserve_last_n, seq_dim=2)

        H = num_heads
        D = head_dim
        mid_seq = k_mid.shape[2]
        M = batch * mid_seq

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Fold batch into the row dimension: [H, M, D]
        Kf = k_mid.permute(1, 0, 2, 3).reshape(H, M, D).float()
        Vf = v_mid.permute(1, 0, 2, 3).reshape(H, M, D).float()

        # Batched Gram + symmetric eigendecomposition.
        G = Kf.transpose(-1, -2) @ Kf                      # [H, D, D]
        w, vecs = SVDCompressor._eigh_batched(G, eigh_compute_device)
        w_desc = w.flip(-1).clamp_min(0.0)                 # [H, D]
        vecs_desc = vecs.flip(-1)                          # [H, D, D]

        # Per-head rank by cumulative-energy rule (matches the SVD path on S**2).
        cs = torch.cumsum(w_desc, dim=-1)                  # [H, D]
        total = cs[:, -1:]                                 # [H, 1]
        taus_t = torch.as_tensor(taus, device=device, dtype=torch.float32).reshape(H, 1)
        r = (cs < taus_t * total).sum(dim=-1) + 1          # [H]
        r = r.clamp(1, D)
        idx = torch.arange(D, device=device).reshape(1, D)
        keep = (idx < r.reshape(H, 1)).to(Kf.dtype)        # [H, D]

        # K' = K (V_r V_r^T)
        Vr_scaled = vecs_desc * keep.unsqueeze(1)          # zero dropped columns
        P = Vr_scaled @ vecs_desc.transpose(-1, -2)        # [H, D, D]
        Kp = Kf @ P                                        # [H, M, D]

        # V_proj = K V_r diag(1/lambda_r) V_r^T (K^T V); invert kept eigenvalues only.
        wmax = w_desc[:, :1]
        invw = keep / torch.clamp(w_desc, min=wmax * 1e-7 + 1e-20)  # [H, D]
        KtV = Kf.transpose(-1, -2) @ Vf                    # [H, D, D]
        T = vecs_desc @ (invw.unsqueeze(-1) * (vecs_desc.transpose(-1, -2) @ KtV))
        Vproj = Kf @ T                                     # [H, M, D]


        # 1-bit residual sketch, batched across all heads (shared projection).
        sketch_code = None
        if value_residual_bits == 1:
            E = (Vf - Vproj).reshape(H * M, D)
            residual_recon, resid_meta = SVDCompressor._reconstruct_qjl_residual(
                E, storage_dtype=dtype, seed=qjl_seed, return_code=return_factors
            )
            sketch_code = resid_meta.pop("_code", None)
            Vrecon = Vproj + residual_recon.reshape(H, M, D)
        else:
            Vrecon = Vproj

        if device.type == "cuda":
            torch.cuda.synchronize()
        layer_svd_time = time.perf_counter() - start

        # Per-head stats: batched norms, single host sync.
        eps = 1e-12
        k_norm = torch.linalg.vector_norm(Kf, dim=(1, 2))
        v_norm = torch.linalg.vector_norm(Vf, dim=(1, 2))
        ke = (torch.linalg.vector_norm(Kf - Kp, dim=(1, 2)) / k_norm.clamp_min(eps))
        vpe = (torch.linalg.vector_norm(Vf - Vproj, dim=(1, 2)) / v_norm.clamp_min(eps))
        ve = (torch.linalg.vector_norm(Vf - Vrecon, dim=(1, 2)) / v_norm.clamp_min(eps))
        ven = ((torch.linalg.vector_norm(Vproj, dim=(1, 2)) ** 2) / (v_norm ** 2).clamp_min(eps))
        erk = torch.gather(cs, 1, (r - 1).reshape(H, 1)).squeeze(1) / total.squeeze(1)

        r_l, ke_l, vpe_l, ve_l, ven_l, erk_l = (
            r.tolist(), ke.tolist(), vpe.tolist(), ve.tolist(), ven.tolist(), erk.tolist()
        )

        if value_residual_bits == 1:
            packed = (M * D + 7) // 8
            norm_bytes = M * bytes_per_element
            resid_comp_bytes = packed + norm_bytes
            resid_corr = {"enabled": True, "bits": 1, "seed": qjl_seed,
                          "code_bytes": packed, "norm_bytes": norm_bytes,
                          "comp_bytes": resid_comp_bytes}
        else:
            resid_comp_bytes = 0
            resid_corr = {"enabled": False, "bits": 0, "seed": qjl_seed,
                          "code_bytes": 0, "norm_bytes": 0, "comp_bytes": 0}

        orig_bytes = M * D * bytes_per_element
        heads_stats = []
        for h in range(H):
            rk = r_l[h]
            comp_bytes_k = (M * rk + rk + rk * D) * bytes_per_element
            comp_bytes_v = (rk * D) * bytes_per_element + resid_comp_bytes
            sk = {
                "tensor": "K", "rank": rk, "max_rank": D,
                "relative_error": ke_l[h], "energy_retained": erk_l[h],
                "seq_len": mid_seq, "head_dim": D,
                "orig_bytes": orig_bytes, "comp_bytes": comp_bytes_k,
                "compression_ratio": orig_bytes / comp_bytes_k if comp_bytes_k > 0 else 1.0,
                "svd_time": layer_svd_time / H, "status": "success",
            }
            sv = {
                "tensor": "V", "rank": rk, "max_rank": D,
                "relative_error": ve_l[h], "projection_relative_error": vpe_l[h],
                "energy_retained": ven_l[h], "seq_len": mid_seq, "head_dim": D,
                "orig_bytes": orig_bytes, "comp_bytes": comp_bytes_v,
                "compression_ratio": orig_bytes / comp_bytes_v if comp_bytes_v > 0 else 1.0,
                "svd_time": 0.0, "residual_time": 0.0,
                "residual_correction": resid_corr, "status": "success",
            }
            heads_stats.append({"head": h, "tau": float(taus[h]), "K": sk, "V": sv})

        if return_factors:
            # K = U Sigma Psi^T, so U = K Psi Sigma^{-1}; dropped columns are zeroed
            # by `keep`, which makes padding every head to the layer-wide r_max exact.
            sigma = torch.sqrt(torch.clamp(w_desc, min=wmax * 1e-7 + 1e-20))   # [H, D]
            U_full = ((Kf @ vecs_desc) / sigma.unsqueeze(1)) * keep.unsqueeze(1)
            sp_full = (sigma * keep).unsqueeze(-1) * vecs_desc.transpose(-1, -2)
            # Stored per head at its own rank: padding every head to the layer
            # maximum would occupy more than the r_h that Eq. (60) charges.
            ranks = r_l
            U_r = [U_full[h, :, :ranks[h]].contiguous() for h in range(H)]
            sigma_psi_t = [sp_full[h, :ranks[h], :].contiguous() for h in range(H)]
            C_V = [(U_r[h].transpose(-1, -2) @ Vf[h]).contiguous() for h in range(H)]
            # .contiguous() on the tails: they are slices of the original key /
            # value tensors, and holding a view keeps the whole prefix allocation
            # alive even though only the last few tokens are wanted.
            factors = {
                "batch": batch, "mid_seq": mid_seq, "num_heads": H, "head_dim": D,
                "U_r": [t.to(dtype) for t in U_r],
                "sigma_psi_t": [t.to(dtype) for t in sigma_psi_t],
                "C_V": [t.to(dtype) for t in C_V], "rank": ranks,
                "sketch": sketch_code, "qjl_seed": qjl_seed,
                "k_post": None if k_last is None else k_last.contiguous(),
                "v_post": None if v_last is None else v_last.contiguous(),
                "k_pre": None if k_first is None else k_first.contiguous(),
                "v_pre": None if v_first is None else v_first.contiguous(),
            }
            # No dense tensors are returned or reassembled: Kp / Vrecon existed
            # only for the per-head error statistics above and are freed here,
            # so at most one layer's worth is ever transient.
            return None, None, heads_stats, layer_svd_time, factors

        # Reassemble [first | middle | last] along the sequence axis.
        Kp_b = Kp.reshape(H, batch, mid_seq, D).permute(1, 0, 2, 3).to(dtype)
        Vr_b = Vrecon.reshape(H, batch, mid_seq, D).permute(1, 0, 2, 3).to(dtype)
        compressed_k = SVDCompressor._join_preserved(k_first, Kp_b, k_last, seq_dim=2)
        compressed_v = SVDCompressor._join_preserved(v_first, Vr_b, v_last, seq_dim=2)

        return compressed_k, compressed_v, heads_stats, layer_svd_time, None

    @staticmethod
    def compress_layer_independent_eigh(
        key: torch.Tensor,
        value: torch.Tensor,
        taus,
        preserve_last_n: int = 0,
        preserve_first_n: int = 0,
        eigh_compute_device: str = "cpu",
    ):
        """Batched-eigh fast path for INDEPENDENT compression (whole layer at once).

        Mathematically equivalent to looping ``compress_head`` (per-head independent
        SVD) over the heads, but replaces the ``2 * num_heads`` per-head SVDs of
        ``[seq, head_dim]`` with two batched eigendecompositions of the
        ``[num_heads, head_dim, head_dim]`` Gram matrices ``K^T K`` and ``V^T V``.

        Unlike shared-basis, K and V each get their OWN basis and OWN rank (chosen
        from their own energy spectrum). No 1-bit residual (matches ``compress_head``,
        which the reference independent path uses). For ``X = U S W^T`` we have
        ``X^T X = W S^2 W^T``, so eigenvectors == right singular vectors and
        eigenvalues == squared singular values, and ``X' = X (W_r W_r^T)`` reproduces
        the rank-r truncated SVD reconstruction ``U_r S_r W_r^T``.

        key, value: [batch, num_heads, seq_len, head_dim]
        taus:       per-head tau, length num_heads
        Returns: (compressed_key, compressed_value, heads_stats, layer_svd_time)
        """
        batch, num_heads, seq_len, head_dim = key.shape
        device = key.device
        dtype = key.dtype
        bytes_per_element = 2 if dtype == torch.float16 else 4
        total_preserve = preserve_first_n + preserve_last_n

        # Nothing to compress: mirror the looped path's "skipped" stats.
        if total_preserve >= seq_len:
            heads_stats = []
            ob = batch * seq_len * head_dim * bytes_per_element
            for h in range(num_heads):
                sk, sv = SVDCompressor._skipped_stats(seq_len, head_dim, ob)
                heads_stats.append({"head": h, "tau": float(taus[h]), "K": sk, "V": sv})
            return key, value, heads_stats, 0.0

        # Split [first | middle | last]; only the middle is compressed.
        k_first, k_mid, k_last = SVDCompressor._split_preserved(
            key, preserve_first_n, preserve_last_n, seq_dim=2)
        v_first, v_mid, v_last = SVDCompressor._split_preserved(
            value, preserve_first_n, preserve_last_n, seq_dim=2)

        H = num_heads
        D = head_dim
        mid_seq = k_mid.shape[2]
        M = batch * mid_seq

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Fold batch into the row dimension: [H, M, D]
        Kf = k_mid.permute(1, 0, 2, 3).reshape(H, M, D).float()
        Vf = v_mid.permute(1, 0, 2, 3).reshape(H, M, D).float()
        taus_t = torch.as_tensor(taus, device=device, dtype=torch.float32).reshape(H, 1)
        idx = torch.arange(D, device=device).reshape(1, D)

        def _project(Xf):
            # Batched low-rank projection X (W_r W_r^T) via eigh of the Gram X^T X.
            # Returns (Xp [H,M,D], rank r [H], energy_retained erk [H]).
            G = Xf.transpose(-1, -2) @ Xf                  # [H, D, D]
            w, vecs = SVDCompressor._eigh_batched(G, eigh_compute_device)
            w_desc = w.flip(-1).clamp_min(0.0)             # [H, D]
            vecs_desc = vecs.flip(-1)                      # [H, D, D]
            cs = torch.cumsum(w_desc, dim=-1)              # [H, D]
            total = cs[:, -1:]                             # [H, 1]
            r = (cs < taus_t * total).sum(dim=-1) + 1      # [H]  (matches searchsorted+1)
            r = r.clamp(1, D)
            keep = (idx < r.reshape(H, 1)).to(Xf.dtype)    # [H, D]
            Wr = vecs_desc * keep.unsqueeze(1)             # zero dropped columns
            P = Wr @ vecs_desc.transpose(-1, -2)           # [H, D, D] = W_r W_r^T
            Xp = Xf @ P                                    # [H, M, D]
            erk = torch.gather(cs, 1, (r - 1).reshape(H, 1)).squeeze(1) / total.squeeze(1)
            return Xp, r, erk

        Kp, rK, erkK = _project(Kf)
        Vp, rV, erkV = _project(Vf)

        if device.type == "cuda":
            torch.cuda.synchronize()
        layer_svd_time = time.perf_counter() - start

        # Per-head stats: batched norms, single host sync.
        eps = 1e-12
        k_norm = torch.linalg.vector_norm(Kf, dim=(1, 2))
        v_norm = torch.linalg.vector_norm(Vf, dim=(1, 2))
        keK = (torch.linalg.vector_norm(Kf - Kp, dim=(1, 2)) / k_norm.clamp_min(eps))
        veV = (torch.linalg.vector_norm(Vf - Vp, dim=(1, 2)) / v_norm.clamp_min(eps))

        rK_l, rV_l = rK.tolist(), rV.tolist()
        keK_l, veV_l = keK.tolist(), veV.tolist()
        erkK_l, erkV_l = erkK.tolist(), erkV.tolist()

        orig_bytes = M * D * bytes_per_element
        per_tensor_time = layer_svd_time / (2 * H)
        heads_stats = []
        for h in range(H):
            rk, rv = rK_l[h], rV_l[h]
            comp_bytes_k = (M * rk + rk + rk * D) * bytes_per_element
            comp_bytes_v = (M * rv + rv + rv * D) * bytes_per_element
            sk = {
                "tensor": "K", "rank": rk, "max_rank": D,
                "relative_error": keK_l[h], "energy_retained": erkK_l[h],
                "seq_len": mid_seq, "head_dim": D,
                "orig_bytes": orig_bytes, "comp_bytes": comp_bytes_k,
                "compression_ratio": orig_bytes / comp_bytes_k if comp_bytes_k > 0 else 1.0,
                "svd_time": per_tensor_time, "status": "success",
            }
            sv = {
                "tensor": "V", "rank": rv, "max_rank": D,
                "relative_error": veV_l[h], "energy_retained": erkV_l[h],
                "seq_len": mid_seq, "head_dim": D,
                "orig_bytes": orig_bytes, "comp_bytes": comp_bytes_v,
                "compression_ratio": orig_bytes / comp_bytes_v if comp_bytes_v > 0 else 1.0,
                "svd_time": per_tensor_time, "status": "success",
            }
            heads_stats.append({"head": h, "tau": float(taus[h]), "K": sk, "V": sv})

        # Reassemble [first | middle | last] along the sequence axis.
        Kp_b = Kp.reshape(H, batch, mid_seq, D).permute(1, 0, 2, 3).to(dtype)
        Vp_b = Vp.reshape(H, batch, mid_seq, D).permute(1, 0, 2, 3).to(dtype)
        compressed_k = SVDCompressor._join_preserved(k_first, Kp_b, k_last, seq_dim=2)
        compressed_v = SVDCompressor._join_preserved(v_first, Vp_b, v_last, seq_dim=2)

        return compressed_k, compressed_v, heads_stats, layer_svd_time
