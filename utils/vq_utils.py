import os
import glob
import numpy as np
import torch
import torch.nn as nn
from collections.abc import Iterable
from sklearn.cluster import MiniBatchKMeans

# Optional: bring in crop-level codec to decode full maps when needed
CODEC_AVAILABLE = False
try:
    import sys
    _here = os.path.dirname(__file__)
    _codec_dir = os.path.abspath(os.path.join(_here, '..', 'LLaVA-NeXT'))
    if os.path.isdir(_codec_dir) and _codec_dir not in sys.path:
        sys.path.insert(0, _codec_dir)
    from crop_feature_codec import CropFeatureCodec  # noqa: E402
    CODEC_AVAILABLE = True
except Exception:
    # Keep running without codec; callers using crop-level features will error if decode is required
    CODEC_AVAILABLE = False


def softmax_to_topk_soft_code(logits, k):
    """
    Sparse Coefficient
    """
    # Apply softmax to get probabilities
    y_soft = logits.softmax(dim=1)  # [batch_size, K]

    values, indices = torch.topk(y_soft, k, dim=1)
    mask = torch.zeros_like(y_soft, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    zero_tensor = torch.full_like(y_soft, 0)
    y_soft_topk = torch.where(mask, y_soft, zero_tensor)
    y_soft_topk = y_soft_topk / (y_soft_topk.sum(dim=1).unsqueeze(1) + 1e-10)
    soft_code_topk = y_soft_topk

    return soft_code_topk

def get_weights_and_indices(logits, k):
    # Apply softmax to get probabilities
    y_soft = logits.softmax(dim=1)  # [batch_size, K]
    values, indices = torch.topk(y_soft, k, dim=1)
    mask = torch.zeros_like(y_soft, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    zero_tensor = torch.full_like(y_soft, 0)
    y_soft_topk = torch.where(mask, y_soft, zero_tensor)
    y_soft_topk = y_soft_topk / (y_soft_topk.sum(dim=1).unsqueeze(1) + 1e-10)
    soft_code_topk = y_soft_topk
    non_zero_mask = soft_code_topk != 0
    weights = soft_code_topk[non_zero_mask].view(soft_code_topk.shape[0], k)
    indices = torch.arange(y_soft_topk.shape[1]).expand_as(soft_code_topk)[non_zero_mask].view(soft_code_topk.shape[0], k)

    return weights.float(), indices.float()


class ResidualVectorQuantizationWithClustering(nn.Module):
    def __init__(self, num_levels, num_clusters, feature_dim, device):
        super(ResidualVectorQuantizationWithClustering, self).__init__()
        self.num_levels = num_levels
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.device = device
        # Store the quantizers for each level
        self.quantizers = []

    def _iter_feature_blocks(self, features):
        """Yield feature batches as contiguous float32 numpy arrays."""
        if torch.is_tensor(features):
            yield np.ascontiguousarray(features.detach().cpu().numpy(), dtype=np.float32)
            return

        if isinstance(features, np.ndarray):
            yield np.ascontiguousarray(features, dtype=np.float32)
            return

        if isinstance(features, LMMFeatureStream):
            for block in features:
                yield np.ascontiguousarray(block, dtype=np.float32)
            return

        if isinstance(features, str):
            raise TypeError("`features` cannot be a string path; pass an iterable of arrays instead.")

        if isinstance(features, Iterable):
            for block in features:
                if torch.is_tensor(block):
                    numpy_block = block.detach().cpu().numpy()
                else:
                    numpy_block = np.asarray(block)
                if numpy_block.ndim == 0:
                    continue
                if numpy_block.ndim == 1:
                    numpy_block = numpy_block.reshape(1, -1)
                yield np.ascontiguousarray(numpy_block, dtype=np.float32)
            return

        raise TypeError(
            "Unsupported feature container type. Expected numpy array, torch tensor, "
            "LMMFeatureStream, or iterable of arrays."
        )

    def _infer_total_samples(self, features):
        if torch.is_tensor(features) or isinstance(features, np.ndarray):
            return int(features.shape[0])

        if isinstance(features, LMMFeatureStream):
            return features.total_features()

        if isinstance(features, (list, tuple)):
            total = 0
            for block in features:
                if torch.is_tensor(block):
                    total += int(block.shape[0])
                else:
                    array = np.asarray(block)
                    if array.ndim == 0:
                        continue
                    total += int(array.shape[0])
            return total

        return None

    def _infer_total_blocks(self, features):
        if torch.is_tensor(features) or isinstance(features, np.ndarray):
            return 1

        if isinstance(features, LMMFeatureStream):
            return len(features)

        if isinstance(features, (list, tuple)):
            return len(features)

        try:
            return len(features)
        except TypeError:
            return None

    def fit_quantizers(self, features):
        """
        Perform clustering to initialize quantizers for each level.

        For the common single-level RVQ (`num_levels == 1`), this routine streams feature
        blocks through ``MiniBatchKMeans.partial_fit`` so that only a small window of data
        is resident in memory at any time.
        """

        self.quantizers = []

        if self.num_levels == 1:
            self._fit_single_level(features)
            return

        # Fallback to the previous multi-level behaviour (materialises features).
        materialised = []
        for block in self._iter_feature_blocks(features):
            materialised.append(block)
        if not materialised:
            raise ValueError("No features provided to fit_quantizers")
        residuals = np.concatenate(materialised, axis=0)

        for level in range(self.num_levels):
            print(
                f"Level {level}: clustering {residuals.shape[0]} points with {self.num_clusters} clusters"
            )
            kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, batch_size=10000, verbose=1)
            kmeans.fit(residuals)
            self.quantizers.append(
                torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)
            )
            quantized = self._quantize_with_centers(residuals, kmeans.cluster_centers_).cpu().numpy()
            residuals = residuals - quantized
            print(f"Level {level} done, residual norm: {np.linalg.norm(residuals, axis=1).mean():.4f}")

    def _fit_single_level(self, features):
        """Stream a single-level quantizer fit using MiniBatchKMeans."""

        kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, batch_size=10000, verbose=1)
        buffer = []
        buffer_count = 0
        fitted = False
        samples_seen = 0
        blocks_seen = 0
        total_samples_hint = self._infer_total_samples(features)
        total_blocks_hint = self._infer_total_blocks(features)

        if total_samples_hint is not None:
            print(
                f"Streaming MiniBatchKMeans initialisation over {total_samples_hint} samples"
            )
        elif total_blocks_hint is not None:
            print(
                f"Streaming MiniBatchKMeans initialisation across {total_blocks_hint} blocks"
            )

        def log_progress():
            if samples_seen == 0 and blocks_seen == 0:
                return

            parts = []
            if total_samples_hint:
                percent = min(100.0, (samples_seen / total_samples_hint) * 100)
                parts.append(
                    f"{samples_seen}/{total_samples_hint} samples ({percent:.2f}%)"
                )
            else:
                parts.append(f"{samples_seen} samples")

            if total_blocks_hint:
                parts.append(f"{blocks_seen}/{total_blocks_hint} blocks")
            else:
                parts.append(f"{blocks_seen} blocks")

            print("  Progress: " + "; ".join(parts))

        for block in self._iter_feature_blocks(features):
            blocks_seen += 1
            if block.size == 0:
                continue
            if block.ndim != 2:
                block = block.reshape(block.shape[0], -1)

            block_len = block.shape[0]

            if not fitted:
                offset = 0

                while offset < block_len:
                    needed = self.num_clusters - buffer_count
                    take_count = min(needed, block_len - offset)
                    if take_count <= 0:
                        break

                    slice_ = block[offset : offset + take_count]
                    buffer.append(slice_)
                    buffer_count += take_count
                    offset += take_count

                    if buffer_count >= self.num_clusters:
                        init_batch = (
                            buffer[0]
                            if len(buffer) == 1
                            else np.concatenate(buffer, axis=0)
                        )
                        kmeans.partial_fit(init_batch)
                        fitted = True
                        buffer.clear()
                        buffer_count = 0

                        if offset < block_len:
                            remainder = block[offset:]
                            if remainder.size:
                                kmeans.partial_fit(remainder)
                        break

                if not fitted:
                    samples_seen += block_len
                    log_progress()
                    continue
            else:
                kmeans.partial_fit(block)

            samples_seen += block_len
            log_progress()

        if not fitted:
            raise ValueError(
                "Insufficient samples to initialise MiniBatchKMeans: "
                f"seen {samples_seen}, need at least {self.num_clusters}."
            )

        print(
            f"Fitted single-level quantizer on {samples_seen} samples with {self.num_clusters} clusters"
        )
        self.quantizers.append(
            torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)
        )

    def _quantize_with_centers(self, data, centers, batch_size=2048):
      """
      Batched quantization: process `data` in chunks to avoid huge distance matrices.
      `data` and `centers` can be numpy arrays or torch tensors. Returns a tensor on self.device.

      关键改进：不一次性将所有data加载到GPU，而是分批加载和处理
      """
      # 首先处理centers - centers通常较小，可以一次性加载到GPU
      if isinstance(centers, np.ndarray):
        centers_tensor = torch.from_numpy(centers).to(self.device)
      elif torch.is_tensor(centers):
        centers_tensor = centers.to(self.device)
      else:
        centers_tensor = torch.tensor(centers, device=self.device)

      # 确定data的形状和类型，但不立即全部加载到GPU
      if isinstance(data, np.ndarray):
        n = data.shape[0]
        data_is_numpy = True
        data_cpu = data
      elif torch.is_tensor(data):
        n = data.shape[0]
        data_is_numpy = False
        data_cpu = data.cpu() if data.is_cuda else data
      else:
        data_cpu = torch.tensor(data)
        n = data_cpu.shape[0]
        data_is_numpy = False

      if n == 0:
        return torch.empty((0, centers_tensor.shape[1]), device=self.device)

      num_centers = centers_tensor.shape[0]

      # 动态调整batch_size，避免OOM
      # 估算距离矩阵大小：batch_size * num_centers * 4 bytes (float32)
      # 设置最大内存使用为1GB（更保守）
      max_memory_mb = 1024
      max_elements = (max_memory_mb * 1024 * 1024) // 4
      safe_batch_size = min(batch_size, max_elements // max(num_centers, 1))
      safe_batch_size = max(1, safe_batch_size)  # 至少为1

      print(f"Quantizing {n} points with {num_centers} centers, using batch_size={safe_batch_size}")

      # 分批处理data，每次只加载一个batch到GPU
      quantized_chunks = []
      for start in range(0, n, safe_batch_size):
        end = min(start + safe_batch_size, n)

        # 只加载当前batch到GPU
        if data_is_numpy:
          chunk = torch.from_numpy(data_cpu[start:end]).to(self.device)
        else:
          chunk = data_cpu[start:end].to(self.device)

        # 如果centers数量仍然很大，进一步分批处理centers
        if num_centers > 10000:
          center_batch_size = 5000
          min_distances = None
          best_indices = None

          for c_start in range(0, num_centers, center_batch_size):
            c_end = min(c_start + center_batch_size, num_centers)
            center_chunk = centers_tensor[c_start:c_end]

            # 计算当前chunk到当前center_chunk的距离
            distances = torch.cdist(chunk, center_chunk, p=2)  # [b, c_batch]
            min_dist_chunk, min_idx_chunk = distances.min(dim=1)  # [b]

            # 调整索引到全局
            min_idx_chunk = min_idx_chunk + c_start

            if min_distances is None:
              min_distances = min_dist_chunk
              best_indices = min_idx_chunk
            else:
              # 更新最小距离和对应的索引
              mask = min_dist_chunk < min_distances
              min_distances[mask] = min_dist_chunk[mask]
              best_indices[mask] = min_idx_chunk[mask]

            # 清理中间变量
            del distances, min_dist_chunk, min_idx_chunk, center_chunk
            torch.cuda.empty_cache()

          indices = best_indices
        else:
          # centers数量不大，直接计算
          distances = torch.cdist(chunk, centers_tensor, p=2)  # [b, k]
          indices = distances.argmin(dim=1)  # [b]
          del distances

        quantized_chunk = centers_tensor[indices]  # [b, dim]
        # 将结果移回CPU以节省GPU内存
        quantized_chunks.append(quantized_chunk.cpu())

        # 清理内存
        del chunk, indices, quantized_chunk
        torch.cuda.empty_cache()

        if (start // safe_batch_size) % 10 == 0:
          print(f"  Processed {end}/{n} points")

      # 在CPU上拼接所有结果
      # 注意：返回CPU上的tensor，避免一次性加载大量数据到GPU
      quantized_data_cpu = torch.cat(quantized_chunks, dim=0)

      return quantized_data_cpu  # 返回CPU tensor，由调用者决定是否需要移到GPU

    def forward(self, features):
        residuals = features
        quantized_outputs = []
        quantization_indices = []

        for level, centers in enumerate(self.quantizers):
            # Calculate distances to each cluster center and get the closest one
            print(level)
            print(torch.norm(centers, dim=1))
            print(torch.norm(residuals, dim=1).mean())
            distances = torch.cdist(residuals, centers, p=2)
            indices = distances.argmin(dim=1)
            # Retrieve quantized values based on closest centers
            quantized = centers[indices]
            # Store the quantized output and indices for each level
            quantized_outputs.append(quantized)
            quantization_indices.append(indices)
            # Update residuals for the next level
            residuals = residuals - quantized        
        quantized_result = sum(quantized_outputs)

        return quantized_result, quantization_indices

def load_2d_language_feature(data_dir, device):
    """
    Load language feature from 2D images
    """
    data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
    for i in range(len(data_names)):
        features = np.load(data_names[i])
        if i == 0:
            data = features
        else:
            data = np.concatenate([data, features], axis=0)
    data = torch.from_numpy(data).to(device) # 对于teatime 一共才[39065,512]这还是multi-scale的。但是对于我们的方法，every pixel diff。how to handle it?（当然他的pixel数量是少的，没准也是可以的）
    
    return data #[N, 512]

class LMMFeatureStream:
    """Iterate over 2D LMM feature tensors without loading everything into RAM."""

    def __init__(self, data_dir, load_func="large"):
        self.data_dir = data_dir
        self.load_func = load_func
        self.data_names = sorted(glob.glob(os.path.join(data_dir, "*.pth")))
        if not self.data_names:
            raise FileNotFoundError(f"No .pth feature files found under {data_dir}")
        self._cached_first = None
        self._block_lengths = None
        self._total_features = None

    def __len__(self):
        return len(self.data_names)

    def _load_feature_block(self, path):
        file = torch.load(path, map_location="cpu")

        # 兼容两种格式：字典格式和直接张量格式
        if isinstance(file, torch.Tensor):
            # 直接是张量格式 [N, C] 或 [H, W, C]
            features = file.numpy()
            if len(features.shape) == 3:
                # [H, W, C] 格式，展平为 [H*W, C]
                H, W, C = features.shape
                features = features.reshape(H * W, C)
            return np.ascontiguousarray(features, dtype=np.float32)

        # 字典格式（仅支持新格式：'feature_maps'，crop-level 存储）
        if not isinstance(file, dict):
            raise ValueError(f"Unsupported file format: {type(file)} at {path}")

        if "feature_maps" not in file:
            raise ValueError("Feature file missing 'feature_maps' (expected crop-level storage)")

        def _extract_by_scale(features_container, scale_name: str):
            """从 crop-level 存储解码并提取有效特征（避免interpolation和averaging）。

            如果缺少指定尺度（例如某些文件没有"Large"），返回 None，让调用方决定跳过。
            """
            if scale_name not in features_container:
                return None  # 缺失该尺度，调用方将跳过

            scale_data = features_container[scale_name]
            if not (isinstance(scale_data, dict) and "crop_features" in scale_data and "image_size" in scale_data):
                raise ValueError(
                    f"Scale '{scale_name}' is not in crop-level format (missing 'crop_features'/'image_size')"
                )

            crop_features = scale_data['crop_features']
            image_size = scale_data['image_size']  # (W, H)

            H, W = image_size[1], image_size[0]
            C = crop_features[0]['feature'].shape[-1]
            feature_map = np.zeros((H, W, C), dtype=np.float32)
            mask = np.zeros((H, W), dtype=bool)

            for crop_data in crop_features:
                feature = crop_data['feature']
                bbox = crop_data['bbox']  # (x, y, w, h)

                feature_np = feature.numpy() if hasattr(feature, 'numpy') else np.asarray(feature)
                x, y, w, h = bbox
                x2, y2 = x + w, y + h

                crop_h, crop_w = int(y2 - y), int(x2 - x)
                if feature_np.shape[0] != crop_h or feature_np.shape[1] != crop_w:
                    # 使用nearest neighbor resize（避免bilinear平滑）
                    feature_torch = torch.from_numpy(feature_np).permute(2, 0, 1).unsqueeze(0)
                    feature_torch = torch.nn.functional.interpolate(
                        feature_torch, size=(crop_h, crop_w), mode='nearest'
                    )
                    feature_np = feature_torch.squeeze(0).permute(1, 2, 0).numpy()

                # 直接覆盖（last-write-wins，不averaging）
                feature_map[y:y2, x:x2] = feature_np
                mask[y:y2, x:x2] = True

            # 只返回有效区域的特征
            return feature_map.reshape(H * W, C)[mask.reshape(H * W)]

        features_container = file["feature_maps"]

        # 加载指定尺度或混合（若某尺度缺失则跳过）
        if self.load_func == "small":
            features = _extract_by_scale(features_container, "Small")
            if features is None:
                features = np.empty((0, 0), dtype=np.float32)
        elif self.load_func == "medium":
            features = _extract_by_scale(features_container, "Medium")
            if features is None:
                features = np.empty((0, 0), dtype=np.float32)
        elif self.load_func == "large":
            features = _extract_by_scale(features_container, "Large")
            if features is None:
                features = np.empty((0, 0), dtype=np.float32)
        elif self.load_func == "hybrid":
            parts = []
            for nm in ("Small", "Medium", "Large"):
                part = _extract_by_scale(features_container, nm)
                if part is not None and part.size:
                    parts.append(part)
            if parts:
                # 不同尺度通常维度一致；若不一致，此处会报错提示
                features = np.concatenate(parts, axis=0)
            else:
                features = np.empty((0, 0), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported load_func: {self.load_func}")

        return np.ascontiguousarray(features, dtype=np.float32)

    def __iter__(self):
        start_idx = 0

        if self._cached_first is not None:
            length = self._cached_first.shape[0]
            if self._block_lengths is None:
                self._block_lengths = [None] * len(self.data_names)
            self._block_lengths[0] = length
            yield self._cached_first
            start_idx = 1

        for idx, path in enumerate(self.data_names[start_idx:], start=start_idx):
            features = self._load_feature_block(path)
            if features.size == 0:
                if self._block_lengths is not None:
                    self._block_lengths[idx] = 0
                continue
            if self._block_lengths is None:
                self._block_lengths = [None] * len(self.data_names)
            self._block_lengths[idx] = features.shape[0]
            yield features

        if self._block_lengths is not None and None not in self._block_lengths:
            self._total_features = int(sum(self._block_lengths))

        self._cached_first = None

    def peek(self):
        """Load (and cache) the first feature block for shape inspection."""

        if self._cached_first is None:
            self._cached_first = self._load_feature_block(self.data_names[0])
        return self._cached_first

    def total_features(self, recompute=False):
        """Return the total number of feature vectors across all blocks."""

        if self._total_features is not None and not recompute:
            return self._total_features

        lengths = []
        cached_first = None
        for idx, path in enumerate(self.data_names):
            features = self._load_feature_block(path)
            length = features.shape[0]
            lengths.append(length)
            if idx == 0:
                cached_first = features

        self._block_lengths = lengths
        self._total_features = int(sum(lengths))

        # Preserve the first block so consumers can iterate without reloading it immediately.
        if cached_first is not None:
            self._cached_first = cached_first

        return self._total_features


def load_2d_lmm_feature(data_dir, device, load_func="hybrid", streaming=True):
    """
    Load (or stream) large multi-modal features from 2D images.
    load_func: "small", "medium", "large", "hybrid"
    streaming: when True, return an iterable that loads blocks on demand; when False,
               materialize all features into a single numpy array (high memory).
    """
    stream = LMMFeatureStream(data_dir, load_func=load_func)

    if streaming:
        print(
            f"Prepared feature stream with {len(stream)} files under {data_dir}; "
            "keeping data on disk to limit RAM usage."
        )
        return stream

    # Materialize all features (may consume large amounts of RAM)
    chunks = [chunk for chunk in stream]
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    data = np.concatenate(chunks, axis=0)
    print(
        f"Loaded {data.shape[0]} features with dimension {data.shape[1]}, keeping on CPU to avoid OOM"
    )
    return data
  
class opt:
    vq_layer_num = 1
    codebook_size = 2048

class dataset:
    # lf_path = "/home/wangyz/data/lerf_ovs/teatime/language_features"
    lf_path = "/home/wangyz/data/lerf_ovs/teatime/llava_features_multiscale"

if __name__ == "__main__":
    # Test the ResidualVectorQuantizationWithClustering
        device = torch.device("cuda")
        # features = load_2d_language_feature(dataset.lf_path, device)
        features = load_2d_lmm_feature(dataset.lf_path, device)
        first_block = features.peek()
        feature_dim = first_block.shape[1] if first_block.size else 0
        print("ResidualVectorQuantizationWithClustering...")
        rvq = ResidualVectorQuantizationWithClustering(
            opt.vq_layer_num, opt.codebook_size, feature_dim, device
        ).to(device)
        print("Fitting quantizers...")
        rvq.fit_quantizers(features)
        print("Stacking...")
        codebooks = torch.stack(rvq.quantizers, dim=0).to(device) # [vq_layer_num, codebook_size, feature_dim] e.g. [1,64,512]
        print(codebooks.shape)
