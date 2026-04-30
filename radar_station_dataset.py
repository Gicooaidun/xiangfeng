from __future__ import annotations

"""读取时空对齐后的雷达-站点网格数据，并构造 PyTorch 训练样本。

预处理 notebook 会把雷达反射率、站点风场和站点降水统一到同一时间轴
和同一 461x461 雷达网格上，输出 `aligned_YYYYmmddHHMMSS.npy` 文件。

每个 aligned 文件形状为 `(6, H, W)`：
    0: 雷达反射率
    1: 10 分钟平均风速
    2: 10 分钟平均风 u 分量
    3: 10 分钟平均风 v 分量
    4: 瞬时最大风速
    5: 1 小时降水

这个 Dataset 的核心工作是：
    1. 按文件名时间排序；
    2. 只在连续 6 分钟片段内滑动取窗口，避免跨大断档构造样本；
    3. 返回过去 sequence_length 帧雷达、当前观测网格，以及可选未来目标帧。
"""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


# aligned_*.npy 的通道名称到下标的映射，和 README/channels.csv 保持一致。
CHANNELS = {
    "radar_reflectivity": 0,
    "wind_speed_avg_10mi": 1,
    "wind_u_avg_10mi": 2,
    "wind_v_avg_10mi": 3,
    "wind_speed_inst_max": 4,
    "pre_1h": 5,
}


@dataclass(frozen=True)
class SampleIndex:
    """一个样本在文件序列中的起止位置。

    start 和 end 都是闭区间下标。例如 sequence_length=20 时，
    end = start + 19，表示输入包含 20 个连续雷达时刻。
    """

    start: int
    end: int


def parse_aligned_time(path: Path) -> datetime:
    """从 `aligned_YYYYmmddHHMMSS.npy` 文件名解析时间戳。"""

    timestamp = path.stem.split("_", 1)[1]
    return datetime.strptime(timestamp, "%Y%m%d%H%M%S")


def _split_contiguous_segments(times: list[datetime], step_minutes: int) -> list[tuple[int, int]]:
    """把时间序列切成若干严格连续片段。

    雷达数据中可能存在从 2024-05-27 到 2024-06-18 这样的大断档。
    如果直接按排序后的文件滑动窗口，会把断档两侧错误地拼成一个样本。
    这里用相邻时间差是否等于 step_minutes 来识别连续片段。
    """

    if not times:
        return []

    expected = timedelta(minutes=step_minutes)
    segments: list[tuple[int, int]] = []
    start = 0

    for i in range(1, len(times)):
        # 一旦相邻文件不是固定 6 分钟间隔，就结束当前连续片段。
        if times[i] - times[i - 1] != expected:
            segments.append((start, i - 1))
            start = i

    segments.append((start, len(times) - 1))
    return segments


def _build_window_indices(
    times: list[datetime],
    sequence_length: int,
    step_minutes: int,
    target_offset: int | None,
) -> list[SampleIndex]:
    """在每个连续片段内生成可用的滑动窗口下标。

    当 target_offset 不为 None 时，窗口后面还必须有足够的未来帧作为
    预测目标。例如 sequence_length=20、target_offset=1 时，一个样本
    需要 20 帧输入加第 21 帧目标。
    """

    segments = _split_contiguous_segments(times, step_minutes)
    indices: list[SampleIndex] = []

    extra = target_offset if target_offset is not None else 0
    required = sequence_length + extra

    for segment_start, segment_end in segments:
        segment_len = segment_end - segment_start + 1
        if segment_len < required:
            # 太短的连续片段无法同时提供输入窗口和目标帧，直接跳过。
            continue

        # last_start 保证 end + target_offset 不会越过当前连续片段。
        last_start = segment_end - required + 1
        for start in range(segment_start, last_start + 1):
            end = start + sequence_length - 1
            indices.append(SampleIndex(start=start, end=end))

    return indices


def _split_indices(
    indices: list[SampleIndex],
    split: Literal["all", "train", "val"],
    val_ratio: float,
    split_mode: Literal["chronological", "random"],
    seed: int,
) -> list[SampleIndex]:
    """把全部样本下标拆成训练集或验证集。

    split_mode="chronological" 时保持时间顺序，前半段训练、后半段验证；
    split_mode="random" 时先用 seed 打乱，再按比例切分。
    """

    if split == "all":
        return indices

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    order = np.arange(len(indices))
    if split_mode == "random":
        # 使用 numpy Generator，避免影响全局随机状态。
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    elif split_mode != "chronological":
        raise ValueError("split_mode must be 'chronological' or 'random'.")

    val_count = max(1, int(round(len(indices) * val_ratio)))
    train_count = len(indices) - val_count

    selected = order[:train_count] if split == "train" else order[train_count:]
    return [indices[i] for i in selected]


class AlignedRadarStationDataset(Dataset):
    """用于读取已对齐雷达和站点网格 npy 文件的 PyTorch Dataset。

    每个 aligned 文件应为 ``(6, H, W)``：

    - channel 0: 雷达反射率
    - channel 1: 10 分钟平均风速
    - channel 2: 平均风 u 分量
    - channel 3: 平均风 v 分量
    - channel 4: 瞬时最大风速
    - channel 5: 1 小时降水

    默认每个样本返回：

    - ``radar``: 连续 20 帧雷达，形状 ``(20, H, W)``
    - ``obs``: 输入窗口最后一帧时刻的 2 个观测网格，形状 ``(2, H, W)``
    - 可选 ``target``: 未来雷达帧，形状 ``(H, W)``

    默认观测通道是 10 分钟平均风速和 1 小时降水。
    """

    def __init__(
        self,
        data_dir: str | Path = "aligned_radar_station_npy",
        sequence_length: int = 20,
        obs_channels: tuple[str | int, ...] = ("wind_speed_avg_10mi", "pre_1h"),
        split: Literal["all", "train", "val"] = "all",
        val_ratio: float = 0.2,
        split_mode: Literal["chronological", "random"] = "chronological",
        seed: int = 42,
        step_minutes: int = 6,
        target_offset: int | None = None,
        mmap_mode: str | None = "r",
        cache_size: int = 128,
        dtype: torch.dtype = torch.float32,
        return_metadata: bool = False,
    ) -> None:
        # 保存基础配置；obs_channels 可传通道名，也可直接传整数下标。
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.obs_channel_indices = tuple(self._resolve_channel(ch) for ch in obs_channels)
        self.step_minutes = step_minutes
        self.target_offset = target_offset
        self.mmap_mode = mmap_mode
        self.cache_size = cache_size
        self.dtype = dtype
        self.return_metadata = return_metadata
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()

        # 基础参数检查，尽早暴露配置错误。
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if len(self.obs_channel_indices) != 2:
            raise ValueError("This dataset expects exactly 2 observation channels.")
        if target_offset is not None and target_offset <= 0:
            raise ValueError("target_offset must be positive when provided.")

        # 文件名里包含时间戳，排序后才能按时间窗口构造样本。
        self.files = sorted(self.data_dir.glob("aligned_*.npy"), key=parse_aligned_time)
        if not self.files:
            raise FileNotFoundError(f"No aligned_*.npy files found in {self.data_dir.resolve()}")

        self.times = [parse_aligned_time(path) for path in self.files]

        # 生成所有合法窗口；这里会自动避开雷达时间断档。
        all_indices = _build_window_indices(
            self.times,
            sequence_length=sequence_length,
            step_minutes=step_minutes,
            target_offset=target_offset,
        )
        if not all_indices:
            raise ValueError("No valid contiguous windows were found.")

        # 根据 split 参数取全量、训练集或验证集。
        self.indices = _split_indices(
            all_indices,
            split=split,
            val_ratio=val_ratio,
            split_mode=split_mode,
            seed=seed,
        )

    @staticmethod
    def _resolve_channel(channel: str | int) -> int:
        """把通道名转换为 aligned 数组中的整数通道下标。"""

        if isinstance(channel, int):
            return channel
        if channel not in CHANNELS:
            raise KeyError(f"Unknown channel {channel!r}. Available: {sorted(CHANNELS)}")
        return CHANNELS[channel]

    def __len__(self) -> int:
        """返回当前 split 中可用样本数量。"""

        return len(self.indices)

    def __getitem__(self, index: int):
        """读取一个训练样本。

        返回字典至少包含：
        - radar: (sequence_length, H, W)
        - obs: (2, H, W)

        如果 target_offset 不为 None，还会包含：
        - target: (H, W)
        """

        sample = self.indices[index]
        frame_indices = range(sample.start, sample.end + 1)

        # 加载输入窗口内所有 aligned 文件；_load_file 内部会使用 mmap 和 LRU 缓存。
        frames = [self._load_file(self.files[i]) for i in frame_indices]

        # 只取每个文件的雷达反射率通道，并堆叠成时间通道。
        radar = np.stack([frame[CHANNELS["radar_reflectivity"]] for frame in frames], axis=0)

        # 站点观测是小时级，同一小时内多个 6 分钟雷达帧会共享同一份站点网格。
        # 这里使用输入窗口最后一帧对应的观测，表示“当前时刻可用的站点信息”。
        obs_frame = frames[-1]
        obs = obs_frame[list(self.obs_channel_indices)]

        # 转为 torch.Tensor，dtype 默认 float32，供 DataLoader 批量拼接。
        item = {
            "radar": torch.as_tensor(np.asarray(radar), dtype=self.dtype),
            "obs": torch.as_tensor(np.asarray(obs), dtype=self.dtype),
        }

        if self.target_offset is not None:
            # target_offset=1 表示 end 后一帧，也就是 6 分钟后的雷达反射率。
            target_index = sample.end + self.target_offset
            target_frame = self._load_file(self.files[target_index])
            target = target_frame[CHANNELS["radar_reflectivity"]]
            item["target"] = torch.as_tensor(np.asarray(target), dtype=self.dtype)

        if self.return_metadata:
            # 调试或可视化时打开 return_metadata，可追踪样本对应的真实时间和文件名。
            item["start_time"] = self.times[sample.start].isoformat(sep=" ")
            item["end_time"] = self.times[sample.end].isoformat(sep=" ")
            item["files"] = [self.files[i].name for i in frame_indices]
            if self.target_offset is not None:
                item["target_time"] = self.times[sample.end + self.target_offset].isoformat(sep=" ")

        return item

    def _load_file(self, path: Path) -> np.ndarray:
        """加载单个 npy 文件，并用简单 LRU 缓存减少重复磁盘读取。

        训练样本之间高度重叠：相邻样本会共享大部分历史帧。缓存最近读过的文件
        可以显著减少 np.load 次数。mmap_mode="r" 会尽量以内存映射方式读取，
        对 461x461 的多通道网格更省内存。
        """

        if self.cache_size <= 0:
            return np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)

        cached = self._cache.get(path)
        if cached is not None:
            # 命中缓存后移动到末尾，表示最近使用。
            self._cache.move_to_end(path)
            return cached

        arr = np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)
        self._cache[path] = arr
        self._cache.move_to_end(path)

        while len(self._cache) > self.cache_size:
            # OrderedDict 头部是最久未使用项，超过容量时淘汰。
            self._cache.popitem(last=False)

        return arr


def make_datasets(
    data_dir: str | Path = "aligned_radar_station_npy",
    sequence_length: int = 20,
    val_ratio: float = 0.2,
    split_mode: Literal["chronological", "random"] = "chronological",
    seed: int = 42,
    target_offset: int | None = None,
    **kwargs,
) -> tuple[AlignedRadarStationDataset, AlignedRadarStationDataset]:
    """使用相同配置创建训练集和验证集。

    除 split 不同外，其余参数完全一致，保证训练/验证样本来自同一数据定义。
    kwargs 会继续传给 AlignedRadarStationDataset，例如 obs_channels、cache_size、
    return_metadata 等。
    """

    train_dataset = AlignedRadarStationDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        split="train",
        val_ratio=val_ratio,
        split_mode=split_mode,
        seed=seed,
        target_offset=target_offset,
        **kwargs,
    )
    val_dataset = AlignedRadarStationDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        split="val",
        val_ratio=val_ratio,
        split_mode=split_mode,
        seed=seed,
        target_offset=target_offset,
        **kwargs,
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    # 直接运行本文件时，打印一个样本的形状和时间范围，用于快速检查数据是否可读。
    train_ds, val_ds = make_datasets(
        sequence_length=20,
        val_ratio=0.2,
        target_offset=1,
        return_metadata=True,
    )

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    sample = train_ds[0]
    print("radar:", sample["radar"].shape)
    print("obs:", sample["obs"].shape)
    print("target:", sample["target"].shape)
    print("time:", sample["start_time"], "->", sample["end_time"])
