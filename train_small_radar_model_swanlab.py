from __future__ import annotations

"""训练一个轻量级雷达临近预报模型。

这个脚本负责把 `radar_station_dataset.py` 生成的样本送入一个小型
Encoder-Decoder CNN，并用 SwanLab 记录训练过程。默认任务是：

    过去 20 帧 6 分钟雷达反射率 + 当前小时 2 个站点观测网格
    -> 预测下一帧 6 分钟雷达反射率

输入文件来自 `aligned_radar_station_npy/`，每个 `aligned_*.npy` 的形状为
`(6, 461, 461)`，其中第 0 通道是雷达反射率，第 1 和第 5 通道默认作为
观测辅助变量：10 分钟平均风速与 1 小时降水。
"""

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from radar_station_dataset import AlignedRadarStationDataset, make_datasets


@dataclass
class TrainConfig:
    """集中管理训练配置，便于命令行参数、日志和 checkpoint 保持一致。"""

    # 数据与输出目录。
    data_dir: str = "aligned_radar_station_npy"
    output_dir: str = "training_outputs"

    # SwanLab 项目与实验名称。
    project: str = "radar-station-nowcasting"
    experiment_name: str = "small-cnn-20radar-2obs"

    # 样本构造：20 帧历史雷达，target_offset=1 表示预测 6 分钟后一帧。
    sequence_length: int = 20
    target_offset: int = 1

    # 训练/验证划分；chronological 更适合时间序列，避免未来信息泄漏。
    val_ratio: float = 0.2
    split_mode: str = "chronological"

    # 作为额外输入的站点观测通道，对应 aligned npy 的通道名。
    obs_channels: tuple[str, str] = ("wind_speed_avg_10mi", "pre_1h")

    # 优化相关超参数。batch_size 默认较小，是为了适配 461x461 大网格。
    batch_size: int = 2
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # 数据读取参数：num_workers=0 在 Windows/Notebook 环境下最稳。
    num_workers: int = 0
    cache_size: int = 128

    # 复现实验、快速抽样调试与混合精度开关。
    seed: int = 42
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    amp: bool = False

    # "cloud" 上传 SwanLab；"local" 本地记录；"disabled" 由 SwanLab 自身处理。
    swanlab_mode: str = "cloud"


class ConvBlock(nn.Module):
    """两层卷积块：Conv -> BN -> GELU 重复两次。

    这个块在编码器、瓶颈层和解码器中复用，用来提取局地空间特征。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallRadarNowcastNet(nn.Module):
    """用于单步雷达临近预报的小型 Encoder-Decoder CNN。

    输入形状:
        (B, 22, 461, 461)
        B 是 batch size；22 = 20 个历史雷达通道 + 2 个观测通道。

    输出形状:
        (B, 1, 461, 461)
        输出为下一帧雷达反射率网格。
    """

    def __init__(self, in_channels: int = 22, base_channels: int = 24) -> None:
        super().__init__()
        # 编码器第一层保持原始分辨率，先从 22 个输入通道中提取浅层空间特征。
        self.enc1 = ConvBlock(in_channels, base_channels)

        # 两次 stride=2 下采样扩大感受野，帮助模型看到更大范围的回波运动。
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)

        # 解码器逐步上采样回 461x461，并和编码器特征拼接保留局地细节。
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # 1x1 卷积把特征压到 1 个通道，即预测的下一帧雷达反射率。
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码路径：保留 e1/e2 用作 skip connection。
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        # 461 不是 4 的倍数，转置卷积上采样后可能差 1 个像素；
        # _resize_like 会对齐到对应编码层的空间尺寸。
        d2 = self.up2(b)
        d2 = _resize_like(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = _resize_like(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 x 的高宽对齐到 ref，处理奇数尺寸下采样/上采样造成的边界差异。"""

    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def set_seed(seed: int) -> None:
    """固定 PyTorch 随机种子，让随机划分和初始化尽量可复现。"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_xy(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """从 Dataset 返回的 batch 中拼出模型输入 x 和监督目标 y。

    x:
        20 帧雷达 + 2 个观测网格在 channel 维拼接。
        形状: (B, 22, 461, 461)

    y:
        下一帧 6 分钟雷达反射率。
        形状: (B, 1, 461, 461)
    """
    # non_blocking=True 配合 DataLoader pin_memory 可减少 CPU->GPU 拷贝等待。
    radar = batch["radar"].to(device, non_blocking=True)
    obs = batch["obs"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True).unsqueeze(1)

    # radar: (B, 20, H, W)，obs: (B, 2, H, W)，拼接后为 (B, 22, H, W)。
    x = torch.cat([radar, obs], dim=1)
    y = target

    # 使用固定的轻量归一化，不依赖额外统计文件；clip 可避免异常观测值主导 loss。
    x = normalize_x(x)
    y = normalize_radar(y)
    return x, y


def normalize_radar(radar: torch.Tensor) -> torch.Tensor:
    """雷达反射率按经验上界 60 缩放，并保留少量负值/超上界空间。"""

    return torch.clamp(radar / 60.0, min=-0.2, max=1.2)


def normalize_x(x: torch.Tensor) -> torch.Tensor:
    """分别归一化输入中的雷达、风速和降水通道。

    通道约定：
    - x[:, :20] 是 20 帧雷达反射率；
    - x[:, 20:21] 是 10 分钟平均风速；
    - x[:, 21:22] 是 1 小时降水。
    """

    radar = normalize_radar(x[:, :20])
    wind_speed = torch.clamp(x[:, 20:21] / 30.0, min=0.0, max=2.0)
    precip = torch.clamp(x[:, 21:22] / 100.0, min=0.0, max=2.0)
    return torch.cat([radar, wind_speed, precip], dim=1)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """只在 pred 和 target 都是有限值的位置计算 MSE。

    如果数据里出现 NaN/Inf，这个 loss 可以避免整批训练被污染。
    """

    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[mask], target[mask])


def maybe_limit_dataset(dataset: AlignedRadarStationDataset, max_samples: int | None):
    """调试时截取前 max_samples 个样本，完整训练时原样返回数据集。"""

    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, range(max_samples))


def make_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """根据配置创建训练和验证 DataLoader。"""

    # make_datasets 会按连续 6 分钟时间窗构造样本，并避免跨雷达断档取样。
    train_ds, val_ds = make_datasets(
        data_dir=config.data_dir,
        sequence_length=config.sequence_length,
        val_ratio=config.val_ratio,
        split_mode=config.split_mode,  # type: ignore[arg-type]
        seed=config.seed,
        target_offset=config.target_offset,
        obs_channels=config.obs_channels,
        cache_size=config.cache_size,
    )

    train_ds = maybe_limit_dataset(train_ds, config.max_train_samples)
    val_ds = maybe_limit_dataset(val_ds, config.max_val_samples)

    # random split 时训练集可 shuffle；chronological split 保持时间顺序更直观。
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=config.split_mode == "random",
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> float:
    """训练一个 epoch，并返回按样本数加权的平均 loss。"""

    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x, y = build_xy(batch, device)
        optimizer.zero_grad(set_to_none=True)

        # autocast 仅在 CUDA + --amp 时启用；CPU 或未开 amp 时等价于普通前向。
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x)
            loss = masked_mse_loss(pred, y)

        # GradScaler 在 use_amp=False 时仍可调用，只是退化为普通 backward/step。
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> float:
    """在验证集上计算平均 loss，不更新模型参数。"""

    model.eval()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x, y = build_xy(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x)
            loss = masked_mse_loss(pred, y)

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def save_loss_curve(history: list[dict[str, float]], output_dir: Path) -> Path:
    """把每个 epoch 的训练/验证 loss 保存成本地 PNG 曲线图。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train")
    plt.plot(epochs, val_loss, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Small Radar Nowcasting Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    curve_path = output_dir / "loss_curve.png"
    plt.savefig(curve_path, dpi=160)
    plt.close()
    return curve_path


def init_swanlab(config: TrainConfig):
    """初始化 SwanLab；未安装时降级为只在本地打印和保存结果。"""

    try:
        import swanlab
    except ImportError:
        print("swanlab is not installed; training will continue without SwanLab logging.")
        return None

    return swanlab.init(
        project=config.project,
        experiment_name=config.experiment_name,
        config=asdict(config),
        mode=config.swanlab_mode,
    )


def train(config: TrainConfig) -> None:
    """完整训练流程：准备数据、模型、优化器，循环训练并保存结果。"""

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 自动选择 GPU；只有在 CUDA 上才启用混合精度，避免 CPU autocast 行为差异。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.amp and device.type == "cuda"

    train_loader, val_loader = make_loaders(config)

    # 输入通道数由历史雷达帧数和观测通道数共同决定，默认 20 + 2 = 22。
    model = SmallRadarNowcastNet(in_channels=config.sequence_length + len(config.obs_channels)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    run = init_swanlab(config)

    print("x = 20 radar frames + 2 observation grids, shape (B, 22, 461, 461)")
    print("y = next 6-minute radar reflectivity frame, shape (B, 1, 461, 461)")
    print("device:", device)
    print("train batches:", len(train_loader), "val batches:", len(val_loader))

    best_val = math.inf
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        # 每轮先训练，再用验证集评估泛化误差。
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, use_amp)
        val_loss = validate(model, val_loader, device, use_amp)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(
            f"epoch {epoch:03d}/{config.epochs:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if run is not None:
            import swanlab

            swanlab.log(metrics)

        # 只保留验证集 loss 最低的模型，避免最后一轮过拟合时覆盖较好权重。
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                output_dir / "best_model.pt",
            )

    curve_path = save_loss_curve(history, output_dir)
    print("saved loss curve:", curve_path)

    if run is not None:
        import swanlab

        # 训练完成后上传 loss 曲线图片，并显式结束 SwanLab run。
        swanlab.log({"loss_curve": swanlab.Image(str(curve_path))})
        swanlab.finish()


def parse_args() -> TrainConfig:
    """解析命令行参数，并转换成 TrainConfig。

    这里没有暴露 sequence_length 和 obs_channels，是因为当前脚本专门对应
    README 中的小模型设定：20 帧雷达 + 风速/降水两个观测通道。
    """

    parser = argparse.ArgumentParser(description="Train a small radar nowcasting model with SwanLab.")
    parser.add_argument("--data-dir", default="aligned_radar_station_npy")
    parser.add_argument("--output-dir", default="training_outputs")
    parser.add_argument("--project", default="radar-station-nowcasting")
    parser.add_argument("--experiment-name", default="small-cnn-20radar-2obs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["chronological", "random"], default="chronological")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--swanlab-mode", default="cloud", choices=["cloud", "local", "disabled"])
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        project=args.project,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        amp=args.amp,
        swanlab_mode=args.swanlab_mode,
    )


if __name__ == "__main__":
    train(parse_args())
