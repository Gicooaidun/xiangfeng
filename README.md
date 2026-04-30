# 雷达、测风、降水数据时空对齐说明

本文档记录 `test` 目录中四类数据的分析、处理方法和输出结果。处理目标是将雷达反射率、站点测风数据、站点降水数据对齐到同一时间轴和同一空间网格，最终输出可直接用于建模或后续分析的 `.npy` 文件。

## 0. 运行环境

推荐使用 Python 3.10 或更新版本。项目主要依赖如下：

```text
numpy       读取和处理 npy 数组
pandas      读取 parquet 表格数据、输出 csv 日志
pyarrow     pandas 读写 parquet 所需后端
scipy       站点到雷达网格的 KDTree 最近邻搜索
matplotlib  保存训练 loss 曲线
torch       Dataset、DataLoader 和 CNN 模型训练
swanlab     可选，用于训练日志记录
```

建议在独立虚拟环境中安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要使用 GPU 训练，请根据本机 CUDA 版本安装对应的 PyTorch。可以先安装 `requirements.txt` 中的通用版本；如果 CUDA 不匹配，再参考 PyTorch 官网命令重新安装 `torch`。

环境检查：

```bash
python -c "import numpy, pandas, scipy, torch, matplotlib; print('env ok', torch.__version__)"
```

## 0.1 数据目录与输入输出清单

由于原始雷达、站点表格和中间 `.npy` 文件体积较大，建议 GitHub 仓库上传代码、notebook、README、`requirements.txt` 和必要的可视化输出，不要上传大体积数据文件或模型权重。使用者需要在本地按下面的目录结构准备数据：

```text
项目根目录/
  test/
    Z9002_*.npy                 原始雷达反射率文件
    wind_YYYYmmddHH.parquet     原始逐小时测风站点文件
    pre_1h_YYYYmmddHH.parquet   原始逐小时降水站点文件
    radar_situation.npz         雷达网格经纬度文件

  test_radar_6min/              第 2 步生成，整 6 分钟雷达文件
  aligned_radar_station_npy/    第 3 步生成，雷达和站点对齐后的建模数据
  training_outputs/             训练脚本生成，模型权重和 loss 曲线
```

每一步的输入输出如下：

| 步骤 | 运行文件 | 输入 | 输出 |
|---|---|---|---|
| 1. 原始数据检查 | `1.read_test_four_types.ipynb` | `test/Z9002_*.npy`、`test/wind_*.parquet`、`test/pre_1h_*.parquet`、`test/radar_situation.npz` | 数据数量、时间范围、字段和网格信息的检查结果 |
| 2. 雷达时间插值 | `2.interpolate_radar_to_6min.ipynb` | `test/Z9002_*.npy` | `test_radar_6min/Z9002_*.npy`、`test_radar_6min/interpolation_log.csv` |
| 3. 站点空间对齐 | `3.align_station_to_radar_grid.ipynb` | `test_radar_6min/Z9002_*.npy`、`test/wind_*.parquet`、`test/pre_1h_*.parquet`、`test/radar_situation.npz` | `aligned_radar_station_npy/aligned_*.npy`、`alignment_log.csv`、`channels.csv` |
| 4. Dataset 读取检查 | `radar_station_dataset.py` | `aligned_radar_station_npy/aligned_*.npy` | 打印 train/val 样本数和单个样本形状 |
| 5. 小模型训练 | `train_small_radar_model_swanlab.py` | `aligned_radar_station_npy/aligned_*.npy` | `training_outputs/best_model.pt`、`training_outputs/loss_curve.png`，可选 SwanLab 日志 |

其中真正用于模型训练的最终数据是：

```text
aligned_radar_station_npy/aligned_*.npy
```

如果只想运行训练脚本，可以不保留原始 `test/` 和 `test_radar_6min/`，但必须提前准备好 `aligned_radar_station_npy/`。如果要从原始数据完整复现流程，则需要保留 `test/` 中的四类原始文件。

## 1. 原始数据

原始数据位于 `test/` 目录。

| 数据类型 | 文件规则 | 实际数量 | 说明 |
|---|---:|---:|---|
| 雷达反射率 | `Z9002_*.npy` | 566 | 二维雷达反射率网格，文件名包含秒级时间戳 |
| 站点测风 | `wind_*.parquet` | 576 | 逐小时站点风速、风向等观测数据 |
| 站点降水 | `pre_1h_*.parquet` | 576 | 逐小时站点累计降水数据 |
| 雷达经纬度 | `radar_situation.npz` | 1 | 雷达网格的经纬度坐标 |

雷达反射率文件的数组形状为：

```text
(461, 461)
```

`radar_situation.npz` 中包含：

```text
lon: (461, 461), 范围约 118.6530 ~ 123.2530
lat: (461, 461), 范围约 28.7700 ~ 33.3700
```

因此雷达数组中每个格点 `(i, j)` 都可以通过 `lon[i, j]` 和 `lat[i, j]` 定位到空间位置。

## 2. 时间范围分析

三类时序数据的时间范围如下：

| 数据类型 | 起始时间 | 结束时间 | 时间分辨率 | 备注 |
|---|---|---|---|---|
| 雷达反射率 | `2024-05-26 00:07:44` | `2024-06-18 06:05:46` | 约 5-6 分钟 | 中间存在大断档 |
| 站点测风 | `2024-05-26 00:00:00` | `2024-06-18 23:00:00` | 1 小时 | 连续 |
| 站点降水 | `2024-05-26 00:00:00` | `2024-06-18 23:00:00` | 1 小时 | 连续 |

雷达反射率实际只覆盖以下日期和时段：

```text
2024-05-26: 235 帧
2024-05-27: 266 帧
2024-06-18: 65 帧
```

测风和降水数据覆盖完整的 24 天小时序列：

```text
2024-05-26 00:00:00 ~ 2024-06-18 23:00:00
共 576 小时
```

因此时空对齐时，不需要使用全部 576 小时站点数据，只需要使用雷达实际覆盖时段对应的小时数据。

## 3. 雷达时间插值

雷达原始时间不是严格整 6 分钟，例如：

```text
04:45, 04:52
```

处理时将雷达反射率插值到整 6 分钟时刻，例如：

```text
04:48, 04:54, 05:00
```

对应 notebook：

```text
interpolate_radar_to_6min.ipynb
```

方法：

1. 从 `Z9002_YYYYmmddHHMMSS.npy` 文件名解析雷达时间。
2. 按时间缺口切分连续片段。
3. 只在相邻雷达帧时间差不超过 `MAX_GAP_MINUTES = 30` 分钟时做线性插值。
4. 对每个目标 6 分钟时刻，找到前后两个雷达帧：

```text
t0 <= t <= t1
```

5. 对每个空间格点做线性插值：

```text
arr(t) = arr(t0) * (1 - w) + arr(t1) * w
w = (t - t0) / (t1 - t0)
```

这样可以避免从 `2024-05-27` 直接跨越大缺测段插值到 `2024-06-18`。

输出目录：

```text
test_radar_6min/
```

实际输出：

```text
538 个 Z9002_*.npy 文件
```

其中：

```text
2024-05-26 00:12:00 ~ 2024-05-27 23:54:00
2024-06-18 00:06:00 ~ 2024-06-18 06:00:00
```

输出日志：

```text
test_radar_6min/interpolation_log.csv
```

## 4. 测风数据字段说明

`wind_*.parquet` 文件共有 7 列：

| 列名 | 含义 |
|---|---|
| `Lat` | 观测站纬度 |
| `Lon` | 观测站经度 |
| `Alti` | 观测站海拔 |
| `WIN_S_Avg_10mi` | 10 分钟平均风速 |
| `WIN_D_Avg_10mi` | 10 分钟平均风向 |
| `WIN_S_Inst_Max` | 瞬时最大风速 |
| `WIN_S_INST_Max_OTime` | 瞬时最大风速出现时间 |

所有 `wind` 文件列名一致，但每个小时的站点集合不完全一致：

```text
行数最少: 14401
行数最多: 21825
平均行数: 19151.15
```

坐标整体范围：

```text
Lon: 113.0000 ~ 123.1519
Lat: 23.0000 ~ 39.4994
Alti: -10.0 ~ 999999.0
```

其中 `Alti = 999999.0` 很可能是缺测或异常填充值。

落在雷达网格范围内的 wind 站点数：

```text
最少: 3078
最多: 4830
平均: 4322.14
```

由于站点集合会随小时变化，空间插值时采用“每个小时单独读取、单独插值”的方式。

## 5. 时间对齐方法

雷达已经被插值到整 6 分钟，而站点测风和降水是整小时数据。

时间匹配规则：

```text
每个 6 分钟雷达时刻匹配其所在小时的 wind 和 pre_1h 文件
```

例如：

```text
radar: 2024-05-26 04:48:00
wind : wind_2024052604.parquet
pre  : pre_1h_2024052604.parquet
```

当前 538 个 6 分钟雷达时刻都可以匹配到对应小时的测风和降水文件。

## 6. 空间对齐方法

空间对齐目标是将站点散点数据插值到雷达经纬度网格上：

```text
站点散点数据 -> 雷达 461 x 461 网格
```

对应 notebook：

```text
align_station_to_radar_grid.ipynb
```

采用方法：

```text
IDW, Inverse Distance Weighting, 反距离加权插值
```

对雷达网格上的每个点，寻找最近的 `K_NEIGHBORS = 8` 个站点，按距离加权平均：

```text
grid_value = sum(station_value / distance^2) / sum(1 / distance^2)
```

参数：

```text
K_NEIGHBORS = 8
IDW_POWER = 2.0
```

如果某个雷达格点刚好与站点坐标重合，则直接使用该站点值。

插值前只保留落在雷达经纬度范围内的站点：

```text
Lon: 118.6530 ~ 123.2530
Lat: 28.7700 ~ 33.3700
```

这样可以避免距离雷达区域很远的站点影响插值结果。

## 7. 风向处理

风向不能直接做角度插值。原因是角度有环绕问题，例如：

```text
359 度和 1 度直接平均会得到 180 度
```

但这两个方向实际上都接近正北。

因此处理时先将平均风速和平均风向转换为风矢量分量：

```text
u = -speed * sin(direction)
v = -speed * cos(direction)
```

其中 `direction` 是气象风向，表示风从哪个方向吹来。

含义：

```text
u > 0: 向东吹
u < 0: 向西吹
v > 0: 向北吹
v < 0: 向南吹
```

然后分别对 `u` 和 `v` 做空间插值。后续如果需要风向，可以再从 `u/v` 反算。

## 8. 最终输出

最终对齐结果输出到：

```text
aligned_radar_station_npy/
```

实际生成：

```text
538 个 aligned_*.npy 文件
```

每个文件对应一个 6 分钟雷达时刻，例如：

```text
aligned_20240526001200.npy
aligned_20240526001800.npy
aligned_20240526002400.npy
```

每个 `.npy` 文件形状为：

```text
(6, 461, 461)
```

6 个通道分别为：

| 通道 | 名称 | 含义 |
|---:|---|---|
| 0 | `radar_reflectivity` | 雷达反射率 |
| 1 | `wind_speed_avg_10mi` | 10 分钟平均风速插值网格 |
| 2 | `wind_u_avg_10mi` | 10 分钟平均风的 u 分量 |
| 3 | `wind_v_avg_10mi` | 10 分钟平均风的 v 分量 |
| 4 | `wind_speed_inst_max` | 瞬时最大风速插值网格 |
| 5 | `pre_1h` | 逐小时降水插值网格 |

同时生成辅助文件：

```text
aligned_radar_station_npy/alignment_log.csv
aligned_radar_station_npy/channels.csv
```

`channels.csv` 记录通道编号和通道名。`alignment_log.csv` 记录每个输出文件对应的雷达文件、测风文件、降水文件、观测小时和站点数量。

## 9. 处理流程复现

推荐按以下顺序运行 notebook：

```text
1. read_test_four_types.ipynb
   读取并检查四类原始文件。

2. interpolate_radar_to_6min.ipynb
   将原始雷达反射率插值到整 6 分钟时刻。

3. align_station_to_radar_grid.ipynb
   将 wind 和 pre_1h 站点数据按小时匹配，并插值到雷达空间网格。
```

运行完成后，应得到：

```text
test_radar_6min/
  538 个 6 分钟雷达 npy 文件

aligned_radar_station_npy/
  538 个时空对齐后的多通道 npy 文件
```

## 10. 注意事项

1. 雷达数据存在大时间断档，不能跨断档插值。
2. 站点数据是小时级，雷达是 6 分钟级，因此同一小时内多个雷达时刻会共享同一个小时的站点插值网格。
3. wind 文件的站点集合不是每小时完全一致，因此不能假设所有 wind 文件只有风速风向不同。
4. 风向需要转换为 `u/v` 分量后再插值，不能直接对角度做平均。
5. 当前空间插值使用 IDW，适合快速生成规则网格；如果后续需要更严格的空间统计建模，可以考虑 Kriging、径向基函数或地形约束插值等方法。

## 11. PyTorch Dataset 与小模型训练

已提供 Dataset 和训练脚本：

```text
radar_station_dataset.py
train_small_radar_model_swanlab.py
```

### Dataset 样本定义

每个对齐后的 `.npy` 文件形状为：

```text
(6, 461, 461)
```

训练样本从连续的 6 分钟对齐文件中构造。

`x` 是模型输入：

```text
x = 20 个连续雷达反射率网格 + 2 个观测站插值网格
x.shape = (B, 22, 461, 461)
```

其中：

```text
前 20 个通道: 连续 20 帧 radar_reflectivity
第 21 个通道: wind_speed_avg_10mi
第 22 个通道: pre_1h
```

`y` 是模型预测目标：

```text
y = 下一帧 6 分钟雷达反射率
y.shape = (B, 1, 461, 461)
```

也就是说，模型任务是：

```text
用过去 20 帧雷达 + 当前小时观测站风速/降水，预测下一帧雷达反射率。
```

默认时间跨度：

```text
20 帧 x 6 分钟 = 120 分钟历史雷达信息
target_offset = 1 表示预测 6 分钟后的雷达反射率
```

### Dataset 读取优化

`radar_station_dataset.py` 中实现了：

```text
np.load(..., mmap_mode="r")
LRU 文件缓存 cache_size
连续时间窗口检查，避免跨断档构造样本
训练 / 验证集划分
```

训练 / 验证划分支持：

```text
chronological: 按时间顺序划分
random: 随机划分
```

### 模型设计

训练脚本中的模型为：

```text
SmallRadarNowcastNet
```

它是一个小型 Encoder-Decoder CNN：

```text
输入:  (B, 22, 461, 461)
输出:  (B, 1, 461, 461)
```

结构设计：

```text
ConvBlock
Downsample Conv
ConvBlock
Downsample Conv
Bottleneck ConvBlock
Upsample
Skip Connection
Upsample
Skip Connection
1x1 Conv 输出下一帧雷达
```

选择该结构的原因：

```text
1. 雷达和观测站数据已经是规则网格，适合用 CNN 提取空间邻域特征。
2. 20 帧雷达作为通道输入，让模型在轻量结构中学习短时演变。
3. Encoder-Decoder 可以扩大感受野，同时恢复到原始 461 x 461 分辨率。
4. Skip connection 有助于保留局地强回波等细节。
```

### 训练配置

默认训练配置：

```text
batch_size = 2
epochs = 5
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
loss = MSE
val_ratio = 0.2
split_mode = chronological
```

归一化方式：

```text
radar_reflectivity / 60
wind_speed_avg_10mi / 30
pre_1h / 100
```

训练日志和 loss 曲线使用 SwanLab 记录，同时本地保存：

```text
training_outputs/loss_curve.png
training_outputs/best_model.pt
```

### 运行训练

完整训练：

```bash
python train_small_radar_model_swanlab.py --epochs 5 --batch-size 2
```

快速跑通流程可以限制样本数：

```bash
python train_small_radar_model_swanlab.py \
  --epochs 2 \
  --batch-size 1 \
  --max-train-samples 16 \
  --max-val-samples 4 \
  --swanlab-mode local
```

如果使用 SwanLab 云端记录，需要先完成 SwanLab 登录或配置。脚本会调用：

```text
swanlab.init(...)
swanlab.log(...)
swanlab.Image(...)
```

如果环境中没有安装 SwanLab，脚本会继续训练，但不会上传日志。
