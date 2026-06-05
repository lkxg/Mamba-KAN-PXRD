# SIMPOD Public Data Processing Pipeline

本文档记录从下载 SIMPOD 公开数据，到整理成本项目训练数据的完整流程。

## 1. 公开数据来源

SIMPOD 全称是 Simulated Powder X-ray Diffraction Open Database，论文为：

- Rincón et al., *Scientific Data* 12, 1186, 2025
- 论文 DOI: `10.1038/s41597-025-05534-3`
- 数据 DOI: `10.57760/sciencedb.09755`
- 官方代码与教程: `https://github.com/BCV-Uniandes/SIMPOD.git`

论文说明 SIMPOD 来源于 COD，截至 2023 年 8 月，最终包含 `467,861` 个晶体结构。数据在 Science Data Bank 公开，组织为两个主要部分：

- JSON 结构数据：每个晶体一个 JSON，包含晶体 ID、空间群、晶胞参数、原子信息和模拟 PXRD 向量。
- PNG 径向图像：与 JSON 一一对应，用于图像模型。

本项目只做一维 PXRD 序列分类，因此只需要下载 JSON 文件。论文也建议如果不需要图像，可以只下载 JSON。

## 2. 下载原始 JSON 数据

从 Science Data Bank 页面下载 SIMPOD 的结构 JSON 数据后，解压到本地目录。项目中的旧脚本默认原始目录为 Windows 路径：

```text
d:/SimPOD/Structures/Structures/
```

其中应包含大量 JSON 文件，例如：

```text
1000000.json
1000002.json
1000003.json
...
```

每个 JSON 至少需要包含：

- `space_group`: 空间群编号，范围 `1-230`
- `intensities`: 一维 PXRD 强度向量，长度 `10824`

本项目最终使用的有效样本数为 `467,861`。

如果在 Linux 机器上复跑，需要先把脚本里的硬编码路径改成本地路径，主要是：

- [analysis/scripts/scan_metadata.py](/home/exouser/code/Mamba-KAN-PXRD/analysis/scripts/scan_metadata.py)
- [analysis/scripts/preprocess.py](/home/exouser/code/Mamba-KAN-PXRD/analysis/scripts/preprocess.py)
- [analysis/scripts/check_dataset.py](/home/exouser/code/Mamba-KAN-PXRD/analysis/scripts/check_dataset.py)
- [analysis/scripts/plot_distributions.py](/home/exouser/code/Mamba-KAN-PXRD/analysis/scripts/plot_distributions.py)
- [analysis/scripts/plot_intensity_curves.py](/home/exouser/code/Mamba-KAN-PXRD/analysis/scripts/plot_intensity_curves.py)

现在仓库里的训练配置默认读取相对路径：

```text
dataset/pxrd.npy
dataset/labels.csv
splits/splits.csv
```

## 3. 扫描元数据

第一步可以快速扫描所有 JSON 的 `space_group`，生成轻量元数据表：

```bash
python3 analysis/scripts/scan_metadata.py
```

脚本做了这些事：

1. 遍历 `Structures/Structures/*.json`
2. 每个文件只读前 `256` 字节
3. 用正则匹配 `"space_group": <number>`
4. 多进程并行扫描
5. 输出 `analysis/metadata.csv`

当前仓库已有的输出为：

```text
analysis/metadata.csv
```

格式：

```csv
ID,space_group
1000000,14
1000002,14
1000003,14
...
```

当前文件统计：

- 行数：`467,862`，其中 1 行表头，`467,861` 个样本
- 大小：约 `5.3 MB`

这个文件主要用于前期分析和画分布图，不是训练时必须读取的文件。

## 4. 预处理为训练数据

核心预处理脚本是：

```bash
python3 analysis/scripts/preprocess.py
```

它把大量 JSON 转换成两个训练用文件：

```text
dataset/pxrd.npy
dataset/labels.csv
```

处理逻辑如下。

### 4.1 列出 JSON 文件

脚本先按文件名排序读取所有 `.json`：

```python
files = sorted(p for p in DATA_DIR.iterdir() if p.suffix == ".json")
```

排序后的文件顺序就是后续 `row` 的顺序。也就是说：

- `row=0` 对应排序后的第 1 个 JSON
- `row=1` 对应排序后的第 2 个 JSON
- 以此类推

### 4.2 分配 `pxrd.npy`

脚本创建一个 NumPy memmap 文件：

```text
shape = (N, 10824)
dtype = float16
```

当前数据为：

```text
dataset/pxrd.npy
shape = (467861, 10824)
dtype = float16
size  = 10,128,255,056 bytes, about 9.5 GiB
```

使用 `float16` 是因为 SIMPOD 的强度已经归一化到 `[0, 1]`，半精度可以显著减少磁盘占用。训练读取时，`src/data.py` 会再转成 `float32` tensor。

### 4.3 并行解析 JSON

每个 worker 解析一批 JSON，提取：

- 文件名 stem 作为 `id`
- `space_group`
- `intensities`

有效性检查：

- `space_group` 必须在 `1-230`
- `intensities` 必须存在
- `intensities` 长度必须等于 `10824`

无效样本会写入全零强度，并在标签中标为：

```text
space_group = 原始值或 -1
crystal_system = Invalid
crystal_system_id = -1
```

当前使用的数据中没有无效样本。

### 4.4 空间群映射到晶系

脚本按国际空间群编号范围映射晶系：

| 晶系 | 空间群范围 | `crystal_system_id` |
|---|---:|---:|
| Triclinic | 1-2 | 0 |
| Monoclinic | 3-15 | 1 |
| Orthorhombic | 16-74 | 2 |
| Tetragonal | 75-142 | 3 |
| Trigonal | 143-167 | 4 |
| Hexagonal | 168-194 | 5 |
| Cubic | 195-230 | 6 |

### 4.5 写出 `labels.csv`

输出格式：

```csv
row,id,space_group,crystal_system,crystal_system_id
0,1000000,14,Monoclinic,1
1,1000002,14,Monoclinic,1
2,1000003,14,Monoclinic,1
...
```

当前文件：

```text
dataset/labels.csv
rows = 467,861
size = about 15 MB
```

晶系分布：

| 晶系 | 样本数 |
|---|---:|
| Monoclinic | 228,011 |
| Triclinic | 113,002 |
| Orthorhombic | 79,990 |
| Tetragonal | 16,335 |
| Trigonal | 13,179 |
| Cubic | 10,357 |
| Hexagonal | 6,987 |

## 5. 检查预处理结果

预处理完成后运行：

```bash
python3 analysis/scripts/check_dataset.py
```

它会检查：

- `pxrd.npy` shape、dtype、大小
- `labels.csv` 列名和样本数
- 晶系分布
- 前几行标签
- 前几条 PXRD 曲线的 min/max/mean/nonzero
- 随机采样 1000 行，检查强度范围是否合理

当前项目里的核心检查结果为：

```text
pxrd.npy: (467861, 10824), float16
labels.csv: 467861 rows
value range: [0, 1]
```

## 6. 生成训练/验证/测试划分

划分脚本是：

```bash
python3 scripts/make_splits.py
```

默认参数：

```text
labels       = dataset/labels.csv
out          = splits/splits.csv
val_frac     = 0.10
test_frac    = 0.10
min_per_class = 10
seed         = 42
```

划分策略在 [src/data.py](/home/exouser/code/Mamba-KAN-PXRD/src/data.py) 的 `make_stratified_split` 中实现：

1. 先过滤 `crystal_system_id >= 0` 的有效样本
2. 按 `space_group` 统计每类样本数
3. 样本数少于 `min_per_class` 的空间群全部放入训练集
4. 其余空间群按 `space_group` 分层抽样
5. 先切出 test，再从剩余数据切出 val
6. 输出 `row,space_group,split,split_id`

当前划分结果：

| split | 样本数 | 覆盖空间群数 |
|---|---:|---:|
| train | 374,294 | 230 |
| val | 46,783 | 223 |
| test | 46,784 | 223 |

因为有 7 个空间群样本数少于 10，它们全部放入训练集，所以 val/test 覆盖 `223/230` 个空间群。

## 7. 训练时如何读取数据

训练脚本读取的是：

```text
dataset/pxrd.npy
dataset/labels.csv
splits/splits.csv
```

数据集类为 [src/data.py](/home/exouser/code/Mamba-KAN-PXRD/src/data.py) 中的 `PXRDDataset`。

读取逻辑：

1. `np.load(..., mmap_mode="r")` 打开 `pxrd.npy`
2. 读取 `labels.csv`
3. 根据 `splits/splits.csv` 传入的 `row` 选择 train/val/test
4. 对 `space_group` 任务，把标签从 `1-230` 转成 `0-229`
5. 对 `crystal_system` 任务，直接使用 `crystal_system_id` 的 `0-6`
6. 每次 `__getitem__` 动态读取一条 PXRD 曲线，并转为 `float32` tensor

这样不需要把 9.5 GiB 的 `pxrd.npy` 一次性加载进内存。

## 8. 可视化分析

预处理后，`analysis/scripts` 里还有两个可视化脚本：

```bash
python3 analysis/scripts/plot_distributions.py
python3 analysis/scripts/plot_intensity_curves.py
```

当前已有图：

```text
analysis/plots/01_space_group_distribution.png
analysis/plots/02_crystal_system_distribution.png
analysis/plots/03_intensity_curves_per_system.png
analysis/plots/04_peak_count_distribution.png
```

这些图主要用于说明：

- 空间群分布是明显长尾
- 晶系分布也不均衡
- 不同晶系的 PXRD 曲线形态差异
- 不同晶系的峰数量分布

## 9. 端到端复现顺序

假设已经从 Science Data Bank 下载并解压 JSON 到本地：

```bash
# 0. 修改 analysis/scripts/*.py 里的 DATA_DIR/OUT_DIR 为本机路径

# 1. 可选：扫描 space_group 元数据
python3 analysis/scripts/scan_metadata.py

# 2. 生成训练数据
python3 analysis/scripts/preprocess.py

# 3. 检查数据
python3 analysis/scripts/check_dataset.py

# 4. 生成划分
python3 scripts/make_splits.py

# 5. 可选：画数据分布和曲线图
python3 analysis/scripts/plot_distributions.py
python3 analysis/scripts/plot_intensity_curves.py

# 6. 训练
python3 scripts/train.py --config configs/default.yaml
```

如果是在当前仓库现有数据上训练，不需要重复前 1-4 步，因为这些文件已经存在：

```text
dataset/pxrd.npy
dataset/labels.csv
splits/splits.csv
```

## 10. 当前仓库数据状态

当前本机数据文件：

| 文件 | 作用 | 当前状态 |
|---|---|---|
| `analysis/metadata.csv` | 轻量元数据，ID + space group | 存在，约 5.3 MB |
| `dataset/pxrd.npy` | 训练用 PXRD 强度矩阵 | 存在，约 9.5 GiB |
| `dataset/labels.csv` | 训练标签与晶系映射 | 存在，约 15 MB |
| `splits/splits.csv` | train/val/test 划分 | 存在，约 7.7 MB |

最终训练集规模：

```text
N = 467,861
signal_length = 10,824
space_group classes = 230
crystal_system classes = 7
```
