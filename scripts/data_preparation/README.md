# 数据预处理脚本使用说明 (Data Preparation)

本目录包含了用于处理 QM9 原始数据集的整合逻辑。通过将原本散乱的脚本整合为模块化工具，提升了数据处理的效率和规范性。

## 目录结构

- `converters.py`: **格式转换工具**。负责 npz, csv, pickle, npy 之间的相互转换。
- `process_data.py`: **核心处理工具**。负责处理分子原子类型编码（Atom Types）和位置坐标（Positions）。
- `clean_data.py`: **数据清洗工具**。处理数据类型转换（如 Object 转 Bool）等清洗操作。
- `utils.py`: **通用工具类**。包含文件读写、目录创建等基础函数。

---

## 快速使用指南

### 1. 格式转换 (`converters.py`)

将 `.npz` 中的特定属性提取为 `.csv`:
```bash
python converters.py npz2csv --input ../../data/raw/train.npz --key alpha --output train_alpha.csv
```

将 `.csv` 封装为 `.pickle`:
```bash
python converters.py csv2pickle --input train_data.csv --output train_data.pickle
```

一键将 `.npz` 中的所有数组转换为 `.pickle` 字典:
```bash
python converters.py npz2pickle --input raw_data.npz --output processed_data.pickle
```

### 2. 分子数据处理 (`process_data.py`)

处理原子序数并生成 One-hot 编码格式 (QM9):
```bash
python process_data.py atom_types --input charges.csv --output atom_types.csv
```

格式化分子位置坐标 (每行三坐标嵌套):
```bash
python process_data.py positions --input raw_pos.csv --output formatted_pos.csv
```

### 3. 数据类型清洗 (`clean_data.py`)

将 Pickle 文件中的 `atom_types` 从 `object` 类型转换为 `bool` 类型，以减少内存占用并适配模型：
```bash
python clean_data.py --input final_data.pickle --key atom_types --output cleaned_data.pickle
```

---

## 开发规范建议
1. **统一路径管理**：脚本中使用相对路径时，请确保工作目录正确。建议通过 `argparse` 从外部传入路径。
2. **逻辑沉淀**：如果发现新的通用处理逻辑，请优先提取到 `utils.py` 或 `process_data.py` 的函数中，避免重复造轮子。
3. **实验代码**：新的尝试性代码请放在 `experimental/` 目录下，完成后再择优整合入核心脚本。
