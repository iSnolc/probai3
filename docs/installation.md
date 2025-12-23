# 安装指南

## 系统要求

- Python 3.10+
- CUDA 11.8+ (可选，用于 GPU 加速)

## 方式一：使用 Conda（推荐）

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate mol_diff
```

## 方式二：使用 pip

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

## 验证安装

```bash
# 检查项目核心导入
python -c "from src.models.ddpm import DDPM; from src.models.egnn_score import EGNNScore; print('Import OK')"

# 运行测试
pytest tests/ -v
```

## 数据准备

1. **获取预处理数据**：
   - 推荐从 [bird001/qm9-for-probai3](https://huggingface.co/datasets/bird001/qm9-for-probai3) 下载。

2. **放置路径**：
   将 `.pickle` 文件放置在 `data/final_data/` 目录下。

```text
data/
└── final_data/
    ├── train.pickle
    └── valid.pickle
```

3. **从头开始处理**（可选）：
   如果您下载的是原始 `.npz` 数据，请使用 `scripts/data_preparation/` 目录下的工具进行转换。详情请参阅该目录下的 `README.md`。

## 常见问题

### PyTorch 无法导入

确保激活了正确的 conda 环境：
```bash
conda activate mol_diff
```

### CUDA 不可用

检查 CUDA 版本兼容性：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
