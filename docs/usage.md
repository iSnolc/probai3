# 使用指南

## 快速开始

### 1. 训练模型

```python
from src.data.qm9 import QM9Dataset
from src.models.ddpm import DDPM
from src.models.egnn_score import EGNNScore
from src.training.trainer import Trainer
from torch_geometric.loader import DataLoader
import torch
import yaml

# 加载数据
train_dataset = QM9Dataset(file_path="./data/final_data/train.pickle")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

valid_dataset = QM9Dataset(file_path="./data/final_data/valid.pickle")
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# 加载配置
with open("./configs/default.yaml", 'r') as f:
    config = yaml.safe_load(f)

training_config = config['Training']

# 从配置文件读取参数
lr = training_config['lr']
epochs = training_config['epochs']
checkpoint_name = training_config['checkpoint_name']

# 初始化评分网络
score = EGNNScore(
    in_node_nf=6,       # 5 (one-hot) + 1 (time)
    hidden_nf=config['EGNN']['hidden_nf'],
    n_layers=config['EGNN']['n_layers'],
    out_node_nf=5
)

# 初始化扩散模型
ddpm = DDPM(
    model=score,
    noise_schedule_type="linear",
    N=config['DDPM']['N']
)

# 训练
trainer = Trainer(ddpm, lr=lr, checkpoints_path=f"./checkpoints/{checkpoint_name}")
trainer.train(train_loader, valid_loader, epochs=epochs, device=torch.device('cuda'))
```

### 2. 生成分子

```python
from src.evaluation.evaluator import Evaluator

# 创建评估器
evaluator = Evaluator(ddpm, valid_loader)

# 生成并评估一批样本
x, h, ptr = evaluator.sample_batch(device=torch.device('cuda'))

# 评估稳定性
atom_stability, mol_stability = evaluator.eval_stability(x, h, ptr)
print(f"原子稳定性: {atom_stability:.2%}")
print(f"分子稳定性: {mol_stability:.2%}")
```

### 3. 可视化

```python
# 可视化生成的分子并保存图片
evaluator.eval_plot(x, h, ptr, max_num_plots=5)
```

## 命令行运行

### 核心脚本说明

- `scripts/train.py`: 使用完整配置进行模型训练。
- `scripts/valid.py`: 对保存的检查点进行全面验证。
- `scripts/demo_train.py`: 使用小规模数据集快速测试流程。
- `scripts/demo_valid.py`: 快速演示评估过程。

### 运行示例

```bash
# 开启完整训练
python scripts/train.py

# 运行演示验证
python scripts/demo_valid.py
```

## 数据准备流程

如果您需要从原始 QM9 数据重新开始，请使用 `scripts/data_preparation/` 下的工具：

1. **转换格式**: `python converters.py npz2pickle --input ...`
2. **特征提取**: `python process_data.py atom_types --input ...`
3. **清洗数据**: `python clean_data.py --input ...`

详情见 [数据处理 README](../scripts/data_preparation/README.md)。
