# API 参考文档

## 模型模块 (src.models)

### DDPM

去噪扩散概率模型。

```python
from src.models import DDPM

ddpm = DDPM(
    model: torch.nn.Module,       # 评分网络
    noise_schedule_type: str = "linear",  # "linear" 或 "cosine"
    N: int = 1000,                # 扩散步数
    t_epsilon: float = 0.001
)
```

**方法**:

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `losses()` | `x, batch, h, context, edge_index` | `Tensor` | 计算 DDPM 损失 |
| `sample()` | `shape, context, edge_index, batch` | `ndarray` | 生成样本 |
| `q_sample()` | `z, t` | `(z_t, epsilon)` | 前向扩散采样 |
| `get_coefs()` | `N, type` | `(betas, alphas, alpha_bars)` | 获取噪声调度系数 |

---

### EGNN

E(n) 等变图神经网络。

```python
from src.models import EGNN

egnn = EGNN(
    in_node_nf: int,      # 输入节点特征维度
    hidden_nf: int,       # 隐藏层维度
    n_layers: int = 3,    # 层数
    out_node_nf: int = None,  # 输出维度
)
```

---

### EGNNScore

用于 DDPM 的评分网络包装。

```python
from src.models import EGNNScore

score = EGNNScore(
    in_node_nf: int,          # 输入维度 (features + time)
    hidden_nf: int,           # 隐藏层维度
    n_layers: int = 3,
    out_node_nf: int = None,
)
```

---

## 训练模块 (src.training)

### Trainer

```python
from src.training import Trainer

trainer = Trainer(
    model: DDPM,
    lr: float = 1e-3,
    checkpoints_path: str = None
)
```

**方法**:

| 方法 | 说明 |
|------|------|
| `train(train_loader, val_loader, epochs, device)` | 训练模型 |
| `save_checkpoint(epoch)` | 保存检查点 |
| `load_checkpoint(path, device)` | 加载检查点，返回 epoch |

---

## 评估模块 (src.evaluation)

### Evaluator

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(ddpm, valid_loader)
```

**方法**:

| 方法 | 返回 | 说明 |
|------|------|------|
| `sample_batch(device)` | `(x, h, ptr)` | 生成一批样本 |
| `eval_stability(x, h, ptr)` | `(atom_st, mol_st)` | 评估稳定性 |
| `eval_plot(x, h, ptr, max_num_plots)` | `None` | 可视化样本 |

---

### check_stability

```python
from src.evaluation.stability_analyze import check_stability

is_stable, num_stable_atoms, total_atoms = check_stability(
    positions: np.ndarray,   # 形状 (N, 3)
    atom_type: np.ndarray,   # 形状 (N,)
    dataset_info: dict = DATASET_INFO
)
```

---

### get_bond_order

```python
from src.evaluation.bond_analyze import get_bond_order

order = get_bond_order(
    atom1: str,      # 原子符号，如 "C"
    atom2: str,      # 原子符号，如 "H"
    distance: float  # 距离（Å）
)
# 返回: 0=无键, 1=单键, 2=双键, 3=三键
```

---

## 数据模块 (src.data)

### AllQM9Dataset

```python
from src.data.all_qm9 import AllQM9Dataset

dataset = AllQM9Dataset(file_path="path/to/data.pickle")
```

### MiniQM9Dataset

```python
from src.data.mini_qm9 import MiniQM9Dataset

dataset = MiniQM9Dataset(file_path="path/to/mini_data.pickle")
```

**Dataset 属性**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `x` | `Tensor` | 原子坐标 |
| `h` | `Tensor` | 原子类型 (one-hot) |
| `edge_index` | `Tensor` | 边索引 |
| `batch` | `Tensor` | 批次索引 |
