import sys
import os

# 将上一级目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.mini_qm9 import MiniQM9Dataset
from src.data.all_qm9 import AllQM9Dataset
from torch_geometric.loader import DataLoader
from src.models.ddpm import DDPM
from src.models.egnn import EGNNScore
from src.training.training_loop import Trainer
import matplotlib.pyplot as plt
import torch
import yaml

train_dataset = MiniQM9Dataset(file_path=f"../data/demo_train_data.pickle")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

valid_dataset = MiniQM9Dataset(file_path=f"../data/demo_valid_data.pickle")
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

# Initialize EGNN
with open("../configs/demo.yaml", 'r') as file:
    config = yaml.safe_load(file)

egnn_config = config['EGNN']
training_config = config['Training'] # 强制要求 Training 块

# 直接从配置读取，缺失将报错
lr = training_config['lr']
epochs = training_config['epochs']
batch_size = training_config['batch_size']
checkpoint_name = training_config['checkpoint_name']
device = training_config['device']




hidden_nf = egnn_config['hidden_nf']
n_layers = egnn_config['n_layers']
score = EGNNScore(in_node_nf=5 + 1,  # 5 for the one hot encoding, 1 for diffusion time
                  hidden_nf=hidden_nf,
                  n_layers=n_layers,
                  out_node_nf=5)  # 5 atom types in QM9

# Initialize DDPM
ddpm_config = config['DDPM']
N = config['DDPM']['N']  # Numbero of noise level, default set to 100
ddpm = DDPM(noise_schedule_type="linear", model=score, N=N)

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt_path = os.path.join(project_root, "checkpoints", training_config['checkpoint_name'])
log_csv_path = os.path.join(project_root, "output", "train_valid_loss.csv")

trainer = Trainer(
    ddpm, 
    lr=lr, 
    checkpoints_path=ckpt_path,
    log_path=log_csv_path
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

trainer.train(train_loader, valid_loader, epochs=epochs, device=torch.device(device))



print(trainer.val_losses)
plt.plot(trainer.val_losses)
