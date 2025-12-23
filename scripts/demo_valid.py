import sys
import os

# 将上一级目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil

from src.data.mini_qm9 import MiniQM9Dataset
from torch_geometric.loader import DataLoader
from src.models.ddpm import DDPM
from src.models.egnn import EGNNScore
from src.training.training_loop import Trainer
from src.evaluation.evaluator import Evaluator
import torch
import yaml

# Initialize EGNN
with open("../configs/demo.yaml", 'r') as file:
    config = yaml.safe_load(file)

egnn_config = config['EGNN']
training_config = config['Training']
device = training_config['device']

# Loading the validation dataset and creating a DataLoader
dataset_valid = MiniQM9Dataset(file_path=f"../data/demo_valid_data.pickle")
loader_valid = DataLoader(dataset_valid, batch_size=training_config['batch_size'], shuffle=False)

hidden_nf = egnn_config['hidden_nf']
n_layers = egnn_config['n_layers']
score = EGNNScore(in_node_nf=5 + 1,  # 5 for the one hot encoding, 1 for diffusion time
                  hidden_nf=hidden_nf,
                  n_layers=n_layers,
                  out_node_nf=5)  # 5 atom types in QM9

# Initialize DDPM and load checkpoint
ddpm_config = config['DDPM']
N = config['DDPM']['N']  # Numbero of noise level, default set to 100
ddpm = DDPM(noise_schedule_type="linear", model=score, N=N)
trainer = Trainer(ddpm)

# 动态计算绝对路径，确保加载正确
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt_path = os.path.join(project_root, "checkpoints", "demo_egnn_checkpoint.pth")
trainer.load_checkpoint(ckpt_path)


# Generate some samples (same as loader_valid.batch_size)
evaluator = Evaluator(ddpm, valid_loader=loader_valid)
x, h, ptr = evaluator.sample_batch(device=torch.device(device))

h_atom_type = torch.argmax(h, dim=1).numpy()
print(len(x), len(h_atom_type), len(h), len(ptr))
# print(x, h_atom_type, ptr)
# Evaluate atom and molecule stabilities.
# For a model trained in few epochs we should expect good atom stability and ver low molecule stability.
# Large Molecule stability would require longer trainings

atom_st, mol_st = evaluator.eval_stability(x, h_atom_type, ptr)
print(f"Atom stability: {atom_st} \t Molecule Stability {mol_st}")

# Print some sample generated with the trained model
evaluator.eval_plot(x, h_atom_type, ptr, max_num_plots=10, save_path="../output/plot_image.png")


