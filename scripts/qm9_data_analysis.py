import sys
import os

# 将项目根目录添加到系统路径中，确保能顺利导入 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.data.mini_qm9 import MiniQM9Dataset
from torch_geometric.loader import DataLoader
from src.evaluation.visualizer import plot_data3d
from src.evaluation.stability_analyze import check_stability
import torch
from tqdm import tqdm

# Fo this workshop we have defined the MiniQM9Dataset. Which defines a subset of QM9 dataset with molecules of maximum 15 atoms.
# Next we initialize the dataset and dataloaders.

# 这段代码是一个完整的分子图数据处理、可视化和稳定性分析流程，主要用于处理和分析化学分子数据集（特别是QM9数据集的一个子集）

dataset_train = MiniQM9Dataset(file_path=f"../data/demo_train_data.pickle")
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

dataset_valid = MiniQM9Dataset(file_path=f"../data/demo_valid_data.pickle")
dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
print(f"Number of training samples: \t{len(dataset_train)} \nNumber of validations samples: \t{len(dataset_valid)}")

# Analyze the structure of a single by loading a sample from the dataloader with batch_size = 1.
for batch in dataloader_train:
    # x: The positions of the atoms in the molecule (coordinates)
    print(f"x (3D coordinates): \t \t \t {batch.x.shape}")

    # h: The one-hot representation of the atom types
    print(f"h (one-hot atom types): \t\t {batch.h.shape}")

    # edge_index: The adjacency matrix of the molecular graph
    print(f"edge_index (adjacency matrix): \t\t {batch.edge_index.shape}")

    # context: The context index of each graph
    print(f"context (mol property: polarizability):\t {batch.context.shape}")

    # batch: A tensor assigning each node to its respective graph in the batch
    print(f"batch (node-to-graph assignment): \t {batch.batch.shape}")
    print(batch.h)
    break

# Load and visualize a few samples from the MiniQM9Dataset
num_plotted_samples = 5
for idx, batch in enumerate(dataset_train):
    # Print the current batch information
    print(batch)

    # Convert the one-hot encoded atom types to integer labels
    h_atom_type = torch.argmax(batch.h, dim=1).numpy()

    # Visualize the molecular structure using plot_data3d
    # batch.x contains the 3D coordinates of the atoms
    # h_atom_type contains the atom type labels
    # spheres_3d=True enables the 3D sphere representation for atoms. Se to False for faster speed.
    # plot_data3d(batch.x, h_atom_type, spheres_3d=True)

    if idx >= num_plotted_samples:
        break

# Initialize a dictionary to store the counts of stable molecules and atoms, as well as the total number of molecules and atoms
st_dict = {"num_stable_mols": 0, "num_mols":0, "num_stable_atoms": 0, "num_atoms": 0}
for idx, batch in enumerate(tqdm(dataloader_valid, desc="Evaluating stability")):

    # Convert the one-hot encoded atom types to integer labels
    h_atom_type = torch.argmax(batch.h, dim=1).numpy()

    # Check the stability of the current molecule and count the number of stable atoms.
    mol_stable, num_stable_atoms, num_atoms = check_stability(batch.x, h_atom_type)

    # Update the stability dictionary with the results
    st_dict["num_stable_mols"] += mol_stable
    st_dict["num_mols"] += 1
    st_dict["num_stable_atoms"] += num_stable_atoms
    st_dict["num_atoms"] += num_atoms


# Calculate the atom and molecule stability ratios and print the results.
atom_st = st_dict['num_stable_atoms']/st_dict['num_atoms'] #原子稳定性分析，分析原子在分子结构中的稳定性
mol_st = st_dict['num_stable_mols']/st_dict['num_mols'] # 分子稳定性分析，计算完全稳定的分子占比
print(f"Atom stability: {atom_st} \t Molecule Stability {mol_st}")