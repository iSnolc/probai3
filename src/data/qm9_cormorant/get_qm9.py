# -*- coding: utf-8 -*-
# @Author: Truman
# @Date:   2024/4/24 16:01
# @Last Modified by:   Truman
# @Last Modified time: 2024-04-24 16:01:57

import torch
import pandas as pd
import torchvision
from torch import nn
from torch.nn import Conv2d, Sequential, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
from src.data.qm9_cormorant.prepare import download_dataset_qm9
from src.data.qm9_cormorant.prepare import gen_splits_gdb9, get_thermo_dict, add_thermo_targets, get_unique_charges
from src.data.qm9_cormorant.prepare.download import download_dataset_md17

datadir = 'qm_cormorant'
dataname = 'gdb9dir'

gdb9dir = 'gdb9dir'

data = 'gdb9dir/dsgdb9nsd.xyz.tar.bz2'

# download_dataset_qm9(datadir='.', dataname='gdb9dir', splits=None, calculate_thermo=True,
#                      exclude=True, cleanup=True)

# splits = gen_splits_gdb9(gdb9dir, cleanup=True)
# print(splits)

therm_energy_dict = get_thermo_dict(gdb9dir, cleanup=True)

data = add_thermo_targets(data, therm_energy_dict)
print(data)

charge_counts = get_unique_charges(len())
