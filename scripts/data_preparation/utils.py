import os
import pandas as pd
import numpy as np
import pickle
import ast

def load_npz(file_path):
    """加载 npz 文件"""
    return np.load(file_path, allow_pickle=True)

def load_csv(file_path, **kwargs):
    """加载 csv 文件"""
    return pd.read_csv(file_path, **kwargs)

def save_csv(df, output_path, index=False):
    """保存为 csv 文件"""
    df.to_csv(output_path, index=index)
    print(f"Saved: {output_path}")

def save_pickle(obj, output_path):
    """保存为 pickle 文件"""
    with open(output_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {output_path}")

def ensure_dir(path):
    """确保目录存在"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

