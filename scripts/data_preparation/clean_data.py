import pandas as pd
import numpy as np
import pickle
import argparse
import os
from utils import save_pickle

def object_to_bool(pickle_path, key, output_pickle):
    """将 pickle 文件中特定键的 object 类型数组强制转换为 bool 类型"""
    print(f"Cleaning types (Object -> Bool) in {pickle_path} for key '{key}'...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    if key not in data:
        print(f"Key '{key}' not found, skipping.")
        return

    nested_arr = data[key]
    # 执行转换
    arr_bool = [arr.astype(bool) for arr in nested_arr]
    data[key] = arr_bool
    
    save_pickle(data, output_pickle)

def main():
    parser = argparse.ArgumentParser(description="Data Cleaning Tool")
    parser.add_argument("--input", required=True, help="Input pickle file")
    parser.add_argument("--key", default="atom_types", help="Key to convert to bool (default: atom_types)")
    parser.add_argument("--output", required=True, help="Output pickle file")

    args = parser.parse_args()
    object_to_bool(args.input, args.key, args.output)

if __name__ == "__main__":
    main()
