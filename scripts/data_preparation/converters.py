import numpy as np
import pandas as pd
import pickle
import os
import argparse
from utils import load_npz, save_csv, save_pickle, ensure_dir

def npz_to_csv(npz_path, key, output_csv):
    """从 npz 文件提取特定键的数据并保存为 csv"""
    print(f"Converting {npz_path} [{key}] to {output_csv}...")
    data = load_npz(npz_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path}. Available keys: {list(data.keys())}")
    
    my_array = data[key]
    # 处理高维数组：reshape 为二维以便存入 CSV
    if my_array.ndim > 2:
        my_array = my_array.reshape(my_array.shape[0], -1)
    
    df = pd.DataFrame(my_array)
    save_csv(df, output_csv)

def csv_to_pickle(csv_path, output_pickle):
    """将 csv 转换为 pickle"""
    print(f"Converting {csv_path} to {output_pickle}...")
    df = pd.read_csv(csv_path)
    save_pickle(df, output_pickle)

def npz_to_pickle(npz_path, output_pickle):
    """将 npz 中的所有数组保存为 pickle 字典"""
    print(f"Converting all arrays from {npz_path} to {output_pickle}...")
    npz_data = load_npz(npz_path)
    data_dict = {f: npz_data[f] for f in npz_data.files}
    save_pickle(data_dict, output_pickle)

def main():
    parser = argparse.ArgumentParser(description="Data Format Converter")
    subparsers = parser.add_subparsers(dest="command")

    # npz2csv
    npz_parser = subparsers.add_parser("npz2csv")
    npz_parser.add_argument("--input", required=True, help="Input .npz file")
    npz_parser.add_argument("--key", required=True, help="Key in npz to extract")
    npz_parser.add_argument("--output", required=True, help="Output .csv file")

    # csv2pickle
    pickle_parser = subparsers.add_parser("csv2pickle")
    pickle_parser.add_argument("--input", required=True, help="Input .csv file")
    pickle_parser.add_argument("--output", required=True, help="Output .pickle file")

    # npz2pickle
    npz_pickle_parser = subparsers.add_parser("npz2pickle")
    npz_pickle_parser.add_argument("--input", required=True, help="Input .npz file")
    npz_pickle_parser.add_argument("--output", required=True, help="Output .pickle file")

    args = parser.parse_args()

    if args.command == "npz2csv":
        npz_to_csv(args.input, args.key, args.output)
    elif args.command == "csv2pickle":
        csv_to_pickle(args.input, args.output)
    elif args.command == "npz2pickle":
        npz_to_pickle(args.input, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
