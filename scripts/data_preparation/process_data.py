import pandas as pd
import numpy as np
import argparse
import os
from utils import load_csv, save_csv

def convert_element_to_array(element):
    """
    根据 QM9 规则将原子序数转换为对应的 One-hot 布尔数组。
    H: 1, C: 6, N: 7, O: 8, F: 9
    """
    switcher = {
        1: [True, False, False, False, False],
        6: [False, True, False, False, False],
        7: [False, False, True, False, False],
        8: [False, False, False, True, False],
        9: [False, False, False, False, True]
    }
    return switcher.get(element, None)

def process_atom_types(input_csv, output_csv):
    """处理原子电荷/类型 CSV 并转换为模型可读格式"""
    print(f"Processing atom types from {input_csv}...")
    df = load_csv(input_csv)
    converted_data = []

    for _, row in df.iterrows():
        row_array = []
        for element in row:
            if element == 0: continue
            converted_element = convert_element_to_array(element)
            if converted_element:
                row_array.append(converted_element)

        if row_array:
            # 格式化为字符串表示以存入 CSV
            array_str = '\n'.join(['[' + ' '.join(str(e) for e in r) + ']' for r in row_array])
            converted_data.append(["[" + array_str + "]"])

    converted_df = pd.DataFrame(converted_data, columns=['Converted Data'])
    save_csv(converted_df, output_csv)

def process_positions(input_csv, output_csv):
    """处理分子位置坐标，将其格式化为嵌套列表字符串"""
    print(f"Processing positions from {input_csv}...")
    df = load_csv(input_csv, header=None)
    
    formatted_data = []
    for _, row in df.iterrows():
        row_list = row.tolist()
        # 移除填充的零
        try:
            zero_index = row_list.index(0)
            valid_data = row_list[:zero_index]
        except ValueError:
            valid_data = row_list
            
        # 每三个坐标 (x,y,z) 构造成一组
        triples = [valid_data[i:i + 3] for i in range(0, len(valid_data), 3) if len(valid_data[i:i + 3]) == 3]
        formatted_row = "[ " + " ".join("[{:.10f} {:.10f} {:.10f}]".format(*t) for t in triples) + " ]"
        formatted_data.append(formatted_row)

    with open(output_csv, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(item.replace('] [', ']\n[') + '\n')
    print(f"Saved formatted positions to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Molecular Data Processor")
    subparsers = parser.add_subparsers(dest="command")

    # atom_types
    atom_parser = subparsers.add_parser("atom_types")
    atom_parser.add_argument("--input", required=True, help="Input CSV with atomic numbers")
    atom_parser.add_argument("--output", required=True, help="Output formatted CSV")

    # positions
    pos_parser = subparsers.add_parser("positions")
    pos_parser.add_argument("--input", required=True, help="Input CSV with raw positions")
    pos_parser.add_argument("--output", required=True, help="Output formatted positions CSV")

    args = parser.parse_args()

    if args.command == "atom_types":
        process_atom_types(args.input, args.output)
    elif args.command == "positions":
        process_positions(args.input, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
