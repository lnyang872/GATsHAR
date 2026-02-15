import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm
import json

# =========================================================================
# ============================== 配置区 ===================================
# =========================================================================

# 包含所有MATLAB输出文件夹的根目录
INPUT_BASE_FOLDER = './min'

# 输出HDF5文件的文件夹
OUTPUT_HDF5_FOLDER = './processed_data1001'

# 节点信息文件 (非常重要，用于确定矩阵的顺序和维度)
NODE_INFO_FILE = './node_info.json'

# =========================================================================

def build_h5_file(h5_path, vol_folders, covol_folders, node_order):
    """
    一个通用的辅助函数，用于构建一个HDF5文件。
    """
    num_nodes = len(node_order)
    node_to_idx = {name: i for i, name in enumerate(node_order)}
    
    print(f"\n--- 正在为 {os.path.basename(h5_path)} 加载数据 ---")
    
    # --- (核心修改) 1. 自动确定基准天数 ---
    final_days_length = None
    # 尝试从第一个可用的波动率文件中确定天数
    for folder in vol_folders:
        if os.path.exists(folder) and os.listdir(folder):
            first_file_path = os.path.join(folder, os.listdir(folder)[0])
            df_first = pd.read_csv(first_file_path, header=None)
            final_days_length = df_first.shape[1]
            print(f"[调试信息] 已自动从文件 '{os.path.basename(first_file_path)}' 确定基准天数为: {final_days_length}")
            break # 找到后即退出循环
            
    if final_days_length is None:
        print("错误：所有波动率文件夹均为空，无法确定基准天数。")
        return

    # a. 加载所有波动率数据
    vol_data = {}
    for folder in vol_folders:
        if not os.path.exists(folder): continue
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                asset_id = os.path.splitext(file)[0]
                if asset_id in node_order:
                    df = pd.read_csv(os.path.join(folder, file), header=None)
                    if df.shape[1] >= final_days_length:
                        # 截取最后 final_days_length 列
                        vol_data[asset_id] = df.iloc[:, -final_days_length:].values
                    else:
                        print(f"  -> 警告: 波动率文件 {file} 的天数（{df.shape[1]}）少于要求的 {final_days_length}，已跳过。")

    # b. 加载所有协波动率数据
    covol_data = {}
    for group, folder in covol_folders.items():
        if not os.path.exists(folder): continue
        print(f"正在加载组: {group}")
        for file in os.listdir(folder):
            if file.endswith('.csv'):
                pair_name = os.path.splitext(file)[0]
                suffixes_to_remove = ['_covol_of_vol', '_covol', '_vol_of_vol', '_vol']
                for suffix in suffixes_to_remove:
                    if pair_name.endswith(suffix):
                        pair_name = pair_name[:-len(suffix)]
                        break
                
                df = pd.read_csv(os.path.join(folder, file), header=None)
                if df.shape[1] >= final_days_length:
                    covol_data[pair_name] = df.iloc[:, -final_days_length:].values
                else:
                    print(f"  -> 警告: 协波动率文件 {file} 的天数（{df.shape[1]}）少于要求的 {final_days_length}，已跳过。")
            
    # --- 2. 确定时间和维度 ---
    if not vol_data:
        print(f"错误: 未能加载任何有效的波动率数据，无法继续。")
        return
        
    num_timesteps = next(iter(vol_data.values())).shape[1]
    num_intraday_points = next(iter(vol_data.values())).shape[0]

    print(f"数据加载完成。所有数据已截取为 {num_timesteps} 天, 每天 {num_intraday_points} 个日内时间点。")
    
    # --- 3. 逐个时间点构建矩阵并写入HDF5 ---
    with h5py.File(h5_path, 'w') as f:
        global_timestep_idx = 0
        desc = f"构建矩阵 ({os.path.basename(h5_path)})"
        for day_idx in tqdm(range(num_timesteps), desc=desc):
            for point_idx in range(num_intraday_points):
                
                matrix = np.zeros((num_nodes, num_nodes))
                
                # a. 填充对角线 (波动率)
                for asset_id, data in vol_data.items():
                    if asset_id in node_to_idx:
                        idx = node_to_idx[asset_id]
                        matrix[idx, idx] = data[point_idx, day_idx]
                
                # b. 填充非对角线 (协波动率)
                for pair_name, data in covol_data.items():
                    try:
                        id1, id2 = pair_name.split('_')
                        if id1 in node_to_idx and id2 in node_to_idx:
                            idx1 = node_to_idx[id1]
                            idx2 = node_to_idx[id2]
                            value = data[point_idx, day_idx]
                            matrix[idx1, idx2] = value
                            matrix[idx2, idx1] = value
                    except (ValueError, KeyError):
                        continue

                # c. 将该时间点的矩阵写入HDF5
                f.create_dataset(str(global_timestep_idx), data=matrix, dtype=np.float64)
                global_timestep_idx += 1

    print(f"成功创建HDF5文件: {os.path.basename(h5_path)}")

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_HDF5_FOLDER):
        os.makedirs(OUTPUT_HDF5_FOLDER)
        
    try:
        with open(NODE_INFO_FILE, 'r', encoding='utf-8') as f:
            node_info = json.load(f)
        NODE_ORDER = node_info['node_order']
    except FileNotFoundError:
        raise Exception(f"错误：找不到节点信息文件 {NODE_INFO_FILE}")

    # --- 任务1: 构建 vol_covol.h5 ---
    vol_covol_h5_path = os.path.join(OUTPUT_HDF5_FOLDER, 'vol_covol.h5')
    vol_folders_1 = [
        os.path.join(INPUT_BASE_FOLDER, 'stock_vol'),
        os.path.join(INPUT_BASE_FOLDER, 'energy_vol')
    ]
    covol_folders_1 = {
        'stock_stock': os.path.join(INPUT_BASE_FOLDER, 'covol_stock_stock'),
        'stock_energy': os.path.join(INPUT_BASE_FOLDER, 'covol_stock_energy')
    }
    build_h5_file(vol_covol_h5_path, vol_folders_1, covol_folders_1, NODE_ORDER)
    
    # --- 任务2: 构建 volvol_covolvol.h5 ---
    volvol_covolvol_h5_path = os.path.join(OUTPUT_HDF5_FOLDER, 'volvol_covolvol.h5')
    vol_folders_2 = [
        os.path.join(INPUT_BASE_FOLDER, 'stock_vol_of_vol'),
        os.path.join(INPUT_BASE_FOLDER, 'energy_vol_of_vol')
    ]
    covol_folders_2 = {
        'stock_stock': os.path.join(INPUT_BASE_FOLDER, 'covol_of_vol_stock_stock'),
        'stock_energy': os.path.join(INPUT_BASE_FOLDER, 'covol_of_vol_stock_energy')
    }
    build_h5_file(volvol_covolvol_h5_path, vol_folders_2, covol_folders_2, NODE_ORDER)

    print("\n--- 所有HDF5文件创建完毕！ ---")