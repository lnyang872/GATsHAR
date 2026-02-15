import h5py
import numpy as np
import os
from tqdm import tqdm
import json

# =========================================================================
# ============================== 配置区 ===================================
# =========================================================================
# 1. 指向需要处理的原始HDF5文件列表
HDF5_FILES_TO_PROCESS = [
    './processed_data1001/vol_covol.h5',
    './processed_data1001/volvol_covolvol.h5'
]

# 2. 指定节点信息文件 (用于区分股票和能源)
NODE_INFO_FILE = './node_info.json'

# 3. 为输出文件添加的后缀
OUTPUT_SUFFIX = '_standardized'
# =========================================================================

def standardize_h5_granularly(file_path, node_info):
    """
    根据资产类型（股票/能源）对HDF5文件中的矩阵进行精细化的Z-score标准化。
    """
    print("="*70)
    print(f"--- 开始处理文件: {os.path.basename(file_path)} ---")
    
    if not os.path.exists(file_path):
        print(f"\n错误: 文件 '{file_path}' 未找到！")
        return

    # --- 准备索引信息 ---
    node_order = node_info['node_order']
    stock_ids = node_info['stock_ids']
    energy_ids = node_info['energy_ids']
    
    stock_indices = [node_order.index(sid) for sid in stock_ids]
    energy_indices = [node_order.index(eid) for eid in energy_ids]

    # --- 步骤 1: 读取所有数据以计算各组的统计量 ---
    print("\n--- 步骤 1/2: 读取数据以计算统计量... ---")
    
    # 为5个不同的组初始化列表
    stock_diagonals, energy_diagonals = [], []
    ss_off_diagonals, ee_off_diagonals, se_off_diagonals = [], [], []

    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            if not keys:
                print("文件为空，跳过。")
                return

            for key in tqdm(keys, desc="  - 读取矩阵"):
                matrix = f[key][()]
                
                # 提取对角线
                stock_diagonals.append(matrix[stock_indices, stock_indices])
                energy_diagonals.append(matrix[energy_indices, energy_indices])
                
                # 提取非对角线
                # Stock-Stock (ss)
                ss_block = matrix[np.ix_(stock_indices, stock_indices)]
                ss_off_diagonals.append(ss_block[np.triu_indices_from(ss_block, k=1)])
                
                # Energy-Energy (ee)
                ee_block = matrix[np.ix_(energy_indices, energy_indices)]
                ee_off_diagonals.append(ee_block[np.triu_indices_from(ee_block, k=1)])
                
                # Stock-Energy (se)
                se_block = matrix[np.ix_(stock_indices, energy_indices)]
                se_off_diagonals.append(se_block.flatten())

        # 合并并计算统计量
        stats = {}
        groups = {
            'stock_diag': np.concatenate(stock_diagonals),
            'energy_diag': np.concatenate(energy_diagonals),
            'ss_off_diag': np.concatenate(ss_off_diagonals),
            'ee_off_diag': np.concatenate(ee_off_diagonals),
            'se_off_diag': np.concatenate(se_off_diagonals)
        }
        
        print("\n统计量计算完成:")
        for name, data in groups.items():
            mean, std = np.mean(data), np.std(data)
            stats[f'{name}_mean'] = float(mean)
            stats[f'{name}_std'] = float(std)
            print(f"  - {name:<15}: Mean={mean:.6e}, Std={std:.6e}")

        # 保存统计量到JSON文件
        base, _ = os.path.splitext(file_path)
        stats_path = f"{base}_stats_granular.json"
        with open(stats_path, 'w') as f_stats:
            json.dump(stats, f_stats, indent=4)
        print(f"详细统计信息已保存至: {stats_path}")

    except Exception as e:
        print(f"\n在步骤1（读取/计算统计量）中发生错误: {e}")
        return

    # --- 步骤 2: 应用标准化并写入新文件 ---
    print("\n--- 步骤 2/2: 应用标准化并写入新文件... ---")
    
    base, ext = os.path.splitext(file_path)
    output_path = f"{base}{OUTPUT_SUFFIX}{ext}"

    try:
        with h5py.File(file_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            keys = list(f_in.keys())
            
            for key in tqdm(keys, desc="  - 标准化矩阵"):
                original_matrix = f_in[key][()]
                standardized_matrix = np.zeros_like(original_matrix, dtype=np.float64)
                
                # a. 标准化对角线
                std_val = stats['stock_diag_std']
                if std_val > 1e-9:
                    standardized_matrix[stock_indices, stock_indices] = (original_matrix[stock_indices, stock_indices] - stats['stock_diag_mean']) / std_val
                
                std_val = stats['energy_diag_std']
                if std_val > 1e-9:
                    standardized_matrix[energy_indices, energy_indices] = (original_matrix[energy_indices, energy_indices] - stats['energy_diag_mean']) / std_val

                # b. 标准化非对角线
                # Stock-Stock
                ss_block = original_matrix[np.ix_(stock_indices, stock_indices)]
                std_val = stats['ss_off_diag_std']
                if std_val > 1e-9:
                    ss_block_std = (ss_block - stats['ss_off_diag_mean']) / std_val
                    standardized_matrix[np.ix_(stock_indices, stock_indices)] = ss_block_std

                # Energy-Energy
                ee_block = original_matrix[np.ix_(energy_indices, energy_indices)]
                std_val = stats['ee_off_diag_std']
                if std_val > 1e-9:
                    ee_block_std = (ee_block - stats['ee_off_diag_mean']) / std_val
                    standardized_matrix[np.ix_(energy_indices, energy_indices)] = ee_block_std

                # Stock-Energy
                se_block = original_matrix[np.ix_(stock_indices, energy_indices)]
                std_val = stats['se_off_diag_std']
                if std_val > 1e-9:
                    se_block_std = (se_block - stats['se_off_diag_mean']) / std_val
                    standardized_matrix[np.ix_(stock_indices, energy_indices)] = se_block_std
                    standardized_matrix[np.ix_(energy_indices, stock_indices)] = se_block_std.T # 保持对称

                # 恢复对角线 (因为块操作会覆盖它们)
                np.fill_diagonal(standardized_matrix, np.diag(standardized_matrix))

                f_out.create_dataset(key, data=standardized_matrix)
                
        print(f"\n标准化成功！输出文件已保存至: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"\n在步骤2（标准化/写入）中发生错误: {e}")
    
    print("="*70)


if __name__ == '__main__':
    try:
        with open(NODE_INFO_FILE, 'r', encoding='utf-8') as f:
            node_information = json.load(f)
    except FileNotFoundError:
        raise Exception(f"错误：找不到节点信息文件 '{NODE_INFO_FILE}'")

    for h5_file in HDF5_FILES_TO_PROCESS:
        standardize_h5_granularly(h5_file, node_information)
        
    print("\n所有文件处理完毕。")
