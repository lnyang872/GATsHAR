import h5py
import numpy as np
import os
from tqdm import tqdm
import json

# =========================================================================
# ============================== Deployment Zone ===================================
# =========================================================================
# 1. List of raw HDF5 files requiring processing
HDF5_FILES_TO_PROCESS = [
    './processed_data1001/vol_covol.h5',
    './processed_data1001/volvol_covolvol.h5'
]

# 2. Designated Node Information File (distinguishing between equities and energy)
NODE_INFO_FILE = './node_info.json'

# 3. The suffix added to the output file
OUTPUT_SUFFIX = '_standardized'
# =========================================================================

def standardize_h5_granularly(file_path, node_info):
    """
    Perform Z-score standardisation on the matrix within the HDF5 file according to asset type (equities/energy).
    """
    print("="*70)
    print(f"--- Commencing processing of the file: {os.path.basename(file_path)} ---")
    
    if not os.path.exists(file_path):
        print(f"\nError: File '{file_path}' not found")
        return

    # --- Prepare index information ---
    node_order = node_info['node_order']
    stock_ids = node_info['stock_ids']
    energy_ids = node_info['energy_ids']
    
    stock_indices = [node_order.index(sid) for sid in stock_ids]
    energy_indices = [node_order.index(eid) for eid in energy_ids]

    # ---  1: Read all data to compute the statistics for each group ---
    print("\n--- Read data to compute statistics... ---")
    
    # Initialise lists for five distinct groups
    stock_diagonals, energy_diagonals = [], []
    ss_off_diagonals, ee_off_diagonals, se_off_diagonals = [], [], []

    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            if not keys:
                print("The file is empty; skipped.")
                return

            for key in tqdm(keys, desc="  - Read Matrix"):
                matrix = f[key][()]
                
                # Extract the diagonal
                stock_diagonals.append(matrix[stock_indices, stock_indices])
                energy_diagonals.append(matrix[energy_indices, energy_indices])
                
                # Extract non-diagonal elements
                # Stock-Stock (ss)
                ss_block = matrix[np.ix_(stock_indices, stock_indices)]
                ss_off_diagonals.append(ss_block[np.triu_indices_from(ss_block, k=1)])
                
                # Energy-Energy (ee)
                ee_block = matrix[np.ix_(energy_indices, energy_indices)]
                ee_off_diagonals.append(ee_block[np.triu_indices_from(ee_block, k=1)])
                
                # Stock-Energy (se)
                se_block = matrix[np.ix_(stock_indices, energy_indices)]
                se_off_diagonals.append(se_block.flatten())

        # Merge and compute statistics
        stats = {}
        groups = {
            'stock_diag': np.concatenate(stock_diagonals),
            'energy_diag': np.concatenate(energy_diagonals),
            'ss_off_diag': np.concatenate(ss_off_diagonals),
            'ee_off_diag': np.concatenate(ee_off_diagonals),
            'se_off_diag': np.concatenate(se_off_diagonals)
        }
        
        print("\nStatistical calculation completed:")
        for name, data in groups.items():
            mean, std = np.mean(data), np.std(data)
            stats[f'{name}_mean'] = float(mean)
            stats[f'{name}_std'] = float(std)
            print(f"  - {name:<15}: Mean={mean:.6e}, Std={std:.6e}")

        # Save statistics to a JSON file
        base, _ = os.path.splitext(file_path)
        stats_path = f"{base}_stats_granular.json"
        with open(stats_path, 'w') as f_stats:
            json.dump(stats, f_stats, indent=4)
        print(f"Detailed statistical information has been saved to: {stats_path}")

    except Exception as e:
        print(f"\nAn error occurred while reading/calculating statistics: {e}")
        return

    # ---  2: Apply standardisation and incorporate into a new document ---
    print("\n---  Apply standardisation and incorporate into a new document... ---")
    
    base, ext = os.path.splitext(file_path)
    output_path = f"{base}{OUTPUT_SUFFIX}{ext}"

    try:
        with h5py.File(file_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            keys = list(f_in.keys())
            
            for key in tqdm(keys, desc="  - Standardisation Matrix"):
                original_matrix = f_in[key][()]
                standardized_matrix = np.zeros_like(original_matrix, dtype=np.float64)
                
                # a. Standardised diagonal
                std_val = stats['stock_diag_std']
                if std_val > 1e-9:
                    standardized_matrix[stock_indices, stock_indices] = (original_matrix[stock_indices, stock_indices] - stats['stock_diag_mean']) / std_val
                
                std_val = stats['energy_diag_std']
                if std_val > 1e-9:
                    standardized_matrix[energy_indices, energy_indices] = (original_matrix[energy_indices, energy_indices] - stats['energy_diag_mean']) / std_val

                # b. Standardised off-diagonal
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

                # Restore the diagonals (as block operations will overwrite them)
                np.fill_diagonal(standardized_matrix, np.diag(standardized_matrix))

                f_out.create_dataset(key, data=standardized_matrix)
                
        print(f"\nStandardisation successful! Output file saved to: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"\nAn error occurred during standardisation/writing: {e}")
    
    print("="*70)


if __name__ == '__main__':
    try:
        with open(NODE_INFO_FILE, 'r', encoding='utf-8') as f:
            node_information = json.load(f)
    except FileNotFoundError:
        raise Exception(f"Error: Node information file '{NODE_INFO_FILE}' not found.")

    for h5_file in HDF5_FILES_TO_PROCESS:
        standardize_h5_granularly(h5_file, node_information)
        
    print("\nAll documents have been processed.")
