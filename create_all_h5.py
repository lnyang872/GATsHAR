import yaml
import os
from utils.datasetchange import FinalHeteroDataset
import pandas as pd 

def create_all_caches():
    """
    A utility script for batch-creating dataset caches for all specified seq_length values.
    It reads the tuning configuration file and generates a dedicated PyTorch Geometric cache folder 
    for each seq_length value therein.
    """
    # --- 1. Load the configuration file for hyperparameter tuning ---
    try:
        config_path = './config/GNN_param_optuna.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            p = yaml.safe_load(f)
        print(f"--- Configuration file loaded successfully: {config_path} ---")
    except FileNotFoundError:
        print(f"Error: Tuning profile not found '{config_path}'.")
        return

    # --- 2. Retrieve all seq_lengths requiring processing ---
    try:
        seq_lengths_to_process = p['hyperparameters']['seq_length'][0]
        if not isinstance(seq_lengths_to_process, list):
            print("Error: The format of seq_length in the configuration file is incorrect.")
            return
        print(f"--- Prepare to create a cache for the following seq_length: {seq_lengths_to_process} ---")
    except (KeyError, IndexError):
        print("Error: The 'seq_length' setting could not be found in the configuration file.")
        return

    # --- 3. Create each cache file in a loop ---
    for seq_length in seq_lengths_to_process:
        print(f"\n=================================================")
        print(f"====== Commencing processing seq_length = {seq_length} ======")
        print(f"=================================================")
        
        # Define the cache path specific to the current seq_length
        root_path = f"processed_data1001/final_hetero_dataset_seq_{seq_length}"
        
        # Check whether the cache already exists
        if os.path.exists(os.path.join(root_path, 'processed', 'data.pt')):
            print(f"--- The cache already exists at '{root_path}', skipping. ---")
            continue
            
        try:
            # Instantiate the Dataset class. If the cache does not exist, it will automatically invoke the process() method.
            # We are here 'pretending' to load it, thereby triggering its creation process.
            print(f"--- Creating a new dataset cache for seq_length={seq_length}... ---")
            dataset = FinalHeteroDataset(
                hdf5_file_vol=p['hdf5_file_vol_std'],
                hdf5_file_volvol=p['hdf5_file_volvol_std'],
                stock_har_rv_folder=p['stock_har_rv_folder'],
                energy_har_rv_folder=p['energy_har_rv_folder'],
                stock_energy_corr_folder=p['stock_energy_corr_folder'],
                energy_energy_corr_folder=p['energy_energy_corr_folder'],
                node_info_file=p['node_info_file'],
                root=root_path,
                seq_length=seq_length,
                intraday_points=p['intraday_points']
            )
            print(f"--- Cache creation for seq_length = {seq_length} successful---")

        except Exception as e:
            print(f"!!!!!! A fatal error occurred while processing seq_length = {seq_length}: {e} !!!!!!")
            # Even if one fails, keep trying the next one.
            continue
            
    print("\n=================================================")
    print("====== All specified cache files have been processed. ======")
    print("=================================================")


if __name__ == '__main__':
    
    create_all_caches()
