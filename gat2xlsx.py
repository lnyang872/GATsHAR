# unified_prediction_script.py

import os
import yaml
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils.losses import QLIKELoss
from utils.datasetchange import FinalHeteroDataset
from utils.modelschange import HeteroGNNModel

# ==============================================================================
# I. Predictive Generative Function
# ==============================================================================

def generate_predictions(model, loader, device, p, stock_ids, diag_mean_high, diag_std_high, diag_mean_low, diag_std_low):
    """
    Generate predictions and return a Pandas DataFrame containing both actual and predicted values.
    """
    model.eval()
    all_preds_h, all_actuals_h, all_preds_l, all_actuals_l = [], [], [], []
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Generating prediction"):
            data = data.to(device)
            pred_high, pred_low = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            if pred_high.numel() == 0: continue
            
            y_high, y_low = data['stock'].y_high, data['stock'].y_low
            
            # anti-standardisation
            pred_high_unstd = (pred_high * torch.tensor(diag_std_high, device=device)) + torch.tensor(diag_mean_high, device=device)
            y_high_unstd = (y_high * torch.tensor(diag_std_high, device=device)) + torch.tensor(diag_mean_high, device=device)
            pred_low_unstd = (pred_low * torch.tensor(diag_std_low, device=device)) + torch.tensor(diag_mean_low, device=device)
            y_low_unstd = (y_low * torch.tensor(diag_std_low, device=device)) + torch.tensor(diag_mean_low, device=device)
            
            all_preds_h.append(pred_high_unstd.cpu())
            all_actuals_h.append(y_high_unstd.cpu())
            all_preds_l.append(pred_low_unstd.cpu())
            all_actuals_l.append(y_low_unstd.cpu())

    # Combine the results of all batches
    preds_h, actuals_h = torch.cat(all_preds_h, dim=0), torch.cat(all_actuals_h, dim=0)
    preds_l, actuals_l = torch.cat(all_preds_l, dim=0), torch.cat(all_actuals_l, dim=0)
    
    # Non-negativity constraint on predicted values (to prevent QLIKE explosion)
    PREDICTION_LOWER_BOUND = 1e-8
    preds_h = torch.clamp(preds_h, min=PREDICTION_LOWER_BOUND)
    preds_l = torch.clamp(preds_l, min=PREDICTION_LOWER_BOUND)
    
    num_stocks = len(stock_ids)
    
    # High-frequency data remodelling
    preds_h = preds_h.view(-1, num_stocks)
    actuals_h = actuals_h.view(-1, num_stocks)
    
    # Low-frequency data reshaping and daily aggregation
    preds_l = preds_l.view(-1, num_stocks)
    actuals_l = actuals_l.view(-1, num_stocks)
    
    intraday_points = p.get('intraday_points', 3)
    num_samples = preds_l.shape[0]
    
    if num_samples > 0 and num_samples % intraday_points == 0:
        num_days = num_samples // intraday_points
        # Low-frequency forecast values shall be taken as the intraday average.
        avg_preds_per_day = preds_l.reshape(num_days, intraday_points, num_stocks).mean(dim=1)
        # The low-frequency actual value shall be taken as the last point within the day.（the RV of T+1d）
        actuals_per_day = actuals_l.reshape(num_days, intraday_points, num_stocks)[:, -1, :]
    else:
        avg_preds_per_day = torch.empty(0, num_stocks)
        actuals_per_day = torch.empty(0, num_stocks)

    # Create a high-frequency DataFrame
    df_h_true = pd.DataFrame(actuals_h.numpy(), columns=stock_ids)
    df_h_pred = pd.DataFrame(preds_h.numpy(), columns=stock_ids)
    df_h_combined = pd.concat([df_h_true, df_h_pred], axis=1, keys=['True', 'Predicted'])

    # Create a low-frequency DataFrame
    df_l_true = pd.DataFrame(actuals_per_day.numpy(), columns=stock_ids)
    df_l_pred = pd.DataFrame(avg_preds_per_day.numpy(), columns=stock_ids)
    df_l_combined = pd.concat([df_l_true, df_l_pred], axis=1, keys=['True', 'Predicted'])

    return df_h_combined, df_l_combined

# ==============================================================================
# II. Excel Combination Functions
# ==============================================================================

def combine_to_excel(df_val, df_test, output_excel_path, file_prefix):
    """
    Convert the wide-format validation and test set DataFrames into long format and save them as separate Excel sheets.
    """
    print(f"\n--- processing {file_prefix} data and write to Excel ---")
    
    all_long_dfs = []
    
    for df_wide, dataset_name in [(df_val, 'Validation'), (df_test, 'Test')]:
        if df_wide.empty:
            print(f"Warning: The DataFrame for {dataset_name} is empty and has been skipped.")
            continue
        df_long = df_wide.stack(level=1)
        df_long = df_long.reset_index()
        df_long = df_long.rename(columns={'level_0': 'Sample_Index', 'level_1': 'stock_id',
                                         'True': '真实值', 'Predicted': '预测值'})
        df_long['Dataset'] = dataset_name
        all_long_dfs.append(df_long)

    if not all_long_dfs:
        print("Error: No valid data available to write to Excel.")
        return

    all_results_df = pd.concat(all_long_dfs, ignore_index=True)
    
    print(f"--- Writing the results to an Excel file: {output_excel_path} ---")
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        # Group by the 'stock_id' column
        for stock_id, group_df in tqdm(all_results_df.groupby('stock_id'), desc="write to Sheet"):
           
            # Select the three columns ultimately required.
            sheet_data = group_df[['Actual value', 'Predicted value', 'Dataset']]
            
            # Write to the sheet named after the stock ID
            sheet_data.to_excel(writer, sheet_name=str(stock_id), index=False)

    print(f"File successfully created: {output_excel_path}")

# ==============================================================================
# III. Main process function (including 70% robustness check )
# ==============================================================================

def main():
    # ==============================================================================
    # ======================== Deployment Zone =====================================
    # ==============================================================================
    
    # Output folder path for the optimal trial
    TRIAL_FOLDER = 'output/GATHAR_with_e_15_h_5154100_tuning/trial_39_20251124-110023' # your path
    
    # Define the output folder where the final saved prediction files will be stored.
    PREDICTION_OUTPUT_FOLDER = 'prediction_results_70h1'
    
    # Define the proportion for robustness checks (70%)
    ROBUSTNESS_PROPORTION = 0.7 
    
    # ==============================================================================

    print(f"--- Loading configuration from the test folder: {TRIAL_FOLDER} ---")
    
    # The full path to the configuration file and model file
    config_path = os.path.join(TRIAL_FOLDER, 'GNN_param_used.yaml')
    model_path = os.path.join(TRIAL_FOLDER, 'best_model.pth')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"Error: 'GNN_param_used.yaml' or 'best_model.pth' could not be found in the folder '{TRIAL_FOLDER}'.")
        print("Please ensure the TRIAL_FOLDER path is correct.")
        return

    # 1. Load the configuration for a specific test
    with open(config_path, 'r', encoding='utf-8') as f:
        p = yaml.safe_load(f)

    # 2. Load the complete dataset
    print("--- Loading dataset ---")
    dataset = FinalHeteroDataset(
        hdf5_file_vol=p['hdf5_file_vol_std'],
        hdf5_file_volvol=p['hdf5_file_volvol_std'],
        stock_har_rv_folder=p['stock_har_rv_folder'],
        energy_har_rv_folder=p['energy_har_rv_folder'],
        stock_energy_corr_folder=p['stock_energy_corr_folder'],
        energy_energy_corr_folder=p['energy_energy_corr_folder'],
        node_info_file=p['node_info_file'],
        root=f"processed_data1001/final_hetero_dataset_seq_{p['seq_length']}",
        seq_length=p['seq_length'],
        intraday_points=p['intraday_points']
    )
    
    # 3. Data truncation
    intraday_points = p.get('intraday_points', 3)
    total_days = len(dataset) // intraday_points
    
    robustness_limit_days = int(ROBUSTNESS_PROPORTION * total_days) 
    dataset = dataset[:robustness_limit_days * intraday_points] # Truncate the dataset
    total_days = robustness_limit_days # Total number of days updated
    
    print(f"--- [Forecast Truncation] Total days limit: {total_days} days ({ROBUSTNESS_PROPORTION * 100:.0f}%) ---")
    
    subset_days = total_days
    subset_dataset = dataset # Truncated dataset
    
    # 4. Partition the dataset (split train/val/test proportionally on the truncated dataset)
    train_days = int(p['train_proportion'] * subset_days)
    validation_days = int(p['validation_proportion'] * subset_days)
    train_size = train_days * intraday_points
    validation_end_idx = (train_days + validation_days) * intraday_points
    
    # Truncate the dataset
    train_dataset = subset_dataset[:train_size]
    validation_dataset = subset_dataset[train_size:validation_end_idx]
    test_dataset = subset_dataset[validation_end_idx:]
    
    print(f"Training set size: {len(train_dataset)} | Validation set size: {len(validation_dataset)} | Test set size: {len(test_dataset)}")
    
    # 5. create DataLoader
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    print(f"Number of samples in the validation set: {len(validation_dataset)}, Number of samples in the test set: {len(test_dataset)}")

    # 6. Loading statistical data and model-building parameters
    with open('./processed_data1001/vol_covol_stats.json', 'r', encoding='utf-8') as f:
        vol_stats = json.load(f)
    diag_mean_high, diag_std_high = vol_stats['stock_diag_mean'], vol_stats['stock_diag_std']
    with open('./processed_data1001/har_rv_stats.json', 'r', encoding='utf-8') as f:
        rv_stats = json.load(f)
    diag_mean_low, diag_std_low = rv_stats['daily_mean'], rv_stats['daily_std']

    with open(p['node_info_file'], 'r', encoding='utf-8') as f:
        node_info = json.load(f)
    stock_ids = node_info['stock_ids']
    num_stocks = len(stock_ids)
    
    # Feature dimension calculations must be consistent with the configuration used during training.
    num_energy = len(node_info.get('energy_ids', []))
    stock_feature_len_base = 3 + 1 + (num_stocks - 1) + num_energy
    stock_feature_dim = stock_feature_len_base * p['seq_length']
    energy_feature_len_base = 3 + 1 + num_stocks + (num_energy - 1)
    energy_feature_dim = energy_feature_len_base * p['seq_length']
    edge_feature_dim = 3 * p['seq_length']

    # 7. Construct the model skeleton and load the weights
    print("--- Building the model and loading pre-trained weights... ---")
    model = HeteroGNNModel(
        stock_feature_dim=stock_feature_dim, energy_feature_dim=energy_feature_dim, 
        edge_feature_dim=edge_feature_dim, heads=p['num_heads'], 
        high_freq_output_dim=num_stocks, low_freq_output_dim=num_stocks,
        hidden_layout=p['hidden_layout'], dropout=p.get('dropout', 0.0), 
        activation=p.get('activation', 'relu'),
        include_energy=p.get('include_energy', True) 
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print("Model loaded successfully.")

    # 8. Generate predictions
    val_h_df, val_l_df = generate_predictions(model, validation_loader, device, p, stock_ids, diag_mean_high, diag_std_high, diag_mean_low, diag_std_low)
    test_h_df, test_l_df = generate_predictions(model, test_loader, device, p, stock_ids, diag_mean_high, diag_std_high, diag_mean_low, diag_std_low)
    
    # 9. Merge and save as Excel
    os.makedirs(PREDICTION_OUTPUT_FOLDER, exist_ok=True)
    
    # High-frequency results retention
    combine_to_excel(
        df_val=val_h_df, 
        df_test=test_h_df, 
        output_excel_path=os.path.join(PREDICTION_OUTPUT_FOLDER, 'GNN_predictions_high_freq.xlsx'),
        file_prefix='high_freq'
    )

    # Low-frequency results retention
    combine_to_excel(
        df_val=val_l_df, 
        df_test=test_l_df, 
        output_excel_path=os.path.join(PREDICTION_OUTPUT_FOLDER, 'GNN_predictions_low_freq.xlsx'),
        file_prefix='low_freq'
    )

    print("\n" + "="*30 + " All forecasting and archiving tasks completed " + "="*30)


if __name__ == '__main__':
    main()
