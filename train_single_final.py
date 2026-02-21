import torch
import numpy as np
import yaml
import os
import json
import math
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils.modelschange import HeteroGNNModel
from utils.datasetchange import FinalHeteroDataset 
from utils.losses import QLIKELoss

def train(p: dict, trial_folder: str) -> dict:
    folder_path = trial_folder 
    with open(os.path.join(folder_path, 'GNN_param_used.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(p, f)
    torch.manual_seed(p['seed'])
    np.random.seed(p['seed'])

    # --- Read configuration switch ---
    include_energy = p.get('include_energy', True)
    use_har = p.get('use_har_features', True)     # Read HAR switch
    
    print(f"--- [configuration] include_energy = {include_energy} ---")
    print(f"--- [configuration] use_har_features = {use_har} ---")

    # --- Dynamically configure the cache path and initialise the Dataset ---
    dataset_root_name = f"final_hetero_dataset_seq_{p['seq_length']}_har_{use_har}"
    
    dataset = FinalHeteroDataset(
        hdf5_file_vol=p['hdf5_file_vol_std'],
        hdf5_file_volvol=p['hdf5_file_volvol_std'],
        stock_har_rv_folder=p['stock_har_rv_folder'],
        energy_har_rv_folder=p['energy_har_rv_folder'],
        stock_energy_corr_folder=p['stock_energy_corr_folder'],
        energy_energy_corr_folder=p['energy_energy_corr_folder'],
        node_info_file=p['node_info_file'],
        root=f"processed_data1001/{dataset_root_name}",     # Path contains HAR status
        seq_length=p['seq_length'],
        intraday_points=p['intraday_points'],
        include_energy=include_energy,
        use_har_features=use_har     # Passed to Dataset
    )
    
    intraday_points = p.get('intraday_points', 3)
    total_days = len(dataset) // intraday_points
    ROBUSTNESS_PROPORTION = 0.7 
    robustness_limit_days = int(ROBUSTNESS_PROPORTION * total_days) 
    dataset = dataset[:robustness_limit_days * intraday_points] 
    total_days = robustness_limit_days 
    print(f"--- [Robustness check] The total number of days is limited to: {total_days} days ({ROBUSTNESS_PROPORTION * 100}%) ---")
    
    train_days = int(p['train_proportion'] * total_days)
    validation_days = int(p['validation_proportion'] * total_days)
    train_size = train_days * intraday_points
    validation_end_idx = (train_days + validation_days) * intraday_points
    train_dataset = dataset[:train_size]
    validation_dataset = dataset[train_size:validation_end_idx]
    test_dataset = dataset[validation_end_idx:]

    train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=p['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False, num_workers=4)
    
    with open(p['node_info_file'], 'r', encoding='utf-8') as f:
        node_info = json.load(f)
    
    num_stocks = len(node_info['stock_ids'])
    
    if include_energy:
        energy_ids = node_info['energy_ids']
        num_energy = len(energy_ids)
    else:
        energy_ids = []
        num_energy = 0
    
    with open('./processed_data1001/vol_covol_stats.json', 'r', encoding='utf-8') as f:
        vol_stats = json.load(f)
    diag_mean_high = vol_stats['stock_diag_mean']
    diag_std_high = vol_stats['stock_diag_std']
    with open('./processed_data1001/har_rv_stats.json', 'r', encoding='utf-8') as f:
        rv_stats = json.load(f)
    diag_mean_low = rv_stats['daily_mean']
    diag_std_low = rv_stats['daily_std']

    # ---Dynamic computation of feature dimensions ---
    # Define the base length for HAR features: 3 when enabled (Rv_d, Rv_w, Rv_m), otherwise 0.
    har_dim = 3 if use_har else 0

    # 1. Calculation of Stock Feature Dimensions
    stock_feature_len_base = har_dim + 1 + (num_stocks - 1)
    
    if include_energy:
        stock_feature_len_base += num_energy
    
    stock_feature_dim = stock_feature_len_base * p['seq_length']

    # 2. Calculation of Energy Characteristic Dimensions
    energy_feature_dim = 0
    if include_energy:
        energy_feature_len_base = har_dim + 1 + num_stocks + (num_energy - 1)
        energy_feature_dim = energy_feature_len_base * p['seq_length']

    edge_feature_dim = 3 * p['seq_length']

    print(f"--- [Dimension Check] HAR dimension: {har_dim}, Total stock feature dimension: {stock_feature_dim}, Total energy feature dimension: {energy_feature_dim} ---")
    
    # --- Model initialisation ---
    model = HeteroGNNModel(
        stock_feature_dim=stock_feature_dim, 
        energy_feature_dim=energy_feature_dim,
        edge_feature_dim=edge_feature_dim, 
        heads=p['num_heads'], 
        high_freq_output_dim=num_stocks, 
        low_freq_output_dim=num_stocks,
        hidden_layout=p['hidden_layout'], 
        dropout=p.get('dropout', 0.0), 
        activation=p.get('activation', 'relu'),
        include_energy=include_energy 
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'])
    elif p['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=p['learning_rate'])

    if p['loss_function'] == 'mse':
        train_criterion = torch.nn.MSELoss(reduction='none')
    elif p['loss_function'] == 'qlike':
        train_criterion = QLIKELoss(reduction='none')
    else:
        raise ValueError(f"Unsupported loss functions: {p['loss_function']}")

    min_validation_loss = float('inf')
    best_epoch = -1
    
    # --- Training ---
    for epoch in range(p['num_epochs']):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred_high, pred_low = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            if pred_high.numel() == 0: continue
            
            y_high, y_low = data['stock'].y_high, data['stock'].y_low
            
            # Use macro averaging
            loss_high = train_criterion(pred_high, y_high).mean(dim=0).mean()
            loss_low = train_criterion(pred_low, y_low).mean(dim=0).mean()
            loss = loss_low
            
            loss.backward()
            optimizer.step()

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for data in validation_loader:
                data = data.to(device)
                pred_high, pred_low = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
                if pred_high.numel() == 0: continue
                
                y_high, y_low = data['stock'].y_high, data['stock'].y_low
                
                loss_high = train_criterion(pred_high, y_high).mean(dim=0).mean()
                loss_low = train_criterion(pred_low, y_low).mean(dim=0).mean()
                loss = loss_low
                
                validation_loss += loss.item()
                
        avg_validation_loss = validation_loss / len(validation_loader) if len(validation_loader) > 0 else float('inf')
        
        if avg_validation_loss < min_validation_loss:
            min_validation_loss = avg_validation_loss
            best_epoch = epoch + 1
            save_path = os.path.join(folder_path, 'best_model.pth')
            torch.save(model.state_dict(), save_path)

    best_model_path = os.path.join(folder_path, 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"Warning: Best model file not found: {best_model_path}")
        return { 'min_validation_loss': float('inf') }

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    def evaluate_on_set(loader, dataset_ref):
        if not loader or len(dataset_ref) == 0:
            return {'mse_h': np.nan, 'rmse_h': np.nan, 'qlike_h': np.nan,
                    'mse_l': np.nan, 'rmse_l': np.nan, 'qlike_l': np.nan}

        mse_criterion = torch.nn.MSELoss(reduction='none')
        qlike_criterion = QLIKELoss(reduction='none')
        all_preds_h, all_actuals_h, all_preds_l, all_actuals_l = [], [], [], []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                pred_high, pred_low = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
                if pred_high.numel() == 0: continue
                
                y_high, y_low = data['stock'].y_high, data['stock'].y_low
                pred_high_unstd = (pred_high * torch.tensor(diag_std_high, device=device)) + torch.tensor(diag_mean_high, device=device)
                y_high_unstd = (y_high * torch.tensor(diag_std_high, device=device)) + torch.tensor(diag_mean_high, device=device)
                pred_low_unstd = (pred_low * torch.tensor(diag_std_low, device=device)) + torch.tensor(diag_mean_low, device=device)
                y_low_unstd = (y_low * torch.tensor(diag_std_low, device=device)) + torch.tensor(diag_mean_low, device=device)
                
                all_preds_h.append(pred_high_unstd); all_actuals_h.append(y_high_unstd)
                all_preds_l.append(pred_low_unstd); all_actuals_l.append(y_low_unstd)
        
        if not all_preds_h: 
             return {'mse_h': np.nan, 'rmse_h': np.nan, 'qlike_h': np.nan,
                     'mse_l': np.nan, 'rmse_l': np.nan, 'qlike_l': np.nan}

        preds_h, actuals_h = torch.cat(all_preds_h, dim=0), torch.cat(all_actuals_h, dim=0)
        preds_l, actuals_l = torch.cat(all_preds_l, dim=0), torch.cat(all_actuals_l, dim=0)
        preds_l = preds_l.view(-1, num_stocks)
        actuals_l = actuals_l.view(-1, num_stocks)
        
        mse_per_stock_h = mse_criterion(preds_h, actuals_h).mean(dim=0)
        qlike_per_stock_h = qlike_criterion(preds_h, actuals_h).mean(dim=0)
        avg_mse_h, avg_qlike_h = mse_per_stock_h.mean().item(), qlike_per_stock_h.mean().item()

        intraday_points = p.get('intraday_points', 3)
        num_samples = preds_l.shape[0]
        num_stocks_l = preds_l.shape[1]
        avg_mse_l, avg_qlike_l = np.nan, np.nan
        if num_samples > 0 and num_samples % intraday_points == 0:
            num_days = num_samples // intraday_points
            avg_preds_per_day = preds_l.reshape(num_days, intraday_points, num_stocks_l).mean(dim=1)
            actuals_per_day = actuals_l.reshape(num_days, intraday_points, num_stocks_l)[:, -1, :]
            mse_per_stock_l = mse_criterion(avg_preds_per_day, actuals_per_day).mean(dim=0)
            qlike_per_stock_l = qlike_criterion(avg_preds_per_day, actuals_per_day).mean(dim=0)
            avg_mse_l, avg_qlike_l = mse_per_stock_l.mean().item(), qlike_per_stock_l.mean().item()
        
        rmse_h = math.sqrt(avg_mse_h) if not np.isnan(avg_mse_h) else np.nan
        rmse_l = math.sqrt(avg_mse_l) if not np.isnan(avg_mse_l) else np.nan
        
        print(f"Evaluation indicators: 'mse_h': {avg_mse_h}, 'rmse_h': {rmse_h}, 'qlike_h': {avg_qlike_h}\n'mse_l': {avg_mse_l}, 'rmse_l': {rmse_l}, 'qlike_l': {avg_qlike_l}")
        return {
            'mse_h': avg_mse_h, 'rmse_h': rmse_h, 'qlike_h': avg_qlike_h,
            'mse_l': avg_mse_l, 'rmse_l': rmse_l, 'qlike_l': avg_qlike_l
        }
            
    validation_metrics = evaluate_on_set(validation_loader, validation_dataset)
    test_metrics = evaluate_on_set(test_loader, test_dataset)
    
    return {
        'min_validation_loss': min_validation_loss, 'best_epoch': best_epoch,
        **{f'validation_{k}': v for k, v in validation_metrics.items()},
        **{f'test_{k}': v for k, v in test_metrics.items()}
    }
