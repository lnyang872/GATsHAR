import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import optuna
import functools

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_qlike(y_true, y_pred, epsilon=1e-5):
    y_true = np.maximum(y_true, epsilon)
    y_pred = np.maximum(y_pred, epsilon)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_layout, output_size, dropout):
        super(MultivariateLSTM, self).__init__()
        self.layers = nn.ModuleList()
        last_size = input_size
        for h_size in hidden_layout:
            self.layers.append(nn.LSTM(last_size, h_size, batch_first=True))
            last_size = h_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(last_size, output_size)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out, _ = layer(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

def prepare_wide_data(stock_folder):
    stock_files = sorted(glob.glob(os.path.join(stock_folder, "*.csv")))
    stock_dfs = []
    for f in stock_files:
        name = os.path.splitext(os.path.basename(f))[0]
        stock_dfs.append(pd.read_csv(f, header=None, names=[name]))
    
    stock_wide = pd.concat(stock_dfs, axis=1)
    stock_names = stock_wide.columns.tolist()
    num_stocks = len(stock_names)

    full_wide = stock_wide.ffill().bfill().fillna(0)
    return full_wide, stock_names, num_stocks

def create_windows(data, window_size, num_target_cols):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size, :])
        y.append(data[i + window_size, :num_target_cols]) 
    return np.array(X), np.array(y)

def calculate_macro_metrics_matrix(y_true, y_pred):
    """Input dimensions (Samples, Stocks), compute indicators by column and calculate the average"""
    num_stocks = y_true.shape[1]
    mse_list, rmse_list, qlike_list = [], [], []
    
    for i in range(num_stocks):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        
        mse = mean_squared_error(col_true, col_pred)
        rmse = np.sqrt(mse) # Single stock RMSE
        qlike = calculate_qlike(col_true, col_pred)
        
        mse_list.append(mse)
        rmse_list.append(rmse)
        qlike_list.append(qlike)
        
    return np.mean(mse_list), np.mean(rmse_list), np.mean(qlike_list)

def objective(trial, X_train, y_train, X_val, y_val, device, input_size, output_size, num_epochs=50, patience=5):
    """
    Optuna objective function, used for training and evaluating a combination of hyperparameters.
    """
    
    # 1. Defining the hyperparameter search space
    hidden_layout = trial.suggest_categorical('hidden_layout', [
        [256, 128, 64], [192, 96, 48], [128, 64, 32],
        [256, 128], [192, 96], [128, 64],
        [256, 256, 256], [128, 128, 128], [64, 64, 64]
    ])
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'RMSprop'])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

    # 2. create DataLoader, Model, Optimizer
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    model = MultivariateLSTM(
        input_size=input_size,
        hidden_layout=hidden_layout,
        output_size=output_size,
        dropout=dropout
    ).to(device)
    
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else: # RMSprop
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    criterion = nn.MSELoss()
    
    # 3. Training and validation
    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx.to(device)), by.to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()
        
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_loss

def inverse_transform_prediction(pred_scaled, y_scaled, full_cols, target_cols):
    dummy = np.zeros((pred_scaled.shape[0], full_cols - target_cols))
    full_pred = np.hstack([pred_scaled, dummy])
    full_true = np.hstack([y_scaled, dummy])
    pred_orig = scaler.inverse_transform(full_pred)[:, :target_cols]
    true_orig = scaler.inverse_transform(full_true)[:, :target_cols]
    return pred_orig, true_orig

def make_long_df(y_true, y_pred, dataset_name, stock_ids_list):
    df_true = pd.DataFrame(y_true, columns=stock_ids_list)
    df_true['time_idx'] = range(len(df_true))
    df_true_long = df_true.melt(id_vars='time_idx', var_name='stock_code', value_name='Actual value')
    
    df_pred = pd.DataFrame(y_pred, columns=stock_ids_list)
    df_pred['time_idx'] = range(len(df_pred))
    df_pred_long = df_pred.melt(id_vars='time_idx', var_name='stock_code', value_name='Predicted value')
    
    res = pd.merge(df_true_long, df_pred_long, on=['time_idx', 'stock_code'])
    res[dataset'] = dataset_name
    return res


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    STOCK_FOLDER = "./new_day_vol"
    WINDOW_SIZE = 22
    NUM_EPOCHS = 50  
    PATIENCE = 5     
    N_TRIALS = 100   
    print("Loading and consolidating stock data...")
    wide_df, stock_ids, num_stocks = prepare_wide_data(STOCK_FOLDER)
    
    subset_len = int(len(wide_df) * 1.0)
    wide_df = wide_df.iloc[:subset_len]
    
    train_len = int(len(wide_df) * 0.7)
    
    global scaler 
    scaler = StandardScaler()
    train_data_raw = wide_df.iloc[:train_len].values
    scaler.fit(train_data_raw)
    data_scaled = scaler.transform(wide_df.values)
    
    X_all, y_all = create_windows(data_scaled, WINDOW_SIZE, num_target_cols=num_stocks)
    
    train_samples = int(len(X_all) * 0.7)
    val_samples = int(len(X_all) * 0.1)
    
    X_train = torch.tensor(X_all[:train_samples], dtype=torch.float32)
    y_train = torch.tensor(y_all[:train_samples], dtype=torch.float32)
    
    X_val = torch.tensor(X_all[train_samples : train_samples + val_samples], dtype=torch.float32)
    y_val = torch.tensor(y_all[train_samples : train_samples + val_samples], dtype=torch.float32)
    
    X_test = torch.tensor(X_all[train_samples + val_samples:], dtype=torch.float32)
    y_test = torch.tensor(y_all[train_samples + val_samples:], dtype=torch.float32)
    
    print(f"Train shape: {X_train.shape}")
    print(f"--- Initiate Optuna hyperparameter tuning (TPE, {N_TRIALS} attempts) ---")
    
    input_size = X_train.shape[2]
    output_size = num_stocks
    
    objective_with_data = functools.partial(
        objective,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        device=device,
        input_size=input_size,
        output_size=output_size,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )
    
    study = optuna.create_study(
        direction='minimize', 
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective_with_data, n_trials=N_TRIALS, show_progress_bar=True)
    
    print("Optimisation complete.")
    print(f"  Best validation set MSE: {study.best_value:.4e}")
    print(f"  Optimal parameter combination: {study.best_params}")
    print("Commencing training of the LSTM model (stocks only) using optimal parameters...")
    best_params = study.best_params
    
    final_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_params['batch_size'], shuffle=True)
    
    model = MultivariateLSTM(
        input_size=input_size,
        hidden_layout=best_params['hidden_layout'],
        output_size=output_size,
        dropout=best_params['dropout']
    ).to(device)
    
    if best_params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    else: # RMSprop
        optimizer = torch.optim.RMSprop(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        
    criterion = nn.MSELoss()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for bx, by in final_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx.to(device)), by.to(device))
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Val Loss: {val_loss:.4e}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_stocks_ONLY.pth')
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("The final model training has been completed, the optimal weights have been loaded for prediction....")
    model.load_state_dict(torch.load('best_lstm_stocks_ONLY.pth'))
    model.eval()
    
    with torch.no_grad():
        val_pred_scaled = model(X_val.to(device)).cpu().numpy()
        y_val_scaled = y_val.cpu().numpy()
        test_pred_scaled = model(X_test.to(device)).cpu().numpy()
        y_test_scaled = y_test.cpu().numpy()

    val_pred_orig, y_val_orig = inverse_transform_prediction(
        val_pred_scaled, y_val_scaled, wide_df.shape[1], num_stocks
    )
    
    test_pred_orig, y_test_orig = inverse_transform_prediction(
        test_pred_scaled, y_test_scaled, wide_df.shape[1], num_stocks
    )

    print("\n" + "="*25 + " LSTM (Stocks Only) Model Performance (Macro Average) " + "="*25)
    
    val_mse, val_rmse, val_qlike = calculate_macro_metrics_matrix(y_val_orig, val_pred_orig)
    print("【Validation】:")
    print(f"  - Macro MSE:     {val_mse:.4e}")
    print(f"  - Macro RMSE:    {val_rmse:.4e}")
    print(f"  - Macro QLIKE:  {val_qlike:.4f}")

    test_mse, test_rmse, test_qlike = calculate_macro_metrics_matrix(y_test_orig, test_pred_orig)
    print("【Test】:")
    print(f"  - Macro MSE:     {test_mse:.4e}")
    print(f"  - Macro RMSE:    {test_rmse:.4e}")
    print(f"  - Macro QLIKE:  {test_qlike:.4f}")
    print("="*50)

    # --- write to Excel ---
    output_dir = 'results_low'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lstm_stocks_ONLY_tuned_predictions.xlsx')

    df_val_final = make_long_df(y_val_orig, val_pred_orig, 'Validation', stock_ids)
    df_test_final = make_long_df(y_test_orig, test_pred_orig, 'Test', stock_ids)
    
    final_results = pd.concat([df_val_final, df_test_final])

    print(f"writing to Excel: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for stock_id, group_df in tqdm(final_results.groupby('stock_code'), desc="write toSheet"):
            group_df[['Actual value', 'Predicted value', 'dataset']].to_excel(writer, sheet_name=str(stock_id), index=False)
    
    print("Saved successfully.")

if __name__ == "__main__":
    main()
