import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit

def calculate_qlike(y_true, y_pred, epsilon=1e-8):
    y_true = np.maximum(y_true, epsilon)
    y_pred = np.maximum(y_pred, epsilon)
    ratio = y_true / y_pred
    qlike = ratio - np.log(ratio) - 1
    return np.mean(qlike)

def load_folder_to_df(path):
    files = sorted(glob(os.path.join(path, '*.csv')))
    dfs = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_csv(f, header=None, names=[name])
            dfs.append(df)
        except:
            pass
    if not dfs: return pd.DataFrame()
    res = pd.concat(dfs, axis=1)
    res.index = range(len(res))
    return res

def build_har_features(series, prefix):
    """Constructing HAR features for a single sequence"""
    df = pd.DataFrame()
    df[f'{prefix}_lag1'] = series.shift(1)
    df[f'{prefix}_week'] = series.shift(1).rolling(5).mean()
    df[f'{prefix}_month'] = series.shift(1).rolling(22).mean()
    return df

def main():
    stock_path = 'new_day_vol'
    energy_path = 'new_day_vol_energy'
    
    print("Loading data..")
    stock_df = load_folder_to_df(stock_path)  
    energy_df = load_folder_to_df(energy_path) 
    
    # Constructing HAR features for all energy future
    print("Constructing HAR features for all energy future...")
    energy_feats = []
    for col in energy_df.columns:
        feats = build_har_features(energy_df[col], prefix=f'ENERGY_{col}')
        energy_feats.append(feats)
    
    if energy_feats:
        all_energy_features = pd.concat(energy_feats, axis=1)
    else:
        all_energy_features = pd.DataFrame()

    # Constructing HAR features for all stocks
    print("Constructing HAR features for all stocks...")
    stock_feats_list = []
    for col in stock_df.columns:
        feats = build_har_features(stock_df[col], prefix=f'STOCK_{col}')
        stock_feats_list.append(feats)
    
    if stock_feats_list:
        all_stock_har_features = pd.concat(stock_feats_list, axis=1)
    else:
        all_stock_har_features = pd.DataFrame()

    print("Commencing individual stock training: HAR-X-AllStocks (Ridge)...")
    
    results_list = []
    
    # --- Macro Indicator Collection List ---
    val_metrics = {'mse': [], 'rmse': [], 'qlike': []}
    test_metrics = {'mse': [], 'rmse': [], 'qlike': []}
    
    cutoff = int(len(stock_df) * 1.0) 
    stock_df = stock_df.iloc[:cutoff]
    all_energy_features = all_energy_features.iloc[:cutoff]
    all_stock_har_features = all_stock_har_features.iloc[:cutoff] 
    
    for stock_col in tqdm(stock_df.columns, desc="Processing Stocks"):
        y = stock_df[stock_col]
        
        # 1. Identify the feature column corresponding to the current stock (y)
        current_stock_prefix = f'STOCK_{stock_col}'
        own_feature_cols = [
            col for col in all_stock_har_features.columns 
            if col.startswith(current_stock_prefix)
        ]
        X_own_stock = all_stock_har_features[own_feature_cols]
        
        # 2. Identify the characteristics of all other stocks
        X_other_stocks = all_stock_har_features.drop(columns=own_feature_cols)
        
        # 3. Merge Features
        X = pd.concat([X_own_stock, X_other_stocks, all_energy_features], axis=1)
        
        
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        if len(X) == 0: continue
            
        n = len(X)
        train_size = int(n * 0.7)
        val_size = int(n * 0.1)
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        X_val = X.iloc[train_size : train_size + val_size]
        y_val = y.iloc[train_size : train_size + val_size]
        
        X_test = X.iloc[train_size + val_size:] 
        y_test = y.iloc[train_size + val_size:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=TimeSeriesSplit(5))
        model.fit(X_train_s, y_train)
        
        # --- Model prediction ---
        pred_val = model.predict(X_val_s)
        pred_test = model.predict(X_test_s)

        PREDICTION_LOWER_BOUND = 1e-8
        pred_val = np.maximum(pred_val, PREDICTION_LOWER_BOUND)
        pred_test = np.maximum(pred_test, PREDICTION_LOWER_BOUND)
        
        # --- Calculate the indicators for individual stocks and store them in a list (Macro Logic) ---
        # Validation set
        v_mse = mean_squared_error(y_val, pred_val)
        val_metrics['mse'].append(v_mse)
        val_metrics['rmse'].append(np.sqrt(v_mse))
        val_metrics['qlike'].append(calculate_qlike(y_val.values, pred_val)) 
        
        # test set
        t_mse = mean_squared_error(y_test, pred_test)
        test_metrics['mse'].append(t_mse)
        test_metrics['rmse'].append(np.sqrt(t_mse))
        test_metrics['qlike'].append(calculate_qlike(y_test.values, pred_test))

        # Collect the results for use in Excel
        res_val = pd.DataFrame({
            'stock_code': stock_col,
            'Actual value': y_val.values,
            'Predicted value': pred_val, 
            'dataset': 'Validation'
        })
        results_list.append(res_val)
        
        res_test = pd.DataFrame({
            'stock_code': stock_col,
            'Actual value': y_test.values,
            'Predicted value': pred_test, 
            'dataset': 'Test'
        })
        results_list.append(res_test)
    
    # --- 3. Overall assessment output (Macro Average) ---
    print("\n" + "="*25 + " HAR-X-AllStocks Model performance (Macro Average) " + "="*25)
    
    print("【Validation set】:")
    print(f"  - MSE:    {np.mean(val_metrics['mse']):.4e}")
    print(f"  - RMSE:   {np.mean(val_metrics['rmse']):.4e}")
    print(f"  - QLIKE: {np.mean(val_metrics['qlike']):.4f}")
    
    print("【Test set】:")
    print(f"  - MSE:    {np.mean(test_metrics['mse']):.4e}")
    print(f"  - RMSE:   {np.mean(test_metrics['rmse']):.4e}")
    print(f"  - QLIKE: {np.mean(test_metrics['qlike']):.4f}")
    print("="*64)
    
    # --- 4. write to Excel ---
    output_dir = 'results_low'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'har_ridge_energy_allstocks_predictions.xlsx')

    final_df = pd.concat(results_list, ignore_index=True)

    print(f"Writing to Excel:  {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for stock_id, group_df in tqdm(final_df.groupby('stock_code'), desc="Writing to Sheet"):
            group_df[['Actual value', 'Predicted value', 'dataset']].to_excel(writer, sheet_name=str(stock_id), index=False)
            
    print(f"Saved successfully.")

if __name__ == "__main__":
    main()
