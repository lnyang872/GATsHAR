import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm
import optuna 
import functools 

SEED = 42

def calculate_qlike(y_true, y_pred, epsilon=1e-8):
    y_true = np.maximum(y_true, epsilon)
    y_pred = np.maximum(y_pred, epsilon)
    ratio = y_true / y_pred
    qlike = ratio - np.log(ratio) - 1
    return np.mean(qlike)

def calculate_macro_metrics(y_true, y_pred, stock_codes):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'code': stock_codes})
    mse_list, rmse_list, qlike_list = [], [], []
    
    # Calculate grouped by stock code
    for code, group in df.groupby('code'):
        mse = mean_squared_error(group['true'], group['pred'])
        rmse = np.sqrt(mse)     # single stock RMSE
        qlike = calculate_qlike(group['true'].values, group['pred'].values)
        
        mse_list.append(mse)
        rmse_list.append(rmse)
        qlike_list.append(qlike)
    
    # Return the average value of the indicator
    return np.mean(mse_list), np.mean(rmse_list), np.mean(qlike_list)

def load_data_from_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    data_dict = {}
    for file_path in files:
        code = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df = pd.read_csv(file_path, header=None, names=['RV'])
            df.index.name = 'time_index'
            data_dict[code] = df
        except pd.errors.EmptyDataError:
            print(f"Warning: File {file_path} is empty and has been skipped.")
    return data_dict

def create_lag_features(df, prefix='RV'):
    """
    Generate lagged features
    """
    df = df.copy()
    col_name = df.columns[0] 
    df[f'{prefix}_lag_day'] = df[col_name].shift(1)
    df[f'{prefix}_lag_week'] = df[col_name].shift(5)
    df[f'{prefix}_lag_month'] = df[col_name].shift(22)
    return df

def process_energy_data(energy_data_dict, max_length):
    """
    Processing energy data: Generate lagged features for all energy sources and merge them.
    """
    energy_features_list = []
    for code, df in energy_data_dict.items():
        # Add specific prefixes to energy features 
        feat_df = create_lag_features(df, prefix=f'ENERGY_{code}')
        cols_to_keep = [c for c in feat_df.columns if 'lag' in c]
        feat_df = feat_df[cols_to_keep]
        energy_features_list.append(feat_df)
        
    if not energy_features_list:
        return pd.DataFrame()

    full_energy_df = pd.concat(energy_features_list, axis=1)
    # Uniform length and padding
    full_energy_df = full_energy_df.reindex(range(max_length)).ffill().fillna(0)
    full_energy_df = full_energy_df.reset_index(drop=True)
    full_energy_df['time_index'] = full_energy_df.index
    return full_energy_df

def process_all_stocks_features(stock_data_dict, max_length):
    """
    Processing all stock features: Generate lagged features for each stock and horizontally merge them into a large wide table 
    """
    stock_features_list = []
    for code, df in stock_data_dict.items():
         # Add specific prefixes to stock features
        feat_df = create_lag_features(df, prefix=f'STOCK_{code}')
        cols_to_keep = [c for c in feat_df.columns if 'lag' in c]
        feat_df = feat_df[cols_to_keep]
        stock_features_list.append(feat_df)
    
    if not stock_features_list:
        return pd.DataFrame()
    
    # Horizontal consolidation of all share features
    full_stocks_df = pd.concat(stock_features_list, axis=1)
    
    # Uniform length and padding
    full_stocks_df = full_stocks_df.reindex(range(max_length)).ffill().fillna(0)
    full_stocks_df = full_stocks_df.reset_index(drop=True)
    full_stocks_df['time_index'] = full_stocks_df.index
    
    return full_stocks_df

def prepare_combined_data(stock_data, energy_data):
    # 1. Calculate the maximum duration
    max_len_stock = max([len(df) for df in stock_data.values()]) if stock_data else 0
    max_len_energy = max([len(df) for df in energy_data.values()]) if energy_data else 0
    total_max_len = max(max_len_stock, max_len_energy)
    
    print(f"Maximum data duration: {total_max_len}")
    
    # 2. Constructing comprehensive energy features (M * 3 columns)
    print("--- Constructing comprehensive energy features ---")
    energy_features_df = process_energy_data(energy_data, total_max_len)
    
    # 3. Constructing comprehensive stock features (N * 3 columns)
    print("--- Constructing comprehensive stock features  ---")
    all_stocks_features_df = process_all_stocks_features(stock_data, total_max_len)
    
    # 4.  Forming a global feature pool
    print("---  Forming a global feature pool ---")
    if not energy_features_df.empty:
        global_features = pd.merge(all_stocks_features_df, energy_features_df, on='time_index', how='left')
    else:
        global_features = all_stocks_features_df
        
    print(f"Global feature matrix dimension: {global_features.shape}")

    # 5. Constructing panel data for training purposes
    all_data_rows = []
    
    print("---  Building the panel dataset ---")
    for stock_code, df in tqdm(stock_data.items(), desc="Processing Stocks"):

        df_target = df[['RV']].copy()
        df_target['stock_code'] = stock_code
        df_target = df_target.reset_index()     # Obtain the time_index
        df_merged = pd.merge(df_target, global_features, on='time_index', how='left')
        
        all_data_rows.append(df_merged)
        
    combined_df = pd.concat(all_data_rows).reset_index(drop=True)
    combined_df.sort_values(by=['time_index', 'stock_code'], inplace=True)
    combined_df = combined_df.dropna()
    index_cols = combined_df[['time_index', 'stock_code']]
    
    # 6. Generate one-hot encoding (to distinguish stock identities)
    print("--- Generate one-hot codes ---")
    features_df = pd.get_dummies(combined_df, columns=['stock_code'], prefix='stock')
    
    # Separate X and y
    # drop 'RV' (Target) and 'time_index' (Not a feature)
    X = features_df.drop(['RV', 'time_index'], axis=1)
    y = features_df['RV']
    
    print(f"Final feature matrix X dimension: {X.shape}")
    
    return X, y, index_cols

def objective(trial, X_train, y_train, X_val, y_val):

    # 1. Defining the hyperparameter search space 
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }

    # 2. Add fixed parameters
    param_grid.update({
        'n_estimators': 1000, 
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': SEED,
        'early_stopping_rounds': 25,
        # 'tree_method': 'hist' 
    })
    
    # 3. Train the model
    model = xgb.XGBRegressor(**param_grid)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False) 
    
    # 4. Evaluate on the validation set and return the metrics
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse

def main():
    stock_folder = "new_day_vol"
    energy_folder = "new_day_vol_energy"
    
    print("Loading data...")
    stock_data = load_data_from_folder(stock_folder)
    energy_data = load_data_from_folder(energy_folder)

    # Call the modified data preparation function
    X_combined, y_combined, index_info = prepare_combined_data(stock_data, energy_data)

    print("--- Splitting the dataset ---")
    subset_end_index = int(len(X_combined) * 1.0) 
    
    X_subset = X_combined.iloc[:subset_end_index]
    y_subset = y_combined.iloc[:subset_end_index]
    index_subset = index_info.iloc[:subset_end_index]
    
    subset_size = len(X_subset)
    train_size = int(subset_size * 0.7)
    val_size = int(subset_size * 0.1)
    
    X_train = X_subset.iloc[:train_size]
    y_train = y_subset.iloc[:train_size]
    
    X_val = X_subset.iloc[train_size : train_size + val_size]
    y_val = y_subset.iloc[train_size : train_size + val_size]
    index_val = index_subset.iloc[train_size : train_size + val_size]
    
    X_test = X_subset.iloc[train_size + val_size:]
    y_test = y_subset.iloc[train_size + val_size:]
    index_test = index_subset.iloc[train_size + val_size:]
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    scaler = StandardScaler()
    print("Standardisation is currently underway (StandardScaler)...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Running Optuna optimisation ---
    print("--- Initiate Optuna hyperparameter tuning  ---")
    
    objective_with_data = functools.partial(objective,
                                            X_train=X_train_scaled,
                                            y_train=y_train,
                                            X_val=X_val_scaled,
                                            y_val=y_val)

    study = optuna.create_study(
        direction='minimize', 
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    
    study.optimize(objective_with_data, n_trials=50)
    
    print(f"Tuning complete. Best validation set MSE:{study.best_value:.4e}")
    print(f"Optimal parameter combination: {study.best_params}")

    best_params = study.best_params
    
    best_params.update({
        'n_estimators': 1000, 
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': SEED,
        'early_stopping_rounds': 25 
    })
    
    model = xgb.XGBRegressor(**best_params)
    
    print("Begin training the final XGBoost model using optimal parameters...")
    model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)], 
              verbose=False)
    
    best_iteration = model.best_iteration
    print(f"The model determined the optimal number of trees via early stopping to be: {best_iteration}")
    
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    print("\n" + "="*25 + " XGBoost Model performance (Macro Average) " + "="*25)
    
    # --- Macro Average 计算 ---
    val_mse, val_rmse, val_qlike = calculate_macro_metrics(y_val, val_pred, index_val['stock_code'])
    print("【 Validation】:")
    print(f"  - Macro MSE:    {val_mse:.4e}")
    print(f"  - Macro RMSE:   {val_rmse:.4e}")
    print(f"  - Macro QLIKE: {val_qlike:.4f}")

    test_mse, test_rmse, test_qlike = calculate_macro_metrics(y_test, test_pred, index_test['stock_code'])
    print("【 Test】:")
    print(f"  - Macro MSE:    {test_mse:.4e}")
    print(f"  - Macro RMSE:   {test_rmse:.4e}")
    print(f"  - Macro QLIKE: {test_qlike:.4f}")
    print("="*50)

    # --- Save results ---
    output_dir = 'results_low'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'xgboost_allstocks_energy_tuned_predictions.xlsx')
    
    res_val = index_val.copy()
    res_val['Actual value'] = y_val.values
    res_val['Predicted value'] = val_pred
    res_val['dataset'] = 'Validation'
    
    res_test = index_test.copy()
    res_test['Actual value'] = y_test.values
    res_test['Predicted value'] = test_pred
    res_test['dataset'] = 'Test'
    
    all_res = pd.concat([res_val, res_test])
    
    print(f"writing to Excel: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for stock_id, group_df in tqdm(all_res.groupby('stock_code'), desc="write to Sheet"):
            group_df[['Actual value', 'Predicted value', 'dataset']].to_excel(writer, sheet_name=str(stock_id), index=False)
            
    print(f"Saved successfully.")

if __name__ == "__main__":
    main()
