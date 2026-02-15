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
    """构建单个序列的 HAR 特征"""
    df = pd.DataFrame()
    df[f'{prefix}_lag1'] = series.shift(1)
    df[f'{prefix}_week'] = series.shift(1).rolling(5).mean()
    df[f'{prefix}_month'] = series.shift(1).rolling(22).mean()
    return df

def main():
    stock_path = 'new_day_vol'
    energy_path = 'new_day_vol_energy'
    
    print("加载数据...")
    stock_df = load_folder_to_df(stock_path)  
    energy_df = load_folder_to_df(energy_path) 
    
    # 1. 构建全量的能源特征
    print("构建能源 HAR 特征...")
    energy_feats = []
    for col in energy_df.columns:
        feats = build_har_features(energy_df[col], prefix=f'ENERGY_{col}')
        energy_feats.append(feats)
    
    if energy_feats:
        all_energy_features = pd.concat(energy_feats, axis=1)
    else:
        all_energy_features = pd.DataFrame()

    # 构建所有股票的全量 HAR 特征 ---
    print("构建所有股票的 HAR 特征...")
    stock_feats_list = []
    for col in stock_df.columns:
        feats = build_har_features(stock_df[col], prefix=f'STOCK_{col}')
        stock_feats_list.append(feats)
    
    if stock_feats_list:
        all_stock_har_features = pd.concat(stock_feats_list, axis=1)
    else:
        all_stock_har_features = pd.DataFrame()

    print("开始逐个股票训练 HAR-X-AllStocks (Ridge)...")
    
    results_list = []
    
    # --- Macro 指标收集列表 ---
    val_metrics = {'mse': [], 'rmse': [], 'qlike': []}
    test_metrics = {'mse': [], 'rmse': [], 'qlike': []}
    
    cutoff = int(len(stock_df) * 1.0) 
    stock_df = stock_df.iloc[:cutoff]
    all_energy_features = all_energy_features.iloc[:cutoff]
    all_stock_har_features = all_stock_har_features.iloc[:cutoff] 
    
    for stock_col in tqdm(stock_df.columns, desc="Processing Stocks"):
        y = stock_df[stock_col]
        
        # 1. 找出当前股票 (y) 对应的特征列
        current_stock_prefix = f'STOCK_{stock_col}'
        own_feature_cols = [
            col for col in all_stock_har_features.columns 
            if col.startswith(current_stock_prefix)
        ]
        X_own_stock = all_stock_har_features[own_feature_cols]
        
        # 2. 找出所有其他股票的特征
        X_other_stocks = all_stock_har_features.drop(columns=own_feature_cols)
        
        # 3. 合并特征
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
        
        # --- 模型预测 ---
        pred_val = model.predict(X_val_s)
        pred_test = model.predict(X_test_s)

        PREDICTION_LOWER_BOUND = 1e-8
        pred_val = np.maximum(pred_val, PREDICTION_LOWER_BOUND)
        pred_test = np.maximum(pred_test, PREDICTION_LOWER_BOUND)
        
        # --- 计算单只股票的指标并存入列表 (Macro Logic) ---
        # 验证集
        v_mse = mean_squared_error(y_val, pred_val)
        val_metrics['mse'].append(v_mse)
        val_metrics['rmse'].append(np.sqrt(v_mse))
        val_metrics['qlike'].append(calculate_qlike(y_val.values, pred_val)) 
        
        # 测试集
        t_mse = mean_squared_error(y_test, pred_test)
        test_metrics['mse'].append(t_mse)
        test_metrics['rmse'].append(np.sqrt(t_mse))
        test_metrics['qlike'].append(calculate_qlike(y_test.values, pred_test))

        # 收集结果用于 Excel
        res_val = pd.DataFrame({
            'stock_code': stock_col,
            '真实值': y_val.values,
            '预测值': pred_val, 
            '数据集': 'Validation'
        })
        results_list.append(res_val)
        
        res_test = pd.DataFrame({
            'stock_code': stock_col,
            '真实值': y_test.values,
            '预测值': pred_test, 
            '数据集': 'Test'
        })
        results_list.append(res_test)
    
    # --- 3. 整体评估输出 (Macro Average) ---
    print("\n" + "="*25 + " HAR-X-AllStocks 模型性能 (Macro Average) " + "="*25)
    
    print("【验证集 Validation】:")
    print(f"  - MSE:    {np.mean(val_metrics['mse']):.4e}")
    print(f"  - RMSE:   {np.mean(val_metrics['rmse']):.4e}")
    print(f"  - QLIKE: {np.mean(val_metrics['qlike']):.4f}")
    
    print("【测试集 Test】:")
    print(f"  - MSE:    {np.mean(test_metrics['mse']):.4e}")
    print(f"  - RMSE:   {np.mean(test_metrics['rmse']):.4e}")
    print(f"  - QLIKE: {np.mean(test_metrics['qlike']):.4f}")
    print("="*64)
    
    # --- 4. 写入 Excel ---
    output_dir = 'results_low'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'har_ridge_energy_allstocks_predictions.xlsx')

    final_df = pd.concat(results_list, ignore_index=True)

    print(f"正在写入 Excel: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for stock_id, group_df in tqdm(final_df.groupby('stock_code'), desc="写入Sheet"):
            group_df[['真实值', '预测值', '数据集']].to_excel(writer, sheet_name=str(stock_id), index=False)
            
    print(f"保存完成。")

if __name__ == "__main__":
    main()