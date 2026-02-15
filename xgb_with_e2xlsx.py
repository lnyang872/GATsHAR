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
    
    # 按股票代码分组计算
    for code, group in df.groupby('code'):
        mse = mean_squared_error(group['true'], group['pred'])
        rmse = np.sqrt(mse) # 单只股票的 RMSE
        qlike = calculate_qlike(group['true'].values, group['pred'].values)
        
        mse_list.append(mse)
        rmse_list.append(rmse)
        qlike_list.append(qlike)
    
    # 返回指标的平均值
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
            print(f"警告：文件 {file_path} 为空，已跳过。")
    return data_dict

def create_lag_features(df, prefix='RV'):
    """
    通用滞后特征生成函数
    """
    df = df.copy()
    col_name = df.columns[0] # 假设只有一列数据
    df[f'{prefix}_lag_day'] = df[col_name].shift(1)
    df[f'{prefix}_lag_week'] = df[col_name].shift(5)
    df[f'{prefix}_lag_month'] = df[col_name].shift(22)
    return df

def process_energy_data(energy_data_dict, max_length):
    """
    处理能源数据：生成所有能源的滞后特征并合并
    """
    energy_features_list = []
    for code, df in energy_data_dict.items():
        # 为能源特征添加特定前缀
        feat_df = create_lag_features(df, prefix=f'ENERGY_{code}')
        cols_to_keep = [c for c in feat_df.columns if 'lag' in c]
        feat_df = feat_df[cols_to_keep]
        energy_features_list.append(feat_df)
        
    if not energy_features_list:
        return pd.DataFrame()

    full_energy_df = pd.concat(energy_features_list, axis=1)
    # 统一长度并填充
    full_energy_df = full_energy_df.reindex(range(max_length)).ffill().fillna(0)
    full_energy_df = full_energy_df.reset_index(drop=True)
    full_energy_df['time_index'] = full_energy_df.index
    return full_energy_df

def process_all_stocks_features(stock_data_dict, max_length):
    """
    生成每只股票的滞后特征，并横向合并成一个巨大的宽表。
    """
    stock_features_list = []
    for code, df in stock_data_dict.items():
        # 为股票特征添加特定前缀，例如 STOCK_000001_lag_day
        feat_df = create_lag_features(df, prefix=f'STOCK_{code}')
        cols_to_keep = [c for c in feat_df.columns if 'lag' in c]
        feat_df = feat_df[cols_to_keep]
        stock_features_list.append(feat_df)
    
    if not stock_features_list:
        return pd.DataFrame()
    
    # 横向合并所有股票的特征
    full_stocks_df = pd.concat(stock_features_list, axis=1)
    
    # 统一长度并填充
    full_stocks_df = full_stocks_df.reindex(range(max_length)).ffill().fillna(0)
    full_stocks_df = full_stocks_df.reset_index(drop=True)
    full_stocks_df['time_index'] = full_stocks_df.index
    
    return full_stocks_df

def prepare_combined_data(stock_data, energy_data):
    # 1. 计算最大时间长度
    max_len_stock = max([len(df) for df in stock_data.values()]) if stock_data else 0
    max_len_energy = max([len(df) for df in energy_data.values()]) if energy_data else 0
    total_max_len = max(max_len_stock, max_len_energy)
    
    print(f"数据最大时间长度: {total_max_len}")
    
    # 2. 构建全量能源特征 (M * 3 列)
    print("--- 构建全量能源特征 ---")
    energy_features_df = process_energy_data(energy_data, total_max_len)
    
    # 3. 构建全量股票特征 (N * 3 列)
    print("--- 构建全量股票特征 (All-Stocks) ---")
    all_stocks_features_df = process_all_stocks_features(stock_data, total_max_len)
    
    # 4. 合并形成全局特征池
    print("--- 合并构建全局特征池 ---")
    if not energy_features_df.empty:
        global_features = pd.merge(all_stocks_features_df, energy_features_df, on='time_index', how='left')
    else:
        global_features = all_stocks_features_df
        
    print(f"全局特征矩阵维度: {global_features.shape}")

    # 5. 构建训练用的面板数据
    all_data_rows = []
    
    print("--- 正在构建面板数据集 (这可能需要一些时间) ---")
    for stock_code, df in tqdm(stock_data.items(), desc="Processing Stocks"):

        df_target = df[['RV']].copy()
        df_target['stock_code'] = stock_code
        df_target = df_target.reset_index() # 获得 time_index
        df_merged = pd.merge(df_target, global_features, on='time_index', how='left')
        
        all_data_rows.append(df_merged)
        
    combined_df = pd.concat(all_data_rows).reset_index(drop=True)
    
    # 排序以确保时间顺序
    combined_df.sort_values(by=['time_index', 'stock_code'], inplace=True)
    
    # 删除包含 NaN 的行
    combined_df = combined_df.dropna()

    # 保存索引信息方便后续还原
    index_cols = combined_df[['time_index', 'stock_code']]
    
    # 6. 生成 One-Hot 编码 (区分股票身份)
    print("--- 生成独热编码 ---")
    features_df = pd.get_dummies(combined_df, columns=['stock_code'], prefix='stock')
    
    # 分离 X 和 y
    # drop 'RV' (Target) and 'time_index' (Not a feature)
    X = features_df.drop(['RV', 'time_index'], axis=1)
    y = features_df['RV']
    
    print(f"最终特征矩阵 X 维度: {X.shape}")
    
    return X, y, index_cols

def objective(trial, X_train, y_train, X_val, y_val):

    # 1. 定义超参数搜索空间 
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

    # 2. 添加固定的参数
    param_grid.update({
        'n_estimators': 1000, 
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': SEED,
        'early_stopping_rounds': 25,
        # 'tree_method': 'hist' # 针对大数据集通常更快
    })
    
    # 3. 训练模型
    model = xgb.XGBRegressor(**param_grid)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False) 
    
    # 4. 在验证集上评估并返回指标
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse

def main():
    stock_folder = "new_day_vol"
    energy_folder = "new_day_vol_energy"
    
    print("正在加载数据...")
    stock_data = load_data_from_folder(stock_folder)
    energy_data = load_data_from_folder(energy_folder)

    # 调用修改后的数据准备函数
    X_combined, y_combined, index_info = prepare_combined_data(stock_data, energy_data)

    print("--- 正在划分数据集 ---")
    # 使用 100% 数据
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
    print("正在进行标准化 (StandardScaler)...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # --- 运行 Optuna 调优 ---
    print("--- 启动 Optuna 超参数调优 ---")
    
    objective_with_data = functools.partial(objective,
                                            X_train=X_train_scaled,
                                            y_train=y_train,
                                            X_val=X_val_scaled,
                                            y_val=y_val)

    study = optuna.create_study(
        direction='minimize', 
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    
    # 减少 trial 次数，因为特征维度增加后训练会变慢
    study.optimize(objective_with_data, n_trials=50)
    
    print(f"调优完成。最佳验证集 MSE: {study.best_value:.4e}")
    print(f"最佳参数组合: {study.best_params}")

    best_params = study.best_params
    
    best_params.update({
        'n_estimators': 1000, 
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': SEED,
        'early_stopping_rounds': 25 
    })
    
    model = xgb.XGBRegressor(**best_params)
    
    print("开始使用最佳参数训练最终的 XGBoost 模型...")
    model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)], 
              verbose=False)
    
    best_iteration = model.best_iteration
    print(f"模型通过早停确定最佳树数量为: {best_iteration}")
    
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    print("\n" + "="*25 + " XGBoost 模型性能 (Macro Average) " + "="*25)
    
    # --- Macro Average 计算 ---
    val_mse, val_rmse, val_qlike = calculate_macro_metrics(y_val, val_pred, index_val['stock_code'])
    print("【验证集 Validation】:")
    print(f"  - Macro MSE:    {val_mse:.4e}")
    print(f"  - Macro RMSE:   {val_rmse:.4e}")
    print(f"  - Macro QLIKE: {val_qlike:.4f}")

    test_mse, test_rmse, test_qlike = calculate_macro_metrics(y_test, test_pred, index_test['stock_code'])
    print("【测试集 Test】:")
    print(f"  - Macro MSE:    {test_mse:.4e}")
    print(f"  - Macro RMSE:   {test_rmse:.4e}")
    print(f"  - Macro QLIKE: {test_qlike:.4f}")
    print("="*50)

    # --- 保存结果 ---
    output_dir = 'results_low'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'xgboost_allstocks_energy_tuned_predictions.xlsx')
    
    res_val = index_val.copy()
    res_val['真实值'] = y_val.values
    res_val['预测值'] = val_pred
    res_val['数据集'] = 'Validation'
    
    res_test = index_test.copy()
    res_test['真实值'] = y_test.values
    res_test['预测值'] = test_pred
    res_test['数据集'] = 'Test'
    
    all_res = pd.concat([res_val, res_test])
    
    print(f"正在写入 Excel: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for stock_id, group_df in tqdm(all_res.groupby('stock_code'), desc="写入Sheet"):
            group_df[['真实值', '预测值', '数据集']].to_excel(writer, sheet_name=str(stock_id), index=False)
            
    print("保存完成。")

if __name__ == "__main__":
    main()