# unified_prediction_script.py

import os
import yaml
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
# 确保以下模块在您的运行环境中可导入
from utils.losses import QLIKELoss
from utils.datasetchange import FinalHeteroDataset
from utils.modelschange import HeteroGNNModel

# ==============================================================================
# I. 预测生成函数
# ==============================================================================

def generate_predictions(model, loader, device, p, stock_ids, diag_mean_high, diag_std_high, diag_mean_low, diag_std_low):
    """
    生成预测并返回包含真实值和预测值的 Pandas DataFrame。
    """
    model.eval()
    all_preds_h, all_actuals_h, all_preds_l, all_actuals_l = [], [], [], []
    
    with torch.no_grad():
        for data in tqdm(loader, desc="正在生成预测"):
            data = data.to(device)
            pred_high, pred_low = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            if pred_high.numel() == 0: continue
            
            y_high, y_low = data['stock'].y_high, data['stock'].y_low
            
            # 反标准化
            pred_high_unstd = (pred_high * torch.tensor(diag_std_high, device=device)) + torch.tensor(diag_mean_high, device=device)
            y_high_unstd = (y_high * torch.tensor(diag_std_high, device=device)) + torch.tensor(diag_mean_high, device=device)
            pred_low_unstd = (pred_low * torch.tensor(diag_std_low, device=device)) + torch.tensor(diag_mean_low, device=device)
            y_low_unstd = (y_low * torch.tensor(diag_std_low, device=device)) + torch.tensor(diag_mean_low, device=device)
            
            all_preds_h.append(pred_high_unstd.cpu())
            all_actuals_h.append(y_high_unstd.cpu())
            all_preds_l.append(pred_low_unstd.cpu())
            all_actuals_l.append(y_low_unstd.cpu())

    # 将所有批次的结果合并
    preds_h, actuals_h = torch.cat(all_preds_h, dim=0), torch.cat(all_actuals_h, dim=0)
    preds_l, actuals_l = torch.cat(all_preds_l, dim=0), torch.cat(all_actuals_l, dim=0)
    
    # 预测值的非负约束 (防止 QLIKE 爆炸)
    PREDICTION_LOWER_BOUND = 1e-8
    preds_h = torch.clamp(preds_h, min=PREDICTION_LOWER_BOUND)
    preds_l = torch.clamp(preds_l, min=PREDICTION_LOWER_BOUND)
    
    num_stocks = len(stock_ids)
    
    # 高频数据重塑
    preds_h = preds_h.view(-1, num_stocks)
    actuals_h = actuals_h.view(-1, num_stocks)
    
    # 低频数据重塑和日度聚合
    preds_l = preds_l.view(-1, num_stocks)
    actuals_l = actuals_l.view(-1, num_stocks)
    
    intraday_points = p.get('intraday_points', 3)
    num_samples = preds_l.shape[0]
    
    if num_samples > 0 and num_samples % intraday_points == 0:
        num_days = num_samples // intraday_points
        # 低频预测值取日内平均
        avg_preds_per_day = preds_l.reshape(num_days, intraday_points, num_stocks).mean(dim=1)
        # 低频真实值取日内最后一个点（T+1d的RV）
        actuals_per_day = actuals_l.reshape(num_days, intraday_points, num_stocks)[:, -1, :]
    else:
        avg_preds_per_day = torch.empty(0, num_stocks)
        actuals_per_day = torch.empty(0, num_stocks)

    # 创建高频 DataFrame
    df_h_true = pd.DataFrame(actuals_h.numpy(), columns=stock_ids)
    df_h_pred = pd.DataFrame(preds_h.numpy(), columns=stock_ids)
    df_h_combined = pd.concat([df_h_true, df_h_pred], axis=1, keys=['True', 'Predicted'])

    # 创建低频 DataFrame
    df_l_true = pd.DataFrame(actuals_per_day.numpy(), columns=stock_ids)
    df_l_pred = pd.DataFrame(avg_preds_per_day.numpy(), columns=stock_ids)
    df_l_combined = pd.concat([df_l_true, df_l_pred], axis=1, keys=['True', 'Predicted'])

    return df_h_combined, df_l_combined

# ==============================================================================
# II. Excel 组合函数
# ==============================================================================

def combine_to_excel(df_val, df_test, output_excel_path, file_prefix):
    """
    将宽格式的验证集和测试集DataFrame转换为长格式，并保存为分Sheet的Excel文件。
    """
    print(f"\n--- 正在处理 {file_prefix} 数据并写入 Excel ---")
    
    all_long_dfs = []
    
    for df_wide, dataset_name in [(df_val, 'Validation'), (df_test, 'Test')]:
        if df_wide.empty:
            print(f"警告：{dataset_name} 的 DataFrame 为空，跳过处理。")
            continue

        # 使用 .stack(level=1) 将股票ID从列标题“堆叠”到行索引中
        df_long = df_wide.stack(level=1)
        
        # 重置索引，将'stock_id'从索引变为普通列
        df_long = df_long.reset_index()
        
        # 重命名列
        df_long = df_long.rename(columns={'level_0': 'Sample_Index', 'level_1': 'stock_id',
                                         'True': '真实值', 'Predicted': '预测值'})
        
        # 添加数据集标识列
        df_long['数据集'] = dataset_name
        
        all_long_dfs.append(df_long)

    if not all_long_dfs:
        print("错误：没有有效数据可以写入Excel。")
        return

    # 3. 合并验证集和测试集
    all_results_df = pd.concat(all_long_dfs, ignore_index=True)
    
    # 4. 将结果写入分Sheet的Excel文件
    print(f"--- 正在将结果写入Excel文件: {output_excel_path} ---")
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        # 按 'stock_id' 列分组
        for stock_id, group_df in tqdm(all_results_df.groupby('stock_id'), desc="写入Sheet"):
            # 选择最终需要的三列
            sheet_data = group_df[['真实值', '预测值', '数据集']]
            
            # 写入以股票ID命名的Sheet，并且不写入pandas的行号索引
            sheet_data.to_excel(writer, sheet_name=str(stock_id), index=False)

    print(f"成功创建文件: {output_excel_path}")

# ==============================================================================
# III. 主流程函数 (已集成 70% 稳健性检查逻辑)
# ==============================================================================

def main():
    # ==============================================================================
    # >>> 1. 用户需要配置的区域 <<<
    # ==============================================================================
    
    # 填入您最佳试验的输出文件夹路径
    TRIAL_FOLDER = 'output/GATHAR_with_e_15_h_5154100_tuning/trial_39_20251124-110023' # <--- 请修改为您的路径
    
    # 定义最终保存预测文件的输出文件夹
    PREDICTION_OUTPUT_FOLDER = 'prediction_results_70h1'
    
    # 定义稳健性检查的比例 (70%)
    ROBUSTNESS_PROPORTION = 0.7 
    
    # ==============================================================================

    print(f"--- 正在从试验文件夹加载配置: {TRIAL_FOLDER} ---")
    
    # 构造配置文件和模型文件的完整路径
    config_path = os.path.join(TRIAL_FOLDER, 'GNN_param_used.yaml')
    model_path = os.path.join(TRIAL_FOLDER, 'best_model.pth')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"错误：在文件夹 '{TRIAL_FOLDER}' 中找不到 'GNN_param_used.yaml' 或 'best_model.pth'。")
        print("请确保TRIAL_FOLDER路径正确。")
        return

    # 1. 加载特定试验的配置
    with open(config_path, 'r', encoding='utf-8') as f:
        p = yaml.safe_load(f)

    # 2. 加载完整数据集
    print("--- 正在加载数据集 ---")
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
    
    # 3. ⭐️ 稳健性检查的数据截断逻辑
    intraday_points = p.get('intraday_points', 3)
    total_days = len(dataset) // intraday_points
    
    robustness_limit_days = int(ROBUSTNESS_PROPORTION * total_days) 
    dataset = dataset[:robustness_limit_days * intraday_points] # ⬅️ 截断数据集
    total_days = robustness_limit_days # ⬅️ 更新总天数
    
    print(f"--- [预测截断] 总天数限制为: {total_days} 天 ({ROBUSTNESS_PROPORTION * 100:.0f}%) ---")
    
    subset_days = total_days
    subset_dataset = dataset # 截断后的数据集
    
    # 4. 划分数据集 (在截断后的数据集上按比例划分 train/val/test)
    train_days = int(p['train_proportion'] * subset_days)
    validation_days = int(p['validation_proportion'] * subset_days)
    train_size = train_days * intraday_points
    validation_end_idx = (train_days + validation_days) * intraday_points
    
    # 划分数据集
    train_dataset = subset_dataset[:train_size]
    validation_dataset = subset_dataset[train_size:validation_end_idx]
    test_dataset = subset_dataset[validation_end_idx:]
    
    print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(validation_dataset)} | 测试集大小: {len(test_dataset)}")
    
    # 5. 创建DataLoader
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    print(f"验证集样本数: {len(validation_dataset)}, 测试集样本数: {len(test_dataset)}")

    # 6. 加载统计数据和模型构建参数
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
    
    # 特征维度计算需要与训练时的配置一致
    num_energy = len(node_info.get('energy_ids', []))
    stock_feature_len_base = 3 + 1 + (num_stocks - 1) + num_energy
    stock_feature_dim = stock_feature_len_base * p['seq_length']
    energy_feature_len_base = 3 + 1 + num_stocks + (num_energy - 1)
    energy_feature_dim = energy_feature_len_base * p['seq_length']
    edge_feature_dim = 3 * p['seq_length']

    # 7. 构建模型骨架并加载权重
    print("--- 正在构建模型并加载预训练权重... ---")
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
    print("模型加载成功。")

    # 8. 生成预测
    val_h_df, val_l_df = generate_predictions(model, validation_loader, device, p, stock_ids, diag_mean_high, diag_std_high, diag_mean_low, diag_std_low)
    test_h_df, test_l_df = generate_predictions(model, test_loader, device, p, stock_ids, diag_mean_high, diag_std_high, diag_mean_low, diag_std_low)
    
    # 9. 合并并保存为Excel
    os.makedirs(PREDICTION_OUTPUT_FOLDER, exist_ok=True)
    
    # 高频结果保存
    combine_to_excel(
        df_val=val_h_df, 
        df_test=test_h_df, 
        output_excel_path=os.path.join(PREDICTION_OUTPUT_FOLDER, 'GNN_predictions_high_freq.xlsx'),
        file_prefix='high_freq'
    )

    # 低频结果保存
    combine_to_excel(
        df_val=val_l_df, 
        df_test=test_l_df, 
        output_excel_path=os.path.join(PREDICTION_OUTPUT_FOLDER, 'GNN_predictions_low_freq.xlsx'),
        file_prefix='low_freq'
    )

    print("\n" + "="*30 + " 所有预测和保存任务完成 " + "="*30)


if __name__ == '__main__':
    main()