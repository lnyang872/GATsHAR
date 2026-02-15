# 超参数调优脚本 (支持连续区间优化)
# -*- coding: utf-8 -*-

import numpy as np
import yaml
import os
import optuna
import pandas as pd
from datetime import datetime
from train_single_final import train
import traceback

def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna的目标函数，为每一次试验（Trial）运行一次完整的训练和评估。
    """
    # 1. 加载基础配置
    with open('config/ablation_no_har.yaml', 'r', encoding='utf-8') as f:
        p = yaml.safe_load(f)
    
    # 2. 根据 YAML 中的类型，为 grid 中列出的参数进行调优
    for param in p['grid']:
        if param in p['hyperparameters']:
            values, dtype = p['hyperparameters'][param]
            
            # --- 核心修改部分 ---
            if param == 'hidden_layout':
                # 特殊处理 hidden_layout
                str_choices = [str(layout) for layout in values]
                suggested_str = trial.suggest_categorical(param, str_choices)
                p[param] = eval(suggested_str)
            
            elif dtype == 'cat':
                # 处理分类参数
                p[param] = trial.suggest_categorical(param, values)
            
            elif dtype == 'float':
                # 新增：处理浮点数（连续区间）参数
                min_val, max_val = values
                # 对学习率使用对数均匀分布进行采样，这在跨数量级搜索时更有效
                if param == 'learning_rate':
                    p[param] = trial.suggest_float(param, min_val, max_val, log=True)
                else:
                    p[param] = trial.suggest_float(param, min_val, max_val, log=False)
            
            elif dtype == 'int':
                # 新增：处理整数（连续区间）参数
                min_val, max_val = values
                p[param] = trial.suggest_int(param, min_val, max_val)
            # --- 修改结束 ---

    # 3. 为本次试验创建唯一的输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_folder = os.path.join('output', f"{p['modelname']}_tuning", f"trial_{trial.number}_{timestamp}")
    os.makedirs(trial_folder, exist_ok=True)

    # 4. 运行训练
    try:
        metrics = train(p=p, trial_folder=trial_folder)
        
        # 5. 保存指标
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr('output_folder', trial_folder)

        # 6. 返回核心优化目标
        return metrics.get('min_validation_loss', float('inf'))

    except Exception as e:
        print(f"错误: Trial #{trial.number} 因异常而失败: {e}")
        print("\n" + "="*25 + " 完整错误追溯 (Traceback) " + "="*25)
        traceback.print_exc()
        print("="*75 + "\n")
        return float('inf')


if __name__ == '__main__':
    with open('config/ablation_no_har.yaml', 'r', encoding='utf-8') as f:
        p = yaml.safe_load(f)
    
    main_output_folder = os.path.join('output', f"{p['modelname']}_tuning")
    os.makedirs(main_output_folder, exist_ok=True)
    
    storage_name = f"sqlite:///{main_output_folder}/optuna_study.db"
    study = optuna.create_study(
        study_name=p['modelname'],
        direction='minimize',
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=p['seed'])
    )
    
    study.optimize(objective, n_trials=p['n_trials'])
    
    # --- 调优结束后，生成最终的总结报告 ---
    print("\n--- 超参数调优完成 ---")
    
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            params = trial.params
            if 'hidden_layout' in params:
                params['hidden_layout'] = str(eval(params['hidden_layout']))
            metrics = trial.user_attrs
            row = {**params, **metrics}
            row['trial_number'] = trial.number
            row['value (best_val_loss)'] = trial.value
            results.append(row)
    
    results_df = pd.DataFrame(results)
    
    metric_cols = [
        'trial_number', 'value (best_val_loss)', 'best_epoch', 
        'validation_rmse_h', 'validation_mse_h', 'validation_qlike_h',
        'validation_rmse_l', 'validation_mse_l', 'validation_qlike_l',
        'test_rmse_h', 'test_mse_h', 'test_qlike_h',
        'test_rmse_l', 'test_mse_l', 'test_qlike_l', 
        'output_folder'
    ]
    param_cols = [col for col in p.get('grid', []) if col in results_df.columns]
    
    final_cols = []
    for col in metric_cols + param_cols:
        if col in results_df.columns:
            final_cols.append(col)
            
    results_df = results_df[final_cols]
    results_df = results_df.sort_values(by='value (best_val_loss)', ascending=True)

    summary_path = os.path.join(main_output_folder, 'tuning_summary.csv')
    results_df.to_csv(summary_path, index=False, float_format='%.12f')
    
    print(f"\n最佳试验结果:")
    best_params_display = study.best_params
    if 'hidden_layout' in best_params_display:
        best_params_display['hidden_layout'] = eval(best_params_display['hidden_layout'])
    print(best_params_display)
    
    print(f"  Value (min_validation_loss): {study.best_trial.value}")
    print(f"\n--- 完整的调优总结报告已保存至: {summary_path} ---")