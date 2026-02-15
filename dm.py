import pandas as pd
import numpy as np
from scipy import stats
import glob
from itertools import combinations
import os

def dm_test_from_diffs(d):
    d = np.asarray(d)
    T = len(d)
    d_mean = np.mean(d)
    
    sample_variance_d = np.var(d, ddof=1) 
    var_d_mean = sample_variance_d / T
    dm_statistic = d_mean / np.sqrt(var_d_mean)
    p_value = 2 * stats.norm.sf(np.abs(dm_statistic))
    return dm_statistic, p_value

def perform_overall_dm_analysis_to_excel():
    """
    主函数，加载所有数据，进行汇总DM检验，并以表格形式输出到Excel文件。
    """
    file_path = '70_low_only_stock'
    all_files = glob.glob(os.path.join(file_path, "*.xlsx"))
    
    if not all_files:
        print("错误：在当前目录下没有找到任何 .xlsx 文件。")
        return

    print(f"找到了 {len(all_files)} 个模型文件: {all_files}\n")

    # 步骤 1: 加载并预处理所有数据
    all_data = {}
    model_names_from_files = []
    for file in all_files:
        model_name = os.path.splitext(os.path.basename(file))[0]
        model_names_from_files.append(model_name)
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df.columns = [col.strip() for col in df.columns]
                
                required_cols = ['真实值', '预测值', '数据集']
                if not all(col in df.columns for col in required_cols):
                    continue
                
                df_processed = df[required_cols]
                df_reversed = df_processed.iloc[::-1].reset_index(drop=True)

                if sheet_name not in all_data:
                    all_data[sheet_name] = {}
                all_data[sheet_name][model_name] = df_reversed
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")

    stock_names = list(all_data.keys())
    if not stock_names:
        print("没有成功加载任何数据，无法进行分析。")
        return
        
    # 获取排序后的模型名称列表，以保证表格顺序一致
    model_names = sorted(model_names_from_files)
    
    # 创建一个Excel写入器
    output_filename = 'DM_Test_Results_l_only_s.xlsx'
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        
        # 步骤 2: 分别对 Validation 和 Test 数据集进行总体检验
        for dataset_type in ['Validation', 'Test']:
            
            print(f"{'='*25} 正在生成: {dataset_type} 集的DM检验表 {'='*25}")
            
            # 创建一个空的DataFrame用于存放结果，行列都是模型名称
            dm_table = pd.DataFrame(index=model_names, columns=model_names, dtype=object)
            
            model_pairs = combinations(model_names, 2)
            
            for model1, model2 in model_pairs:
                
                combined_loss_diffs = []
                
                # 步骤 3: 遍历所有股票，收集损失差异序列
                for stock in stock_names:
                    if model1 not in all_data[stock] or model2 not in all_data[stock]:
                        continue

                    df1 = all_data[stock][model1]
                    df2 = all_data[stock][model2]

                    # 找到当前股票上所有模型的共同索引
                    indices_sets = [set(all_data[stock][m][all_data[stock][m]['数据集'] == dataset_type].index) for m in model_names if m in all_data[stock]]
                    if not indices_sets: continue
                    common_indices = sorted(list(set.intersection(*indices_sets)))

                    if len(common_indices) < 2:
                        continue

                    actuals = df1.loc[common_indices, '真实值'].values
                    pred1 = df1.loc[common_indices, '预测值'].values
                    pred2 = df2.loc[common_indices, '预测值'].values
                    
                    # 损失差异: d = e1^2 - e2^2. 
                    loss_diff = (np.abs(actuals - pred1)**2 - np.abs(actuals - pred2)**2)
                    combined_loss_diffs.append(loss_diff)

                # 步骤 4: 拼接并执行一次DM检验
                if not combined_loss_diffs:
                    continue

                final_d_series = np.concatenate(combined_loss_diffs)
                dm_stat, p_value = dm_test_from_diffs(final_d_series, h=1)
                
                # 步骤 5: 格式化结果并填充到表格中
                if not np.isnan(dm_stat):
                    # 添加星号表示5%水平下显著
                    significance_star = "*" if p_value < 0.05 else ""
                    # 格式化DM统计量，保留两位小数
                    formatted_stat = f"{dm_stat:.2f}{significance_star}"
                    dm_table.loc[model1, model2] = formatted_stat
                else:
                    dm_table.loc[model1, model2] = "N/A"

            # 步骤 6: 打印并保存到Excel
            print(f"\n--- {dataset_type} Set Results ---")
            print(dm_table.fillna('').to_string())
            
            # 将DataFrame写入Excel的一个sheet
            dm_table.to_excel(writer, sheet_name=f'{dataset_type}_DM_Test', index=True)
            print(f"\n结果已写入Excel文件 '{output_filename}' 的 '{dataset_type}_DM_Test' 工作表中。")
        
    print(f"\n{'='*20} 所有分析完成 {'='*20}")

if __name__ == '__main__':
    perform_overall_dm_analysis_to_excel()