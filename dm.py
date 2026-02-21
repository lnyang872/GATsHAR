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
    Load all data, conduct a summary DM test, and output the results in tabular form to an Excel file.
    """
    file_path = '70_low_only_stock'
    all_files = glob.glob(os.path.join(file_path, "*.xlsx"))
    
    if not all_files:
        print("Error: No .xlsx files were found in the current directory.")
        return

    print(f"Found {len(all_files)} model files: {all_files}\n")

    # 1. Load and preprocess all data
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
                
                required_cols = ['Actual value', 'Predicted value', 'Dataset']
                if not all(col in df.columns for col in required_cols):
                    continue
                
                df_processed = df[required_cols]
                df_reversed = df_processed.iloc[::-1].reset_index(drop=True)

                if sheet_name not in all_data:
                    all_data[sheet_name] = {}
                all_data[sheet_name][model_name] = df_reversed
        except Exception as e:
            print(f"An error occurred while processing file {file}: {e}")

    stock_names = list(all_data.keys())
    if not stock_names:
        print("No data has been successfully loaded")
        return
        
    # Obtain the sorted list of model names to ensure consistent table order.
    model_names = sorted(model_names_from_files)
    
    # Create an Excel writer
    output_filename = 'DM_Test_Results_l_only_s.xlsx'
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        
        # 2. Perform overall testing on both the validation and test datasets respectively.
        for dataset_type in ['Validation', 'Test']:
            
            print(f"{'='*25} Generating: DM test table for {dataset_type} dataset {'='*25}")

            dm_table = pd.DataFrame(index=model_names, columns=model_names, dtype=object)
            
            model_pairs = combinations(model_names, 2)
            
            for model1, model2 in model_pairs:
                
                combined_loss_diffs = []
                
                # 3. Iterate through all stocks to collect loss difference sequences
                for stock in stock_names:
                    if model1 not in all_data[stock] or model2 not in all_data[stock]:
                        continue

                    df1 = all_data[stock][model1]
                    df2 = all_data[stock][model2]

                    # Find the common indices for all models on the current stock
                    indices_sets = [set(all_data[stock][m][all_data[stock][m]['dataset'] == dataset_type].index) for m in model_names if m in all_data[stock]]
                    if not indices_sets: continue
                    common_indices = sorted(list(set.intersection(*indices_sets)))

                    if len(common_indices) < 2:
                        continue

                    actuals = df1.loc[common_indices, 'Actual value'].values
                    pred1 = df1.loc[common_indices, 'Predicted value'].values
                    pred2 = df2.loc[common_indices, 'Predicted value'].values
                    
                    # loss difference: d = e1^2 - e2^2. 
                    loss_diff = (np.abs(actuals - pred1)**2 - np.abs(actuals - pred2)**2)
                    combined_loss_diffs.append(loss_diff)

                # 4. Concatenate and execute a DM verification once
                if not combined_loss_diffs:
                    continue

                final_d_series = np.concatenate(combined_loss_diffs)
                dm_stat, p_value = dm_test_from_diffs(final_d_series, h=1)
                
                # 5. Format the results and populate the table
                if not np.isnan(dm_stat):
                    # The asterisk indicates significance at the 5% level.
                    significance_star = "*" if p_value < 0.05 else ""
                    # Format DM statistics to two decimal places
                    formatted_stat = f"{dm_stat:.2f}{significance_star}"
                    dm_table.loc[model1, model2] = formatted_stat
                else:
                    dm_table.loc[model1, model2] = "N/A"

            # 6. Print and save to Excel
            print(f"\n--- {dataset_type} Set Results ---")
            print(dm_table.fillna('').to_string())
            
            # Write the DataFrame to a sheet in Excel
            dm_table.to_excel(writer, sheet_name=f'{dataset_type}_DM_Test', index=True)
            print(f"\nThe results have been written to the '{dataset_type}_DM_Test' worksheet in the Excel file '{output_filename}'.")
        
    print(f"\n{'='*20} All analyses completed {'='*20}")

if __name__ == '__main__':
    perform_overall_dm_analysis_to_excel()
