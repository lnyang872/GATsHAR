import yaml
import os
from utils.datasetchange import FinalHeteroDataset
# 导入 pandas 是因为 FinalHeteroDataset 内部依赖它
import pandas as pd 

def create_all_caches():
    """
    一个用于批量创建所有指定seq_length的数据集缓存的工具脚本。
    它会读取调优配置文件，并为其中的每一个seq_length值生成一个专属的
    PyTorch Geometric缓存文件夹。
    """
    # --- 1. 加载超参数调优的配置文件 ---
    try:
        config_path = './config/GNN_param_optuna.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            p = yaml.safe_load(f)
        print(f"--- 成功加载配置文件: {config_path} ---")
    except FileNotFoundError:
        print(f"错误：找不到调优配置文件 '{config_path}'。")
        return

    # --- 2. 获取所有需要处理的 seq_length ---
    try:
        seq_lengths_to_process = p['hyperparameters']['seq_length'][0]
        if not isinstance(seq_lengths_to_process, list):
            print("错误：配置文件中的 seq_length 格式不正确。")
            return
        print(f"--- 准备为以下 seq_length 创建缓存: {seq_lengths_to_process} ---")
    except (KeyError, IndexError):
        print("错误：在配置文件中找不到 'seq_length' 的设置。")
        return

    # --- 3. 循环创建每一个缓存文件 ---
    for seq_length in seq_lengths_to_process:
        print(f"\n=================================================")
        print(f"====== 开始处理 seq_length = {seq_length} ======")
        print(f"=================================================")
        
        # 定义当前seq_length专属的缓存路径
        root_path = f"processed_data1001/final_hetero_dataset_seq_{seq_length}"
        
        # 检查缓存是否已经存在
        if os.path.exists(os.path.join(root_path, 'processed', 'data.pt')):
            print(f"--- 缓存已存在于 '{root_path}'，跳过。 ---")
            continue
            
        try:
            # 实例化Dataset类。如果缓存不存在，它会自动调用process()方法
            # 我们在这里“假装”要加载它，从而触发它的创建过程。
            print(f"--- 正在为 seq_length={seq_length} 创建新的数据集缓存... ---")
            dataset = FinalHeteroDataset(
                hdf5_file_vol=p['hdf5_file_vol_std'],
                hdf5_file_volvol=p['hdf5_file_volvol_std'],
                stock_har_rv_folder=p['stock_har_rv_folder'],
                energy_har_rv_folder=p['energy_har_rv_folder'],
                stock_energy_corr_folder=p['stock_energy_corr_folder'],
                energy_energy_corr_folder=p['energy_energy_corr_folder'],
                node_info_file=p['node_info_file'],
                root=root_path,
                seq_length=seq_length,
                intraday_points=p['intraday_points']
            )
            print(f"--- seq_length = {seq_length} 的缓存创建成功！---")

        except Exception as e:
            print(f"!!!!!! 处理 seq_length = {seq_length} 时发生严重错误: {e} !!!!!!")
            # 即使一个失败了，也继续尝试下一个
            continue
            
    print("\n=================================================")
    print("====== 所有指定的缓存文件均已处理完毕。 ======")
    print("=================================================")


if __name__ == '__main__':
    # 确保你的 utils.datasetchange 脚本使用的是“离线”版本
    # (即 process() 方法内有拼接逻辑的版本)
    create_all_caches()
