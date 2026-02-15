import os
import json
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, HeteroData

class FinalHeteroDataset(InMemoryDataset):
    """
    用于构建异构图序列的数据集类。
    """
    
    def __init__(self, hdf5_file_vol, hdf5_file_volvol, stock_har_rv_folder, 
                 energy_har_rv_folder, stock_energy_corr_folder, energy_energy_corr_folder, 
                 node_info_file, root, transform=None, pre_transform=None, 
                 seq_length=15, intraday_points=3, 
                 include_energy=True,      # 能源开关（是否包含能源相关的信息，默认为true）
                 use_har_features=True):   # HAR特征开关（是否包含HAR特征，默认为true）
        
        # --- 基础设置 ---
        self.hdf5_file_vol = hdf5_file_vol
        self.hdf5_file_volvol = hdf5_file_volvol
        self.stock_har_rv_folder = stock_har_rv_folder
        self.node_info_file = node_info_file
        self.intraday_points = intraday_points
        self.seq_length = seq_length
        
        # --- 保存开关状态 ---
        self.include_energy = include_energy 
        self.use_har_features = use_har_features 

        # --- 根据 include_energy 设置能源路径 ---
        if self.include_energy:
            self.energy_har_rv_folder = energy_har_rv_folder
            self.stock_energy_corr_folder = stock_energy_corr_folder
            self.energy_energy_corr_folder = energy_energy_corr_folder
        else:
            self.energy_har_rv_folder = None
            self.stock_energy_corr_folder = None
            self.energy_energy_corr_folder = None
            
        # --- 加载节点元信息 ---
        with open(node_info_file, 'r', encoding='utf-8') as f:
            self.node_info = json.load(f)
        
        self.stock_ids = self.node_info['stock_ids']
        self.energy_ids = self.node_info['energy_ids'] if self.include_energy else []
        self.node_order = self.node_info['node_order'] 
        
        self.num_stocks = len(self.stock_ids)
        self.num_energy = len(self.energy_ids)
        self.num_nodes = len(self.node_order)

        # --- 索引映射 ---
        self.node_to_idx = {name: i for i, name in enumerate(self.node_order)}
        self.stock_indices = torch.tensor([self.node_to_idx[sid] for sid in self.stock_ids], dtype=torch.long)
        
        if self.include_energy:
            self.energy_indices = torch.tensor([self.node_to_idx[eid] for eid in self.energy_ids], dtype=torch.long)
        else:
            self.energy_indices = torch.tensor([], dtype=torch.long)

        # --- 初始化父类 ---
        super().__init__(root, transform, pre_transform)
        
        # --- 加载缓存 ---
        if os.path.exists(self.processed_paths[0]):
            print(f"--- 正在从缓存目录 '{self.processed_dir}' 加载数据... ---")
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("--- 缓存数据加载成功。 ---")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _load_features_to_dict(self, folder_path, file_ids):
        if folder_path is None or not os.path.exists(folder_path):
            return {}
            
        feature_dict = {}
        for asset_id in file_ids:
            try:
                fpath = next(os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith(asset_id))
                df = pd.read_csv(fpath, header=None, index_col=0, parse_dates=True)
                feature_dict[asset_id] = df
            except StopIteration:
                print(f"警告：在文件夹 {folder_path} 中找不到资产 {asset_id} 的文件。")
        return feature_dict

    def process(self):
        print("\n==========================================================")
        print("====== 开始执行 process() 方法 (构建最终版缓存)... ======")
        print(f"====== [配置] 能源节点 (include_energy): {'启用' if self.include_energy else '禁用'} ======")
        print(f"====== [配置] HAR特征 (use_har_features): {'启用' if self.use_har_features else '禁用'} ======")
        print("==========================================================")
        
        # --- 1. 加载 HAR 数据 ---
        all_har_features = {
            **self._load_features_to_dict(self.stock_har_rv_folder, self.stock_ids)
        }
        if self.include_energy:
            all_har_features.update(self._load_features_to_dict(self.energy_har_rv_folder, self.energy_ids))
        
        if not all_har_features:
            raise FileNotFoundError("未能在指定路径找到任何 HAR-RV 特征文件。")
            
        valid_dates = next(iter(all_har_features.values())).index
        
        # --- 2. 加载相关性 ---
        corr_df_dict = {}
        all_corr_folders = []
        if self.include_energy:
            all_corr_folders = [self.stock_energy_corr_folder, self.energy_energy_corr_folder]
        
        for folder in all_corr_folders:
            if not os.path.exists(folder): continue
            for pair_file in os.listdir(folder):
                if pair_file.endswith('.csv'):
                    pair_name = os.path.splitext(pair_file)[0]
                    path = os.path.join(folder, pair_file)
                    corr_values = pd.read_csv(path, header=None).values.flatten()
                    if len(corr_values) == len(valid_dates):
                        corr_df_dict[pair_name] = pd.Series(corr_values, index=valid_dates)
        
        all_corr_df = pd.DataFrame(corr_df_dict)

        # --- 3. 时间步映射 ---
        all_hf_steps = []
        for date in valid_dates:
            day_start_index = valid_dates.get_loc(date) * self.intraday_points
            for point_idx in range(self.intraday_points):
                all_hf_steps.append((day_start_index + point_idx, date))
        
        data_list = []
        skipped_samples_count = 0
        
        if self.include_energy:
            relations = [('stock', 'to', 'stock'), ('stock', 'to', 'energy'), ('energy', 'to', 'stock')]
        else:
            relations = [('stock', 'to', 'stock')]
        
        # --- 4. 遍历构建 ---
        with h5py.File(self.hdf5_file_vol, 'r') as f_vol, h5py.File(self.hdf5_file_volvol, 'r') as f_volvol:
            
            iterator_range = len(all_hf_steps) - self.seq_length - self.intraday_points
            
            for i in tqdm(range(iterator_range), desc="正在生成异构图序列"):
                
                seq_data_list = []
                is_valid_sequence = True
                feature_len_per_step = -1 

                for j in range(self.seq_length):
                    hf_global_idx, current_date = all_hf_steps[i + j]
                    
                    try:
                        cov_matrix = np.array(f_vol[str(hf_global_idx)])
                        volvol_matrix = np.array(f_volvol[str(hf_global_idx)])
                    except KeyError:
                        is_valid_sequence = False; break
                        
                    data = HeteroData()
                    node_features_t = []
                    
                    for node_idx, node_id in enumerate(self.node_order):
                        
                        # === 临时列表，用于按顺序收集该节点的特征 ===
                        features_to_concat = []

                        if node_id in self.stock_ids:
                            # --- A. 股票节点 ---
                            
                            # 条件性添加 HAR 特征
                            if self.use_har_features:
                                f_har = all_har_features[node_id].loc[current_date].values
                                features_to_concat.append(f_har)
                            
                            # 基础特征
                            f_vi = np.array([cov_matrix[node_idx, node_idx]])
                            stock_local_idx = self.stock_ids.index(node_id)
                            f_cij_ss = np.delete(cov_matrix[node_idx, self.stock_indices], stock_local_idx)
                            
                            features_to_concat.append(f_vi)
                            features_to_concat.append(f_cij_ss)
                            
                            # 能源相关性
                            if self.include_energy:
                                corr_values_se = [ all_corr_df.get(f"{node_id}_{eid}", all_corr_df.get(f"{eid}_{node_id}", pd.Series([0])))[current_date] for eid in self.energy_ids ]
                                features_to_concat.append(np.array(corr_values_se))
                            
                            # 合并
                            final_features = np.concatenate(features_to_concat)
                            node_features_t.append(final_features)
                            
                            if feature_len_per_step == -1:
                                feature_len_per_step = len(final_features)

                        elif self.include_energy and node_id in self.energy_ids:
                            # --- B. 能源节点 ---
                            
                            # 条件性添加 HAR 特征
                            if self.use_har_features:
                                f_har = all_har_features[node_id].loc[current_date].values
                                features_to_concat.append(f_har)
                            
                            f_vi = np.array([cov_matrix[node_idx, node_idx]])
                            f_cij_es = cov_matrix[node_idx, self.stock_indices]
                            
                            corr_values_ee = []
                            for other_eid in self.energy_ids:
                                if node_id == other_eid: continue
                                pair1, pair2 = f"{node_id}_{other_eid}", f"{other_eid}_{node_id}"
                                corr_series = all_corr_df.get(pair1, all_corr_df.get(pair2, pd.Series([0])))
                                corr_values_ee.append(corr_series[current_date])
                            
                            features_to_concat.extend([f_vi, f_cij_es, np.array(corr_values_ee)])
                            
                            node_features_t.append(np.concatenate(features_to_concat))
                        
                        elif (not self.include_energy) and (node_id in self.energy_ids):
                            # --- C. 能源节点 (占位符模式) ---
                            # 当 include_energy=False 时，我们需要填补 0，保持 HDF5 索引对齐
                            
                            if feature_len_per_step == -1:
                                # [修改] 计算占位符长度时，必须考虑 HAR 开关
                                base_har_len = 3 if self.use_har_features else 0 
                                base_vi_len = 1
                                base_cij_ss_len = self.num_stocks - 1
                                feature_len_per_step = base_har_len + base_vi_len + base_cij_ss_len
                            
                            node_features_t.append(np.zeros(feature_len_per_step))

                    # --- 赋值特征 ---
                    x_all = torch.tensor(np.array(node_features_t), dtype=torch.float)
                    data['stock'].x = x_all[self.stock_indices]
                    
                    if self.include_energy:
                        data['energy'].x = x_all[self.energy_indices]
                    
                    # --- 构建边 ---
                    adj_matrix = volvol_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)
                    global_edge_index = torch.from_numpy(np.vstack(np.where(adj_matrix != 0))).long()
                    
                    global_edge_attr = None
                    if global_edge_index.numel() > 0:
                        variances = torch.tensor(np.diag(volvol_matrix), dtype=torch.float)
                        source_vars = variances[global_edge_index[0]]
                        target_vars = variances[global_edge_index[1]]
                        covars = torch.tensor(adj_matrix[global_edge_index[0], global_edge_index[1]], dtype=torch.float)
                        global_edge_attr = torch.stack([covars, source_vars, target_vars], dim=1)
                    
                    for src, rel, dst in relations:
                        if global_edge_index.numel() == 0: continue
                        
                        global_src_indices = self.stock_indices if src == 'stock' else self.energy_indices
                        global_dst_indices = self.stock_indices if dst == 'stock' else self.energy_indices
                        
                        mask = torch.isin(global_edge_index[0], global_src_indices) & torch.isin(global_edge_index[1], global_dst_indices)
                        
                        if mask.sum() == 0: continue

                        src_map = {idx.item(): i for i, idx in enumerate(global_src_indices)}
                        dst_map = {idx.item(): i for i, idx in enumerate(global_dst_indices)}
                        
                        local_edge_index = global_edge_index[:, mask]
                        
                        local_edge_index_0 = torch.tensor([src_map[idx.item()] for idx in local_edge_index[0]])
                        local_edge_index_1 = torch.tensor([dst_map[idx.item()] for idx in local_edge_index[1]])
                        local_edge_index = torch.stack([local_edge_index_0, local_edge_index_1], dim=0)
                            
                        data[src, rel, dst].edge_index = local_edge_index
                        data[src, rel, dst].edge_attr = global_edge_attr[mask]
                    
                    # --- 标签 y ---
                    try:
                        next_hf_global_idx, _ = all_hf_steps[i + j + 1]
                        next_cov_matrix = np.array(f_vol[str(next_hf_global_idx)])
                        y_high_all = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float)
                        data['stock'].y_high = y_high_all[self.stock_indices]
                        
                        next_day_date = all_hf_steps[i + j + self.intraday_points][1]
                        y_low_values = [all_har_features[sid].iloc[:, 0].loc[next_day_date] for sid in self.stock_ids]
                        data['stock'].y_low = torch.tensor(y_low_values, dtype=torch.float)
                    except (KeyError, IndexError):
                        is_valid_sequence = False; break
                    
                    seq_data_list.append(data)

                if not is_valid_sequence:
                    continue
                
                # --- 5. 聚合序列 ---
                final_data = HeteroData()
                final_data['stock'].x = torch.cat([d['stock'].x for d in seq_data_list], dim=1)
                
                if self.include_energy:
                    final_data['energy'].x = torch.cat([d['energy'].x for d in seq_data_list], dim=1)
                
                last_data = seq_data_list[-1]
                
                for rel in last_data.edge_types:
                    final_data[rel].edge_index = last_data[rel].edge_index
                    edge_attr_seq = []
                    single_step_edge_dim = 3
                    
                    for d in seq_data_list:
                        if rel in d and 'edge_attr' in d[rel]:
                            edge_attr_seq.append(d[rel].edge_attr)
                        else:
                            num_edges = last_data[rel].edge_index.size(1)
                            placeholder = torch.zeros(num_edges, single_step_edge_dim, dtype=torch.float)
                            edge_attr_seq.append(placeholder)
                    
                    final_data[rel].edge_attr = torch.cat(edge_attr_seq, dim=1)

                final_data['stock'].y_high = last_data['stock'].y_high
                final_data['stock'].y_low = last_data['stock'].y_low
                
                total_edges = sum(index.size(1) for index in final_data.edge_index_dict.values())
                
                if total_edges > 0:
                    data_list.append(final_data)
                else:
                    skipped_samples_count += 1

        if skipped_samples_count > 0:
            print(f"\n[数据处理警告] 由于没有任何边，已跳过 {skipped_samples_count} 个样本。")

        if not data_list:
            raise ValueError("未能成功创建任何有效的图样本。")

        print("\n正在整理和保存最终数据集...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"数据集缓存已成功创建并保存到 {self.processed_paths[0]}。")
        print("==========================================================")