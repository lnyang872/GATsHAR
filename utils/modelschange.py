import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, GATConv
from typing import List

class HeteroGNNModel(torch.nn.Module):
    def __init__(self, stock_feature_dim: int, energy_feature_dim: int, edge_feature_dim: int, 
                 heads: int, high_freq_output_dim: int, low_freq_output_dim: int,
                 hidden_layout: List[int], dropout: float = 0.0, activation: str = 'relu',
                 include_energy: bool = True,       # 能源开关（是否包含能源相关的信息，默认为true）
                 use_har_features: bool = True):    # HAR特征开关（是否包含HAR特征，默认为true）

        super().__init__()

        if not hidden_layout:
            raise ValueError("hidden_layout 列表不能为空！")
            
        self.include_energy = include_energy 
        self.use_har_features = use_har_features 

        # --- 设置激活函数 ---
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu
        elif activation == 'tanh':
            self.activation_fn = F.tanh
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        print(f"--- [模型初始化] 能源节点: {'启用' if self.include_energy else '禁用'}")
        print(f"--- [模型初始化] HAR 特征: {'启用' if self.use_har_features else '禁用'}")
        print(f"--- [模型初始化] 输入维度 -> Stock: {stock_feature_dim}, Energy: {energy_feature_dim}")
        
        first_layer_dim = hidden_layout[0]
        
        lin_modules = {
            'stock': Linear(stock_feature_dim, first_layer_dim)
        }
        
        if self.include_energy:
            if energy_feature_dim <= 0:
                 raise ValueError("配置错误：include_energy=True，但计算出的 energy_feature_dim <= 0。")
            lin_modules['energy'] = Linear(energy_feature_dim, first_layer_dim)
            
        self.lin_dict = nn.ModuleDict(lin_modules)

        # --- 构建GNN层和关系 ---
        self.convs = nn.ModuleList()
        num_layers = len(hidden_layout)
        
        # GNN层的输入维度
        in_channels = first_layer_dim
        
        # 根据开关定义图关系
        if self.include_energy:
            self.relations = [('stock', 'to', 'stock'), ('stock', 'to', 'energy'), ('energy', 'to', 'stock')]
        else:
            self.relations = [('stock', 'to', 'stock')]
        
        print(f"--- [模型初始化] GNN 关系架构: {self.relations} ---")
        
        for i in range(num_layers):
            out_channels = hidden_layout[i]
            
            conv = HeteroConv({
                rel: GATConv(
                    in_channels, 
                    out_channels, 
                    heads=heads, 
                    concat=True, 
                    edge_dim=edge_feature_dim, 
                    add_self_loops=False, 
                    dropout=dropout
                )
                for rel in self.relations
            }, aggr='sum')
            
            self.convs.append(conv)
            
            # 为下一层更新输入维度 (因为 GATConv concat=True)
            in_channels = out_channels * heads
            
        # --- 预测头 ---
        final_embedding_dim = hidden_layout[-1] * heads
        self.high_freq_head = Linear(final_embedding_dim, 1)
        self.low_freq_head = Linear(final_embedding_dim, 1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # --- 初始线性变换 ---
        x_dict = {
            node_type: self.activation_fn(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.lin_dict # 确保只处理存在的节点类型
        }
        
        # --- GNN层 ---
        for conv in self.convs:
            
            # 筛选出当前模型层支持的边属性
            current_edge_attr_dict = {k: v for k, v in edge_attr_dict.items() if k in self.relations}
            
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=current_edge_attr_dict)
            x_dict = {key: self.activation_fn(x) for key, x in x_dict.items()}

        if 'stock' not in x_dict:
            # 调试与错误处理逻辑
            print("\n" + "="*50)
            print("====== [调试模式] 致命错误：'stock' 键在GNN层后丢失！ ======")
            print(f"当前 x_dict 中剩余的键: {list(x_dict.keys())}")
            print(f"导致此问题的批次中，edge_index_dict 包含的边关系: {list(edge_index_dict.keys())}")
            print(f"模型配置的关系: {self.relations}")
            print("="*50 + "\n")
            return torch.tensor([]), torch.tensor([])
            
        final_stock_embedding = x_dict['stock']
        
        pred_high = self.high_freq_head(final_stock_embedding)
        pred_low = self.low_freq_head(final_stock_embedding)
        
        return pred_high.squeeze(-1), pred_low.squeeze(-1)