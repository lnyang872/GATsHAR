import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, GATConv
from typing import List

class HeteroGNNModel(torch.nn.Module):
    def __init__(self, stock_feature_dim: int, energy_feature_dim: int, edge_feature_dim: int, 
                 heads: int, high_freq_output_dim: int, low_freq_output_dim: int,
                 hidden_layout: List[int], dropout: float = 0.0, activation: str = 'relu',
                 include_energy: bool = True,        # Energy switch (whether it contains energy-related information; default is true)
                 use_har_features: bool = True):     # HAR Feature Switch (indicates whether HAR features are included; default is true)
        super().__init__()

        if not hidden_layout:
            raise ValueError("The hidden_layout list must not be empty")
            
        self.include_energy = include_energy 
        self.use_har_features = use_har_features 

        # --- Set up the activation function ---
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu
        elif activation == 'tanh':
            self.activation_fn = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        print(f"--- [Model initialisation] Energy switch: {'on' if self.include_energy else 'off'}")
        print(f"--- [Model initialisation] HAR Feature Switch: {'on' if self.use_har_features else 'off'}")
        print(f"--- [Model initialisation] Input dimension -> Stock: {stock_feature_dim}, Energy: {energy_feature_dim}")
        
        first_layer_dim = hidden_layout[0]
        
        lin_modules = {
            'stock': Linear(stock_feature_dim, first_layer_dim)
        }
        
        if self.include_energy:
            if energy_feature_dim <= 0:
                 raise ValueError(Configuration error:include_energy=True，but energy_feature_dim <= 0。")
            lin_modules['energy'] = Linear(energy_feature_dim, first_layer_dim)
            
        self.lin_dict = nn.ModuleDict(lin_modules)

        # --- Constructing GNN Layers and relevance ---
        self.convs = nn.ModuleList()
        num_layers = len(hidden_layout)
        
        # Input dimension of the GNN layer
        in_channels = first_layer_dim
        
        # According to the switch definition diagram relationship
        if self.include_energy:
            self.relations = [('stock', 'to', 'stock'), ('stock', 'to', 'energy'), ('energy', 'to', 'stock')]
        else:
            self.relations = [('stock', 'to', 'stock')]
        
        print(f"--- [Model initialisation] GNN Relational Architecture: {self.relations} ---")
        
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
            
            # Update the input dimension for the next layer (as GATConv concat=True)
            in_channels = out_channels * heads
            
        # --- Prediction Header ---
        final_embedding_dim = hidden_layout[-1] * heads
        self.high_freq_head = Linear(final_embedding_dim, 1)
        self.low_freq_head = Linear(final_embedding_dim, 1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # --- Initial linear transformation ---
        x_dict = {
            node_type: self.activation_fn(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.lin_dict     # Ensure that only existing node types are processed
        }
        
        # --- GNN layer  ---
        for conv in self.convs:
            
            # Filter edge attributes supported by the current model layer
            current_edge_attr_dict = {k: v for k, v in edge_attr_dict.items() if k in self.relations}
            
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=current_edge_attr_dict)
            x_dict = {key: self.activation_fn(x) for key, x in x_dict.items()}

        if 'stock' not in x_dict:
            # Debugging and Error Handling Logic
            print("\n" + "="*50)
            print("====== [Debug mode] error：The 'stock' key is lost after the GNN layer ======")
            print(f"Remaining keys in x_dict: {list(x_dict.keys())}")
            print(f"In the batch causing this issue, the edge relationships contained within edge_index_dict are: {list(edge_index_dict.keys())}")
            print(f"Relationship of model configurations: {self.relations}")
            print("="*50 + "\n")
            return torch.tensor([]), torch.tensor([])
            
        final_stock_embedding = x_dict['stock']
        
        pred_high = self.high_freq_head(final_stock_embedding)
        pred_low = self.low_freq_head(final_stock_embedding)
        
        return pred_high.squeeze(-1), pred_low.squeeze(-1)
