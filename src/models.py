import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINEConv, JumpingKnowledge
from torch.nn import BatchNorm1d, Dropout, Sequential, ReLU, Linear, ModuleList, LayerNorm
from typing import List


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, dropout_p=0.3):
        super().__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.norm3 = LayerNorm(hidden_channels)

        self.dropout = Dropout(dropout_p)

        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)  # critical for diversity
        x = self.linear(x)

        return x


class GINEGCN(torch.nn.Module):
    def __init__(self, num_node_features, edge_dim, hidden_channels=128):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels

        def mlp(in_ch, out_ch):
            return Sequential(
                Linear(in_ch, self.hidden_channels),
                ReLU(),
                Linear(self.hidden_channels, out_ch),
            )

        self.conv1 = GINEConv(
            nn=mlp(self.num_node_features, self.hidden_channels),
            edge_dim=edge_dim,
        )
        self.conv2 = GINEConv(
            nn=mlp(self.hidden_channels, self.hidden_channels),
            edge_dim=edge_dim,
        )
        self.conv3 = GINEConv(
            nn=mlp(self.hidden_channels, self.hidden_channels),
            edge_dim=edge_dim,
        )

        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)

        self.dropout = Dropout(p=0.2)
        self.linear = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.linear(x)

        return x


import torch
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN_CPS(torch.nn.Module):
    """
    Final GCN architecture optimized for:
    - Ensemble diversity
    - CPS stability
    - EMA teacher compatibility
    """

    def __init__(self, num_node_features, hidden_channels=128, dropout_p=0.3):
        super().__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.norm3 = LayerNorm(hidden_channels)

        self.dropout = Dropout(dropout_p)

        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.linear(x)
        return x
