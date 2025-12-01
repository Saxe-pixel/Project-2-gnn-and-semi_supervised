import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINEConv, JumpingKnowledge
from torch.nn import BatchNorm1d, Dropout, Sequential, ReLU, Linear, ModuleList
from typing import List


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)

        # Regularization
        self.dropout = Dropout(p=0.2)

        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Pool + readout
        x = global_mean_pool(x, batch)
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


class OptimizedGINEGNN(torch.nn.Module):
    """
    High-capacity GNN tailored for QM9-style molecular regression.

    Design choices:
    - GINEConv with learned edge embeddings (bond features).
    - Deep stack of message passing layers with residual connections.
    - Jumping Knowledge over all layers for stable deep training.
    - Two-stage MLP readout for strong expressiveness.
    """
    def __init__(
        self,
        num_node_features: int,
        edge_dim: int,
        hidden_channels: int = 256,
        num_layers: int = 6,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Encode raw node and edge features into a common hidden space.
        self.node_encoder = Linear(num_node_features, hidden_channels)
        self.edge_encoder = Linear(edge_dim, hidden_channels)

        # GINEConv stack with residual connections.
        self.convs = ModuleList()
        self.norms = ModuleList()

        def make_mlp() -> Sequential:
            return Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )

        for _ in range(num_layers):
            self.convs.append(GINEConv(nn=make_mlp(), edge_dim=hidden_channels))
            self.norms.append(BatchNorm1d(hidden_channels))

        # Jumping Knowledge over all layers (concatenate).
        self.jk = JumpingKnowledge(mode="cat")

        jk_out_dim = hidden_channels * num_layers

        # Two-layer MLP readout in graph space.
        self.readout = Sequential(
            Linear(jk_out_dim, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(p=dropout_rate),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, 1),
        )

        self.dropout = Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        # Encode features.
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        layer_outputs: List[torch.Tensor] = []

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection
            layer_outputs.append(x)

        # Aggregate over layers (Jumping Knowledge) and graphs (global pooling).
        x_jk = self.jk(layer_outputs)
        x_graph = global_mean_pool(x_jk, batch)

        out = self.readout(x_graph)
        return out
