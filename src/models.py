import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINEConv
from torch.nn import BatchNorm1d, Dropout, Sequential, ReLU, Linear


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
    def __init__(self, num_node_features, hidden_channels=128):
        super().__init__()
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None

        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)

        self.dropout = Dropout(p=0.2)
        self.linear = Linear(hidden_channels, 1)

    def _build_convs(self, edge_dim: int, device):
        """Build GINEConv layers once we know edge_attr dimensionality."""
        def mlp(in_ch, out_ch):
            return Sequential(
                Linear(in_ch, self.hidden_channels),
                ReLU(),
                Linear(self.hidden_channels, out_ch),
            )

        self.conv1 = GINEConv(
            nn=mlp(self.num_node_features, self.hidden_channels),
            edge_dim=edge_dim,
        ).to(device)
        self.conv2 = GINEConv(
            nn=mlp(self.hidden_channels, self.hidden_channels),
            edge_dim=edge_dim,
        ).to(device)
        self.conv3 = GINEConv(
            nn=mlp(self.hidden_channels, self.hidden_channels),
            edge_dim=edge_dim,
        ).to(device)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        if edge_attr is None:
            raise ValueError(
                "GINEGCN expects `data.edge_attr` to be present (edge features), "
                "but got None."
            )

        # Lazily construct conv layers once we know edge_attr dimension
        if self.conv1 is None:
            edge_dim = edge_attr.size(-1)
            self._build_convs(edge_dim, device=x.device)

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
