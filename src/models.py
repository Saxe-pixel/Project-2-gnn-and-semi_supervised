import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINEConv
from torch.nn import BatchNorm1d, Dropout, Sequential, ReLU, Linear


class GCN_original(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN_original, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GCN must have at least 2 layers")

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # first layer input -> hidden
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # hidden layers 
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        self.dropout = Dropout(p=dropout)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # stacked GCN blocks
        num_blocks = len(self.convs)
        for idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if idx < num_blocks - 1:
                x = self.dropout(x)

        # Pool and readout
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
            raise ValueError("GINEGCN expects data.edge_attr to be present.")

        # construct conv layers once we know edge_attr dimension
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
