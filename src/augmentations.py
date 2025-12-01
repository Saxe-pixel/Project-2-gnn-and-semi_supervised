import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import random


def drop_edges(edge_index, edge_attr=None, drop_prob=0.05):
    """Drops a percentage of edges uniformly and keeps edge_attr in sync."""
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_prob

    edge_index = edge_index[:, keep_mask]

    if edge_attr is not None:
        # Ensure edge_attr matches the kept edges
        edge_attr = edge_attr[keep_mask]

    return edge_index, edge_attr


def mask_node_features(x, mask_prob=0.05):
    """Randomly mask some node features."""
    mask = (torch.rand_like(x) < mask_prob).float()
    return x * (1 - mask)


def coordinate_noise(pos, noise_std=0.05):
    """Jitter 3D coordinates slightly."""
    return pos + torch.randn_like(pos) * noise_std


def augment_batch(data, drop_edge_p=0.05, mask_feature_p=0.05, noise_std=0.05):
    """Graph augmentation compatible with PyTorch Geometric Batch."""
    data = data.clone()

    # 1. Node feature masking
    data.x = mask_node_features(data.x, mask_feature_p)

    # 2. Edge dropout (keep edge_attr aligned with edge_index)
    edge_attr = getattr(data, "edge_attr", None)
    data.edge_index, edge_attr = drop_edges(data.edge_index, edge_attr, drop_edge_p)
    if edge_attr is not None:
        data.edge_attr = edge_attr

    # 3. Coordinate Jitter (if present)
    if hasattr(data, 'pos') and data.pos is not None:
        data.pos = coordinate_noise(data.pos, noise_std)

    return data
