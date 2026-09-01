# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """Three-layer GraphSAGE with LayerNorm.

    LayerNorm normalizes per node across the feature dimension — no running
    statistics, no train/eval difference, and correct behaviour at batch size 1.
    This makes ONNX export straightforward and single-transaction inference
    identical to batch inference.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.LayerNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)


class FraudGNN(nn.Module):
    """GraphSAGE encoder + MLP classifier head."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        embed_dim: int = 64,
        head_hidden: int = 48,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden, embed_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, x, edge_index):
        emb = self.encoder(x, edge_index)
        return self.classifier(emb), emb
