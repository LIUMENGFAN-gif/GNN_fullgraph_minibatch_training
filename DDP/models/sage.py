import torch.nn as nn
from dgl.nn import SAGEConv
import torch.nn.functional as F
import dgl
import torch
import tqdm

class SageModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, drop_rate=0):
        super(SageModel, self).__init__()
        self.sages = nn.ModuleList()
        self.sages.append(SAGEConv(in_feats, h_feats, aggregator_type="mean"))
        self.norms = nn.ModuleList()
        self.norms.append(nn.LayerNorm(h_feats))
        for _ in range(num_layers - 2):
            self.sages.append(SAGEConv(h_feats, h_feats, aggregator_type="mean"))
            self.norms.append(nn.LayerNorm(h_feats))
        self.sages.append(SAGEConv(h_feats, num_classes, aggregator_type="mean"))
        self.drop_rate = drop_rate

    def forward(self, mfgs, h):
        for i, sages in enumerate(self.sages[:-1]):
            h_dst = h[: mfgs[i].num_dst_nodes()]
            h = sages(mfgs[i], (h, h_dst))
            h = F.dropout(h, p=self.drop_rate, training=self.training)
            h = self.norms[i](h)
            h = F.relu(h)
        h_dst = h[: mfgs[-1].num_dst_nodes()]
        h = self.sages[-1](mfgs[-1], (h, h_dst))
        return h
    
class SAGE_one_layer(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_layers, activation
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_classes, "mean"))
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            h = self.activation(h)
        return h
