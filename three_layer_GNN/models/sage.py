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


class SageModelfull(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, drop_rate=0):
        super(SageModelfull, self).__init__()
        self.sages = nn.ModuleList()
        self.sages.append(SAGEConv(in_feats, h_feats, aggregator_type="mean"))
        self.norms = nn.ModuleList()
        self.norms.append(nn.LayerNorm(h_feats))
        for _ in range(num_layers - 2):
            self.sages.append(SAGEConv(h_feats, h_feats, aggregator_type="mean"))
            self.norms.append(nn.LayerNorm(h_feats))
        self.sages.append(SAGEConv(h_feats, num_classes, aggregator_type="mean"))
        self.drop_rate = drop_rate

    def forward(self, g, h):
        for i, sages in enumerate(self.sages[:-1]):
            h = sages(g, h)
            h = F.dropout(h, p=self.drop_rate, training=self.training)
            h = self.norms[i](h)
            h = F.relu(h)
        h = self.sages[-1](g, h)
        return h

class SAGE(nn.Module):
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


class SAGE_three_layers(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norms.append(nn.LayerNorm(n_hidden))
        self.layers.append(SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, "mean"))
            self.norms.append(nn.LayerNorm(n_hidden))
        self.layers.append(SAGEConv(n_hidden, n_classes, "mean"))
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.norms[l](h)
                h = self.activation(h)
                
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
            ).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler,
                batch_size=50000,
                shuffle=True,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes]
                h_dst = h[: block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.norms[l](h)
                    h = self.activation(h)
                    

                y[output_nodes] = h

            x = y
        return y