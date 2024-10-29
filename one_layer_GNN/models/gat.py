import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl
import tqdm


class GatModel(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_layers,activation,num_heads=4,feat_drop=0,attn_drop=0, negative_slope=0.2
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, n_classes, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False,allow_zero_in_degree=True))
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            h = h.mean(1)
            h = self.activation(h)
        return h
    
    def inference(self, g, x, device):
        for l, layer in enumerate(self.layers):
            y = th.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
            ).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=50000,
                shuffle=True,
                drop_last=False,
                num_workers=4,
                device=device,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int()
                h = x[input_nodes]
                h_dst = h[: block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                h = h.mean(1)
                h = self.activation(h)
                y[output_nodes] = h

            x = y
        return y


class GAT_full(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_layers,activation,num_heads=4,feat_drop=0,attn_drop=0, negative_slope=0.2
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, n_classes, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False,allow_zero_in_degree=True))
        self.activation = activation

    def forward(self, graph, x):
        h = x
        h = self.layers[0](graph, h) 
        h = h.mean(1)
        h = self.activation(h)
        return h