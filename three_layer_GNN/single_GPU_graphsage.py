import argparse
import time

import dgl
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
from set_seed import set_seed
import logging
from loadgraph import load_graph


# class SAGE(nn.Module):
#     def __init__(
#         self, in_feats, n_hidden, n_classes, n_layers, activation
#     ):
#         super().__init__()
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_classes = n_classes
#         self.layers = nn.ModuleList()
#         self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
#         for i in range(1, n_layers - 1):
#             self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
#         self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
#         self.activation = activation

#     def forward(self, blocks, x):
#         h = x
#         for l, (layer, block) in enumerate(zip(self.layers, blocks)):
#             # We need to first copy the representation of nodes on the RHS from the
#             # appropriate nodes on the LHS.
#             # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
#             # would be (num_nodes_RHS, D)
#             h_dst = h[: block.num_dst_nodes()]
#             # Then we compute the updated representation on the RHS.
#             # The shape of h now becomes (num_nodes_RHS, D)
#             h = layer(block, (h, h_dst))
#             if l != len(self.layers) - 1:
#                 h = self.activation(h)
#         return h

#     def inference(self, g, x, device):
#         """
#         Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
#         g : the entire graph.
#         x : the input of entire node set.
#         The inference code is written in a fashion that it could handle any number of nodes and
#         layers.
#         """
#         # During inference with sampling, multi-layer blocks are very inefficient because
#         # lots of computations in the first few layers are repeated.
#         # Therefore, we compute the representation of all nodes layer by layer.  The nodes
#         # on each layer are of course splitted in batches.
#         # TODO: can we standardize this?
#         for l, layer in enumerate(self.layers):
#             y = th.zeros(
#                 g.num_nodes(),
#                 self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
#             ).to(device)

#             sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
#             dataloader = dgl.dataloading.DataLoader(
#                 g,
#                 th.arange(g.num_nodes()),
#                 sampler,
#                 batch_size=50000,
#                 shuffle=True,
#                 drop_last=False,
#                 num_workers=args.num_workers,
#             )

#             for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
#                 block = blocks[0].int().to(device)

#                 h = x[input_nodes]
#                 h_dst = h[: block.num_dst_nodes()]
#                 h = layer(block, (h, h_dst))
#                 if l != len(self.layers) - 1:
#                     h = self.activation(h)

#                 y[output_nodes] = h

#             x = y
#         return y


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.norms = nn.ModuleList()
        self.norms.append(nn.LayerNorm(n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.norms.append(nn.LayerNorm(n_hidden))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
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
                num_workers=args.num_workers,
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


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return (
        compute_acc(pred[val_nid], labels[val_nid]),
        compute_acc(pred[test_nid], labels[test_nid]),
        pred,
    )


# def load_subtensor(nfeat, labels, seeds, input_nodes):
#     """
#     Extracts features and labels for a set of nodes.
#     """
#     batch_inputs = nfeat[input_nodes]
#     batch_labels = labels[seeds]
#     return batch_inputs, batch_labels

def load_subtensor(nfeat, labels, seeds, input_nodes, n_classes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels_original = labels[seeds].to(device)
    batch_labels = F.one_hot(batch_labels_original, n_classes).float()
    return batch_inputs, batch_labels, batch_labels_original

#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
    fanout=[int(fanout) for fanout in args.fan_out.split(",")]
    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        fanout
    )
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        device=device,
    )
    set_seed(42)

    # Define model and optimizer
    model = SAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
    )
    nfeat = nfeat.to(device)
    labels = labels.to(device)
    model = model.to(device)
    # loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    best_eval_acc = 0
    best_test_acc = 0
    iteration=0
    total_time = 0
    log_name=f'log.log'
    for epoch in range(args.num_epochs):

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            t1 = time.time()
            num_samples = input_nodes.shape[0] 

            # copy block to gpu
            blocks = [blk.int() for blk in blocks]

            # Load the input features as well as output labels
            # batch_inputs, batch_labels = load_subtensor(
            #     nfeat, labels, seeds, input_nodes
            # )
            batch_inputs, batch_labels_original, batch_labels = load_subtensor(
                nfeat, labels, seeds, input_nodes,n_classes, device
            )

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            # loss = loss_fcn(batch_pred, batch_labels)
            loss = loss_fcn(batch_pred, batch_labels_original)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2=time.time()
            total_time+=t2-t1
            iteration+=1
            logging.basicConfig(filename=log_name, level=logging.INFO)
            logging.info(
                "Epoch {:05d} | iteration {:05d} | Step {:05d} | Loss {:.4f} | Time {:4f}".format(
                    epoch,
                    iteration,
                    step,
                    loss.item(),
                    total_time
                )
            )
            if iteration % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else 0
                )
                eval_acc, test_acc, pred = evaluate(
                    model, g, nfeat, labels, val_nid, test_nid, device
                )
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_test_acc = test_acc
                logging.basicConfig(filename=log_name, level=logging.INFO)
                logging.info(
                    "Epoch {:05d} | iteration {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} |Best Eval Acc {:.4f} |Test Acc {:.4f}|Real Test Acc {:.4f}| GPU {:.1f} MB | Time {:4f}".format(
                        epoch,
                        iteration,
                        step,
                        loss.item(),
                        acc.item(),
                        best_eval_acc,
                        best_test_acc,
                        test_acc,
                        gpu_mem_alloc,
                        total_time
                    )
                )
            
            if iteration>=args.num_iterations:
                break
        if iteration>=args.num_iterations:
            break
    return best_test_acc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num-epochs", type=int, default=1000000)
    argparser.add_argument("--num-iterations", type=int, default=5000)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=3)
    argparser.add_argument("--fan-out", type=str, default="5,10,15")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--raw_dir", type=str, default="raw_dataset")
    argparser.add_argument("--name", type=str, default="ogbn-products")
    argparser.add_argument("--log_name", type=str, default="")
    argparser.add_argument("--log-every", type=int, default=200)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--save-pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device("cuda:%d" % args.gpu)
    else:
        device = th.device("cpu")

    # load ogbn-products data
    if args.name=='ogbn-products':
        data = DglNodePropPredDataset(name="ogbn-products", root="raw_dataset")
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]
        nfeat = graph.ndata.pop("feat")
        labels = labels[:, 0]

        in_feats = nfeat.shape[1]
        n_classes = (labels.max() + 1).item()
    elif args.name=='reddit':
        graph, in_feats, n_classes  = load_graph(args.name, args.raw_dir)
        train_idx = graph.ndata['train_mask'].nonzero().squeeze()
        val_idx = graph.ndata['val_mask'].nonzero().squeeze()
        test_idx = graph.ndata['test_mask'].nonzero().squeeze()
        nfeat = graph.ndata.pop("feat")
        labels = graph.ndata.pop("label")
        graph.ndata.pop('train_mask')
        graph.ndata.pop('val_mask')
        graph.ndata.pop('test_mask')
    elif args.name=='ogbn-arxiv':
        data = DglNodePropPredDataset(name="ogbn-arxiv", root="raw_dataset")
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]
        # add reverse edges
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

        # add self-loop
        graph = graph.remove_self_loop().add_self_loop()
        nfeat = graph.ndata.pop("feat")
        labels = labels[:, 0]

        in_feats = nfeat.shape[1]
        n_classes = (labels.max() + 1).item()
    graph.create_formats_()
    # Pack data
    data = (
        train_idx,
        val_idx,
        test_idx,
        in_feats,
        labels,
        n_classes,
        nfeat,
        graph,
    )

    run(args, device, data)