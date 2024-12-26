import argparse
import time
import dgl.nn.pytorch as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset
from set_seed import set_seed
import logging
from loadgraph import load_graph

class SageModelfull(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers,dropout=0.5):
        super(SageModelfull, self).__init__()
        self.sages = nn.ModuleList()
        self.sages.append(dglnn.SAGEConv(in_feats, h_feats, aggregator_type="mean"))
        self.norms = nn.ModuleList()
        self.norms.append(nn.LayerNorm(h_feats))
        for _ in range(num_layers - 2):
            self.sages.append(dglnn.SAGEConv(h_feats, h_feats, aggregator_type="mean"))
            self.norms.append(nn.LayerNorm(h_feats))
        self.sages.append(dglnn.SAGEConv(h_feats, num_classes, aggregator_type="mean"))
        self.dropout = dropout

    def forward(self, g, h):
        for i, sages in enumerate(self.sages[:-1]):
            h = sages(g, h)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.sages[-1](g, h)
        return h


@th.no_grad()
def test(model, g, nfeat, eval_nid,test_nid, labels):
    model.eval()

    logits = model(g, nfeat)
    eval_acc=compute_acc(logits[eval_nid], labels[eval_nid])
    test_acc=compute_acc(logits[test_nid], labels[test_nid])

    return eval_acc, test_acc

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
    g=g.to(device)
    nfeat=nfeat.to(device)
    labels=labels.to(device)
    set_seed(42)

    # Define model and optimizer
    model = SageModelfull(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        args.drop_out,
    )
    
    model = model.to(device)
    # loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = nn.MSELoss()
    new_labels = F.one_hot(labels, n_classes).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    best_eval_acc = 0
    best_test_acc = 0
    total_time = 0
    log_name=f'log.log'
    for epoch in range(args.num_epochs):
        t1 = time.time()
        model.train()
        logits = model(g, nfeat)
        #loss = loss_fcn(logits[train_nid], labels[train_nid])
        loss = loss_fcn(logits[train_nid], new_labels[train_nid])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2=time.time()
        total_time+=t2-t1
        eval_acc, test_acc=test(model, g, nfeat, val_nid,test_nid, labels)
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_test_acc = test_acc
        gpu_mem_alloc = (
                th.cuda.max_memory_allocated() / 1000000
                if th.cuda.is_available()
                else 0
            )
        logging.basicConfig(filename=log_name, level=logging.INFO)
        logging.info(
            "Epoch {:05d} | Loss {:.4f} |Best Eval Acc {:.4f} |Test Acc {:.4f}|Real Test Acc {:.4f}| GPU {:.1f} MB | Time {:4f}".format(
                epoch,
                loss.item(),
                best_eval_acc,
                best_test_acc,
                test_acc,
                gpu_mem_alloc,
                total_time
            )
        )
    return best_test_acc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num-epochs", type=int, default=2000)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=3)
    argparser.add_argument("--raw_dir", type=str, default="raw_dataset")
    argparser.add_argument("--name", type=str, default="ogbn-products")
    argparser.add_argument("--log_name", type=str, default="")
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--drop-out", type=float, default=0)
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