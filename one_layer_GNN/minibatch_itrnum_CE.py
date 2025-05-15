import argparse
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset
from loadgraph import load_graph
from models import SageModel, GcnModel, GatModel
from set_seed import set_seed

def load_subtensor(nfeat, labels, seeds, input_nodes, n_classes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds]
    batch_labels = F.one_hot(batch_labels, n_classes).float().to(device)
    return batch_inputs, batch_labels


#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
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
    if args.model == "sage":
        # Define model and optimizer
        model = SageModel(
            in_feats,
            n_classes,
            args.num_layers,
            F.relu,
        )
    elif args.model == "gcn":
        model = GcnModel(
            in_feats,
            n_classes,
            args.num_layers,
            F.relu,
        )
    elif args.model == "gat":
        model = GatModel(
            in_feats,
            n_classes,
            args.num_layers,
            F.elu,
        )
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    iteration_num=0
    epoch_continue=True
    nfeat=nfeat.to(device)
    labels=labels.to(device)
    for epoch in range(args.num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # copy block to gpu
            blocks = [blk.int() for blk in blocks]
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(
                nfeat, labels, seeds, input_nodes,n_classes, device
            )

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = F.binary_cross_entropy_with_logits(batch_pred, (batch_labels + 1) / 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration_num+=1
            print(
                "Epoch {:05d} | Step {:05d} | iter {:06d}| bt {:03d} | lr {:4f}| fo {:01d} |Loss {:.4f}".format(
                    epoch,
                    step,
                    iteration_num,
                    args.batch_size,
                    args.lr,
                    int(args.fan_out),
                    loss.item(),
                )
                )
            if loss.item()<=args.epsilon:
                epoch_continue=False
                break
        if not epoch_continue:
            break
    print(f"learning rate: {args.lr} iteration_num: {iteration_num}")
            

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--model", type=str, default="sage")
    argparser.add_argument("--num-epochs", type=int, default=100000)
    argparser.add_argument("--fan-out", type=str, default="5")
    argparser.add_argument("--batch-size", type=int, default=5000)
    argparser.add_argument("--lr", type=float, default=0.015)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--split_ratio", type=float, default=0)
    argparser.add_argument("--name", type=str, default="ogbn-arxiv")
    argparser.add_argument("--epsilon", type=float, default=0.025)
    argparser.add_argument("--raw_dir", type=str, default="dataset_dir")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device("cuda:%d" % args.gpu)
    else:
        device = th.device("cpu")

    if args.name=='reddit':
        graph, in_feats, n_classes  = load_graph(args.name, args.raw_dir)
        train_idx = graph.ndata['train_mask'].nonzero().squeeze()
        val_idx = graph.ndata['val_mask'].nonzero().squeeze()
        test_idx = graph.ndata['test_mask'].nonzero().squeeze()
        nfeat = graph.ndata.pop("feat")
        labels = graph.ndata.pop("label")
        graph.ndata.pop('train_mask')
        graph.ndata.pop('val_mask')
        graph.ndata.pop('test_mask')
    elif args.name=='ogbn-products':
        data = DglNodePropPredDataset(name="ogbn-products", root=args.raw_dir)
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
    elif args.name=='ogbn-arxiv':
        data = DglNodePropPredDataset(name="ogbn-arxiv", root="/nfs/mfliu/datasets")
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

        # add self-loop
        graph = graph.remove_self_loop().add_self_loop()
        nfeat = graph.ndata.pop("feat")
        labels = labels[:, 0]

        in_feats = nfeat.shape[1]
        n_classes = (labels.max() + 1).item()
    graph.create_formats_()
    labels = th.where(labels < int(n_classes / 2), -th.ones_like(labels), th.ones_like(labels))
    
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
    print("done")
    # # Run 10 times
    run(args, device, data)