import argparse
import time
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset
from loadgraph import load_graph
from set_seed import set_seed
import time
from models import SageModel, GcnModel, GatModel


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels_original, val_nid,device):
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels_original[val_nid])


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
    nfeat = nfeat.to(device)
    labels = labels.to(device)
    loss_fcn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    iteration_num=0
    eval_acc_list=[]
    epoch_continue=True
    eval_acc= evaluate(model, g, nfeat, labels, val_nid, device)
    eval_acc_list.append(eval_acc.item())
    time_list=[0]
    time_start = time.time()
    for epoch in range(args.num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # copy block to gpu
            blocks = [blk.int() for blk in blocks]
            # Load the input features as well as output labels
            batch_inputs, batch_labels, batch_labels_original = load_subtensor(
                nfeat, labels, seeds, input_nodes,n_classes, device
            )

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration_num%args.log_every==0 and iteration_num!=0:
                time_end = time.time()
                time_list.append(time_end-time_start)
                eval_acc= evaluate(
                            model, g, nfeat, labels, val_nid, device
                        )
                eval_acc_list.append(eval_acc.item())
                print(
                    "Epoch {:05d} | Step {:05d} | iter {:06d}| bt {:03d} | lr {:4f}| fo {:01d} |Loss {:.8f} |Eval acc {:.4f}".format(
                        epoch,
                        step,
                        iteration_num,
                        args.batch_size,
                        args.lr,
                        int(args.fan_out),
                        loss.item(),
                        eval_acc.item()
                    )
                    )
                time_end2=time.time()
                time_start=time_start-(time_end2-time_end)
                if eval_acc.item()>=args.eval_acc_threshold:
                    epoch_continue=False
                    break
            iteration_num+=1
        if not epoch_continue:
            break
    print(f"name: {args.name} time: {time_list[-1]} acc: {eval_acc_list[-1]}")
            

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num-epochs", type=int, default=100000)
    argparser.add_argument("--model", type=str, default="sage")
    argparser.add_argument("--fan-out", type=str, default="5")
    argparser.add_argument("--batch-size", type=int, default=500)
    argparser.add_argument("--lr", type=float, default=0.025)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--split_ratio", type=float, default=0)
    argparser.add_argument("--name", type=str, default="reddit")
    argparser.add_argument("--eval_acc_threshold", type=float, default=0.6)
    argparser.add_argument("--raw_dir", type=str, default="dataset_dir")
    argparser.add_argument("--log-every", type=int, default=20)
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
        graph=graph.add_self_loop()
        nfeat = graph.ndata.pop("feat")
        labels = labels[:, 0]

        in_feats = nfeat.shape[1]
        n_classes = (labels.max() + 1).item()
    elif args.name=='ogbn-arxiv':
        data = DglNodePropPredDataset(name="ogbn-arxiv", root=args.raw_dir)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]
        graph=graph.add_self_loop()
        nfeat = graph.ndata.pop("feat")
        labels = labels[:, 0]

        in_feats = nfeat.shape[1]
        n_classes = (labels.max() + 1).item()
    
    graph.create_formats_()

    if args.split_ratio>0:
        print(f"before split: train_nids: {train_idx.shape[0]}, test_nids: {test_idx.shape[0]}")
        test_part_idx=int(test_idx.shape[0]*args.split_ratio)
        new_test_nids = test_idx[test_part_idx:].clone()
        new_train_nids = th.cat([train_idx, test_idx[:test_part_idx].clone()])
        train_idx = new_train_nids
        test_idx = new_test_nids
        print(f"after split: train_nids: {new_train_nids.shape[0]}, test_nids: {new_test_nids.shape[0]}")

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
    # # # Run 10 times
    run(args, device, data)