import os
os.environ["DGLBACKEND"] = "pytorch"
from models import SAGE_one_layer as SAGE
from loadgraph import load_graph
import argparse
from trainer import run_graphsage
from ogb.nodeproppred import DglNodePropPredDataset
import torch.nn.functional as F
import torch
from set_seed import set_seed
# Say you have four GPUs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--fanouts", type=str, default="5")
    parser.add_argument("--num_epochs", type=int, default=100000000)
    parser.add_argument("--name", type=str, default="ogbn-products")
    parser.add_argument("--raw_dir", type=str, default="datasets_dir")
    parser.add_argument("--start_dev", type=int, default=0)
    parser.add_argument("--port", type=str, default="12345")
    parser.add_argument("--val_acc_target", type=float, default=0.5)
    parser.add_argument("--replica_num", type=int, default=1)
    args = parser.parse_args()
    args.fanouts = [int(x) for x in args.fanouts.split(',')]
    if args.name=='ogbn-products':
        data = DglNodePropPredDataset(name="ogbn-products", root="/nfs/mfliu/datasets")
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
    graph.create_formats_()
    
    if args.replica_num>0:
        print(f"before split: train_nids: {train_idx.shape[0]}")
        train_num=train_idx.shape[0]*args.replica_num
        new_train_nids = torch.cat([train_idx, test_idx[:train_num].clone()])
        train_idx = new_train_nids
        print(f"after split: train_nids: {new_train_nids.shape[0]}")

    num_gpus = args.num_gpus
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
    set_seed(42)

    model = model = SAGE(in_feats,n_classes,args.num_layers,F.relu,)
    import torch.multiprocessing as mp
    mp.spawn(run_graphsage, args=(data, model, args, list(range(num_gpus)),), nprocs=num_gpus)