import os
os.environ["DGLBACKEND"] = "pytorch"
from models import SageModel, GcnModel
from loadgraph import load_graph
import argparse
from trainer import run
import json
import torch
# Say you have four GPUs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_feats", type=int, default=512)
    parser.add_argument("--fanouts", type=str, default="5,10,15")
    parser.add_argument("--num_epochs", type=int, default=10000000)
    parser.add_argument("--name", type=str, default="ogbn-products")
    parser.add_argument("--raw_dir", type=str, default="datasets_dir")
    parser.add_argument("--start_dev", type=int, default=0) #from zero
    parser.add_argument("--port", type=str, default="12345")
    parser.add_argument("--val_acc_target", type=float, default=0.95)
    args = parser.parse_args()
    args.fanouts = [int(x) for x in args.fanouts.split(',')]
    g, num_features, num_classes  = load_graph(args.name, args.raw_dir)

    num_gpus = args.num_gpus
    g.create_formats_()

    model = SageModel(num_features, args.hidden_feats, num_classes, args.num_layers, args.drop_rate)
    import torch.multiprocessing as mp
    mp.spawn(run, args=(g, model, args, list(range(num_gpus)),), nprocs=num_gpus)