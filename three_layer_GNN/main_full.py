import os
os.environ["DGLBACKEND"] = "pytorch"
from models import SageModelfull,GCNModelfull
from loadgraph import load_graph
import argparse
from trainer import run_full

# Say you have four GPUs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_feats", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100000)
    parser.add_argument("--name", type=str, default="reddit")
    parser.add_argument("--raw_dir", type=str, default="dataset_dir")
    parser.add_argument("--start_dev", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_acc_target", type=float, default=0.95)
    args = parser.parse_args()
    g, num_features, num_classes  = load_graph(args.name, args.raw_dir)
    g.create_formats_()

    
    model = SageModelfull(num_features, args.hidden_feats, num_classes, args.num_layers, args.drop_rate)
    run_full(g, model, args)