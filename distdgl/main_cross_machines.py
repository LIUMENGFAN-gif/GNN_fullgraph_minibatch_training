import os
os.environ["DGLBACKEND"] = "pytorch"
from models import SageModel
import argparse
from trainer import run_across_machines as run
import json
import dgl
import torch as th
import os
import socket
import sys
import numpy as np
import random


def set_comm():
    host_name = socket.gethostname()
    print(f"{host_name}: Initializing DistDGL.")
    dgl.distributed.initialize(ip_config=sys.path[0]+'/ip_config.txt')
    print("done")
    print("start init process group")
    th.distributed.init_process_group(
        backend="gloo",
        init_method="env://",
    )
    print("initialized")

def set_graph(args, part_config_path, info_path):
    g = dgl.distributed.DistGraph(graph_name=args.name, part_config=part_config_path)
    print("graph downloaded")
    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
    )
    val_nid = dgl.distributed.node_split(
        g.ndata["val_mask"], pb, force_even=True
    )
    test_nid = dgl.distributed.node_split(
        g.ndata["test_mask"], pb, force_even=True
    )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    num_train_local = len(np.intersect1d(train_nid.numpy(), local_nid))
    num_val_local = len(np.intersect1d(val_nid.numpy(), local_nid))
    num_test_local = len(np.intersect1d(test_nid.numpy(), local_nid))
    print(
        f"part {g.rank()}, train: {len(train_nid)} (local: {num_train_local}), "
        f"val: {len(val_nid)} (local: {num_val_local}), "
        f"test: {len(test_nid)} (local: {num_test_local})"
    )

    del local_nid

    with open(info_path, "r") as f:
        info = json.load(f)
    num_features = info["num_features"]
    num_classes = info["num_classes"]

    return g, train_nid, val_nid, test_nid, num_features, num_classes

def set_device(args):
    if args.num_gpus == 0:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id+args.start_dev))
    return device

# Say you have four GPUs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--drop_rate", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_feats", type=int, default=1024)
    parser.add_argument("--fanouts", type=str, default="5,10,15")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--name", type=str, default="ogbn-products")
    parser.add_argument("--raw_dir", type=str, default="dir")
    parser.add_argument("--model_dir", type=str, default="dir")
    parser.add_argument("--start_dev", type=int, default=0)
    parser.add_argument("--not_time_record", type=bool, default=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_parts", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--isClient", type=bool, default=False)
    parser.add_argument("--num_gpus", type=int, default=1)
    seed=0
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)

    args = parser.parse_args()
    args.fanouts = [int(x) for x in args.fanouts.split(',')]
    part_config_path=f'{args.name}/bidirected_{args.num_parts}part_data/{args.name}.json'
    info_path=f'{args.name}/info.json'

    set_comm()

    g, train_nid, val_nid, test_nid, num_features, num_classes=set_graph(args, part_config_path, info_path)

    device=set_device(args)
    print(f"device: {device}")
    

    # Pack data.
    data = train_nid, val_nid, test_nid, g

    # best_hyperparam_path = f"/nfs/mfliu/besthyper/{args.name}_best_hyperparams_{args.batch_size}.json"
    # if os.path.exists(best_hyperparam_path):
    #     with open(best_hyperparam_path, "r") as f:
    #         best_hyperparameters = json.load(f)
    #     args.lr = best_hyperparameters["lr"]
    #     args.drop_rate = best_hyperparameters["drop_rate"]
    #     args.weight_decay = best_hyperparameters["weight_decay"]
    #     args.fanouts = best_hyperparameters["fanouts"]
    #     print("Use best hyperparams")
    
    model = SageModel(num_features, args.hidden_feats, num_classes, args.num_layers, args.drop_rate)
    print("model done")

    run(args, model, data, device)