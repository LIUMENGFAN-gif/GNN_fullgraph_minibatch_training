import os
os.environ["DGLBACKEND"] = "pytorch"
from loadgraph import load_graph
import argparse
import dgl
import json

# Say you have four GPUs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ogbn-papers100M")
    parser.add_argument("--raw_dir", type=str, default="/nfs/mfliu/datasets")
    parser.add_argument("--num_parts", type=int, default=4)
    args = parser.parse_args()
    g, num_features, num_classes  = load_graph(args.name, args.raw_dir)
    print(args.name)
    dgl.distributed.partition_graph(g, graph_name=args.name, num_parts=args.num_parts,
                                out_path=f'/nfs/mfliu/partdata/{args.name}/nonbidirected_{args.num_parts}part_data',
                                balance_ntypes=g.ndata['train_mask'],
                                balance_edges=True,
                                return_mapping=True)
    with open(f'/nfs/mfliu/partdata/{args.name}/info.json', "w") as f:
        json.dump({"num_features": int(num_features), "num_classes": int(num_classes)}, f)