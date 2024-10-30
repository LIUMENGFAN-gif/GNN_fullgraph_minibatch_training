# Copyright (c) 2022 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Union, Dict
from argparse import ArgumentParser
import os
import logging
import time
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import dgl  # type: ignore
import json
import sar


parser = ArgumentParser(
    description="GNN training on node classification tasks in homogeneous graphs")


parser.add_argument(
    "--partitioning-json-file",
    type=str,
    default="",
    help="Path to the .json file containing partitioning information "
)

parser.add_argument('--ip-file', default='./ip_file', type=str,
                    help='File with ip-address. Worker 0 creates this file and all others read it ')


parser.add_argument('--backend', default='gloo', type=str, choices=['ccl', 'nccl', 'mpi', 'gloo'],
                    help='Communication backend to use '
                    )

parser.add_argument(
    "--cpu-run", action="store_true",
    help="Run on CPUs if set, otherwise run on GPUs "
)


parser.add_argument('--train-iters', default=100, type=int,
                    help='number of training iterations ')

parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
    help="learning rate"
)


parser.add_argument('--rank', default=0, type=int,
                    help='Rank of the current worker ')

parser.add_argument('--world-size', default=2, type=int,
                    help='Number of workers ')

parser.add_argument('--hidden-layer-dim', default=256, type=int,
                    help='Dimension of GNN hidden layer')


class GCN(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_layers, activation
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.pytorch.GraphConv(in_feats, n_classes))
        self.activation = activation

    def forward(self, graph, x):
        h = x
        h = self.layers[0](graph, h) 
        h = self.activation(h)
        return h


def main():
    args = parser.parse_args()
    print('args', args)

    use_gpu = args.cpu_run#torch.cuda.is_available() and not args.cpu_run
    device = 'cpu'#torch.device('cuda' if use_gpu else 'cpu')
    print("device", device)
    # Obtain the ip address of the master through the network file system
    master_ip_address = sar.nfs_ip_init(args.rank, args.ip_file)
    print('master_ip_address', master_ip_address)
    sar.initialize_comms(args.rank,
                         args.world_size,
                         master_ip_address,
                         args.backend)

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(
        args.partitioning_json_file, args.rank, device)

    # Obtain train,validation, and test masks
    # These are stored as node features. Partitioning may prepend
    # the node type to the mask names. So we use the convenience function
    # suffix_key_lookup to look up the mask name while ignoring the
    # arbitrary node type
    masks = {}
    for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                                       ['train_indices', 'val_indices', 'test_indices']):
        boolean_mask = sar.suffix_key_lookup(partition_data.node_features,
                                             mask_name)
        masks[indices_name] = boolean_mask.nonzero(
            as_tuple=False).view(-1).to(device)

    labels = sar.suffix_key_lookup(partition_data.node_features,
                                   'labels').long().to(device)

    # Obtain the number of classes by finding the max label across all workers
    print(partition_data.node_features.keys())
    num_labels = labels.max() + 1
    print("1 num_labels", num_labels)
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True)
    print("2 num_labels", num_labels)
    num_labels = num_labels.item() 
    
    features = sar.suffix_key_lookup(partition_data.node_features, 'features').to(device)
    full_graph_manager = sar.construct_full_graph(partition_data).to(device)

    #We do not need the partition data anymore
    del partition_data
    print("features", features.size())
    gnn_model = GCN(
        features.size(1),
        num_labels,
        1,
        F.relu,
    )
    print('model', gnn_model)

    # Synchronize the model parmeters across all workers
    sar.sync_params(gnn_model)

    # Obtain the number of labeled nodes in the training
    # This will be needed to properly obtain a cross entropy loss
    # normalized by the number of training examples
    n_train_points = torch.LongTensor([masks['train_indices'].numel()])
    sar.comm.all_reduce(n_train_points, op=dist.ReduceOp.SUM, move_to_comm_device=True)
    n_train_points = n_train_points.item()
    loss_fcn = nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)
    best_val_acc = 0
    best_test_acc=0
    for train_iter_idx in range(args.train_iters):
        # Train
        logits = gnn_model(full_graph_manager, features)
        new_labels = F.one_hot(labels[masks['train_indices']], num_labels).float().to(device)
        loss = loss_fcn(logits[masks['train_indices']], new_labels)/n_train_points

        optimizer.zero_grad()
        loss.backward()
        # Do not forget to gather the parameter gradients from all workers
        sar.gather_grads(gnn_model)
        optimizer.step()

        # Calculate accuracy for train/validation/test
        results = []
        for indices_name in ['train_indices', 'val_indices', 'test_indices']:
            n_correct = (logits[masks[indices_name]].argmax(1) ==
                         labels[masks[indices_name]]).float().sum()
            results.extend([n_correct, masks[indices_name].numel()])

        acc_vec = torch.FloatTensor(results)
        # Sum the n_correct, and number of mask elements across all workers
        sar.comm.all_reduce(acc_vec, op=dist.ReduceOp.SUM, move_to_comm_device=True)
        (train_acc, val_acc, test_acc) =  \
            (acc_vec[0] / acc_vec[1],
             acc_vec[2] / acc_vec[3],
             acc_vec[4] / acc_vec[5])
        if val_acc>best_val_acc:
            best_val_acc=val_acc.item()
            best_test_acc=test_acc.item()
        result_message = (
            f"iteration [{train_iter_idx}/{args.train_iters}] | "
        )
        result_message += ', '.join([
            f"train loss={loss:.4f}, "
            f"Accuracy: "
            f"train={train_acc:.4f} "
            f"valid={val_acc:.4f} "
            f"test={test_acc:.4f} "
            f"best test acc={best_test_acc:.4f} "
            f" |"
        ])
        print(result_message, flush=True)
        
    with open(f"results/GCN_testacc_full_lr{args.lr}_iteration.json", "w") as f:
        json.dump([best_test_acc], f)
    print(f"learning rate: {args.lr} best test acc: {best_test_acc} ")
    


if __name__ == '__main__':
    main()
