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
import psutil
import time
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import dgl  # type: ignore
from dgl.heterograph import DGLBlock  # type: ignore
import json
from set_seed import set_seed
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

parser.add_argument(
    "--precompute-batches", action="store_true",
    help="Precompute the batches "
)


parser.add_argument(
    "--optimized-batches-cache",
    type=str,
    default="",
    help="Prefix of the files used to store precomputed batches "
)


parser.add_argument('--train-iters', default=1000000, type=int,
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

parser.add_argument('--batch-size', default=5000, type=int,
                    help='per worker batch size ')

parser.add_argument('--num-workers', default=0, type=int,
                    help='number of dataloader workers ')

parser.add_argument('--fanout', type=str, default='5',
                    help='fanouts for sampling ')

parser.add_argument("--epsilon", type=float, default=0.0225)

parser.add_argument('--max-collective-size', default=0, type=int,
                    help='The maximum allowed size of the data in a collective. \
If a collective would communicate more than this maximum, it is split into multiple collectives.\
Collective calls with large data may cause instabilities in some communication backends  ')


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_layers, activation
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.pytorch.SAGEConv(in_feats, n_classes, "mean"))
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            h = self.activation(h)
        return h


def main():
    # psutil.Process().cpu_affinity([8])
    args = parser.parse_args()
    print('args', args)

    use_gpu = args.cpu_run#torch.cuda.is_available() and not args.cpu_run
    device = torch.device('cpu')#torch.device('cuda') if use_gpu else torch.device('cpu')

    if args.rank == -1:
        # Try to infer the worker's rank from environment variables
        # created by mpirun or similar MPI launchers
        args.rank = int(os.environ.get("PMI_RANK", -1))
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])

    # os.environ.putenv("GOMP_CPU_AFFINITY", "10,11")
    # os.environ.putenv("OMP_NUM_THREADS", "16")
    # Obtain the ip address of the master through the network file system
    master_ip_address = sar.nfs_ip_init(args.rank, args.ip_file)
    sar.initialize_comms(args.rank,
                         args.world_size, master_ip_address,
                         args.backend)
    print("done")

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(
        args.partitioning_json_file, args.rank, torch.device('cpu'))

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
    print(partition_data.node_features.keys())
    # Obtain the number of classes by finding the max label across all workers
    num_labels = labels.max() + 1
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX,
                        move_to_comm_device=True)
    num_labels = num_labels.item()

    features = sar.suffix_key_lookup(
        partition_data.node_features, 'features').to(device)  # keep features always on CPU
    full_graph_manager = sar.construct_full_graph(
        partition_data)  # Keep full graph on CPU

    node_ranges = partition_data.node_ranges
    del partition_data
    print("data loaded")
    set_seed(42)
    gnn_model = SAGE(
        features.size(1),
        num_labels,
        1,
        F.relu,
    )
    print('model', gnn_model)

    # Synchronize the model parmeters across all workers
    sar.sync_params(gnn_model)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)

    neighbor_sampler = sar.core.sampling.DistNeighborSampler([int(fanout) for fanout in args.fanout.split(",")],
                                                             input_node_features={
                                                                 'features': features},
                                                             output_node_features={
                                                                 'labels': labels},
                                                             output_device=device
                                                             )

    train_nodes = masks['train_indices'] + node_ranges[sar.rank()][0]

    dataloader = sar.core.sampling.DataLoader(
        full_graph_manager,
        train_nodes,
        neighbor_sampler,
        args.batch_size,
        shuffle=True,
        precompute_optimized_batches=args.precompute_batches,
        optimized_batches_cache=(
            args.optimized_batches_cache if args.optimized_batches_cache else None),
        num_workers=args.num_workers)


    for k in list(masks.keys()):
        masks[k] = masks[k].to(device)
    loss_fcn = nn.MSELoss(reduction='mean')
    n_train_points = torch.LongTensor([masks['train_indices'].numel()])
    sar.comm.all_reduce(n_train_points, op=dist.ReduceOp.SUM, move_to_comm_device=True)
    n_train_points = n_train_points.item()
    this_points=torch.LongTensor([masks['train_indices'].numel()])
    epoch_continue=True
    itration_num=0
    total_time = 0
    time_list=[0]
    val_acc_list=[0]
    for train_iter_idx in range(args.train_iters):
        gnn_model.train()
        sar.Config.max_collective_size = 0
        for step, blocks in enumerate(dataloader):
            # print('block edata', [block.edata[dgl.EID] for block in blocks])
            blocks = [b.to(device) for b in blocks]
            new_labels=F.one_hot(blocks[-1].dstdata['labels'], num_labels).float()
            batch_pred = gnn_model(blocks, blocks[0].srcdata['features'])
            loss = loss_fcn(batch_pred, new_labels)
            optimizer.zero_grad()
            loss.backward()
            sar.gather_grads(gnn_model)
            optimizer.step()
            sum_loss=loss*this_points[0]
            sar.comm.all_reduce(sum_loss, op=dist.ReduceOp.SUM, move_to_comm_device=True)
            average_loss = sum_loss/n_train_points
            result_message = (
                f"step/iteration [{step}/{train_iter_idx}/{args.train_iters}] | "
            )
            result_message += ', '.join([
                f"average train loss={average_loss:.8f}, "
                f" |"
            ])
            print(result_message, flush=True)
            itration_num+=1
            if average_loss<=args.epsilon:
                epoch_continue=False
                break
        if not epoch_continue:
            break
    with open(f"results/SAGE_loss_bt{args.world_size*args.batch_size}_fo{args.fanout}_ep{args.epsilon}_lr{args.lr}_iteration.json", "w") as f:
        json.dump([itration_num], f)
    print(f"learning rate: {args.lr} iteration_num: {itration_num}")


if __name__ == '__main__':
    main()
