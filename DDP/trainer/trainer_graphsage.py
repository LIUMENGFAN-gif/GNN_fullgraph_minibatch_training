import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import tqdm
import json
import time
import torch.nn as nn
import torch.nn.functional as F

def set_device(args, proc_id, devices):
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port=args.port
    )
    if torch.cuda.device_count() < 1:
        device = torch.device("cpu")
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id,
        )
    else:
        torch.cuda.set_device(dev_id+args.start_dev)
        device = torch.device("cuda:" + str(dev_id+args.start_dev))
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id,
        )
    return device

def samplers(g, train_nids, valid_nids, args, device):
    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    sampler = dgl.dataloading.NeighborSampler(args.fanouts)
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        g,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=args.batch_size,  # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0,  # Number of sampler processes
    )
    val_sampler = dgl.dataloading.NeighborSampler([-1])
    valid_dataloader = dgl.dataloading.DataLoader(
        g,
        valid_nids,
        val_sampler,
        device=device,
        use_ddp=True,
        batch_size=50000,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_dataloader, valid_dataloader


def evaluation(valid_dataloader, model, proc_id,nfeat,n_classes,val_nid, labels, device):
    model.eval()
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, seeds, blocks in tq:
            # copy block to gpu
            blocks = [blk.int() for blk in blocks]
            # Load the input features as well as output labels
            batch_inputs, batch_labels, batch_labels_original = load_subtensor(
                nfeat, labels, seeds, input_nodes,n_classes, device
            )
            batch_pred = model(blocks, batch_inputs)
            correct_num=(torch.argmax(batch_pred, dim=1) == batch_labels_original).float().sum()
            torch.distributed.all_reduce(correct_num)
            val_acc=correct_num.item()/val_nid.shape[0]

    model.train()

    return val_acc

def load_subtensor(nfeat, labels, seeds, input_nodes, n_classes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels_original = labels[seeds].to(device)
    batch_labels = F.one_hot(batch_labels_original, n_classes).float()
    return batch_inputs, batch_labels, batch_labels_original       



def train_with_evaluation(data, train_dataloader, valid_dataloader, model, opt, proc_id, args, device):
    isStop=False
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
    nfeat = nfeat.to(device)
    labels = labels.to(device)
    loss_fcn = nn.MSELoss()
    time_list=[0]
    t3=0
    itr=0
    # Copied from previous tutorial with changes highlighted.
    for epoch in range(args.num_epochs):
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            t1=time.time()
            # copy block to gpu
            blocks = [blk.int() for blk in blocks]
            # Load the input features as well as output labels
            batch_inputs, batch_labels, batch_labels_original = load_subtensor(
                nfeat, labels, seeds, input_nodes,n_classes, device
            )
            
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t2=time.time()
            t3=t3+t2-t1
            time_list.append(t3)
            if itr%2000==0 and itr!=0:
                val_acc=evaluation(valid_dataloader, model, proc_id,nfeat,n_classes, val_nid, labels, device)
                print(f"itr: {itr}, loss: {loss.item()}, val_acc: {val_acc}, time: {time_list[-1]}")
                if val_acc>args.val_acc_target:
                    isStop=True
                    break
            itr+=1

        if isStop:
            break
    print(f"proc_id: {proc_id}, time: {time_list[-1]}")


def run_graphsage(proc_id, data, model, args, devices):
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
    # Initialize distributed training context.
    device=set_device(args, proc_id, devices)

    train_dataloader, valid_dataloader = samplers(g,train_nid, val_nid, args, device)

    model = model.to(device)
    # Wrap the model with distributed data parallel module.
    if device == torch.device("cpu"):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    

    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    train_with_evaluation(data, train_dataloader, valid_dataloader, model, opt, proc_id, args, device)
    