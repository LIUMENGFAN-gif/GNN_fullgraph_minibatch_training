import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import tqdm
import json
import time
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
            backend="gloo",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id,
        )
    return device

def samplers(g, args, device):
    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    train_nids = g.ndata['train_mask'].nonzero().squeeze()
    valid_nids = g.ndata['val_mask'].nonzero().squeeze()
    test_nids = g.ndata['test_mask'].nonzero().squeeze()
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
    val_sampler = dgl.dataloading.NeighborSampler([-1,-1,-1])
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
    test_sampler = dgl.dataloading.NeighborSampler([-1,-1,-1])
    test_dataloader = dgl.dataloading.DataLoader(
        g,
        test_nids,
        test_sampler,
        device=device,
        use_ddp=True,
        batch_size=50000,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_dataloader, valid_dataloader,test_dataloader

def evaluation(valid_dataloader, model, epoch, proc_id, args, device):
    model.eval()
    # Evaluate on only the first GPU.
    predictions = []
    labels = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata["feat"]
            labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
            predictions.append(
                model(mfgs, inputs).argmax(1).cpu().numpy()
            )
            # Clear unimportant cache
            torch.cuda.empty_cache()
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        #average accuracy across all GPUs
        accuracy_tensor = torch.tensor(accuracy).to(device)
        torch.distributed.all_reduce(accuracy_tensor)
        average_accuracy = accuracy_tensor.item() / torch.distributed.get_world_size()
        if proc_id==0:
            print(f'Epoch {epoch} batch size: {args.batch_size} Average accuracy: {average_accuracy}')
        
            
    return average_accuracy
       

def train_with_evaluation(train_dataloader, valid_dataloader, model, opt, best_accuracy, proc_id, args, device):
    total_time = 0
    itr=0
    is_stop=False
    # Copied from previous tutorial with changes highlighted.
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                t1 = time.time()
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]
                predictions = model(mfgs, inputs)
                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                t2 = time.time()
                total_time += t2 - t1
                torch.distributed.all_reduce(loss)
                loss=loss/args.num_gpus
                tq.set_postfix(
                    {"loss": "%.03f" % loss.item()},
                    refresh=False,
                )

                val_acc=evaluation(valid_dataloader, model, epoch, proc_id, args, device)
                print(f"itr:{itr},loss:{loss}, Validation accuracy: {val_acc}, total_time: {total_time}")
                itr+=1
                if val_acc>=args.val_acc_target:
                    is_stop=True
                    break
            if is_stop:
                break
        

def run(proc_id, g, model, args, devices):
    # Initialize distributed training context.
    device=set_device(args, proc_id, devices)

    train_dataloader, valid_dataloader, test_dataloader = samplers(g, args, device)

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
    opt = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    best_accuracy = 0
    

    train_with_evaluation(train_dataloader, valid_dataloader, model, opt, best_accuracy, proc_id, args, device)
    #test_dataloader
    # train_with_evaluation(train_dataloader, test_dataloader, model, opt, best_accuracy, proc_id, args, device)