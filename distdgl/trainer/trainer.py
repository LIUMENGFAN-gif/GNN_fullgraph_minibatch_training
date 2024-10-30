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
            backend="nccl",
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
    val_sampler = dgl.dataloading.NeighborSampler(args.fanouts)
    valid_dataloader = dgl.dataloading.DataLoader(
        g,
        valid_nids,
        val_sampler,
        device=device,
        use_ddp=True,
        batch_size=10000,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_dataloader, valid_dataloader

def evaluation(epochs_no_improve, valid_dataloader, model, best_accuracy, best_model_path, epoch, proc_id, args, device):
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
            print(f'Epoch {epoch} batch size: {args.batch_size} Average accuracy: {average_accuracy}, Best accuracy so far: {best_accuracy}, epochs_no_improve: {epochs_no_improve}')
        if best_accuracy < average_accuracy:
            best_accuracy = average_accuracy
            torch.save(model.state_dict(), best_model_path)
            with open(best_model_path.replace(".pt", ".json"), "w") as f:
                json.dump({"epoch": epoch, "val accuracy": average_accuracy}, f)
            epochs_no_improve = 0
        else:
            # If the validation accuracy is not better than the best accuracy,
            # increment the count of epochs with no improvement
            epochs_no_improve += 1
            # If the count of epochs with no improvement is greater than our set patience,
            # stop the training
            if epochs_no_improve >= args.patience:
                print('Early stopping!')
                return best_accuracy, epochs_no_improve, True
            
    return best_accuracy, epochs_no_improve, False

def train_with_evaluation(train_dataloader, valid_dataloader, model, opt, best_accuracy, best_model_path, proc_id, args, device):
    epochs_no_improve = 0

    # Copied from previous tutorial with changes highlighted.
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]
                predictions = model(mfgs, inputs)
                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                accuracy = sklearn.metrics.accuracy_score(
                    labels.cpu().numpy(),
                    predictions.argmax(1).detach().cpu().numpy(),
                )
                tq.set_postfix(
                    {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                    refresh=False,
                )

        best_accuracy, epochs_no_improve, isStop=evaluation(epochs_no_improve, valid_dataloader, model, best_accuracy, best_model_path, epoch, proc_id, args, device)
        if isStop:
            break

def train_without_evaluation(train_dataloader, model, opt, args):
    start_time = time.time()
    time_stamp = start_time
    time_list=[]
    loss_list=[]
    train_acc_list=[]
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata["feat"]
                labels = mfgs[-1].dstdata["label"]
                predictions = model(mfgs, inputs)
                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                opt.step()
                accuracy = sklearn.metrics.accuracy_score(
                    labels.cpu().numpy(),
                    predictions.argmax(1).detach().cpu().numpy(),
                )
                train_acc_list.append(accuracy)
                tq.set_postfix(
                    {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                    refresh=False,
                )
                time_list.append(time.time()-time_stamp)
    with open(args.model_dir+f'/{args.name}_time_lr{args.lr}_ngpu{args.num_gpus}_bt{args.batch_size}_wd{args.weight_decay}_dp{args.drop_rate}_fo{"_".join(map(str, args.fanouts))}.json', 'w') as f:
        json.dump(time_list, f)
    with open(args.model_dir+f'/{args.name}_loss_lr{args.lr}_ngpu{args.num_gpus}_bt{args.batch_size}_wd{args.weight_decay}_dp{args.drop_rate}_fo{"_".join(map(str, args.fanouts))}.json', 'w') as f:
        json.dump(loss_list, f)
    with open(args.model_dir+f'/{args.name}_train_acc_lr{args.lr}_ngpu{args.num_gpus}_bt{args.batch_size}_wd{args.weight_decay}_dp{args.drop_rate}_fo{"_".join(map(str, args.fanouts))}.json', 'w') as f:
        json.dump(train_acc_list, f)

def run(proc_id, g, model, args, devices):
    # Initialize distributed training context.
    device=set_device(args, proc_id, devices)

    train_dataloader, valid_dataloader = samplers(g, args, device)

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
    best_model_path = args.model_dir + f'/{args.name}_model_lr{args.lr}_ngpu{args.num_gpus}_bt{args.batch_size}_dp{args.drop_rate}_wd{args.weight_decay}_fo{"_".join(map(str, args.fanouts))}.pt'
    if args.evaluation:
        train_with_evaluation(train_dataloader, valid_dataloader, model, opt, best_accuracy, best_model_path, proc_id, args, device)
    else:
        train_without_evaluation(train_dataloader, model, opt, args)