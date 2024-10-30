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
import torch
import tqdm
import sklearn.metrics
import torch.nn.functional as F
import wandb

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

def set_device(args,g):
    if args.num_gpus == 0:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus + args.start_dev
        device = th.device("cuda:" + str(dev_id))
    return device

def set_dataloaders(args, g, train_nid, valid_nid):
    train_sampler = dgl.dataloading.NeighborSampler(args.fanouts)
    train_dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        train_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_sampler = dgl.dataloading.NeighborSampler([-1,-1,-1])
    valid_dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        valid_nid,
        valid_sampler,
        batch_size=10000,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, valid_dataloader

def set_model_and_opt(num_features, num_classes, args,g,train_nid,val_nid,device):
    model = SageModel(num_features, args.hidden_feats, num_classes, args.num_layers, args.drop_rate)
    print("model done")
    train_dataloader, valid_dataloader = set_dataloaders(args, g, train_nid, val_nid)
    model = model.to(device)
    # Wrap the model with distributed data parallel module.
    if args.num_gpus == 0:
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    # Define optimizer
    opt = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    return model, opt, train_dataloader, valid_dataloader

def each_epoch_train_and_evaluation(model, train_dataloader,g,device,epoch,args,valid_dataloader,opt):
    model.train()
    with tqdm.tqdm(train_dataloader) as tq, model.join():
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # feature copy from CPU to GPU takes place here
            inputs = g.ndata["feat"][input_nodes].to(device)
            labels = g.ndata["label"][output_nodes].to(device).long()
            mfgs = [mfg.to(device) for mfg in mfgs]
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
    print("Evaluating...")
    model.eval()
    # Evaluate on only the first GPU.
    predictions = []
    labels = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = g.ndata["feat"][input_nodes].to(device)
            labels.append(g.ndata["label"][output_nodes].to(device).long().cpu().numpy())
            mfgs = [mfg.to(device) for mfg in mfgs]
            predictions.append(
                model(mfgs, inputs).argmax(1).cpu().numpy()
            )
            # Clear unimportant cache
            torch.cuda.empty_cache()
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        #average accuracy across all GPUs
        accuracy_tensor = torch.tensor(accuracy)
        torch.distributed.all_reduce(accuracy_tensor)
        average_accuracy = accuracy_tensor.item() / torch.distributed.get_world_size()
        print(f'Epoch {epoch} batch size: {args.batch_size} Average accuracy: {average_accuracy}')
        g.barrier()
    return average_accuracy

def train(config=None):
    rank = dgl.distributed.get_rank()
    g, train_nid, val_nid, _, num_features, num_classes=set_graph(args, part_config_path, info_path)
    device=set_device(args,g)
    if rank==0:
        with wandb.init(project=f'{args.batch_size*args.num_gpus*args.num_parts}-{args.name}', config=config):
            config = wandb.config
            config_dict = dict(config)
            print(f"rank:{rank} {g.rank()} config_dict {config_dict}")
            th.distributed.broadcast_object_list([config_dict], src=0)
            g.barrier()
            args.weight_decay = config.weight_decay
            args.lr=config.lr
            args.drop_rate=config.drop_rate
            args.fanouts=config.fanouts
            model, opt, train_dataloader, valid_dataloader=set_model_and_opt(num_features, num_classes, args, g, train_nid, val_nid, device)
            for epoch in tqdm.tqdm(range(args.num_epochs)):
                average_accuracy=each_epoch_train_and_evaluation(model, train_dataloader,g,device,epoch,args,valid_dataloader,opt)
                wandb.log({"val_acc": average_accuracy})
    else:
        # Initialize config_dict as an empty list
        # Receive config from rank 0
        config_dict_receive = [{}]
        th.distributed.broadcast_object_list(config_dict_receive, src=0)
        config_dict = config_dict_receive[0]
        print(f"rank:{rank} {g.rank()} config_dict_receive {config_dict_receive}")
        args.weight_decay = config_dict["weight_decay"]
        args.lr=config_dict["lr"]
        args.drop_rate=config_dict["drop_rate"]
        args.fanouts=config_dict["fanouts"]
        g.barrier()
        model, opt, train_dataloader, valid_dataloader=set_model_and_opt(num_features, num_classes, args, g, train_nid, val_nid, device)
        for epoch in tqdm.tqdm(range(args.num_epochs)):
            average_accuracy=each_epoch_train_and_evaluation(model, train_dataloader,g,device,epoch,args,valid_dataloader,opt)

# Say you have four GPUs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--drop_rate", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_feats", type=int, default=256)
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
    parser.add_argument("--count", type=int, default=2)

    args = parser.parse_args()
    args.fanouts = [int(x) for x in args.fanouts.split(',')]
    part_config_path=f'{args.name}/{args.num_parts}part_data/{args.name}.json'
    info_path=f'{args.name}/info.json'

    #before running
    set_comm()
    rank = dgl.distributed.get_rank()
    if rank==0:
        #wandb set
        sweep_config = {'method': 'grid'}
        # early_terminate = {'type': 'hyperband', 's': 2, 'eta': 3, 'max_iter': args.num_epochs}
        metric = {'name':'val_acc', 'goal':'maximize'}
        sweep_config['metric'] = metric
        # sweep_config['early_terminate'] = early_terminate
        parameters_dict = {
            'drop_rate': {'values': [0,0.2,0.5,0.7]},
            'weight_decay': {'values': [0.001, 0.0001, 0]},
            'lr': {'values': [0.001, 0.01]},
            'fanouts': {'values': [[5, 10, 15], [10, 20, 30]]},
            'epochs':{'value':args.num_epochs},
        }

        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project=f'{args.batch_size*args.num_gpus*args.num_parts}-{args.name}')
        wandb.agent(sweep_id, function=train)
    else:
        for i in range(args.count):
            train()
    