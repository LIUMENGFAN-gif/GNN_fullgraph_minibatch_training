import os
os.environ["DGLBACKEND"] = "pytorch"
from models import SageModel
from loadgraph import load_graph
import argparse
import wandb
import dgl
import numpy as np
import sklearn.metrics
import torch
import tqdm
import json
import time
import torch.nn.functional as F


def samplers(g, args, batchsize, fanouts, device):
    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    train_nids = g.ndata['train_mask'].nonzero().squeeze()
    valid_nids = g.ndata['val_mask'].nonzero().squeeze()
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        g,  # The graph
        train_nids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=False,  # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batchsize,  # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0,  # Number of sampler processes
    )
    val_sampler = dgl.dataloading.NeighborSampler(fanouts)
    valid_dataloader = dgl.dataloading.DataLoader(
        g,
        valid_nids,
        val_sampler,
        device=device,
        use_ddp=False,
        batch_size=10000,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_dataloader, valid_dataloader

def evaluation(valid_dataloader, model, device):
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
        return accuracy


def train(config=None):
    
    g, num_features, num_classes  = load_graph(args.name, args.raw_dir)
    g.create_formats_()

    with wandb.init(project=f'{args.project_name}', config=config):
        config=wandb.config
        model = SageModel(num_features, args.hidden_feats, num_classes, args.num_layers)
        device="cuda:"+str(args.start_dev)
        train_dataloader, valid_dataloader = samplers(g, args, config.batchsize, config.fanouts, device)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(),lr=config.lr, weight_decay=config.weight_decay)

        # Copied from previous tutorial with changes highlighted.
        for _ in tqdm.tqdm(range(config.epochs)):
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
                    wandb.log({"train_loss": loss})
                    wandb.log({"train_acc": accuracy})

            accuracy=evaluation(valid_dataloader, model,  device)
            wandb.log({"val_acc": accuracy})
    


# Say you have four GPUs.
def main():
    sweep_config = {'method': 'bayes'}
    early_terminate = {'type': 'hyperband', 's': 2, 'eta': 3, 'max_iter': args.num_epochs}
    metric = {'name':'val_acc', 'goal':'maximize'}
    sweep_config['metric'] = metric
    sweep_config['early_terminate'] = early_terminate


    parameters_dict = {
        'weight_decay': {'values': [0.001, 0.0001]},
        'lr': {'values': [0.0001,0.001, 0.01, 0.02, 0.03]},
        'batchsize': {'values': [1000, 5000, 10000, 40000, 80000]},
        'fanouts': {'values': [[5, 10, 15], [10, 20, 30]]},
        'epochs':{'value':args.num_epochs},
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=f'mfliu-hku-{args.batch_size}-{args.name}')
    wandb.agent(sweep_id, function=train, count=35)

    # Retrieve the best hyperparameters
    api = wandb.Api()
    sweep = api.sweep(f'{args.project_name}/' + sweep_id)
    sweep_best_run=sweep.best_run()
    best_hyperparameters=sweep_best_run.config


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--project_name", type=str, default="project_name")
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--hidden_feats", type=int, default=256)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--name", type=str, default="ogbn-products")
parser.add_argument("--raw_dir", type=str, default="datasets_dir")
parser.add_argument("--start_dev", type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    main()