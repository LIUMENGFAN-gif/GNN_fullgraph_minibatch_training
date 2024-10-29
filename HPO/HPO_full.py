import os
os.environ["DGLBACKEND"] = "pytorch"
from models import SageModelfull
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

def train(config=None):
    device="cuda:"+str(args.start_dev)
    g, num_features, num_classes  = load_graph(args.name, args.raw_dir)
    g=g.to(device)
    g.create_formats_()
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    valid_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    

    with wandb.init(project=f'{args.project_name}', config=config):
        config=wandb.config
        model = SageModelfull(num_features, args.hidden_feats, num_classes, args.num_layers)
        
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(),lr=config.lr, weight_decay=config.weight_decay)

        # Copied from previous tutorial with changes highlighted.
        for _ in tqdm.tqdm(range(config.epochs)):
            model.train()
            logits = model(g, features)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            pred= logits.argmax(1)        
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[valid_mask] == labels[valid_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
            wandb.log({"train_loss": loss})
            wandb.log({"train_acc": train_acc})
            wandb.log({"val_acc": val_acc})
            wandb.log({"test_acc": test_acc})

# Say you have four GPUs.
def main():
    sweep_config = {'method': 'grid'}
    metric = {'name':'val_acc', 'goal':'maximize'}
    sweep_config['metric'] = metric


    parameters_dict = {
        'weight_decay': {'values': [0.001, 0.0001]},
        'lr': {'values': [0.0001,0.001, 0.01, 0.02, 0.03]},
        'epochs':{'value':args.num_epochs},
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=f'mfliu-hku-full-{args.name}')
    wandb.agent(sweep_id, function=train)

    # Retrieve the best hyperparameters
    api = wandb.Api()
    sweep = api.sweep(f'{args.project_name}/' + sweep_id)
    sweep_best_run=sweep.best_run()
    best_hyperparameters=sweep_best_run.config


parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--hidden_feats", type=int, default=256)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--project_name", type=str, default="project_name")
parser.add_argument("--name", type=str, default="ogbn-products")
parser.add_argument("--raw_dir", type=str, default="datasets_dir")
parser.add_argument("--start_dev", type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    main()