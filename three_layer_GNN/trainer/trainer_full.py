import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
import tqdm
import time
import torch.nn.functional as F

def train_with_evaluation(g, model, opt, args, device):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    valid_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    total_time=0
    # Copied from previous tutorial with changes highlighted.
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        model.train()
        t1=time.time()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        pred= logits.argmax(1)        
        opt.zero_grad()
        loss.backward()
        opt.step()
        t2=time.time()
        total_time+=t2-t1
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()

        #evluation
        model.eval()
        with torch.no_grad():
            logits = model(g, features)
            pred = logits.argmax(1)
            val_acc = (pred[valid_mask] == labels[valid_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
            print(f'Epoch {epoch} loss:{loss.item()} train_acc: {train_acc} val_acc: {val_acc} test_acc: {test_acc} total_time:{total_time}')
            if val_acc >= args.val_acc_target:
                break


def run_full(g, model, args):
    # Initialize distributed training context.
    device=torch.device("cuda:" + str(args.start_dev))

    model = model.to(device)
    g=g.to(device)

    # Define optimizer
    opt = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    train_with_evaluation(g, model, opt, args, device)
    