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

def count_corrects(label, pred) -> int:
    return ((pred == label) + 0.0).sum().item()

def evaluation(epochs_no_improve, g, valid_dataloader, test_dataloader, model, best_accuracy, best_model_path, epoch, args, device):
    print("Evaluating...")
    model.eval()
    # Evaluate
    predictions_valid = []
    labels_valid = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = g.ndata["feat"][input_nodes].to(device)
            labels_valid.append(g.ndata["label"][output_nodes].to(device).long().cpu().numpy())
            mfgs = [mfg.to(device) for mfg in mfgs]
            predictions_valid.append(
                model(mfgs, inputs).argmax(1).cpu().numpy()
            )
            # Clear unimportant cache
            torch.cuda.empty_cache()
        predictions_valid = np.concatenate(predictions_valid)
        labels_valid = np.concatenate(labels_valid)
        num_corrects_valid = torch.tensor(count_corrects(labels_valid, predictions_valid))
        torch.distributed.all_reduce(num_corrects_valid)
        num_all_valid = torch.tensor(labels_valid.shape[0])
        torch.distributed.all_reduce(num_all_valid)
        average_accuracy_valid = num_corrects_valid.item() / num_all_valid.item()

    #test
    predictions_test = []
    labels_test = []
    with tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = g.ndata["feat"][input_nodes].to(device)
            labels_test.append(g.ndata["label"][output_nodes].to(device).long().cpu().numpy())
            mfgs = [mfg.to(device) for mfg in mfgs]
            predictions_test.append(
                model(mfgs, inputs).argmax(1).cpu().numpy()
            )
            # Clear unimportant cache
            torch.cuda.empty_cache()
        predictions_test = np.concatenate(predictions_test)
        labels_test = np.concatenate(labels_test)
        num_corrects_test = torch.tensor(count_corrects(labels_test, predictions_test))
        torch.distributed.all_reduce(num_corrects_test)
        num_all_test = torch.tensor(labels_test.shape[0])
        torch.distributed.all_reduce(num_all_test)
        average_accuracy_test = num_corrects_test.item() / num_all_test.item()
    print(f'Epoch {epoch} batch size: {args.batch_size*args.num_parts} valid accuracy: {average_accuracy_valid}, test accuracy:{average_accuracy_test}, Best accuracy so far: {best_accuracy}, epochs_no_improve: {epochs_no_improve}')
    if best_accuracy < average_accuracy_valid:
        best_accuracy = average_accuracy_valid
        torch.save(model.state_dict(), best_model_path)
        with open(best_model_path.replace(".pt", ".json"), "w") as f:
            json.dump({"epoch": epoch, "val_acc": average_accuracy_valid, "test_acc":average_accuracy_test}, f)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= args.patience:
            print('Early stopping!')
            return best_accuracy, epochs_no_improve, True
        
    return best_accuracy, epochs_no_improve, False

def train_with_evaluation(g, train_dataloader, valid_dataloader, test_dataloader, model, opt, lr_scheduler, best_accuracy, best_model_path, args, device):
    epochs_no_improve = 0

    for epoch in tqdm.tqdm(range(args.num_epochs)):
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
                lr_scheduler.step(loss)
                lr_tensor = torch.tensor(opt.param_groups[0]['lr'])
                torch.distributed.all_reduce(lr_tensor)
                opt.param_groups[0]['lr'] = lr_tensor.item()/args.num_parts
                accuracy = sklearn.metrics.accuracy_score(
                    labels.cpu().numpy(),
                    predictions.argmax(1).detach().cpu().numpy(),
                )
                tq.set_postfix(
                    {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy, "lr": "%.05f" % opt.param_groups[0]['lr']},
                    refresh=False,
                )

        best_accuracy, epochs_no_improve, isStop=evaluation(epochs_no_improve, g, valid_dataloader, test_dataloader, model, best_accuracy, best_model_path, epoch, args, device)
        g.barrier()
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
        with tqdm.tqdm(train_dataloader) as tq, model.join():
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

def set_dataloaders(args, g, train_nid, valid_nid, test_nid):
    train_sampler = dgl.dataloading.NeighborSampler(args.fanouts)
    train_dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        train_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_test_fanouts=[-1]*args.num_layers
    valid_sampler = dgl.dataloading.NeighborSampler(valid_test_fanouts)
    valid_dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        valid_nid,
        valid_sampler,
        batch_size=10000,
        shuffle=False,
        drop_last=False,
    )
    test_sampler = dgl.dataloading.NeighborSampler(valid_test_fanouts)
    test_dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        test_nid,
        test_sampler,
        batch_size=10000,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, valid_dataloader, test_dataloader

def run_across_machines(args, model, data, device):
    # Initialize distributed training context.
    train_nid, val_nid, test_nid, g=data
    train_dataloader, valid_dataloader, test_dataloader = set_dataloaders(args, g, train_nid, val_nid, test_nid)
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
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, factor=0.8,
                                                                  patience=1000, verbose=True)
    

    best_accuracy = 0
    best_model_path = args.model_dir + f'/{args.name}_model_lr{args.lr}_ngpu{args.num_gpus}_bt{args.batch_size*args.num_parts}_dp{args.drop_rate}_wd{args.weight_decay}_fo{"_".join(map(str, args.fanouts))}_h{args.hidden_feats}.pt'
    if args.not_time_record:
        train_with_evaluation(g, train_dataloader, valid_dataloader, test_dataloader, model, opt, lr_scheduler, best_accuracy, best_model_path, args, device)
    else:
        train_without_evaluation(train_dataloader, model, opt, args)