import torch
from ogb.nodeproppred import DglNodePropPredDataset
from dgl import DGLHeteroGraph
from dgl.data import RedditDataset
from ogb.lsc import MAG240MDataset
import dgl
import numpy as np
import networkx as nx

def process_obg_dataset(dataset: str, raw_dir: str) -> DGLHeteroGraph:
    '''
    process the ogb dataset, return a dgl graph with node features, labels and train/val/test masks.
    '''
    data = DglNodePropPredDataset(name=dataset, root=raw_dir)
    graph, labels = data[0]
    labels = labels[:, 0]
    # split the dataset into tain/val/test before partitioning
    splitted_idx = data.get_idx_split()
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    num_features = graph.ndata["feat"].shape[1]
    num_classes = (labels.max() + 1).item()
    return graph, num_features, num_classes

def process_obg_paper_dataset(dataset: str, raw_dir: str) -> DGLHeteroGraph:
    '''
    process the ogb dataset, return a dgl graph with node features, labels and train/val/test masks.
    '''
    print("loading ogb-papers100M")
    data = DglNodePropPredDataset(name=dataset, root=raw_dir)
    graph, labels = data[0]
    print("convert to undirected graph")
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    print("loading features")
    labels = labels[:, 0]
    print("adding labels and masks to graph")
    # split the dataset into tain/val/test before partitioning
    splitted_idx = data.get_idx_split()
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    print("adding labels and masks to graph")
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    num_features = graph.ndata["feat"].shape[1]
    sorted_values, _ = torch.topk(labels[~torch.isnan(labels)],k=1)
    num_classes = sorted_values[0]+1
    print(f"graph.ndata['train_mask'] shape: {graph.ndata['train_mask'].shape}")
    return graph, num_features, num_classes

def process_reddit_dataset(raw_dir: str) -> DGLHeteroGraph:
    data = RedditDataset(raw_dir=raw_dir)
    graph = data[0]
    num_features = graph.ndata["feat"].shape[1]
    num_classes = (graph.ndata['label'].max() + 1).item()
    return graph, num_features, num_classes



def load_graph(name: str, raw_dir: str) -> DGLHeteroGraph:
    if name=="ogbn-products":
        return process_obg_dataset(name, raw_dir)
    elif name=="reddit":
        return process_reddit_dataset(raw_dir)
    elif name=="ogbn-papers100M":
        return process_obg_paper_dataset(name, raw_dir)