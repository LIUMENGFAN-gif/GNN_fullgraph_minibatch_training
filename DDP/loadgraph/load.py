import torch
from ogb.nodeproppred import DglNodePropPredDataset
from dgl import DGLHeteroGraph
from dgl.data import RedditDataset
import dgl
import numpy as np
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

def process_reddit_dataset(raw_dir: str) -> DGLHeteroGraph:
    data = RedditDataset(raw_dir=raw_dir)
    graph = data[0]
    num_features = graph.ndata["feat"].shape[1]
    num_classes = (graph.ndata['label'].max() + 1).item()
    return graph, num_features, num_classes



def load_graph(name: str, raw_dir: str, graph_name="powerlaw_graph") -> DGLHeteroGraph:
    if name=="ogbn-products":
        return process_obg_dataset(name, raw_dir)
    elif name=="reddit":
        return process_reddit_dataset(raw_dir)