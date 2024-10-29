import networkx as nx
import dgl
import numpy as np
import torch
import numpy as np
from scipy.stats import powerlaw
# Number of communities
num_nodes =10000
num_features=100
n_classes=50
datadir='datadir'

# Define block sizes
block_sizes = [int(num_nodes/n_classes)]*n_classes
label = np.zeros((num_nodes, n_classes))
for i, block_size in enumerate(block_sizes):
    start = sum(block_sizes[:i])
    end = start + block_size
    label[start:end, i] = 1
average_degree=100
while average_degree>69:
    p_matrix = np.full((len(block_sizes), len(block_sizes)),0.0001)
    diagonal = powerlaw.rvs(0.5, size=n_classes)
    np.fill_diagonal(p_matrix,diagonal)
    # Generate the graph
    powerlaw_graph = nx.stochastic_block_model(block_sizes, p_matrix)


# regular graph
feature = np.random.normal(0, 1, (num_nodes, num_features))
# Generate a regular graph
regular_graph = nx.random_regular_graph(round(average_degree), num_nodes)
print("regular graph done")

# Save the graphs
dgl_regular_graph = dgl.from_networkx(regular_graph)
dgl_regular_graph.ndata["feat"] = torch.tensor(feature, dtype=torch.float32)
dgl_regular_graph.ndata["label"] = torch.tensor(label, dtype=torch.float32)
print("dgl_regular_graph done")
dgl.save_graphs(f'{datadir}/regular_graph.bin', [dgl_regular_graph])
print("regular graph saved")
dgl_powerlaw_graph = dgl.from_networkx(powerlaw_graph)
dgl_powerlaw_graph.ndata["feat"] = torch.tensor(feature, dtype=torch.float32)
dgl_powerlaw_graph.ndata["label"] = torch.tensor(label, dtype=torch.float32)
print("dgl_powerlaw_graph done")
dgl.save_graphs(f'{datadir}/powerlaw_graph.bin', [dgl_powerlaw_graph])
print("powerlaw graph saved")