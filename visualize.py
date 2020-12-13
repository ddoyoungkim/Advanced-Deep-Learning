
import torch
from torch_geometric.data import Data

import networkx as nx
import matplotlib.pyplot as plt

def graph_visualize(h, color, epoch=None, loss=None):
    """
    data = Data(x=x, edge_index=edge_index,)
    G = to_networkx(data, to_undirected=True)
    visualize(G, color=data.y)
    cf) from torch_geometric.utils import to_networkx
    """
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=.5, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()
    