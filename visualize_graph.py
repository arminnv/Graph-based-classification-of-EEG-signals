import networkx as nx
import matplotlib.pyplot as plt

# Define your adjacency matrices (replace these with your own matrices)
import numpy as np



#thresholds = [0, 0, 0, 0, 0]



# Create a figure with 5 subplots in a row
import pandas as pd
import scipy

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

options = {
    'node_color':[(0.3,0.3,0.5)],
    'node_size':20
}

electrode_names = ['AFz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz',
                   'CP4', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO4', 'O1', 'O2']

exclusions = [1, 5, 9]

for idx in sorted(exclusions, reverse=True):
    electrode_names.pop(idx)

A0 = scipy.io.loadmat(f'A0.mat')['A0']
A1 = scipy.io.loadmat(f'A1.mat')['A1']
A_mean = (A0+A1)/2
A0 = A0 - A_mean
A0 = A0>0
A0 = A0/np.max(A0)*0.3
A1 = A1 - A_mean
A1 = A1>0
A1 = A1/np.max(A1)*0.3

graph_names = ['SUB', 'HAND', 'Average']

num_channels = len(electrode_names)
All_electrodes = pd.read_csv('AllElectrodes.csv')

node_pos = {}
rows = np.array([])
labels = {}
for i in range(len(electrode_names)):
    name = electrode_names[i]
    labels[i] = name
    row = np.where(All_electrodes['labels'] == name)[0]
    electrode = All_electrodes.iloc[row]
    node_pos[i] = [float(electrode['Y']),float(electrode['X'])]
print(node_pos)

As = [A0, A1, A_mean]
for i in range(3):
        #print(adj_matrix)
        A = As[i]
        G = nx.from_numpy_array(np.array(A))
        #G = nx.Graph(np.array(adj_matrix[j]))  # Create a graph from the adjacency matrix
        #pos = nx.circular_layout(G)  # Layout algorithm (you can choose others)
        print(G)

        # Draw the graph on the subplot
        #nx.draw(G, pos, ax=axs[i, j], with_labels=True, node_color='skyblue', node_size=30, font_size=10, edge_color=[(0.3,0.3,0.5)])
        axs[i].set_title(f'{graph_names[i]}', size=20)
        #axs[0].set_title(f'class {i + 1}, {band_names[j]}')

        values = []
        keys = []
        n_nodes = A.shape[0]
        n_edges = n_nodes*(n_nodes-1)/2 + n_nodes
        for m in range(n_nodes):
            for p in range(m, n_nodes):
                keys.append((m, p))
                values.append(np.abs(A[m, p]))


        cent = {keys[i]: values[i] for i in range(len(keys))}
        print(cent)
        print(len(cent))
        #node_pos = nx.circular_layout(G)  # Any layout will work here, including nx.spring_layout(G)
        nx.draw_networkx_nodes(G, ax=axs[i], pos=node_pos, **options)  # draw nodes

        print(values)
        [nx.draw_networkx_edges(G, ax=axs[i], pos=node_pos, edgelist=[key], edge_color=[(0.3,0.3,0.5)],
                                alpha=np.amin([value, 1]), width=4) for key,value in cent.items()]  # loop through edges and draw them
        nx.draw_networkx_labels(G, node_pos, labels, font_size=16, font_color='r', ax=axs[2])

# Adjust the layout
plt.tight_layout()

# Display the plots
plt.show()