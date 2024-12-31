import networkx as nx
import matplotlib.pyplot as plt

# Define your adjacency matrices (replace these with your own matrices)
import numpy as np
from preprocess import fbands, subjects, sessions, A, y
from channels import electrodes
from feature_extraction import thresholds


band_names = list(fbands.keys())

#thresholds = [0, 0, 0, 0, 0]

adjacency_matrices = [[], [], [], [], []]
print(A.shape[0])

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        #nmax = np.partition(A[i, j, :, :].flatten(), -100)[-100]
        A[i, j][abs(A[i, j]) < thresholds[j]] = 0

for i in range(1,6):
    index = np.where(y == i)[0]
    for j in range(A.shape[1]):
        adjacency_matrices[i-1].append(np.mean(A[index, j], axis=0))

# Create a figure with 5 subplots in a row
fig, axs = plt.subplots(5, 5, figsize=(15, 15))

options = {
    'node_color':[(0.3,0.3,0.5)],
    'node_size':20
}
coordinates = list(electrodes.values())
node_pos = {i: coordinates[i] for i in range(len(coordinates))}
# Iterate through each adjacency matrix and create a graph
for i, adj_matrix in enumerate(adjacency_matrices):
    for j in range(A.shape[1]):
        #print(adj_matrix)
        G = nx.from_numpy_array(np.array(adj_matrix[j]))
        #G = nx.Graph(np.array(adj_matrix[j]))  # Create a graph from the adjacency matrix
        #pos = nx.circular_layout(G)  # Layout algorithm (you can choose others)
        print(G)

        # Draw the graph on the subplot
        #nx.draw(G, pos, ax=axs[i, j], with_labels=True, node_color='skyblue', node_size=30, font_size=10, edge_color=[(0.3,0.3,0.5)])
        #axs[i, j].set_title(f'class {i + 1}, {band_names[j]}')
        axs[i, j].set_title(f'class {i + 1}, {band_names[j]}')

        values = []
        keys = []
        n_nodes = adj_matrix[j].shape[0]
        n_edges = n_nodes*(n_nodes-1)/2 + n_nodes
        for m in range(n_nodes):
            for p in range(m, n_nodes):
                keys.append((m, p))
                values.append(np.abs(adj_matrix[j][m, p]))


        cent = {keys[i]: values[i] for i in range(len(keys))}
        print(cent)
        print(len(cent))
        node_pos = nx.circular_layout(G)  # Any layout will work here, including nx.spring_layout(G)
        nx.draw_networkx_nodes(G, ax=axs[i, j], pos=node_pos, **options)  # draw nodes
        print(values)
        [nx.draw_networkx_edges(G, ax=axs[i, j], pos=node_pos, edgelist=[key], edge_color=[(0.3,0.3,0.5)],
                                alpha=np.amin([value, 1]), width=4) for key,value in cent.items()]  # loop through edges and draw them

# Adjust the layout
plt.tight_layout()

# Display the plots
plt.show()