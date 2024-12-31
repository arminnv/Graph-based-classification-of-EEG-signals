import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB

from preprocess import Eig, y, X, A, fbands
from mrmr import mrmr_classif

"""
+Radius and Diameter
Total Variation
+Clustering Coefficient
+Local Efficiency
+Eigen Value Centrality
+Betweenness Centrality
Closeness Centrality
+Path Length
Graph Energy
"""


def get_topological_features(G):
    X = []
    #eig_cent = nx.eigenvector_centrality_numpy(G, weight=None, max_iter=50, tol=0)
    # Dictionary of nodes with eigenvector centrality as the value.

    #betw_cent = nx.betweenness_centrality(G, normalized=True)
    # Dictionary of nodes with betweenness centrality as the value.

    clust_coef = nx.clustering(G, nodes=None)

    loc_eff = nx.local_efficiency(G)
    # Returns the average local efficiency of the graph

    #radius = nx.radius(G, e=None, usebounds=False, weight=None)

    #diameter = nx.diameter(G, e=None, usebounds=False, weight=None)

    #avg_path = nx.average_shortest_path_length(G, weight=None, method=None)

    #X += list(eig_cent.values())
    #X += list(sorted(eig_cent))
    #X += list(betw_cent.values())
    #X += list(sorted(betw_cent))
    X += [np.array(list(clust_coef.values())).mean()]
    X.append(loc_eff)
    #X = np.array(X)

    return X




#thresholds = [0.8, 0.7, 0.7, 0.6, 0.7]
thresholds = [0.7, 0.5, 0.5, 0.5, 0.5]
#thresholds = [0, 0, 0, 0, 0]
#G = nx.from_numpy_array(np.array(adj_matrix[j]))

n = y.shape[0]
n_test = int(n/11)
n_eig = A.shape[2]
#Eig = []
#E_avg = np.zeros([5, 5, A.shape[2]])
n_classes = np.zeros([5, 1])

#A = A[:10]
#y = y[:10]
for j in range(A.shape[1]):
        #nmax = np.partition(A[i, j, :, :].flatten(), -100)[-100]
        A[:, j][abs(A[:, j]) < thresholds[j]] = 0

X = []
for i in range(A.shape[0]):
    M = []
    #E = np.zeros([5, n_eig])
    for j in range(A.shape[1]):
        G = nx.from_numpy_array(np.array(A[i, j]))
        #M += get_topological_features(G)
        L = csgraph.laplacian(A[i, j], normed=False)
        eigenvalues, eigenvectors = np.linalg.eig(L)
        eigenvalues = np.real(eigenvalues)
        #E[j] = eigenvalues[:n_eig + 1]
        #E_avg[y[i] - 1, j] += np.real(eigenvalues)
        M += eigenvalues.tolist()
        #M.append()
    n_classes[y[i] - 1] += 1
    #Eig.append(E)
    X.append(np.array(M))

"""
for i in range(5):
    E_avg[i] /= n_classes[i]

colors = {1: "blue", 2: "red", 3: "green", 4: "purple", 5: "lime"}
fig, axs = plt.subplots(5, 1, figsize=(15, 15))
for i in range(5):
    for k in range(A.shape[1]):
        axs[k].plot(E_avg[i, k], color=colors[i+1], linewidth=0.5)
#plt.show()

colors = {1: "blue", 2: "red", 3: "green", 4: "black", 5: "pink"}
fig, axs = plt.subplots(5, 1, figsize=(15, 15))
for i in range(100):
    for k in range(A.shape[1]):
        axs[k].plot(Eig[i][k], color=colors[y[i]], linewidth=0.1)
#plt.show()
Eig = np.real(np.array(Eig))

"""
#Eig -= np.mean(Eig, axis=0)
#Eig = Eig.reshape(Eig.shape[0], -1)
#print("Eig", Eig.shape)
"""
E_test = Eig[-n_test:]
y_test = y[-n_test:]
E_val = Eig[-2*n_test: -n_test]
y_val = y[-2*n_test: -n_test]
E_train = Eig[: -2*n_test]
y_train = y[: -2*n_test]
A_train = A.reshape([n, A.shape[1]*A.shape[2]*A.shape[2]])[: -2*n_test]
A_val = A.reshape([n, A.shape[1]*A.shape[2]*A.shape[2]])[-2*n_test: -n_test]
"""
X = np.array(X)
columns = []
for i in range(X.shape[1]):
    columns.append(i)

print(X.shape)

print(X[0])
df = pd.DataFrame(X, columns = columns)
print(len(df))
selected_features = mrmr_classif(X=df, y=y, K=30)
#print(selected_features)
#X_train = E_train[:, selected_features]
#X_val = E_test[:, selected_features]
X = X[:, selected_features]
