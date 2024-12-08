# import libraries
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch_geometric.data import Data, DataLoader
# from torch_geometric.nn import GCNConv
import numpy as np
from numpy import product
from scipy import signal
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import torch.optim as optim
from tednet.tnn import tensor_train
# define hyperparameters
#from torch_geometric.graphgym import GCNConv
#from torch_geometric.nn import BatchNorm

alpha = 1  # regularization coefficient
num_epochs = 100  # number of training epochs
batch_size = 32  # batch size
learning_rate = 0.01  # learning rate
num_classes = 5  # number of classes
num_features = 64  # number of features per node
num_hidden = 32  # number of hidden units per layer

# load EEG dataset
# assume the dataset is a list of Data objects, each with x, edge_index, edge_attr, and y attributes
# x is a [num_nodes, num_features] tensor of node features
# edge_index is a [2, num_edges] tensor of edge indices
# edge_attr is a [num_edges] tensor of edge attributes (coherency values)
# y is a [1] tensor of the class label

subjects = ['E']
#subjects = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L']
sessions = [1]  # , 2]
fs = 256
t_start = int(4.5 * fs)  # 6
t_end = int(7.5 * fs)  # 10
exclusions = []

# Delta Theta Alpha Beta Gamma
fbands = {'Delta': (0.5, 4),
          'Theta': (4, 8),
          'Alpha': (8, 12),
          'Beta': (12, 35),
          'Gamma': (35, 100)}

electrode_names = ['AFz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz',
                   'CP4', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'PO4', 'O1', 'O2']
num_channels = len(electrode_names)
All_electrodes = pd.read_csv('AllElectrodes.csv')

electrodes = pd.DataFrame()
rows = np.array([])
for i in range(len(electrode_names)):
    name = electrode_names[i]
    row = np.where(All_electrodes['labels'] == name)[0]
    electrodes = pd.concat([electrodes, All_electrodes.iloc[row]], ignore_index=True)
print(electrodes)

X = []
y = []
for subject in subjects:
    for session in sessions:
        data = scipy.io.loadmat(f'{subject}.mat')['data']
        data = data[0, session - 1]
        trial = data['trial'][0, 0].T[0].astype(int)

        x = data['X'][0, 0]
        x = np.delete(x, exclusions, axis=1)
        print(x.shape)
        x = np.array([x[trial[i] + t_start: trial[i] + t_end] for i in range(len(trial))])

        band_powers = np.zeros([x.shape[0], len(list(fbands.keys())), num_channels])
        for i in range(x.shape[0]):
            signal = x[i].T
            psd_per_channel = []
            for channel in range(num_channels):
                freqs, psd = scipy.signal.welch(signal[channel], fs, nperseg=4 * fs)
                psd_per_channel.append(psd)

            # Calculate average power within each band for each channel
            m_band = 0
            for band_name, (fmin, fmax) in fbands.items():
                band_powers[i, m_band] = np.mean([psd[(freqs >= fmin) & (freqs <= fmax)] for psd in psd_per_channel])
                m_band += 1

        X.append(band_powers)
        y.append(data['y'][0, 0].T[0].astype(int))
        print(len(trial))

X = np.concatenate(X, axis=0)
X = np.transpose(X, (0, 2, 1))
y = np.concatenate(y, axis=0)
print("X", X.shape)
n_channels = X.shape[1]

A = np.ones([n_channels, n_channels])
for i in range(n_channels-1):
    for j in range(i+1, n_channels):
        v1 = np.array([electrodes.iloc[i]['X'], electrodes.iloc[i]['Y'], electrodes.iloc[i]['Z']])
        v2 = np.array([electrodes.iloc[j]['X'], electrodes.iloc[j]['Y'], electrodes.iloc[j]['Z']])
        A[i, j] = A[j, i] = 1/np.linalg.norm(v1 - v2)

A[A < 0.00001] = 0
A /= np.max(A)
print(np.max(A))
A += np.eye(len(A))

print("A", A.shape)
print(X.shape)
q = y
y = np.matrix([[int(i + 1 == u) for i in range(max(y))] for u in y])

print(torch.tensor(q-1) == torch.argmax(torch.tensor(y), 1))

def loss_function(output, in1, out1, in2, out2, in3, out3, label):
    # compute the cross entropy loss
    ce_loss = F.cross_entropy(output, label)
    # compute the fisher linear discriminant difference between input features and output features of the first conv layer
    label = torch.argmax(label, dim=1)
    fld_diff1 = graph_fisher_discriminant(in1, out1, label)
    # compute the fisher linear discriminant difference between input features and output features of the second conv layer
    fld_diff2 = graph_fisher_discriminant(in2, out2, label)
    fld_diff3 = graph_fisher_discriminant(in3, out3, label)
    # compute the sum of the fisher linear discriminant differences
    fld_sum = fld_diff1 + fld_diff2 + fld_diff3
    # compute the total loss
    total_loss = ce_loss - alpha * fld_sum
    # return the total loss
    return total_loss


# define the fisher linear discriminant function
# the fisher linear discriminant is a measure of feature separability
# it is defined as the ratio of the between-class scatter to the within-class scatter
def graph_fisher_discriminant(x_in, x_out, label):

    x_in = torch.flatten(x_in, 1)
    x_out = torch.flatten(x_out, 1)

    num_classes = len(torch.unique(label))

    gfd = fisher_score(x_out, label, x_out.shape[1], num_classes) - fisher_score(x_in, label, x_in.shape[1], num_classes)

    return gfd


def fisher_score(x, labels, num_features, num_classes):
    # Compute class means
    class_means = torch.zeros(num_classes, num_features).to(device)
    for c in range(num_classes):
        class_means[c] = torch.mean(x[labels == c], dim=0)

    # Compute within-class scatter matrix
    within_class_scatter = torch.zeros(num_classes)
    for c in range(num_classes):
        diff = torch.flatten(x[labels == c] - class_means[c])
        within_class_scatter[c] = torch.inner(diff, diff)

    # Compute overall mean
    #overall_mean = torch.mean(x, dim=0)

    # Compute between-class scatter matrix
    fisher_score = 0
    for c1 in range(num_classes-1):
        for c2 in range(c1+1, num_classes):
            diff = torch.flatten(class_means[c1] - class_means[c2])
            fisher_score +=  torch.inner(diff, diff) / (within_class_scatter[c1] + within_class_scatter[c2])

    # Compute Fisher score
    #fisher_scores = sum_between_class_scatter / sum_within_class_scatter

    return fisher_score


class GCN(nn.Module):
    def __init__(self, in_channels, A):
        super().__init__()
        """
        
        self.conv1 = torch.nn.Conv2d(1, 5, (1, 8), stride=(1, 4), bias=True, padding=(0, 1))

        self.conv2 = torch.nn.Conv2d(5, 5, (1, 8), stride=(1, 4), bias=True, padding=(0, 1))

        self.conv3 = torch.nn.Conv2d(5, 5, (1, 3), stride=(1, 2), bias=True, padding=(0, 1))
        """

        """
        self.conv1 = torch.nn.Conv1d(in_channels, in_channels, 5, stride=2, bias=True, groups=in_channels, padding=1)

        self.conv2 = torch.nn.Conv1d(in_channels, in_channels, 5, stride=2, bias=True, groups=in_channels, padding=1)

        self.conv3 = torch.nn.Conv1d(in_channels, in_channels, 5, stride=2, bias=True, groups=in_channels, padding=1)
        """

        self.l1 = nn.Linear(5, 32)

        self.l2 = nn.Linear(32, 32)

        self.l3 = nn.Linear(32, 32)

        self.flat = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.00)
        self.fc1 = tensor_train.TTLinear([10, 6, 4, 4], [4, 4, 4, 4], [6, 6, 6])
        self.fc2 = tensor_train.TTLinear([4, 4, 4, 4], [5, 1, 1, 1], [6, 6, 6])
        self.dropout2 = nn.Dropout(p=0.3)
        self.A = A

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params}")

    def forward(self, x):
        """
        in1 = x
        out1 = self.A @ self.conv1(in1)
        in2 = self.dropout1(F.relu(out1))
        out2 = self.A @ self.conv2(in2)
        in3 = self.dropout1(F.relu(out2))
        out3 = self.A @ self.conv3(in3)

        """
        #x = x[:, None, :, :]
        in1 = x
        out1 = self.A @ self.l1(in1)
        in2 = self.dropout1(F.relu(out1))
        out2 = self.A @ self.l2(in2)
        in3 = self.dropout1(F.relu(out2))
        out3 = self.A @ self.l3(in3)

        x = self.flat(self.dropout1(F.relu(out3)))

        x = F.relu(self.fc1(self.dropout2(x)))
        x = F.relu(self.fc2(self.dropout2(x)))


        return x, in1, out1, in2, out2, in3, out3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(y, dtype=torch.float32, device=device)
D = np.diag(sum(A, 1))
#A = np.eye(len(A)) + np.sqrt(np.linalg.inv(D)) @ A @ np.sqrt(np.linalg.inv(D))
A = torch.tensor(A, dtype=torch.float32, device=device)
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

model = GCN(n_channels, A)

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

batch_size = 30
n_batch = int(len(y_train)/batch_size)
print(n_batch)

n_epochs = 20000
for epoch in range(n_epochs):
    for b in range(n_batch):
        # Forward pass
        outputs, in1, out1, in2, out2, in3, out3 = model(X_train[batch_size*b:batch_size*(b+1)])
        #loss = loss_fn(outputs, y_train[batch_size*b:batch_size*(b+1)])
        #print(y_train[batch_size*b:batch_size*(b+1)])
        loss = loss_function(outputs, in1, out1, in2, out2, in3, out3, y_train[batch_size*b:batch_size*(b+1)])
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    y_train_pred, in1, out1, in2, out2, in3, out3 = model(X_train)
    print((torch.argmax(y_train_pred, 1) == torch.argmax(y_train, 1)).sum().item()/len(X_train))
    correct = 0
    total = 0
    with torch.no_grad():
        outputs, in1, out1, in2, out2, in3, out3 = model(X_test)
        total += y_test.size(0)
        correct += (torch.argmax(outputs.data, 1) == torch.argmax(y_test, dim=1)).sum().item()

    print('Epoch %d: Accuracy: %d %%' % (epoch,(100 * correct / total)))

print('Finished Training')

# define the loss function
# the loss function is cross entropy plus alpha times the sum of fisher linear discriminant difference between output features and input features of each conv layer



# define a function to compute the coherence between two signals
# coherence is a measure of the linear correlation between two signals in the frequency domain
def coherence(x1, x2, fs, nperseg):
    # compute the power spectral density of the signals
    f, pxx1 = welch(x1, fs, nperseg=nperseg)
    f, pxx2 = welch(x2, fs, nperseg=nperseg)
    # compute the cross spectral density of the signals
    f, pxy = welch(x1, x2, fs, nperseg=nperseg)
    # compute the coherence
    cxy = np.abs(pxy) ** 2 / (pxx1 * pxx2)
    # return the coherence
    return cxy


# define a function to compute the graph weights from the EEG signals
# the graph weights are the average coherence between each pair of channels
def graph_weights(signals, fs, nperseg):
    # get the number of channels
    num_channels = signals.shape[0]
    # initialize the graph weights matrix
    weights = np.zeros((num_channels, num_channels))
    # loop over the pairs of channels
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            # compute the coherence between the channels
            cxy = coherence(signals[i], signals[j], fs, nperseg)
            # compute the average coherence
            avg_cxy = np.mean(cxy)
            # assign the average coherence to the graph weights matrix
            weights[i, j] = avg_cxy
            weights[j, i] = avg_cxy
    # return the graph weights matrix
    return weights


# define a function to compute the node features from the EEG signals
# the node features are the power spectral density of each channel
def node_features(signals, fs, nperseg):
    # get the number of channels
    num_channels = signals.shape[0]
    # initialize the node features matrix
    features = np.zeros((num_channels, num_features))
    # loop over the channels
    for i in range(num_channels):
        # compute the power spectral density of the channel
        f, pxx = welch(signals[i], fs, nperseg=nperseg)
        # normalize the power spectral density
        pxx = pxx / np.sum(pxx)
        # truncate or pad the power spectral density to match the number of features
        if len(pxx) > num_features:
            pxx = pxx[:num_features]
        elif len(pxx) < num_features:
            pxx = np.pad(pxx, (0, num_features - len(pxx)))
        # assign the power spectral density to the node features matrix
        features[i] = pxx
    # return the node features matrix
    return features


# define the graph convolutional neural network
class GCN(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(GCN, self).__init__()
        # define two graph convolutional layers
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)
        # define a dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_attr):
        # apply the first graph convolutional layer
        x1 = self.conv1(x, edge_index, edge_attr)
        # apply ReLU activation
        x1 = F.relu(x1)
        # apply dropout
        x1 = self.dropout(x1)
        # apply the second graph convolutional layer
        x2 = self.conv2(x1, edge_index, edge_attr)
        # return the output and the intermediate features
        return x2, x1


# create an instance of the network
model = GCN(num_features, num_hidden, num_classes)
# move the model to the device (CPU or GPU)




