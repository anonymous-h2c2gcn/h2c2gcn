"""
File: src/models.py
    - Model architecture
"""
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, graph_start_idx):
        """
        :param nfeat: number of input features
        :param nhid: number of hidden units
        :param nclass: number of class
        :param dropout: dropout probability
        :param graph_start_idx: graph start index
        """
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.svm = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.graph_start_idx = graph_start_idx

    def forward(self, x, adj):
        """
        Forward propagation
        :param x: feature
        :param adj: adjacency matrix
        :return: log softmax probability
        """
        # layer 1
        x = F.relu(self.gc1(x, adj[0]))
        x = F.dropout(x, self.dropout, training=self.training)

        # layer 2
        x = F.relu(self.gc2(x, adj[1]))
        x = F.dropout(x, self.dropout, training=self.training)

        # layer 3
        x = self.gc2(x, adj[1])
        x = F.relu(x)

        # svm
        x = self.svm(x[self.graph_start_idx:])
        x = F.log_softmax(x, dim=1)

        return x
