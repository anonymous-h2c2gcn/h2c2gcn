"""
File: src/mrcct.py
    - Constructs multi-resolution cross-context tree (MRCCT)
"""
import random
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch

from utils import download


def readlines_in_path(path):
    """
    Read a file and return all lines as a list
    :param path: file to read
    :return: list of lines in the file
    """
    file = open(path)
    lines = file.readlines()
    line_lst = []
    for line in lines:
        line_lst.append(line.rstrip())
    return line_lst


def load_data(dataset):
    """
    Reads 4 files containing information about all graphs in a dataset
    :param dataset: dataset type
    :return: dictionary containing information from the graphs
    """
    path = './data/' + dataset + '/raw/'
    print('Loading {} dataset...'.format(dataset))
    file_list = ['node_labels.txt', 'graph_labels.txt', 'graph_indicator.txt', 'A.txt']
    path_list = list()
    for file in file_list:
        path_list.append(path + dataset + '_' + file)
    data_dict = {'node_cnt':       0,
                 'graph_cnt':      0,
                 'node_label_cnt': 0,
                 'subgraph_cnt':   0,
                 'node':           dict(),
                 'graph':          dict(),
                 'subgraph':       dict()}

    # load node labels
    unique_node_label_set = set()
    lines = readlines_in_path(path_list[0])
    for id, line in enumerate(lines):
        id += 1
        label = int(line) + 1
        if label not in unique_node_label_set:
            unique_node_label_set.add(label)
            data_dict['node_label_cnt'] += 1
        data_dict['node'][id] = {'label': label, 'neighbor': set()}
    data_dict['node_labels'] = sorted(list(unique_node_label_set))

    # load graph labels
    lines = readlines_in_path(path_list[1])
    graph_labels_set = set()
    for id, line in enumerate(lines):
        id += 1
        label = int(line)
        data_dict['graph'][id] = {'label': label}
        if label not in graph_labels_set:
            graph_labels_set.add(label)
        data_dict['graph_labels'] = sorted(list(graph_labels_set))

    # load graph indicator
    lines = readlines_in_path(path_list[2])
    for id, line in enumerate(lines):
        id += 1
        label = int(line)
        data_dict['node'][id]['graph'] = label

    # load A
    lines = readlines_in_path(path_list[3])
    for line in lines:
        split = line.split(',')
        node1 = int(split[0])
        node2 = int(split[1])
        data_dict['node'][node1]['neighbor'].add(node2)
        data_dict['node'][node2]['neighbor'].add(node1)

    return data_dict


def extract_subgraph(data_dict, k=2):
    """
    Extracts unique subgraphs within all graphs
    :param data_dict: dictionary of graphs
    :param k: k-hop neighborhood
    :return: updated dictionary of graphs
    """
    subgraph_start_id = 1
    unique_subgraph_set = set()
    subgraph_reverse_dict = dict()
    data_dict['total_edge_cnt'] = 0
    node_set = set()
    graph_set = set()
    for node_id in list(data_dict['node'].keys()):
        graph_id = data_dict['node'][node_id]['graph']
        subgraph_token = tokenize_subgraph(node_id, data_dict, k)
        if subgraph_token == -1:
            continue
        if subgraph_token not in unique_subgraph_set:
            unique_subgraph_set.add(subgraph_token)
            subgraph_reverse_dict[subgraph_token] = subgraph_start_id
            data_dict['subgraph'][subgraph_start_id] = {'token': subgraph_token,
                                                        'node':  set(),
                                                        'graph': set()}
            data_dict['subgraph'][subgraph_start_id]['node'].add(node_id)
            data_dict['subgraph'][subgraph_start_id]['graph'].add(graph_id)
            data_dict['total_edge_cnt'] += 2
            data_dict['subgraph_cnt'] += 1
            subgraph_start_id += 1
        else:
            subgraph_id = subgraph_reverse_dict[subgraph_token]
            data_dict['subgraph'][subgraph_id]['node'].add(node_id)
            data_dict['subgraph'][subgraph_id]['graph'].add(graph_id)
            data_dict['total_edge_cnt'] += 2
        if node_id not in node_set:
            node_set.add(node_id)
        if node_id not in graph_set:
            graph_set.add(graph_id)
    data_dict['node_lst'] = sorted(list(node_set))
    data_dict['graph_lst'] = sorted(list(graph_set))
    data_dict['node_cnt'] = len(node_set)
    data_dict['graph_cnt'] = len(graph_set)
    data_dict['total_node_cnt'] = len(node_set) + data_dict['subgraph_cnt'] + len(graph_set)
    return data_dict


def tokenize_subgraph(node, data_dict, k):
    """
    Tokenizes an extracted subgraph
    :param node: node to extract subgraph on
    :param data_dict: dictionary of graphs
    :param k: how many hops to extract
    :return: subgraph challenge token string
    """
    token = str(data_dict['node'][node]['label'])
    neighbors_set = deepcopy(data_dict['node'][node]['neighbor'])
    neighbors_set.add(node)
    neighbors = list(data_dict['node'][node]['neighbor'])
    if len(neighbors) == 0:
        return -1
    labels = list()
    for neighbor in neighbors:
        labels.append(data_dict['node'][neighbor]['label'])
    labels, neighbors = (list(t) for t in zip(*sorted(zip(labels, neighbors))))

    token += str(labels).replace(' ', '')
    if k == 2:
        current_token = ''
        cnt = 0
        for child in neighbors:
            one_hop_neighbors = list(data_dict['node'][child]['neighbor'])
            one_hop_labels = list()
            for one_hop in one_hop_neighbors:
                if one_hop not in neighbors_set:
                    one_hop_labels.append(data_dict['node'][one_hop]['label'])
                    cnt += 1
            current_token += str(one_hop_labels).replace(' ', '')
        if cnt == 0:
            return token
        else:
            token += '[' + current_token + ']'

    return token


def encode_onehot(node_label, node_labels):
    """
    Encodes a label into a one hot vector
    :param node_label: node label
    :param node_labels: list of node labels
    :return: the index of the list
    """
    onehot = [0] * len(node_labels)
    onehot[node_labels.index(node_label)] = 1
    return onehot


def normalize(mx):
    """
    Row-normalize sparse matrix
    :param mx: matrix to normalize
    :return: normalized matrix
    """
    rowsum = np.array(mx.sum(1))
    power = np.where(rowsum > 0, -1, 1)
    r_inv = np.power(rowsum, power).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor
    :param sparse_mx: scipy sparse matrix
    :return: torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def to_tensors(data_dict):
    """
    Transforms MRCCT to tensors
    :param data_dict: MRCCT dictionary
    :return: adjs: list of three adjacency matrix for each layer,
            features: feature matrix for objects,
            labels: label matrix for graph objects,
            idx_train: indexes of training set,
            idx_val: indexes of validation set,
            idx_test: indexes of test set,
            graph_start_idx: which index the graph starts from
    """
    _ = list()
    node_labels = data_dict['node_labels']
    for node_id in data_dict['node_lst']:
        node_label = data_dict['node'][node_id]['label']
        _.append(encode_onehot(node_label, node_labels))
    for i in range(data_dict['subgraph_cnt'] + data_dict['graph_cnt']):
        _.append([0] * len(node_labels))
    features = sp.csr_matrix(_, dtype=np.float32)

    graph_lst = data_dict['graph_lst']
    graph_labels = data_dict['graph_labels']
    labels = list()
    for graph_id in graph_lst:
        graph_label = data_dict['graph'][graph_id]['label']
        if graph_labels.index(graph_label) == 0:
            onehot_graph = [1, -1]
        else:
            onehot_graph = [-1, 1]
        labels.append(onehot_graph)

    # adj
    A = list()
    for i in range(3):
        A.append(np.zeros((data_dict['total_node_cnt'], data_dict['total_node_cnt'])))
    node_lst = data_dict['node_lst']
    graph_lst = data_dict['graph_lst']
    node_cnt = data_dict['node_cnt']
    subgraph_cnt = data_dict['subgraph_cnt']
    # layer 1 node -> subgraph
    for subgraph_idx, subgraph_id in enumerate(data_dict['subgraph'].keys()):
        subgraph_idx += node_cnt
        for node_id in list(data_dict['subgraph'][subgraph_id]['node']):
            node_idx = node_lst.index(node_id)
            # layer 1, 2, and 3 node -> subgraph
            A[0][node_idx, subgraph_idx] = 1
            A[1][node_idx, subgraph_idx] = 1
            A[2][node_idx, subgraph_idx] = 1
            # layer 2 and 3 node <- subgraph
            A[1][subgraph_idx, node_idx] = 1
            A[2][subgraph_idx, node_idx] = 1
        for graph_id in list(data_dict['subgraph'][subgraph_id]['graph']):
            graph_idx = graph_lst.index(graph_id) + node_cnt + subgraph_cnt
            # layer 2 and 3 subgraph -> graph
            A[1][subgraph_idx, graph_idx] = 1
            A[2][subgraph_idx, graph_idx] = 1
            # layer 3 subgraph <- graph
            A[2][graph_idx, subgraph_idx] = 1
    adjs = list()
    for adj in A:
        _ = sp.coo_matrix(adj, dtype=np.float32)
        adjs.append(sparse_mx_to_torch_sparse_tensor(normalize(_)))

    graph_cnt = data_dict['graph_cnt']
    graph_lst = data_dict['graph_lst']
    graph_idx_lst = list(range(len(graph_lst)))
    random.shuffle(graph_idx_lst)

    test_cnt = int(round(graph_cnt / 5, 0))
    remain_cnt = graph_cnt - test_cnt
    valid_cnt = int(round(remain_cnt / 5, 0))
    train_cnt = remain_cnt - valid_cnt

    idx_train = graph_idx_lst[:train_cnt]
    idx_val = graph_idx_lst[train_cnt: train_cnt + valid_cnt]
    idx_test = graph_idx_lst[train_cnt + valid_cnt: graph_cnt]

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    graph_start_idx = data_dict['node_cnt'] + data_dict['subgraph_cnt']

    return adjs, features, labels, idx_train, idx_val, idx_test, graph_start_idx


def mrcct(dataset='NCI1', k=2):
    """
    Creates required inputs for Hierarchical GCN
    :param dataset: dataset type
    :param k: number of hops
    :return: adjs: list of three adjacency matrix for each layer,
            features: feature matrix for objects,
            labels: label matrix for graph objects,
            idx_train: indexes of training set,
            idx_val: indexes of validation set,
            idx_test: indexes of test set,
            graph_start_idx: which index the graph starts from
    """
    download('./data', dataset)
    data_dict = load_data(dataset=dataset)
    data_dict = extract_subgraph(data_dict, k=k)
    return to_tensors(data_dict)
