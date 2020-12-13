"""
File: src/untils.py
    - Utility function for H2C2GCN
"""
from torch_geometric.datasets import TUDataset


def accuracy(output, labels):
    """
    Calculates accuracy
    :param output: model output
    :param labels: ground truth
    :return: mean accuracy
    """
    preds = output.max(1)[1].type_as(labels)
    labels = labels.max(1)[1]

    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def download(data_dir, dataset='PROTEINS'):
    """
    Downloads dataset
    :param data_dir: directory
    :param dataset: dataset to download
    """
    TUDataset(root=data_dir, name=dataset, use_node_attr=True)
