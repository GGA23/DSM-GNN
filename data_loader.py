import torch
import numpy as np
from pathlib import Path
import networkx as nx
import dgl
from dgl import DGLGraph
import pickle as pkl
import sys
import scipy.sparse as sp
from data_process import (
    normalize_adj,
    eliminate_self_loops_adj,
    largest_connected_components,
    binarize_labels,
    normalized_adjacency,

)
from utils import random_planetoid_splits
from sklearn import preprocessing
import os
small_data = ["cora", "citeseer", "pubmed"]
large_data = ["coauthor-cs","coauthor-phy", "a-computer", "a-photo"]
hete_data = ["chameleon","squirrel", "texas", "cornell","wisconsin"]
def load_data(dataset, args, repeat):
    if dataset in small_data:
        return load_small_data(
            dataset
        )
    elif dataset in large_data:
        return load_large_data(
            dataset,
            args.seed,
            args.labelrate_train,
            args.labelrate_val,
        )
    elif dataset in hete_data:
        return load_hete_data(
            dataset,
            repeat,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def load_small_data(dataset):
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    # if _log is not None:
    #     _log.info('Loading dataset %s.' % dataset)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    data_path = './data/planetoid'
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        os.path.join(data_path, "ind.{}.test.index".format(dataset))
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # cast!!!
    adj = adj.astype(np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #对称化
    features = features.tocsr()
    features = features.astype(np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    adj = normalize_adj(adj)
    adj_sp = adj.tocoo()
    g = dgl.graph((adj_sp.row, adj_sp.col))
    g.ndata["feat"] = features
    dense_adj_tensor = torch.from_numpy(adj_sp.toarray().astype(np.float32))
    dense_adj_tensor = torch.FloatTensor(dense_adj_tensor)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #labels_one_hot = labels
    #labels_one_hot = torch.FloatTensor(labels_one_hot)
    labels = np.argmax(labels, axis=-1)
    labels = torch.LongTensor(labels)
    #print(labels)
    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return dense_adj_tensor,  g, labels, idx_train, idx_val, idx_test
def load_large_data(dataset, seed, labelrate_train, labelrate_val):
    data_path = Path.cwd().joinpath("./data", f"{dataset}.npz")
    if os.path.isfile(data_path):
        data = load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")

    # remove self loop and extract the largest CC
    data = data.standardize()
    adj, features, labels = data.unpack()

    labels = binarize_labels(labels)

    random_state = np.random.RandomState(seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(
        random_state, labels, labelrate_train, labelrate_val
    )
    #labels_one_hot = labels
    #labels_one_hot = torch.FloatTensor(labels_one_hot)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels.argmax(axis=1))

    adj = normalize_adj(adj)
    adj_sp = adj.tocoo()
    g = dgl.graph((adj_sp.row, adj_sp.col))
    g.ndata["feat"] = features
    sp_adj_tensor = torch.from_numpy(adj_sp.toarray().astype(np.float32))
    sp_adj_tensor = torch.FloatTensor(sp_adj_tensor)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return sp_adj_tensor,  g, labels, idx_train, idx_val, idx_test


def load_hete_data(dataset, repeat):
    path = './hete_data/{}/'.format(dataset)

    f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
    l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
    test = np.loadtxt(path + '{}test.txt'.format(repeat), dtype=int)
    train = np.loadtxt(path + '{}train.txt'.format(repeat), dtype=int)
    val = np.loadtxt(path + '{}val.txt'.format(repeat), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    #mask = feature_mask(features, rate)
    #apply_feature_mask(features, mask)
    nclass = len(set(l.tolist()))
    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train)
    print(idx_train.shape)
    idx_test = torch.LongTensor(idx_test)
    print(idx_test.shape)
    idx_val = torch.LongTensor(idx_val)
    print(idx_val.shape)
    label = torch.LongTensor(np.array(l))

    #label_oneHot = torch.FloatTensor(to_categorical(l)).to(device)

    struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    adj = normalize_adj(sadj)
    adj_sp = adj.tocoo()
    g = dgl.graph((adj_sp.row, adj_sp.col))
    g.ndata["feat"] = features
    dense_adj_tensor = torch.from_numpy(adj_sp.toarray().astype(np.float32))
    dense_adj_tensor = torch.FloatTensor(dense_adj_tensor)
    print(dataset, nclass)


    return dense_adj_tensor, g, label, idx_train, idx_val, idx_test

class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form."""

    def __init__(
            self,
            adj_matrix,
            attr_matrix=None,
            labels=None,
            node_names=None,
            attr_names=None,
            class_names=None,
            metadata=None,
    ):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError(
                "Adjacency matrix must be in sparse format (got {0} instead)".format(
                    type(adj_matrix)
                )
            )

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError(
                    "Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(
                        type(attr_matrix)
                    )
                )

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency and attribute matrices don't agree"
                )

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the label vector don't agree"
                )

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the node names don't agree"
                )

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError(
                    "Dimensions of the attribute matrix and the attribute names don't agree"
                )

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G.adj_matrix = eliminate_self_loops_adj(G.adj_matrix)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        if "attr_data" in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix(
                (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
                shape=loader["attr_shape"],
            )
        elif "attr_matrix" in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader["attr_matrix"]
        else:
            attr_matrix = None

        if "labels_data" in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix(
                (
                    loader["labels_data"],
                    loader["labels_indices"],
                    loader["labels_indptr"],
                ),
                shape=loader["labels_shape"],
            )
        elif "labels" in loader:
            # Labels are stored as a numpy array
            labels = loader["labels"]
        else:
            labels = None

        node_names = loader.get("node_names")
        attr_names = loader.get("attr_names")
        class_names = loader.get("class_names")
        metadata = loader.get("metadata")

    return SparseGraph(
        adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata
    )

def sample_per_class(
        random_state, labels, num_examples_per_class, forbidden_indices=None
):
    """
    Used in get_train_val_test_split, when we try to get a fixed number of examples per class
    """

    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [
            random_state.choice(
                sample_indices_per_class[class_index],
                num_examples_per_class,
                replace=False,
            )
            for class_index in range(len(sample_indices_per_class))
        ]
    )

def get_train_val_test_split(
        random_state,
        labels,
        train_examples_per_class=None,
        val_examples_per_class=None,
        test_examples_per_class=None,
        train_size=None,
        val_size=None,
        test_size=None,
):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))
    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False
        )

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state,
            labels,
            val_examples_per_class,
            forbidden_indices=train_indices,
        )
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(
            random_state,
            labels,
            test_examples_per_class,
            forbidden_indices=forbidden_indices,
        )
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert (
                len(np.concatenate((train_indices, val_indices, test_indices)))
                == num_samples
        )

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def load_out_t(out_t_dir):
    return torch.from_numpy(np.load(out_t_dir.joinpath("out.npz"))["arr_0"]) #soft_label


def load_out_emb_t(out_t_dir):
    return torch.from_numpy(np.load(out_t_dir.joinpath("out_emb_list.npz"))["arr_0"]) #hidden_emb of the last layer
