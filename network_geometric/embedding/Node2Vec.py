# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-18 23:54:31
from scipy import sparse
import numpy as np
import gensim
from tqdm import tqdm
import torch
from torch.optim import AdamW, Adam, SGD, SparseAdam
from torch.utils.data import DataLoader

from .utils import utils
from .models.word2vec import Word2Vec
from .loss.node2vec_triplet import Node2VecTripletLoss
from .dataset.triplet_dataset import TripletDataset
from .train import train
from .utils.node_sampler import ConfigModelNodeSampler
from .utils.random_walks import RandomWalkSampler



#
# Base class
#
class Node2Vec:
    """node2vec implementation

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
    restart_prob : float (optional, default 0)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(
        self,
        num_walks=10,  # number of walkers per node
        walk_length=80,  # number of walks per walker
        p=1.0,  # bias parameter
        q=1.0,  # bias parameter
        window=10,  # context window size
        vector_size=64,  # embedding dimension
        ns_exponent=0.75,  # exponent for negative sampling
        alpha=0.025,  # learning rate
        epochs=1,  # epochs
        negative=1,  # number of negative samples per positive sample
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.rw_params = {
            "p": p,
            "q": q,
            "walk_length": walk_length,
        }
        self.ns_exponent = ns_exponent
        self.alpha = alpha
        self.epochs = epochs
        self.negative = negative
        self.num_walks = num_walks
        self.num_nodes = None
        self.vector_size = vector_size
        self.sentences = None
        self.model = None
        self.window = window

    def fit(self, net):
        """Estimating the parameters for embedding."""
        net = self.homogenize_net_data_type(net)
        self.num_nodes = net.shape[0]
        self.sampler = RandomWalkSampler(net, **self.rw_params)
        self.noise_sampler = ConfigModelNodeSampler(ns_exponent=self.ns_exponent)
        self.noise_sampler.fit(net)

    def transform(self, vector_size=None, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if vector_size is None:
            vector_size = self.vector_size

        if self.out_vec is None:
            self.update_embedding(vector_size)
        elif self.out_vec.shape[1] != vector_size:
            self.update_embedding(vector_size)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        pass

    def homogenize_net_data_type(self, net):
        """Convert to the adjacency matrix in form of sparse.csr_matrix.
        :param net: adjacency matrix
        :type net: np.ndarray or csr_matrix
        :return: adjacency matrix
        :rtype: sparse.csr_matrix
        """
        if sparse.issparse(net):
            if type(net) == "scipy.sparse.csr.csr_matrix":
                return net
            return sparse.csr_matrix(net)
        elif "numpy.ndarray" == type(net):
            return sparse.csr_matrix(net)
        else:
            ValueError(
                "Unexpected data type {} for the adjacency matrix".format(type(net))
            )


class GensimNode2Vec(Node2Vec):
    def __init__(self, **params):
        super().__init__(**params)
        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": self.epochs,
            "workers": 4,
            "negative": self.negative,
            "alpha": self.alpha,
            "ns_exponent": self.ns_exponent,
        }

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        self.w2vparams["window"] = self.window
        self.w2vparams["vector_size"] = dim

        def pbar(it):
            return tqdm(it, desc="Training", total=self.num_walks * self.num_nodes)

        starting_nodes = np.kron(
            np.arange(self.num_nodes),
            np.ones(self.num_walks),
        ).astype(np.int64)
        np.random.shuffle(starting_nodes)

        seqs = self.sampler.sampling(starting_nodes).tolist()
        self.model = gensim.models.Word2Vec(sentences=pbar(seqs), **self.w2vparams)

        num_nodes = len(self.model.wv.index_to_key)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv[i]
            self.out_vec[i, :] = self.model.syn1neg[self.model.wv.key_to_index[i]]



class TorchNode2Vec(Node2Vec):
    def __init__(
        self,
        batch_size=256,
        device="cpu",
        buffer_size=100000,
        context_window_type="double",
        miniters=200,
        num_workers=1,
        alpha=1e-3,
        learn_outvec=True,
        **params,
    ):
        """Residual2Vec based on the stochastic gradient descent.
        :param window_length: length of the context window, defaults to 10
        :type window_length: int
        :param batch_size: Number of batches for the SGD, defaults to 4
        :type batch_size: int
        :param num_walks: Number of random walkers per node, defaults to 100
        :type num_walks: int
        :param walk_length: length per walk, defaults to 80
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        :param buffer_size: Buffer size for sampled center and context pairs, defaults to 10000
        :type buffer_size: int, optional
        :param context_window_type: The type of context window. `context_window_type="double"` specifies a context window that extends both left and right of a focal node. context_window_type="left" and ="right" specifies that extends left and right, respectively.
        :type context_window_type: str, optional
        :param miniter: Minimum number of iterations, defaults to 200
        :type miniter: int, optional
        """
        super().__init__(**params)
        self.noise_sampler = ConfigModelNodeSampler(self.ns_exponent)

        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.miniters = miniters
        self.context_window_type = context_window_type
        self.num_workers = num_workers
        self.alpha = alpha
        self.learn_outvec = learn_outvec

    def fit(self, adjmat):
        """Learn the graph structure to generate the node embeddings.
        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: numpy.ndarray or scipy sparse matrix format (csr).
        :return: self
        :rtype: self
        """

        # Convert to scipy.sparse.csr_matrix format
        adjmat = utils.to_adjacency_matrix(adjmat)

        # Set up the graph object for efficient sampling
        self.adjmat = adjmat
        self.n_nodes = adjmat.shape[0]
        self.noise_sampler.fit(adjmat)
        return self

    def update_embedding(self, dim):
        """Generate embedding vectors.
        :param dim: Dimension
        :type dim: int
        :return: Embedding vectors
        :rtype: numpy.ndarray of shape (num_nodes, dim), where num_nodes is the number of nodes.
          Each ith row in the array corresponds to the embedding of the ith node.
        """

        # Set up the embedding model
        PADDING_IDX = self.n_nodes
        model = Word2Vec(
            vocab_size=self.n_nodes + 1,
            embedding_size=dim,
            padding_idx=PADDING_IDX,
            learn_outvec=self.learn_outvec,
        )
        model = model.to(self.device)
        loss_func = Node2VecTripletLoss(n_neg=self.negative)

        # Set up the Training dataset
        adjusted_num_walks = np.ceil(
            self.num_walks
            * np.maximum(
                1,
                self.batch_size
                * self.miniters
                / (self.n_nodes * self.num_walks * self.rw_params["walk_length"]),
            )
        ).astype(int)
        self.rw_params["num_walks"] = adjusted_num_walks
        seq_sampler = RandomWalkSampler(
            adjmat, **self.rw_params
        )
        dataset = TripletDataset(
            n_nodes = adjmat.shape[0],
            seq_sampler = seq_sampler,
            noise_sampler = self.noise_sampler,
            window_length=self.window,
            padding_id=PADDING_IDX,
            buffer_size=self.buffer_size,
            context_window_type=self.context_window_type,
            epochs=self.epochs,
            negative=self.negative,
        )

        train(
            model=model,
            dataset=dataset,
            loss_func=loss_func,
            batch_size=self.batch_size,
            device=self.device,
            learning_rate=self.alpha,
            num_workers=self.num_workers,
        )
        model.eval()
        self.in_vec = model.ivectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        self.out_vec = model.ovectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        return self.in_vec
