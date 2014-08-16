import hashlib, random
from collections import defaultdict
from joblib import Parallel, delayed

from scipy.sparse import csc_matrix, issparse
import numpy as np
import networkx as nx
import prc

def run_prc(adj_matrix, initial_order, n_clusters=2):
    adj_matrix = csc_matrix(adj_matrix)
    #assert issparse(adj_matrix)
    N = adj_matrix.shape[0]
    order= prc.createOrder(initial_order)
    labels = prc.ivec([0] * N)
    policy = prc.prcPolicyStruct()

    prc.sparse_pinchRatioClustering(adj_matrix, order, labels,
                                    n_clusters, policy)
    return np.fromiter(labels, dtype=int)

def hash_array(a):
    if issparse(a):
        b = a.toarray()
    else:
        b = a
    return hashlib.sha1(b).hexdigest()

## use transductive classifier on sample of points;
## used for joblib parallelization
def classify_samples(data, labels, unmarked_idxs,
                     sample_size, n_runs, n_clusters):
    unmarked_point_probs = {}
    all_idxs = range(len(unmarked_idxs))
    random.shuffle(all_idxs)
    keep_raw_idxs = sorted(all_idxs[:sample_size])
    delete_raw_idxs = sorted(all_idxs[sample_size:])
    keep_idxs, delete_idxs = (unmarked_idxs[keep_raw_idxs],
                              unmarked_idxs[delete_raw_idxs])

    bagging_graph = nx.from_scipy_sparse_matrix(data)
    bagging_graph.remove_nodes_from(delete_idxs)
    bagging_adj_matrix = nx.to_scipy_sparse_matrix(bagging_graph)
    bagging_labels = np.delete(labels, delete_idxs, 0)
    bagging_unmarked_idxs = np.where(
        bagging_labels[:, 0] == -1)[0]

    clf = TransductiveClassifier(n_runs, n_clusters)
    clf.fit(bagging_adj_matrix, bagging_labels)
    assert len(keep_idxs) == len(bagging_unmarked_idxs)
    for i, idx in enumerate(keep_idxs):
        unmarked_point_probs[idx] = clf.transduction_[
            bagging_unmarked_idxs[i]]

    return unmarked_point_probs
    
class TransductiveBaggingClassifier(object):
    def __init__(self, base_n_runs, base_n_clusters,
                 n_models, sample_ratio, n_jobs=1):
        ## values to pass TransductiveClassifier
        self.n_runs = base_n_runs
        self.n_clusters = base_n_clusters
        ## number of times to draw samples, percent of all unmarked nodes
        ## to put in each sample
        self.n_models = n_models
        self.sample_ratio = sample_ratio
        self.n_jobs = n_jobs

    def fit(self, data, labels):
        if not issparse(data):
            data = csc_matrix(data)

        unmarked_idxs = np.where(labels[:, 0] == -1)[0]
        sample_size = int(self.sample_ratio * len(unmarked_idxs))

        all_unmarked_point_probs = Parallel(n_jobs=self.n_jobs)(
            delayed(classify_samples)(data, labels, unmarked_idxs,
                                      sample_size, self.n_runs,
                                      self.n_clusters)
            for i in range(self.n_models))

        unmarked_point_probs = defaultdict(list)
        for upps in all_unmarked_point_probs:
            for k, v in upps.items():
                unmarked_point_probs[k].append(v)
        
        new_labels = labels.copy().astype(float)
        for i in unmarked_idxs:
            new_labels[i] = np.mean(unmarked_point_probs[i], 0)

        assert not (new_labels == -1.0).any()
        self.transduction_ = new_labels
        
        return self

class TransductiveClassifier(object):
    def __init__(self, n_runs=1, n_clusters=1000,
                 cluster_algo=run_prc):
        ## number of runs to average probs. over
        self.n_runs = n_runs
        ## number of clusters to look for in each run;
        ## a large value splits at every opportunity
        self.n_clusters = n_clusters
        self.cluster_algo = cluster_algo
        self.array_hash = None
        ## number of 
        self.n_all_unlabeled = 0
        
    def _fit(self, data):
        if not issparse(data):
            data = csc_matrix(data)
        assert (data.diagonal() == 0.0).all()
        graph = nx.from_scipy_sparse_matrix(data)
        self.graph_components = nx.connected_component_subgraphs(graph)

        clf = self.cluster_label_family = defaultdict(list)
        for cpt in self.graph_components:
            for i in range(self.n_runs):
                adj_matrix = nx.to_scipy_sparse_matrix(cpt)
                ordering = np.arange(adj_matrix.shape[0])
                np.random.shuffle(ordering)
                clf[cpt].append(self.cluster_algo(
                    adj_matrix, ordering, self.n_clusters))

    def fit(self, data, labels):
        if self.array_hash is None:
            self._fit(data)
            self.array_hash = hash_array(data)
        else:
            assert hash_array(data) == self.array_hash

        assert data.shape[0] == labels.shape[0]
        
        if len(labels.shape) < 2:
            new_labels = labels.copy().reshape((len(labels), 1)).astype(float)
        else:
            new_labels = labels.copy().astype(float)

        avg_labels = [new_labels.copy() for i in range(self.n_runs)]
        for cpt in self.graph_components:
            for i in range(self.n_runs):
                self._predict_one_cpt(
                    cpt, self.cluster_label_family[cpt][i],
                    avg_labels[i])

        new_labels = np.zeros_like(new_labels)
        for i in range(self.n_runs):
            new_labels += avg_labels[i]
        new_labels /= self.n_runs

        assert not (new_labels == -1).any()
        
        if len(labels.shape) < 2:
            self.transduction_ = new_labels.reshape((len(labels),))
        else:
            self.transduction_ = new_labels
            
        return self
            
    def _predict_one_cpt(self, nx_cpt, cluster_labels, labels):
        for label_index in range(labels.shape[1]):
            marked_labels = labels[:, label_index]
            marked_labels = marked_labels[marked_labels != -1]
            
            default_label = (marked_labels.sum()
                             / float(len(marked_labels)))
            
            for cl in np.unique(cluster_labels):
                value = self._compute_cluster_proportion(
                    cl, label_index, nx_cpt, cluster_labels,
                    labels, default_label)
                
                for i, node in enumerate(nx_cpt.nodes()):
                    if (cluster_labels[i] == cl and
                        labels[node, label_index] == -1):
                        labels[node, label_index] = value

    def _compute_cluster_proportion(
            self, parent, label_index, nx_cpt,
            cluster_labels, labels, default_label):
        if parent == 1: # only one cluster; compute local average
            lls = labels[nx_cpt.nodes(), label_index] # local labels
            lls = lls[lls != -1]
            if len(lls):
                local_default_label = lls.sum() / float(len(lls))
            else:
                self.n_all_unlabeled += 1
                local_default_label = default_label
            
            return local_default_label
            
        ## if they're all unlabeled:
        if (labels[nx_cpt.nodes(), label_index] == -1).all():
            self.n_all_unlabeled += 1
            return default_label

        ones, total = 0.0, 0.0
        for i, node in enumerate(nx_cpt.nodes()):
            if (labels[node, label_index] != -1 and
                cluster_labels[i] == parent):
                ones += labels[node, label_index]
                total += 1.0
        if total > 0.0:
            return ones / total 
        else:
            self.n_all_unlabeled += 1
            return default_label
