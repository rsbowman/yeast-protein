import itertools
from collections import defaultdict
from math import floor

from scipy.sparse import csc_matrix, lil_matrix
import numpy as np
import networkx as nx
import prc

def run_prc(adj_matrix, initial_order, n_clusters=2):
    #adj_matrix = csc_matrix(adjacency_matrix)
    N = adj_matrix.shape[0]
    order= prc.createOrder(initial_order)
    labels = prc.ivec([0] * N)
    policy = prc.prcPolicyStruct()

    prc.pinchRatioClustering(adj_matrix, order, labels,
                             n_clusters, policy)
    return np.fromiter(labels, dtype=int)

def run_ipr(adj_matrix, initial_order, n_clusters):
    N = adj_matrix.shape[0]
    order = prc.createOrder(initial_order)
    labels = prc.ivec([0] * N)
    policy = prc.iprPolicyStruct()
    policy.iprMaxIterations = 10
    policy.iprConvergenceThreshold = 1.0
    prc.ipr(adj_matrix, order, labels, n_clusters, policy)
    return np.fromiter(labels, dtype=int)
    
def child_labels_for(parent, max_label):
    child_labels = set()
    child_queue = [parent]
    while child_queue:
        nxt = child_queue.pop()
        for cl in (nxt, 2*nxt, 2*nxt + 1):
            if cl <= max_label:
                child_labels.add(cl)
                if cl > nxt:
                    child_queue.append(cl)
    return child_labels

## XXX: This doesn't really work:
class TransductiveGlobClassifier(object):
    def __init__(self, clusterer):
        self.clusterer = clusterer
        
    def fit(self, data, labels):
        all_labels = set(np.unique(labels))
        marked_labels = sorted(all_labels - set([-1]))
        n_classes = len(marked_labels)
        unlabeled_indices = np.where(labels == -1)[0]
        n_unlabeled = (labels == -1).sum()

        unlabeled_idx_to_adj_idx = {}
        for i, unlabeled_idx in enumerate(unlabeled_indices):
            unlabeled_idx_to_adj_idx[unlabeled_idx] = i + n_classes
            
        assert (len(all_labels) == len(marked_labels) + 1 and
                len(marked_labels) >= 2)

        adjacency_matrix_size = n_unlabeled + n_classes
        adjacency_matrix = np.zeros((adjacency_matrix_size,
                                     adjacency_matrix_size))
        print adjacency_matrix.shape
        
        for i in unlabeled_indices:
            for j in unlabeled_indices:
                if i != j:
                    adjacency_matrix[unlabeled_idx_to_adj_idx[i],
                                     unlabeled_idx_to_adj_idx[j]] = data[i, j]
                    adjacency_matrix[unlabeled_idx_to_adj_idx[j],
                                     unlabeled_idx_to_adj_idx[i]] = data[j, i]
        for i in range(n_classes):
            label = marked_labels[i]
            label_indices = np.where(labels == label)[0]
            for j in unlabeled_indices:
                weight = data[j, label_indices].sum()
                adjacency_matrix[i, unlabeled_idx_to_adj_idx[j]] = weight
                adjacency_matrix[unlabeled_idx_to_adj_idx[j], i] = weight

        self.clusterer.fit(adjacency_matrix)

        new_labels = self._infer_labels(labels, self.clusterer.labels_,
                                        n_classes, marked_labels)
        
        assert not (new_labels == -1).any()
        self.transduction_ = new_labels

        return self

    def _infer_labels(self, labels, cluster_labels, n_classes,
                      marked_labels):
        new_labels = labels.copy()
        unique_cluster_labels = np.unique(cluster_labels)

        ## first n_classes labels should be distinct
        ## in practice this doesn't happen
        if len(np.unique(cluster_labels[:n_classes])) != n_classes:
            print "WARNING: more than one glob in cluster!"
            print cluster_labels
            raise Exception()

        cluster_label_to_input_label = {}
        for i in range(n_classes):
            cluster_label_to_input_label[cluster_labels[i]] = marked_labels[i]
        for i in unlabeled_indices:
            new_labels[i] = cluster_label_to_input_label[cluster_labels[i]]

        return new_labels
        
def most_common_element(arr):
    counts = defaultdict(int)
    for l in arr:
        counts[l] += 1
    items = counts.items()
    return sorted(counts.items(),
                  key=lambda x: -x[1])[0][0]
    
class TransductiveCliqueClassifier(object):
    def __init__(self, connection_prob=0.5):
        ## the probability that two nodes w/ same label
        ## will be connected by an edge of weight 1:
        self.connection_prob = connection_prob

    def fit(self, data, labels):
        ## unlabeled points marked with -1, will be assigned labels
        ## data is adjacency matrix
        all_labels = set(np.unique(labels))
        marked_labels = all_labels - set([-1])
        n_classes = len(marked_labels)
        assert (len(all_labels) == len(marked_labels) + 1 and
                len(marked_labels) >= 2)
        ##overall_most_common_label = most_common_element(labels[labels != -1])

        #print "  -- creating adjacency matrix"
        adjacency_matrix = np.zeros(data.shape)
        #print "creating cliques"
        for l in marked_labels:
            indices = np.where(labels == l)[0]
            for i in indices:
                for j in indices:
                    if i != j and np.random.random() < self.connection_prob:
                        adjacency_matrix[i, j] = 1.0
                        adjacency_matrix[j, i] = 1.0

        #print "creating unalbeled conns."
        for i in np.where(labels == -1)[0]:
            for j in range(data.shape[0]):
                if i != j:
                    adjacency_matrix[i, j] = data[i, j]
                    adjacency_matrix[j, i] = data[j, i]

        # g = nx.from_numpy_matrix(adjacency_matrix)
        # n_cpts = len(nx.connected_components(g))
        # if n_cpts > 2:
        #     print "graph has {} components!".format(n_cpts)
        #print "  -- fitting clusterer"

        ## XXX: doesn't work.....
        n_tries = 20
        ordering = np.arange(adjacency_matrix.shape[0])
        new_labels, n_try = None, 0
        while new_labels is None and n_try < n_tries:
            cluster_labels = run_prc(adjacency_matrix, ordering)
            new_labels = self._infer_labels_or_none(
                labels, cluster_labels)
            n_try += 1
            np.random.shuffle(ordering)

        if new_labels is None:
            #print "  XX still no labels"
            raise IndexError()
            
        ## XXX: use sk_prc, old
        ## self.clusterer.fit(adjacency_matrix)
        ## cluster_labels = self.clusterer.labels_
        
        #print "  -- computing per cluster class labels"
        ## compute class labels per cluster

        ## write out dot file:
        # with open("xductive-graph.dot", "w") as f:
        #     colors = ["black", "red", "blue", "green"]
        #     g = nx.from_numpy_matrix(adjacency_matrix)
        #     for n in g.nodes():
        #         if labels[n] == -1:
        #             g.node[n]["shape"] = "box"
        #         else:
        #             g.node[n]["shape"] = "circle"
        #         g.node[n]["color"] = colors[new_labels[n]]
        #     nx.write_dot(g, f)
            
        ## ensure we assigned labels to all unlabeled points
        assert not (new_labels == -1).any()
        self.transduction_ = new_labels # this is what sklearn does...
        
        return self

    def _infer_labels_or_none(self, labels, cluster_labels):
        try:
            return self._infer_labels(labels, cluster_labels)
        except IndexError:
            return None
            
    def _infer_labels(self, labels, cluster_labels):
        new_labels = labels.copy()

        unique_cluster_labels = np.unique(cluster_labels)

        # print "XXX"
        # print np.vstack((cluster_labels, labels))
        for cl in unique_cluster_labels:
            marked_in_cluster = labels[cl == cluster_labels]
            marked_in_cluster = marked_in_cluster[marked_in_cluster != -1]
            mce = most_common_element(marked_in_cluster)
            
            for i in np.where(cl == cluster_labels)[0]:
                if labels[i] == -1:
                    new_labels[i] = mce
        return new_labels

class TransductiveClassifier(object):
    def __init__(self, n_runs=1, n_clusters=1000,
                 cluster_algo=run_prc):
        ## number of runs to average probs. over
        self.n_runs = n_runs
        ## number of clusters to look for in each run;
        ## a large value splits at every opportunity
        self.n_clusters = n_clusters
        self.cluster_algo = cluster_algo
        
    def fit(self, data, labels):
        """ data is an adjacency matrix, vertices with label -1
        will be assigned labels
        """
        assert (data.diagonal() == 0.0).all()
        graph = nx.from_numpy_matrix(data)
        components = nx.connected_component_subgraphs(graph)
        if len(labels.shape) < 2:
            new_labels = labels.copy().reshape((len(labels), 1)).astype(float)
        else:
            new_labels = labels.copy().astype(float)

        avg_labels = [new_labels.copy() for i in range(self.n_runs)]
        for cpt in components:
            for i in range(self.n_runs):
                self._fit_one_cpt(cpt, avg_labels[i])

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

    def _fit_one_cpt(self, nx_cpt, labels):
        adj_matrix = nx.to_numpy_matrix(nx_cpt).A
        ordering = np.arange(adj_matrix.shape[0])
        np.random.shuffle(ordering)
        n_clusters = self.n_clusters
        ## other values tried for n_clusters: 1000 and
        ##   np.clip(len(nx_cpt.nodes()) / 10, 2, 100)
        cluster_labels = self.cluster_algo(adj_matrix, ordering, n_clusters)

        # print "cpt size {}, clusters {}".format(
        #     len(nx_cpt.nodes()), sorted(np.unique(cluster_labels)))
        
        for label_index in range(labels.shape[1]):
            marked_labels = labels[:, label_index]
            marked_labels = marked_labels[marked_labels != -1]
            
            default_label = (marked_labels.sum()
                             / float(len(marked_labels)))
            # default_label = most_common_element(
            #     set(labels[:, label_index]) - set([-1]))
            
            for cl in np.unique(cluster_labels):
                value = self._compute_cluster_proportion_parent(
                    cl, label_index, nx_cpt, cluster_labels,
                    labels, default_label)
                
                for i, node in enumerate(nx_cpt.nodes()):
                    if (cluster_labels[i] == cl and
                        labels[node, label_index] == -1):
                        labels[node, label_index] = value

    def _compute_cluster_proportion_parent(
            self, parent, label_index, nx_cpt,
            cluster_labels, labels, default_label):
        if parent == 1:
            lls = labels[nx_cpt.nodes(), label_index] # local labels
            lls = lls[lls != -1]
            if len(lls):
                print "XXX: local default"
                local_default_label = lls.sum() / float(len(lls))
            else:
                print "YYY: global default"
                local_default_label = default_label
            # print ("XXX: pc label 1, cpt size {}, "
            #        "index {}, loc. def. label {:.3f}, "
            #        "def. label {:.3f}").format(
            #            len(nx_cpt.nodes()), label_index,
            #            local_default_label, default_label)
            # print "clusters {}, labels {}".format(
            #     cluster_labels, labels[nx_cpt.nodes(), label_index])
            
            return local_default_label
            
        ## if they're all unlabeled:
        if (labels[nx_cpt.nodes(), label_index] == -1).all():
            print "ZZZ: all node in cpt. unlabeled"
            return default_label

        max_cluster_label = cluster_labels.max()
        children_clusters = child_labels_for(parent, max_cluster_label)
        #if len(children_clusters.intersection(set(cluster_labels))) != 1:
        #    print "XXXXXXXXX"
        ones, total = 0.0, 0.0
        for i, node in enumerate(nx_cpt.nodes()):
            if (labels[node, label_index] != -1 and
                cluster_labels[i] in children_clusters):
                ones += labels[node, label_index]
                total += 1.0
        if total > 0.0:
            print "WWW: actually assigned probs."
            return ones / total
        else:
            #return default_label
            new_parent = int(floor(parent / 2.0))
            return self._compute_cluster_proportion_parent(
                new_parent, label_index, nx_cpt,
                cluster_labels, labels, default_label)
        
    def _compute_cluster_proportion_new(self, cluster, label_index,
                                        nx_cpt, cluster_labels, labels,
                                        default_label):
        ## if they're all unlabeled:
        if (labels[nx_cpt.nodes(), label_index] == -1).all():
            print "XXX: all nodes unlabeled"
            return default_label
                     
        ones, total = 0.0, 0.0
        for i, node in enumerate(nx_cpt.nodes()):
            if (cluster_labels[i] == cluster and
                labels[node, label_index] != -1):
                ones += labels[node, label_index]
                total += 1.0
        if total > 0.0:
            value = ones / total
        else:
            ## compute proportion of parent cluster
            parent_cluster = int(floor(cluster / 2.0))
            ones, total = 0.0, 0.0
            for i, node in enumerate(nx_cpt.nodes()):
                if (cluster_labels[i] in [2*parent_cluster,
                                          2*parent_cluster + 1] and
                    labels[node, label_index] != -1):
                    ones += labels[node, label_index]
                    total += 1.0
            if total > 0.0:
                value = ones / total
                #print "XXX found parent cluster w/ labels"
            else:
                #print "XXX totally unlabeled parent cluster"
                value = default_label
        return value

    def _compute_cluster_proportion(self, cluster, label_index,
                                    nx_cpt, cluster_labels, labels,
                                    default_label):
        ones, total = 0.0, 0.0
        for i, node in enumerate(nx_cpt.nodes()):
            if (cluster_labels[i] == cluster and
                labels[node, label_index] != -1):
                ones += labels[node, label_index]
                total += 1.0
        if total > 0.0:
            value = ones / total
        else:
            value = default_label
            
        return value

        
    def _fit_one_cpt_proportion(self, nx_cpt, labels):
        adj_matrix = nx.to_numpy_matrix(nx_cpt).A
        ordering = np.arange(adj_matrix.shape[0])
        cluster_labels = run_prc(adj_matrix, ordering, 5)

        for label_index in range(labels.shape[1]):
            lbls = labels[:, label_index] # HACK
            average_ones = (lbls == 1).sum() / float((lbls != -1).sum())

            for cl in np.unique(cluster_labels):
                ones, total = 0.0, 0.0
                for i, node in enumerate(nx_cpt.nodes()):
                    if (cluster_labels[i] == cl and
                        labels[node, label_index] != -1):
                        ones += labels[node, label_index]
                        total += 1
                ones_proportion = ones / total
                value = int(ones_proportion > average_ones)
                for i, node in enumerate(nx_cpt.nodes()):
                    if (cluster_labels[i] == cl and
                        labels[node, label_index] == -1):
                        labels[node, label_index] = value
            
            # ## find out which has more 1's
            # max_ones_cl = -1
            # max_ones_proportion = 0.0
            # for cl in np.unique(cluster_labels):
            #     ones = 0.0
            #     for i, node in enumerate(nx_cpt.nodes()):
            #         if (cluster_labels[i] == cl and
            #             labels[node, label_index] != -1):
            #             ones += labels[node, label_index]
            #     ones_proportion = ones / (cluster_labels == cl).sum()
            #     if ones_proportion > max_ones_proportion:
            #         max_ones_proportion = ones_proportion
            #         max_ones_cl = cl

            # for i, node in enumerate(nx_cpt.nodes()):
            #     if labels[node, label_index] == -1:
            #         if cluster_labels[i] == max_ones_cl:
            #             labels[node, label_index] = 1
            #         else:
            #             labels[node, label_index] = 0
        
    # def _fit_one_cpt_old(self, nx_cpt, labels):
    #     adj_matrix = nx.to_numpy_matrix(nx_cpt).A
    #     ordering = np.arange(adj_matrix.shape[0])
    #     cluster_labels = run_prc(adj_matrix, ordering, 3)

    #     for label_index in range(labels.shape[1]):
    #         default_label = most_common_element(
    #             set(labels[:, label_index]) - set([-1]))
    #         for cl in np.unique(cluster_labels):
    #             marked_in_cluster = []
    #             for i, node in enumerate(nx_cpt.nodes()):
    #                 if (cluster_labels[i] == cl and
    #                     labels[node, label_index] != -1):
    #                     marked_in_cluster.append(labels[node, label_index])
    #             if marked_in_cluster:
    #                 mce = most_common_element(marked_in_cluster)
    #             else:
    #                 mce = default_label

    #             for node in nx_cpt.nodes():
    #                 if labels[node, label_index] == -1:
    #                     labels[node, label_index] = mce


class TransductiveAnchoredClassifier(object):
    def fit(self, data, labels):
        assert len(labels.shape) == 1
        assert (data.diagonal() == 0.0).all()
        graph = nx.from_numpy_matrix(data)
        components = nx.connected_component_subgraphs(graph)
        new_labels = labels.copy()
        
        for cpt in components:
            self._fit_one_cpt(cpt, new_labels)

        assert not (new_labels == -1).any()
        self.transduction_ = new_labels
        return self

    def _fit_one_cpt(self, nx_cpt, labels):
        n = len(nx_cpt.nodes())
        adj_matrix = np.zeros((n + 2, n + 2), dtype=float)
        adj_matrix[1:-1, 1:-1] = nx.to_numpy_matrix(nx_cpt).A

        ## add edges to anchored nodes; node 0 is label 0, node -1 is label 1
        for i, node in enumerate(nx_cpt.nodes()):
            if labels[node] == 0:
                adj_matrix[0, i+1] = adj_matrix[i+1, 0] = 1.0
            elif labels[node] == 1:
                adj_matrix[-1, i+1] = adj_matrix[i+1, -1] = 1.0

        ## run PRC
        ## create order object with usePrefix=True
        order = prc.OrderObject(range(1, n + 1), True)
        #order = prc.createOrder(np.arange(n + 2)) # leave out 0 and n + 1
        prc_labels = prc.ivec([0] * (n + 2))
        policy = prc.prcPolicyStruct()
        
        prc.pinchRatioClustering(adj_matrix, order, prc_labels, 2, policy)
        cluster_labels = np.fromiter(prc_labels, dtype=int)

        ordering = np.fromiter(order.vdata, dtype=int)
        zero_class_cluster_label = cluster_labels[ordering[0]]
        one_class_cluster_label = cluster_labels[ordering[-1]]
        
        # print adj_matrix
        # print cluster_labels
        # print "order", np.fromiter(order.vdata, dtype=int)
        # print "boundary", np.fromiter(order.b.b, dtype=float)

        ## if no clusters, assign most common label
        if zero_class_cluster_label == one_class_cluster_label:
            marked_labels = labels[nx_cpt.nodes()]
            marked_labels = marked_labels[marked_labels != -1]
            common_label = most_common_element(marked_labels)
            if common_label == -1:
                print "XXX", labels[nx_cpt.nodes()]
            for i, node in enumerate(nx_cpt.nodes()):
                if labels[node] == -1:
                    labels[node] = common_label
        else:
            for i, node in enumerate(nx_cpt.nodes()):
                if labels[node] == -1:
                    if cluster_labels[i + 1] == zero_class_cluster_label:
                        labels[node] = 0
                    elif cluster_labels[i + 1] == one_class_cluster_label:
                        labels[node] = 1
                    else:
                        print "BAD CLUSTER LABEL!!!!"
        
