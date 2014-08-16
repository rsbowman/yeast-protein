import sys, itertools, time, multiprocessing
import numpy as np
import networkx as nx
import scipy.io

#from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.cross_validation import KFold, StratifiedKFold

from transduction import TransductiveBaggingClassifier

np.random.seed(111)
n_cpus = multiprocessing.cpu_count()
n_jobs = min(n_cpus, 8)

def format_seconds(secs):
    if secs < 60:
        return "{}s".format(int(secs))
    else:
        m, s = divmod(secs, 60)
        return "{}m {}s".format(int(m), int(s))

def sparse_fill_diag(M, value):
    for i in range(M.shape[0]):
        M[i, i] = value

def flatten(l):
    return [item for sublist in l for item in sublist]

def compute_n_components(adj_matrix):
    g = nx.from_scipy_sparse_matrix(adj_matrix)
    cpts = nx.connected_component_subgraphs(g)
    return len(cpts)

def remove_small_components(full_adj_matrix, labels, min_nodes):
    ## get rid of components with fewer than min_nodes nodes
    g = nx.from_scipy_sparse_matrix(full_adj_matrix)
    cpt_nodes = nx.connected_components(g)
    nodes = []
    for cpt in cpt_nodes:
        if len(cpt) >= min_nodes:
            nodes.extend(cpt)
    subgraph = g.subgraph(nodes)
    return (nx.to_scipy_sparse_matrix(subgraph, format="csc"),
            labels[subgraph.nodes()])

def main(argv):
    matlab_data = scipy.io.loadmat("multi_biograph.mat")
    W = matlab_data["W"]
    W1, W2 = W[0, 0], W[0, 1]
    W3, W4 = W[0, 2], W[0, 3]
    W5 = W[0, 4]

    for M in (W1, W2, W3, W4, W5):
        sparse_fill_diag(M, 0.0)
    
    all_labels = matlab_data["yMat"]
    all_labels[all_labels == -1] = 0

    n_folds = 5
    n_bagging_models = 25
    #sample_ratio = 0.5
    n_kfold_trials = 3
    
    integrated_network = (W1 + W2 + W3 + W4 + W5) / 5.0

    for name, graph in [("W1", W1), ("W2", W2), ("W3", W3), ("W4", W4),
                        ("integrated", integrated_network)]:
        print_graph_scores(name, graph, all_labels, n_folds, n_bagging_models,
                           sample_ratio, n_kfold_trials)
    return 0

def print_graph_scores(name, graph, all_labels, n_folds, n_bagging_models,
                       sample_ratio, n_kfold_trials):
    adj_matrix, labels = remove_small_components(
        graph, all_labels, 2)
    start_time = time.time()
    scores = compute_scores(adj_matrix, labels, n_folds,
                            n_bagging_models, sample_ratio,
                            n_kfold_trials)
    end_time = time.time()

    print("{}, {} folds, {} bagging models,"
          " sample ratio {}, {} kfold trials").format(
              name, n_folds, n_bagging_models,
              sample_ratio, n_kfold_trials)
    n_cpts = compute_n_components(adj_matrix)
    print("graph has {} components, {} vertices, {} edges".format(
        n_cpts, adj_matrix.shape[0],
        (adj_matrix.data != 0).sum() / 2))
    for row in range(len(scores)):
        print("cls {:>2} AUC {:.3f} ({:.3f})".format(
            row, np.mean(scores[row]), np.std(scores[row])))
    print("Average ROC {:.3f} ({:.3f}), run took {}".format(
        np.mean(flatten(scores)), np.std(flatten(scores)),
        format_seconds(end_time - start_time)))
    print
            

def compute_scores(adj_matrix, labels, n_folds, n_bagging_models,
                   sample_ratio, n_kfold_trials):
    clf = TransductiveBaggingClassifier(1, -1, n_bagging_models,
                                        sample_ratio, n_jobs)
    predicted_vs_true_tuples = []
    for i in range(n_kfold_trials):
        for train_idxs, test_idxs in KFold(
                labels.shape[0],
                n_folds=n_folds,
                shuffle=True):
            predict_labels = labels.copy()
            predict_labels[test_idxs] = -1
            clf.fit(adj_matrix, predict_labels)
            predicted = clf.transduction_
            predicted_vs_true_tuples.append(
                (predicted[test_idxs], labels[test_idxs]))
            # print predicted[test_idxs]
            # print labels[test_idxs]

    all_scores = []
    for label_index in range(labels.shape[1]):
        scores = []
        for pred_labels, true_labels in predicted_vs_true_tuples:
            try:
                scores.append(metrics.roc_auc_score(
                    true_labels[:, label_index],
                    pred_labels[:, label_index]))
            except ValueError, e:
                print("Error computing roc_auc_score, label={}: {}".format(label_index, e))
                
        all_scores.append(scores)
                
    return all_scores

if __name__ == '__main__':
    sys.exit(main(sys.argv))
