import sys, itertools
import numpy as np
import networkx as nx
import scipy.io

from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.semi_supervised import LabelSpreading

from sk_prc import similarity
from sk_prc.cluster import PinchRatioCppClustering

from transduction import TransductiveCliqueClassifier, \
    TransductiveGlobClassifier, TransductiveClassifier, \
    TransductiveAnchoredClassifier, run_ipr
from propogation import SpectralPropogation

np.random.seed(111)

def sparse_matrix_iterate(x):
    cx = x.tocoo()    
    for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
        yield (i,j,v)

def labeled_unlabeled_component_nodes(adj_matrix, labels):
    g = nx.from_numpy_matrix(adj_matrix)
    for n in g.nodes():
        g.node[n]["label"] = labels[n]
    cpts = nx.connected_component_subgraphs(g)
    n_unlabeled, n_labeled = 0, 0
    n_unlabeled_nodes = 0
    for cpt in cpts:
        #print "CPT: " + ", ".join(str(cpt.node[n]["label"]) for n in cpt.nodes())
        if all(cpt.node[n]["label"] == -1 for n in cpt.nodes()):
            n_unlabeled += 1
            n_unlabeled_nodes += len(cpt)
        else:
            n_labeled += 1
    return n_labeled, n_unlabeled, len(g), n_unlabeled_nodes
    

def balanced_accuracy(y_true, y_pred):
    conf = metrics.confusion_matrix(y_true, y_pred)
    return (0.5*conf[0,0]/(conf[0,0]+conf[0,1]) +
            0.5*conf[1,1]/(conf[1,0]+conf[1,1]))

def remove_small_components(full_adj_matrix, labels, min_nodes):
    ## get rid of components with fewer than min_nodes nodes
    g = nx.from_scipy_sparse_matrix(full_adj_matrix)
    cpt_nodes = nx.connected_components(g)
    nodes = []
    for cpt in cpt_nodes:
        if len(cpt) >= min_nodes:
            nodes.extend(cpt)
    subgraph = g.subgraph(nodes)
    return nx.to_scipy_sparse_matrix(subgraph), labels[subgraph.nodes()]

def largest_connected_component(adj_matrix, labels):
    g = nx.from_scipy_sparse_matrix(adj_matrix)
    cpts = nx.connected_component_subgraphs(g)

    return nx.to_scipy_sparse_matrix(cpts[0]), labels[cpts[0].nodes()]

def compute_n_components(adj_matrix):
    g = nx.from_scipy_sparse_matrix(adj_matrix)
    cpts = nx.connected_component_subgraphs(g)
    return len(cpts)

def sparse_fill_diag(M, value):
    for i in range(M.shape[0]):
        M[i, i] = value
    
def main_integrated_component(argv):
    matlab_data = scipy.io.loadmat("multi_biograph.mat")
    W = matlab_data["W"]
    W1= W[0,0]
    W2 = W[0,1]
    W3, W4 = W[0, 2], W[0, 3]
    W5 = W[0, 4]

    for M in (W1, W2, W3, W4, W5):
        sparse_fill_diag(M, 0.0)
    
    all_labels = matlab_data["yMat"]
    all_labels[all_labels == -1] = 0

    integrated_network = (W1 + W2 + W3 + W4 + W5) / 5.0

    adj_matrix, labels = largest_connected_component(
        integrated_network, all_labels)
    
    n_folds, n_runs = 5, 3
    n_clusters = 1000
    print "integrated network has {} nodes, {} edges".format(
        adj_matrix.shape[0], (adj_matrix.data != 0).sum() / 2)
    print_scores(adj_matrix, labels, n_folds, n_runs, 3, n_clusters)
        
def main_transduction(argv):
    matlab_data = scipy.io.loadmat("multi_biograph.mat")
    W = matlab_data["W"]
    W1= W[0,0]
    W2 = W[0,1]
    W3, W4 = W[0, 2], W[0, 3]
    W5 = W[0, 4]

    for M in (W1, W2, W3, W4, W5):
        sparse_fill_diag(M, 0.0)

    ## to check:
    assert (W1[5, 5] == 0.0 and W4[33, 33] == 0.0)
    
    all_labels = matlab_data["yMat"]
    all_labels[all_labels == -1] = 0

    # integrated_largest_cpt, all_labels = largest_connected_component(
    #     integrated_network, all_labels)
    
    # g = nx.from_numpy_matrix(W1)
    # cpts = nx.connected_component_subgraphs(g)

    # for i, cpt in enumerate(cpts[:10]):
    #     cpt_labels = all_labels[cpt.nodes(), 11]
    #     print "cpt {} has {} 0s, {} 1s".format(
    #         i, (cpt_labels == 0).sum(), (cpt_labels == 1).sum())
        
    # # print labels for cpt 2
    # for label_index in range(13):
    #     cpt1_labels = all_labels[cpts[1].nodes(), label_index]
    #     print "weird cpt, label index {} has {} 0s, {} 1s".format(
    #         label_index,
    #         (cpt1_labels == 0).sum(),
    #         (cpt1_labels == 1).sum())
    
    # for i, cpt in enumerate(cpts[:10]):
    #     adj = nx.to_numpy_matrix(cpt).A
    #     print "cpt {} has {} nodes, {} edges, {} >.9 weight".format(
    #         i, len(cpt.nodes()), len(cpt.edges()),
    #         (adj > 0.9).sum() / 2.)
    # # print "largest cpt has {} nodes, {} edges".format(
    # #     len(largest_cpt.nodes()), len(largest_cpt.edges()))
    
    # adj_matrix = nx.to_numpy_matrix(largest_cpt).A
    # #all_labels = all_labels[largest_cpt.nodes()]

    # #write out dot file:
    # g = cpts[7]
    # label_index = 0
    # with open("xductive-graph.dot", "w") as f:
    #     colors = ["black", "red", "blue", "green"]
    #     for n in g.nodes():
    #         if all_labels[n, label_index] == -1:
    #             g.node[n]["shape"] = "box"
    #         else:
    #             g.node[n]["shape"] = "circle"
    #         if all_labels[n, label_index] != -1:
    #             g.node[n]["color"] = colors[all_labels[n, label_index]]
    #     nx.write_dot(g, f)
    # return 0

    min_nodes = 6
    n_folds, n_runs = 5, 5
    print ">> n_runs", n_runs
    for i, full_adj_matrix in enumerate((W1,)): # W2, W3, W4)): #, W5):
        print "W{} --------------".format(i + 1)
        adj_matrix, labels = remove_small_components(
            full_adj_matrix, all_labels, min_nodes)
        n_cpts = compute_n_components(adj_matrix)
        print ("subgraph w/ min_nodes {} has {} components, "
               "{} nodes, {} edges").format(
                   min_nodes, n_cpts, adj_matrix.shape[0],
                   (adj_matrix.data != 0).sum() / 2)

        print_scores(adj_matrix, labels, n_folds, n_runs, 3, 1000)

def compute_balance_cutoff(labels, balance):
    """ compute d such that (labels > d).sum()/len(labels) is 1 - balance 
    """
    n = len(labels)
    for cutoff in np.linspace(1, 0, 100):
        if (labels > cutoff).sum() >= n * (1 - balance):
            return cutoff
    ## shouldn't get here...
    return 0.5
        
def print_scores(adj_matrix, all_labels, n_folds,
                 n_prc_runs, n_kfold_trials, n_clusters):
    clf = TransductiveClassifier(n_runs=n_prc_runs,
                                 n_clusters=n_clusters)
    #clf = SpectralPropogation()

    predicted_true_tuples = []
    for i in range(n_kfold_trials):
        for train_idxs, test_idxs in KFold(
                all_labels.shape[0],
                n_folds=n_folds,
                shuffle=True):
            predict_labels = all_labels.copy()
            predict_labels[test_idxs] = -1
            clf.fit(adj_matrix, predict_labels)
            predicted = clf.transduction_
            predicted_true_tuples.append((clf.transduction_[test_idxs],
                                          all_labels[test_idxs]))

    all_scores = []
    for label_index in range(all_labels.shape[1]):
        labels = all_labels[:, label_index]
        l1 = (labels == 0).sum()
        l2 = (labels == 1).sum()
        # if l1 > l2:
        #     most_common_label = 0
        # else:
        #     most_common_label = 1
        label_balance = l1/float(l1+l2)
        scores, guess_scores = [], []
        accuracy_scores, ari_scores = [], []
        for pred_labels, true_labels in predicted_true_tuples:
            try:
                scores.append(metrics.roc_auc_score(true_labels[:, label_index],
                                                    pred_labels[:, label_index]))
            except ValueError:
                print "  --> {} 0s, {} 1s".format(l1, l2)

            cutoff = compute_balance_cutoff(pred_labels[:, label_index],
                                            label_balance)
            try:
                accuracy_scores.append(
                    balanced_accuracy(true_labels[:, label_index],
                                      pred_labels[:, label_index] > cutoff))
            except IndexError:
                print "BAD ACCURACY SCORE!"
                
            try:
                ari_scores.append(
                    metrics.adjusted_rand_score(
                        true_labels[:, label_index],
                        pred_labels[:, label_index] > cutoff))
            except:
                print "BAD ARI SCORE!"
                        
            # print "  {} predicted 1, {} true 1".format(
            #     pred_labels[:, label_index].sum(),
            #     true_labels[:, label_index].sum())

        all_scores.extend(scores)
        print ("cls {:>2} AUC {:.3f} ({:.3f}),"
               " acc. {:.3f} ({:.3f}),"
               " ARI {:.3f} ({:.3f}), bal. {:.3f}".format(
                   label_index, np.mean(scores), np.std(scores),
                   np.mean(accuracy_scores), np.std(accuracy_scores),
                   np.mean(ari_scores), np.std(ari_scores),
                   label_balance))
    print "Average ROC {:.3f} ({:.3f})".format(np.mean(all_scores),
                                               np.std(all_scores))

def main_anchored_classifier(argv):
    matlab_data = scipy.io.loadmat("multi_biograph.mat")
    W = matlab_data["W"]
    W1 = W[0,0].toarray()
    np.fill_diagonal(W1, 0)

    all_labels = matlab_data["yMat"]
    all_labels[all_labels == -1] = 0

    n_folds = 3
    clf = TransductiveAnchoredClassifier()
    #adj_matrix, all_labels = largest_connected_component(W1, all_labels)
    adj_matrix, all_labels = remove_small_components(
        W1, all_labels, 5)
    
    for label_index in (1, 2):
        labels = all_labels[:, label_index]
        accuracy_scores, ari_scores = [], []
        roc_scores = []
        for train_idxs, test_idxs in KFold(
                labels.shape[0], n_folds=n_folds):
            predict_labels = labels.copy()
            predict_labels[test_idxs] = -1
            clf.fit(adj_matrix, predict_labels)
            predicted = clf.transduction_
            
            accuracy_scores.append(
                balanced_accuracy(labels[test_idxs],
                                  predicted[test_idxs]))
            ari_scores.append(
                metrics.adjusted_rand_score(
                    labels[test_idxs],
                    predicted[test_idxs]))

            roc_scores.append(
                metrics.roc_auc_score(labels[test_idxs],
                                      predicted[test_idxs]))

        ## print roc_scores, accuracy_scores, ari_scores
        print "class {:>2} ROC {:.3f}, b. accuracy {:.3f}, ARI {:.3f}".format(
            label_index, np.mean(roc_scores),
            np.mean(accuracy_scores), np.mean(ari_scores))

    return 0
            
def main_explore_graph(argv):
    matlab_data = scipy.io.loadmat("multi_biograph.mat")
    W = matlab_data["W"]
    labels = matlab_data["yMat"]
    labels[labels == -1] = 0
    n_folds = 3
    W1 = W[0, 0].toarray()

    for label_index in (0,):# 1, 2, 5, 11, 12):
        print "label index", label_index
        for train_idxs, test_idxs in KFold(W1.shape[0], n_folds=n_folds):
            g = nx.from_numpy_matrix(W1)
            predict_labels = labels[:, label_index].copy()
            predict_labels[test_idxs] = -1
            for n in g.nodes():
                g.node[n]["label"] = predict_labels[n]

            connected_subgraphs = nx.connected_component_subgraphs(g)
            print "{}".format(", ".join(str(len(subg.nodes()))
                                            for subg in connected_subgraphs))
            n_labeled_components = 0
            n_unlabeled_nodes = 0
            for subg in connected_subgraphs:
                n_labeled = 0
                for n in subg.nodes():
                    if subg.node[n]["label"] != -1:
                        n_labeled += 1
                        break
                
                if n_labeled != 0:
                    n_labeled_components += 1
                else:
                    n_unlabeled_nodes += len(subg.nodes())
                    print "{}".format(", ".join(str((n, subg.node[n]["label"]))
                                                   for n in subg.nodes()))
            print "  {} labeled components of {}; {}/{} unlabeled nodes".format(
                n_labeled_components,
                len(connected_subgraphs),
                n_unlabeled_nodes,
                len(test_idxs))

def main(argv):
    matlab_data = scipy.io.loadmat("multi_biograph.mat")
    W = matlab_data["W"]
    labels = matlab_data["yMat"]
    W1 = W[0,0]

    ## explore large connected components
    graph = nx.Graph()
    graph.add_weighted_edges_from(list(
        itertools.ifilter(lambda x: x[0] != x[1],
                          sparse_matrix_iterate(W1))))
    components = nx.connected_component_subgraphs(graph)
    # print "connected components of sizes {}".format(
    #     ", ".join(str(len(c.nodes())) for c in components))
    big = components[1]
    W1p = nx.to_numpy_matrix(big)
    
    spec = SpectralClustering(2, affinity="precomputed")
    spec.fit(W1p)
    guesses = spec.labels_

    for l_index in range(labels.shape[1]):
        labs = labels[big.nodes(), l_index]
        print "Spectral ARI {:.3f} ({}/{} label +1)".format(
            metrics.adjusted_rand_score(guesses, labs),
            (labs == 1).sum(), labs.shape[0])
    
    return 0

if __name__ == '__main__':
#    sys.exit(main_transduction(sys.argv))
    sys.exit(main_transduction(sys.argv))
    #sys.exit(main_anchored_classifier(sys.argv))
    #sys.exit(main_integrated_component(sys.argv))
