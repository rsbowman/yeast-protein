from unittest import TestCase, main
import numpy as np
import networkx as nx

from sk_prc import similarity
from sk_prc.cluster import PinchRatioCppClustering

from transduction import child_labels_for, remove_rows_cols

from transduction import TransductiveCliqueClassifier, \
    TransductiveGlobClassifier, TransductiveClassifier, \
    TransductiveAnchoredClassifier, \
    TransductiveBaggingClassifier

class UtilTests(TestCase):
    def test_chlid_labels(self):
        self.assertEqual(sorted(child_labels_for(3, 11)), [3, 6, 7])
        self.assertEqual(sorted(child_labels_for(1, 15)), range(1, 16))
        self.assertEqual(sorted(child_labels_for(4, 8)), [4, 8])

    def test_remove_rows_cols(self):
        a = np.arange(16)
        a.shape = (4,4)
        b = remove_rows_cols(a, [0, 2])
        self.assertEqual(b.shape, (2,2))
        self.assertEqual(b[0, 0], 5)
        self.assertEqual(b[1, 1], 15)

class TransductionTests(TestCase):
    def dont_test_clique_transduction(self):
        prc_clusterer = PinchRatioCppClustering(2, similarity.AdjacencyMatrix())
        c = TransductiveCliqueClassifier(prc_clusterer)
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        labels = np.array([0, 0, 0, 1, -1, 1])
        adjacency_matrix = nx.to_numpy_matrix(g).A
        c.fit(adjacency_matrix, labels)
        self.assertEquals(c.transduction_[4], 1)

    def _test_glob_classifier(self):
        cg = TransductiveGlobClassifier(prc_clusterer)
        cg.fit(adjacency_matrix, labels)
        self.assertEquals(c.transduction_[3], 1)

    def test_transduction(self):
        clf = TransductiveClassifier()
        g = nx.Graph()
        # graph of two cpts, 0, 1, 2 and 3, 4
        g.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4)])
        labels = np.array([0, 0, -1, 1, -1])
        adjacency_matrix = nx.to_numpy_matrix(g).A
        clf.fit(adjacency_matrix, labels)
        np.testing.assert_array_equal(
            clf.transduction_,
            np.array([0, 0, 0, 1, 1]))

    def test_transduction_bagging(self):
        clf = TransductiveBaggingClassifier(1, 1000, 6, 0.5)
        g = nx.Graph()
        # graph of three cpts, 0, 1, 2 and 3, 4, 5
        g.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5)])
        labels = np.array([[0, 0, -1, 1, 1, -1],
                           [1, 1, -1, 0, 0, -1]]).T
        adjacency_matrix = nx.to_numpy_matrix(g).A
        clf.fit(adjacency_matrix, labels)
        np.testing.assert_array_equal(
            clf.transduction_,
            np.array([[0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 0]]).T)

    def test_transduction_bagging2(self):
        clf = TransductiveBaggingClassifier(1, 1000, 6, 0.5)
        g = nx.Graph()
        ## two squares w/ vertices joined 
        g.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
                          (3, 4),
                          (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)])
        labels = np.array([-1, 0, 0, 0, 1, 1, 1, -1], ndmin=2).T
        adjacency_matrix = nx.to_numpy_matrix(g).A
        clf.fit(adjacency_matrix, labels)
        np.testing.assert_array_equal(
            clf.transduction_,
            np.array([0,0,0,0,1,1,1,1], ndmin=2).T)
        
    def test_anchored(self):
        clf = TransductiveAnchoredClassifier()
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        labels = np.array([0, 0, 0, 1, -1, 1])
        adjacency_matrix = nx.to_numpy_matrix(g).A
        clf.fit(adjacency_matrix, labels)
        self.assertEquals(clf.transduction_[4], 1)
        
if __name__ == '__main__':
    main()
