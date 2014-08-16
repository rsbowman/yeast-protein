from unittest import TestCase, main
import numpy as np
import networkx as nx

from sk_prc import similarity
from sk_prc.cluster import PinchRatioCppClustering

from transduction import TransductiveClassifier, \
    TransductiveBaggingClassifier

class TransductionTests(TestCase):
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
        
if __name__ == '__main__':
    main()
