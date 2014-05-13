from unittest import TestCase, main
import numpy as np
import networkx as nx

from sk_prc import similarity
from sk_prc.cluster import PinchRatioCppClustering

from transduction import child_labels_for

from transduction import TransductiveCliqueClassifier, \
    TransductiveGlobClassifier, TransductiveClassifier, \
    TransductiveAnchoredClassifier

class UtilTests(TestCase):
    def test_chlid_labels(self):
        self.assertEqual(sorted(child_labels_for(3, 11)), [3, 6, 7])
        self.assertEqual(sorted(child_labels_for(1, 15)), range(1, 16))
        self.assertEqual(sorted(child_labels_for(4, 8)), [4, 8])
        
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
