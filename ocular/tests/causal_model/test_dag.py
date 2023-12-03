import unittest

from ocular.causal_model import dag

class TestDAG(unittest.TestCase):
    def setUp(self):
        self.nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
        self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
        self.causal_graph = dag.CausalGraph(self.nodes, self.features)
    
    def test_dag(self):
        self.assertEqual(len(self.causal_graph.dag.nodes), len(self.features))
        self.assertIn('X1', self.causal_graph.dag.nodes)
        for feature in self.features:
            self.assertIn(feature, self.causal_graph.dag.nodes)

    def test_root(self):
        self.assertEqual('X0', self.causal_graph.root[0])
        self.assertEqual('X1', self.causal_graph.root[1])

    def test_ancestor(self):
        self.assertIn('X0', self.causal_graph.ancestors['X2'])
        self.assertIn('X1', self.causal_graph.ancestors['X2'])
        self.assertNotIn('X2', self.causal_graph.ancestors['X2'])

        self.assertIn('X0', self.causal_graph.ancestors['X5'])
        self.assertIn('X1', self.causal_graph.ancestors['X5'])
        self.assertIn('X2', self.causal_graph.ancestors['X5'])
        self.assertIn('X3', self.causal_graph.ancestors['X5'])
        self.assertNotIn('X4', self.causal_graph.ancestors['X5'])
        self.assertNotIn('X5', self.causal_graph.ancestors['X5'])

    def test_ancestors_findex(self):
        self.assertIn(0, self.causal_graph.ancestors_findex['X2'])
        self.assertIn(1, self.causal_graph.ancestors_findex['X2'])
        self.assertNotIn(2, self.causal_graph.ancestors_findex['X2'])

        self.assertIn(0, self.causal_graph.ancestors_findex['X5'])
        self.assertIn(1, self.causal_graph.ancestors_findex['X5'])
        self.assertIn(2, self.causal_graph.ancestors_findex['X5'])
        self.assertIn(3, self.causal_graph.ancestors_findex['X5'])
        self.assertNotIn(4, self.causal_graph.ancestors_findex['X5'])
        self.assertNotIn(5, self.causal_graph.ancestors_findex['X5'])

    def test_features(self):
        for feature in self.features:
            self.assertIn(feature, self.causal_graph.features)

    def test_get_shortest_path(self):
        self.assertEqual(['X1', 'X2', 'X3', 'X5'], self.causal_graph.get_shortest_path('X1', 'X5'))
        self.assertEqual(['X1', 'X2', 'X3'], self.causal_graph.get_shortest_path('X1', 'X3'))
        self.assertEqual(['X1'], self.causal_graph.get_shortest_path('X1', 'X1'))
        self.assertEqual(None, self.causal_graph.get_shortest_path('X4', 'X5'))

    def test_get_node_parent(self):
        self.assertEqual(['X0', 'X1'], self.causal_graph.get_node_parent('X2'))
        self.assertEqual(['X2'], self.causal_graph.get_node_parent('X3'))
        self.assertEqual(['X2'], self.causal_graph.get_node_parent('X4'))
        self.assertEqual(['X3'], self.causal_graph.get_node_parent('X5'))

    def test_predecessors(self):
        self.assertEqual(['X0', 'X1'], self.causal_graph.predecessors('X2'))
        self.assertEqual(['X2'], self.causal_graph.predecessors('X3'))
        self.assertEqual(['X2'], self.causal_graph.predecessors('X4'))
        self.assertEqual(['X3'], self.causal_graph.predecessors('X5'))
        
if __name__ == '__main__': # pragma: no cover
    unittest.main()