import unittest

from ocular.fcm_generation import noise_model_fitting
from ocular.concept_drift_detector import check_concept_drift
from ocular.concept_drift_detector import compute_model_rmse

import copy
import numpy as np
import pandas as pd
import datetime

from ocular.causal_model import dag

from sklearn.metrics import mean_squared_error


class TestConceptDriftDetector(unittest.TestCase):
	def setUp(self):
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.causal_graph = dag.CausalGraph(nodes, self.features)

		target_node = 'X5'
		n_samples = 100
		np.random.seed(13)
		self.data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		all_ancestors_of_node = self.causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]

		self.noise_models = noise_model_fitting(self.data, 
											self.causal_graph, 
				                            m_samples = 0.75, 
				                            target_node = target_node,
				                            sorted_nodes = sorted_nodes)

	def test_compute_model_rmse(self):
		model_rmse = compute_model_rmse(self.data, self.noise_models, self.causal_graph)
		self.assertIn('X0', model_rmse)
		self.assertIn('X1', model_rmse)
		self.assertIn('X2', model_rmse)
		self.assertIn('X3', model_rmse)
		self.assertIn('X5', model_rmse)

	def test_check_concept_drift(self):
		prev_model_rmse = compute_model_rmse(self.data, self.noise_models, self.causal_graph)
		data2 = pd.DataFrame(np.random.randint(0,100,size=(50, len(self.features))), columns=self.features)
		error_threshold_change=0.1
		model_rmse_change = check_concept_drift(data2, self.noise_models, self.causal_graph, prev_model_rmse, error_threshold_change)
		self.assertIn('X0', model_rmse_change)
		self.assertIn('X1', model_rmse_change)
		self.assertIn('X2', model_rmse_change)
		self.assertIn('X3', model_rmse_change)
		self.assertIn('X5', model_rmse_change)

	def test_check_concept_drift2(self):
		prev_model_rmse = compute_model_rmse(self.data, self.noise_models, self.causal_graph)
		np.random.seed(1)
		data2 = pd.DataFrame(np.random.randint(-1,1,size=(50, len(self.features))), columns=self.features)
		error_threshold_change=0.1
		model_rmse_change = check_concept_drift(data2, self.noise_models, self.causal_graph, prev_model_rmse, error_threshold_change)
		self.assertIn('X0', model_rmse_change)
		self.assertIn('X1', model_rmse_change)
		self.assertIn('X2', model_rmse_change)
		self.assertIn('X3', model_rmse_change)
		self.assertIn('X5', model_rmse_change)

		self.assertNotEqual(model_rmse_change['X0'], self.noise_models['X0'])
		self.assertNotEqual(model_rmse_change['X1'], self.noise_models['X1'])

if __name__ == '__main__': # pragma: no cover
	unittest.main()
