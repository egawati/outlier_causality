import unittest

from ocular.fcm_generation import noise_model_fitting
from ocular.fcm_generation import noise_model_generation
from ocular.fcm_generation import noise_model_and_samples_generation

from ocular.concept_drift_detector import compute_model_rmse

from ocular.noise_data_generation import generate_noise_and_node_samples

from dowhy.gcm import MedianCDFQuantileScorer

import copy
import numpy as np
import pandas as pd
import datetime

from ocular.causal_model import dag


class TestFCMGeneration(unittest.TestCase):
	def setUp(self):
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.causal_graph = dag.CausalGraph(nodes, self.features)

	def test_noise_model_fitting(self):
		target_node = 'X5'
		n_samples = 100
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		all_ancestors_of_node = self.causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]

		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
				                            m_samples = 0.75, 
				                            target_node = target_node,
				                            sorted_nodes = sorted_nodes)
		self.assertIn('X0', sorted_nodes)
		self.assertIn('X1' , sorted_nodes)
		self.assertIn('X2', sorted_nodes)
		self.assertIn('X3', sorted_nodes)
		self.assertIn('X5', sorted_nodes)
		self.assertNotIn('X4', sorted_nodes)

		noise_models_keys = sorted(list(noise_models.keys()))
		self.assertListEqual(noise_models_keys,sorted_nodes)

	def test_noise_model_generation(self):
		target_node = 'X5'
		n_samples = 100
		np.random.seed(13)
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		all_ancestors_of_node = self.causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]

		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
				                            m_samples = 0.75, 
				                            target_node = target_node,
				                            sorted_nodes = sorted_nodes)
		snoise_models = {}
		snoise_models[0] = noise_models

		model_rmse = compute_model_rmse(data, noise_models, self.causal_graph)
		smodel_rmse = {}
		smodel_rmse[0] = model_rmse

		basic_time = datetime.datetime.now().timestamp()
		active_fcms = {0 : {'start_ts' : basic_time - 2, 
						    'end_ts' : basic_time}}
		slide_size = 2 #in second
		window_size = 4 #in second
		max_fcm_number = window_size//slide_size + 1
		start_ts = basic_time

		np.random.seed(14)
		data2 = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		active_fcms, snoise_models = noise_model_generation(data=data2, 
						   causal_graph=self.causal_graph, 
						   m_samples=0.75, 
						   target_node=target_node, 
						   sorted_nodes=sorted_nodes,
						   active_fcms=active_fcms,
						   max_fcm_number=max_fcm_number,
						   snoise_models=snoise_models,
						   smodel_rmse=smodel_rmse,
						   start_ts=start_ts,
						   slide_size=slide_size,
						   error_threshold_change=0.1,
						   )
		self.assertIn(0, active_fcms)
		self.assertIn(0, snoise_models)
		self.assertIn('X0', snoise_models[0])
		self.assertIn('X1', snoise_models[0])
		self.assertIn('X2', snoise_models[0])
		self.assertIn('X3', snoise_models[0])
		self.assertIn('X5', snoise_models[0])

		self.assertIn(1, active_fcms)
		self.assertIn(1, snoise_models)
		self.assertIn('X0', snoise_models[1])
		self.assertIn('X1', snoise_models[1])
		self.assertIn('X2', snoise_models[1])
		self.assertIn('X3', snoise_models[1])
		self.assertIn('X5', snoise_models[1])

	def test_noise_model_generation2(self):
		target_node = 'X5'
		n_samples = 100
		np.random.seed(13)
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		all_ancestors_of_node = self.causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]

		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
				                            m_samples = 0.75, 
				                            target_node = target_node,
				                            sorted_nodes = sorted_nodes)
		snoise_models = {}
		snoise_models[0] = noise_models

		model_rmse = compute_model_rmse(data, noise_models, self.causal_graph)
		smodel_rmse = {}
		smodel_rmse[0] = model_rmse

		basic_time = datetime.datetime.now().timestamp()
		active_fcms = {0 : {'start_ts' : basic_time - 2, 
						    'end_ts' : basic_time}}
		slide_size = 2 #in second
		window_size = 4 #in second
		max_fcm_number = window_size//slide_size + 1
		start_ts = basic_time

		np.random.seed(13)
		data2 = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		active_fcms, snoise_models = noise_model_generation(data=data2, 
						   causal_graph=self.causal_graph, 
						   m_samples=0.75, 
						   target_node=target_node, 
						   sorted_nodes=sorted_nodes,
						   active_fcms=active_fcms,
						   max_fcm_number=max_fcm_number,
						   snoise_models=snoise_models,
						   smodel_rmse=smodel_rmse,
						   start_ts=start_ts,
						   slide_size=slide_size,
						   error_threshold_change=0.1,
						   )
		self.assertIn(0, active_fcms)
		self.assertIn(0, snoise_models)
		self.assertIn('X0', snoise_models[0])
		self.assertIn('X1', snoise_models[0])
		self.assertIn('X2', snoise_models[0])
		self.assertIn('X3', snoise_models[0])
		self.assertIn('X5', snoise_models[0])

		self.assertNotIn(1, active_fcms)
		self.assertNotIn(1, snoise_models)

		self.assertEqual(active_fcms[0]['end_ts'], start_ts+slide_size)

	def test_noise_model_and_samples_generation(self):
		target_node = 'X5'
		n_samples = 100
		num_noise_samples = 150
		m_samples = 0.75
		np.random.seed(13)
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		all_ancestors_of_node = self.causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]

		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
				                            m_samples = 0.75, 
				                            target_node = target_node,
				                            sorted_nodes = sorted_nodes)
		snoise_models = {}
		snoise_models[0] = noise_models

		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
									            self.causal_graph, 
									            target_node, 
									            sorted_nodes, 
									            num_noise_samples)
		snoise_samples = {}
		snoise_samples[0] = noise_samples

		model_rmse = compute_model_rmse(data, noise_models, self.causal_graph)
		smodel_rmse = {}
		smodel_rmse[0] = model_rmse

		## now we can train outlier_scorer using node_samples
		outlier_scorer = MedianCDFQuantileScorer()
		outlier_scorer.fit(node_samples[target_node])
		soutlier_scorers = {}
		soutlier_scorers[0] = outlier_scorer

		basic_time = datetime.datetime.now().timestamp()
		active_fcms = {0 : {'start_ts' : basic_time - 2, 
						    'end_ts' : basic_time}}
		slide_size = 2 #in second
		window_size = 4 #in second
		max_fcm_number = window_size//slide_size + 1
		start_ts = basic_time

		np.random.seed(14)
		data2 = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		result = noise_model_and_samples_generation(data=data2, 
												   causal_graph=self.causal_graph, 
												   m_samples=m_samples, 
												   target_node=target_node, 
												   sorted_nodes=sorted_nodes,
												   active_fcms=active_fcms,
												   max_fcm_number=max_fcm_number,
												   snoise_models=snoise_models,
												   smodel_rmse=smodel_rmse,
												   start_ts=start_ts,
												   slide_size=slide_size,
												   snoise_samples=snoise_samples,
												   soutlier_scorers=soutlier_scorers,
												   num_noise_samples = num_noise_samples,
												   error_threshold_change=0.1,
												   )
		active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = result
		
		self.assertIn(0, active_fcms)
		self.assertIn(0, snoise_models)
		self.assertIn('X0', snoise_models[0])
		self.assertIn('X1', snoise_models[0])
		self.assertIn('X2', snoise_models[0])
		self.assertIn('X3', snoise_models[0])
		self.assertIn('X5', snoise_models[0])

		self.assertNotIn(1, active_fcms)
		self.assertNotIn(1, snoise_models)

	def test_noise_model_and_samples_generation2(self):
		target_node = 'X5'
		n_samples = 100
		num_noise_samples = 150
		m_samples = 0.75
		np.random.seed(13)
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		all_ancestors_of_node = self.causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]

		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
				                            m_samples = 0.75, 
				                            target_node = target_node,
				                            sorted_nodes = sorted_nodes)
		snoise_models = {}
		snoise_models[0] = noise_models

		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
									            self.causal_graph, 
									            target_node, 
									            sorted_nodes, 
									            num_noise_samples)
		snoise_samples = {}
		snoise_samples[0] = noise_samples

		model_rmse = compute_model_rmse(data, noise_models, self.causal_graph)
		smodel_rmse = {}
		smodel_rmse[0] = model_rmse

		## now we can train outlier_scorer using node_samples
		outlier_scorer = MedianCDFQuantileScorer()
		outlier_scorer.fit(node_samples[target_node])
		soutlier_scorers = {}
		soutlier_scorers[0] = outlier_scorer

		basic_time = datetime.datetime.now().timestamp()
		active_fcms = {0 : {'start_ts' : basic_time - 2, 
						    'end_ts' : basic_time}}
		slide_size = 2 #in second
		window_size = 4 #in second
		max_fcm_number = window_size//slide_size + 1
		start_ts = basic_time

		np.random.seed(13)
		data2 = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		result = noise_model_and_samples_generation(data=data2, 
												   causal_graph=self.causal_graph, 
												   m_samples=m_samples, 
												   target_node=target_node, 
												   sorted_nodes=sorted_nodes,
												   active_fcms=active_fcms,
												   max_fcm_number=max_fcm_number,
												   snoise_models=snoise_models,
												   smodel_rmse=smodel_rmse,
												   start_ts=start_ts,
												   slide_size=slide_size,
												   snoise_samples=snoise_samples,
												   soutlier_scorers=soutlier_scorers,
												   num_noise_samples = num_noise_samples,
												   error_threshold_change=0.1,
												   )
		active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = result
		
		self.assertIn(0, active_fcms)
		self.assertIn(0, snoise_models)
		self.assertIn('X0', snoise_models[0])
		self.assertIn('X1', snoise_models[0])
		self.assertIn('X2', snoise_models[0])
		self.assertIn('X3', snoise_models[0])
		self.assertIn('X5', snoise_models[0])

		self.assertNotIn(1, active_fcms)
		self.assertNotIn(1, snoise_models)

	def test_noise_model_and_samples_generation3(self):
		target_node = 'X5'
		n_samples = 100
		num_noise_samples = 150
		m_samples = 0.75
		np.random.seed(13)
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		all_ancestors_of_node = self.causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]

		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
				                            m_samples = 0.75, 
				                            target_node = target_node,
				                            sorted_nodes = sorted_nodes)
		snoise_models = {}
		snoise_models[0] = noise_models

		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
									            self.causal_graph, 
									            target_node, 
									            sorted_nodes, 
									            num_noise_samples)
		snoise_samples = {}
		snoise_samples[0] = noise_samples

		model_rmse = compute_model_rmse(data, noise_models, self.causal_graph)
		smodel_rmse = {}
		smodel_rmse[0] = model_rmse

		## now we can train outlier_scorer using node_samples
		outlier_scorer = MedianCDFQuantileScorer()
		outlier_scorer.fit(node_samples[target_node])
		soutlier_scorers = {}
		soutlier_scorers[0] = outlier_scorer

		basic_time = datetime.datetime.now().timestamp()
		active_fcms = {0 : {'start_ts' : basic_time - 2, 
						    'end_ts' : basic_time}}
		slide_size = 2 #in second
		window_size = 4 #in second
		max_fcm_number = window_size//slide_size + 1
		start_ts = basic_time

		np.random.seed(17)
		data2 = pd.DataFrame(np.random.randint(300,310,size=(n_samples, len(self.features))), columns=self.features)
		result = noise_model_and_samples_generation(data=data2, 
												   causal_graph=self.causal_graph, 
												   m_samples=m_samples, 
												   target_node=target_node, 
												   sorted_nodes=sorted_nodes,
												   active_fcms=active_fcms,
												   max_fcm_number=max_fcm_number,
												   snoise_models=snoise_models,
												   smodel_rmse=smodel_rmse,
												   start_ts=start_ts,
												   slide_size=slide_size,
												   snoise_samples=snoise_samples,
												   soutlier_scorers=soutlier_scorers,
												   num_noise_samples = num_noise_samples,
												   error_threshold_change=0.1,
												   )
		active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = result
		
		self.assertIn(0, active_fcms)
		self.assertIn(0, snoise_models)
		self.assertIn('X0', snoise_models[0])
		self.assertIn('X1', snoise_models[0])
		self.assertIn('X2', snoise_models[0])
		self.assertIn('X3', snoise_models[0])
		self.assertIn('X5', snoise_models[0])

		self.assertIn(1, active_fcms)
		self.assertIn(1, snoise_models)
		
if __name__ == '__main__': # pragma: no cover
	unittest.main()
