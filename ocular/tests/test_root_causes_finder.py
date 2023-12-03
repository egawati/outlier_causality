import unittest
import numpy as np
import pandas as pd
import datetime

from dowhy.gcm import MedianCDFQuantileScorer

from ocular.causal_model import dag

from ocular.noise_data_generation import generate_noise_and_node_samples

from ocular.fcm_generation import noise_model_fitting

from ocular.root_causes_finder import find_outlier_root_causes
from ocular.root_causes_finder import find_outliers_root_causes_paralel


class TestRootCausesFinder(unittest.TestCase):
	def setUp(self):
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.causal_graph = dag.CausalGraph(nodes, self.features)
		self.target_node = 'X5'

		all_ancestors_of_node = self.causal_graph.ancestors[self.target_node]
		all_ancestors_of_node.update({self.target_node})
		self.sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]

		self.shapley_config = None
		self.attribute_mean_deviation = None

		self.basic_time = datetime.datetime.now()
		self.active_slides = {0 : {'start_ts' : self.basic_time - datetime.timedelta(seconds=5), 'end_ts' : self.basic_time},
							  1 : {'start_ts' : self.basic_time , 'end_ts' : self.basic_time + datetime.timedelta(seconds=5)}
							 }

		n_samples = 100
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), 
						    columns=self.features)
		
		
		## first we need to generate noise_models
		noise_models = noise_model_fitting(data, 
										self.causal_graph, 
										m_samples,
										self.target_node, 
										self.sorted_nodes)

		
		num_noise_samples = 150
		## next we generate noise_samples and node_samples based on the generated noise_models
		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
							            self.causal_graph, 
							            self.target_node, 
							            self.sorted_nodes, 
							            num_noise_samples)
		
		
		## now we can train outlier_scorer using node_samples
		outlier_scorer = MedianCDFQuantileScorer()
		outlier_scorer.fit(node_samples[self.target_node])

		self.snoise_models = {0 : noise_models}
		self.snoise_samples = {0 : noise_samples}	
		self.soutlier_scorer = {0 : outlier_scorer}				 

	def test_find_outlier_root_causes(self):
		outlier = {'values' : np.random.randint(1000,10000,size=(1, len(self.sorted_nodes))),
		           'event_ts' : self.basic_time + datetime.timedelta(seconds=1)}
		
		outlier_ts, contributions_dict = find_outlier_root_causes(outlier, 
                             self.snoise_models,
                             self.snoise_samples,
                             self.soutlier_scorer, 
                             self.causal_graph, 
                             self.active_slides, 
                             self.sorted_nodes,
                             self.target_node,
                             self.shapley_config,
                             self.attribute_mean_deviation
                             )
		self.assertEqual(outlier_ts, outlier['event_ts'])
		self.assertIn('X0', contributions_dict)
		self.assertIn('X1', contributions_dict)
		self.assertIn('X2', contributions_dict)
		self.assertIn('X3', contributions_dict)
		self.assertIn('X5', contributions_dict)
		print(contributions_dict)

	def test_find_outliers_root_causes_paralel(self):
		outliers = [{ 'values' : np.random.randint(1000,10000,size=(1, len(self.sorted_nodes))),
		           	  'event_ts' : self.basic_time + datetime.timedelta(seconds=1)},
		            { 'values' : np.random.randint(1000,10000,size=(1, len(self.sorted_nodes))),
		              'event_ts' : self.basic_time + datetime.timedelta(seconds=2)}]
		n_jobs = -1
		contributions = find_outliers_root_causes_paralel(outliers, 
                             self.snoise_models,
                             self.snoise_samples,
                             self.soutlier_scorer, 
                             self.causal_graph, 
                             self.active_slides, 
                             self.sorted_nodes,
                             self.target_node,
                             self.shapley_config,
                             self.attribute_mean_deviation,
                             n_jobs
                             )
		self.assertEqual(len(contributions), 2)

		outlier_ts = list(contributions.keys())

		self.assertEqual(outlier_ts[0], outliers[0]['event_ts'])
		self.assertEqual(outlier_ts[1], outliers[1]['event_ts'])

		check = dir(self.snoise_models[0]['X0'])
		print(check)
		print(self.snoise_models[0]['X0']._parameters)


if __name__ == '__main__': # pragma: no cover
	unittest.main()
