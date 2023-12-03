import unittest
import os
import numpy as np
import pandas as pd

from dowhy import gcm
from dowhy.gcm import MedianCDFQuantileScorer

from ocular.causal_model import dag

from ocular.outlier_score import compute_it_score
from ocular.outlier_score import _relative_frequency
from ocular.outlier_score import node_outlier_contribution_scores

from ocular.noise_data_generation import data_dict_to_data_df

from ocular.noise_data_generation import generate_noisedf_from_data

from ocular.noise_data_generation import generate_noise_and_node_samples

from ocular.noise_data_generation import get_target_data_from_noise_arr

from ocular.fcm_generation import noise_model_fitting

class TestOutlierScore(unittest.TestCase):
	def test_compute_it_score_one_dimension(self):
		data_to_train = np.random.rand(30,1)
		outlier_scoring_function = MedianCDFQuantileScorer()
		outlier_scoring_function.fit(data_to_train)

		samples = np.random.rand(10,1)
		outlier = np.random.rand(1,1)

		self.assertEqual(samples.shape[0], 10)
		self.assertEqual(samples.shape[1], 1)

		self.assertEqual(outlier.shape[0], 1)
		self.assertEqual(outlier.shape[1], 1)

		it_score = compute_it_score(outlier, samples, outlier_scoring_function)
		self.assertIsInstance(it_score, float)

	def test__relative_frequency(self):
		conditions = np.array([1,0,0,0,1,1,1,1,1,0])
		self.assertEqual(len(conditions),10)
		self.assertEqual(np.sum(conditions),6)
		freq = (6+0.5)/(10+0.5)
		r_freq = _relative_frequency(conditions)
		self.assertEqual(freq, r_freq)

	def test_node_outlier_contribution_scores(self):
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		causal_graph = dag.CausalGraph(nodes, features)
		target_node = 'X5'
		
		n_samples = 100
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(features))), 
						    columns=features)
		
		all_ancestors_of_node = causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in causal_graph.sorted_nodes if node in all_ancestors_of_node]

		## first we need to generate noise_models
		noise_models = noise_model_fitting(data, 
										causal_graph, 
										m_samples,
										target_node, 
										sorted_nodes)

		
		num_noise_samples = 150
		## next we generate noise_samples and node_samples based on the generated noise_models
		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
							            causal_graph, 
							            target_node, 
							            sorted_nodes, 
							            num_noise_samples)
		
		
		## now we can train outlier_scorer using node_samples
		outlier_scorer = MedianCDFQuantileScorer()
		outlier_scorer.fit(node_samples[target_node])

		
		## suppose there are two outliers
		outliers = pd.DataFrame(np.random.randint(300,1000,size=(2,len(features))), 
								columns=features)
		

		## outlier_noises can have less columns than outliers since we only care about the nodes that has path to target_node
		outlier_noises = generate_noisedf_from_data(outliers, 
							 noise_models, 
							 causal_graph, 
							 sorted_nodes, 
							 target_node)
		out_noises_arr = outlier_noises.to_numpy()

		results = node_outlier_contribution_scores(outlier_noises=out_noises_arr,
                                    noise_samples=noise_samples,
                                    outlier_scorer=outlier_scorer,
                                    attribute_mean_deviation=False,
                                    noise_models=noise_models,
                                    causal_graph=causal_graph,
                                    sorted_nodes=sorted_nodes,
                                    target_node=target_node,
                                    shapley_config = None)
		self.assertEqual(results.shape[0], 2)
		self.assertEqual(results.shape[1], len(sorted_nodes))

	def test_node_outlier_contribution_scores2(self):
		inlier_filepath = os.path.join(os.path.dirname(__file__), 'inlier.csv')
		self.assertIn('inlier.csv', inlier_filepath)

		gcm.config.disable_progress_bars()
		nodes = [('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
		features = ['X1', 'X2', 'X3', 'X4', 'X5']
		causal_graph = dag.CausalGraph(nodes, features)
		target_node = 'X5'

		data = pd.read_csv(inlier_filepath, sep=',')
		m_samples = 1

		all_ancestors_of_node = causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		sorted_nodes = [node for node in causal_graph.sorted_nodes if node in all_ancestors_of_node]
		print(f'sorted_nodes is {sorted_nodes}')
		## first we need to generate noise_models
		noise_models = noise_model_fitting(data, 
		                                causal_graph, 
		                                m_samples,
		                                target_node, 
		                                sorted_nodes)


		num_noise_samples = 200
		## next we generate noise_samples and node_samples based on the generated noise_models
		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
		                                causal_graph, 
		                                target_node, 
		                                sorted_nodes, 
		                                num_noise_samples)


		## now we can train outlier_scorer using node_samples
		outlier_scorer = MedianCDFQuantileScorer()
		outlier_scorer.fit(node_samples[target_node])

		## suppose there is one outlier
		outlier_filepath = os.path.join(os.path.dirname(__file__), 'outlier.csv')
		self.assertIn('outlier.csv', outlier_filepath)
		outliers = pd.read_csv(outlier_filepath, sep=',')

		## outlier_noises can have less columns than outliers since we only care about the nodes that has path to target_node
		outlier_noises = generate_noisedf_from_data(outliers, 
		                     noise_models, 
		                     causal_graph, 
		                     sorted_nodes, 
		                     target_node)
		out_noises_arr = outlier_noises.to_numpy()

		results = node_outlier_contribution_scores(outlier_noises=out_noises_arr,
		                            noise_samples=noise_samples,
		                            outlier_scorer=outlier_scorer,
		                            attribute_mean_deviation=False,
		                            noise_models=noise_models,
		                            causal_graph=causal_graph,
		                            sorted_nodes=sorted_nodes,
		                            target_node=target_node,
		                            shapley_config = None)

		self.assertGreater(results[0,0], results[0,1])
		
if __name__ == '__main__': # pragma: no cover
    unittest.main()