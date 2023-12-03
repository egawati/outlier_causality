import unittest

from ocular.fcm_generation import noise_model_fitting

from ocular.noise_data_generation import generate_noise_and_node_samples
from ocular.noise_data_generation import generate_data_from_noise_samples
from ocular.noise_data_generation import generate_noisedf_from_data
from ocular.noise_data_generation import generate_noises_from_data
from ocular.noise_data_generation import data_dict_to_data_df

from ocular.noise_data_generation import get_target_data_from_noise_arr

import copy
import numpy as np
import pandas as pd
import datetime

from ocular.causal_model import dag


class TestModelGeneration2(unittest.TestCase):
	def setUp(self):
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		target_node = 'X5'
		self.causal_graph = dag.CausalGraph(nodes, self.features)
		all_ancestors_of_node = self.causal_graph.ancestors[target_node]
		all_ancestors_of_node.update({target_node})
		self.sorted_nodes = [node for node in self.causal_graph.sorted_nodes if node in all_ancestors_of_node]


	def test_generate_noise_and_node_samples(self):
		n_samples = 100
		target_node = 'X5'
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
											m_samples,
											target_node,
											self.sorted_nodes)

		num_noise_samples = 1500
		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
							            self.causal_graph, 
							            target_node, 
							            self.sorted_nodes,
							            num_noise_samples)
		
		drawn_samples = pd.DataFrame(np.empty((num_noise_samples, len(self.sorted_nodes))), columns=self.sorted_nodes)
		drawn_noise_samples = pd.DataFrame(np.empty((num_noise_samples, len(self.sorted_nodes))), columns=self.sorted_nodes)
		for node in self.sorted_nodes:
			drawn_samples[node] = node_samples[node]
			drawn_noise_samples[node] = noise_samples[node]

		self.assertEqual(drawn_samples.shape[0], num_noise_samples)
		self.assertEqual(drawn_samples.shape[1], len(self.sorted_nodes))

		self.assertEqual(drawn_noise_samples.shape[0], num_noise_samples)
		self.assertEqual(drawn_noise_samples.shape[1], len(self.sorted_nodes))

	def test_generate_data_from_noise_samples(self):
		n_samples = 100
		target_node = 'X5'
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
											m_samples,
											target_node,
											self.sorted_nodes)

		num_noise_samples = 1500
		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
							            self.causal_graph, 
							            target_node, 
							            self.sorted_nodes,
							            num_noise_samples)
		
		
		new_node_samples = generate_data_from_noise_samples(noise_samples, 
								     noise_models, 
								     self.causal_graph, 
								     target_node, 
								     self.sorted_nodes)

		drawn_samples = pd.DataFrame(np.empty((num_noise_samples, len(self.sorted_nodes))), columns=self.sorted_nodes)
		for node in self.sorted_nodes:
			drawn_samples[node] = new_node_samples[node]
		self.assertEqual(drawn_samples.shape[0], num_noise_samples)
		self.assertEqual(drawn_samples.shape[1], len(self.sorted_nodes))

	def test_generate_noisedf_from_data(self):
		n_samples = 100
		target_node = 'X5'
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
											m_samples,
											target_node,
											self.sorted_nodes)

		ndata = pd.DataFrame(np.random.randint(0,100,size=(5, len(self.features))), columns=self.features)
		
		noise_df = generate_noisedf_from_data(ndata, 
								 noise_models, 
								 self.causal_graph, 
								 self.sorted_nodes, 
								 target_node)
		self.assertEqual(noise_df.shape[0], 5)
		self.assertEqual(noise_df.shape[1], len(self.sorted_nodes))

	def test_generate_noises_from_data(self):
		n_samples = 100
		target_node = 'X5'
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
											m_samples,
											target_node,
											self.sorted_nodes)

		ndata = pd.DataFrame(np.random.randint(0,100,size=(5, len(self.features))), columns=self.features)
		noises = generate_noises_from_data(ndata, 
								 noise_models, 
								 self.causal_graph, 
								 self.sorted_nodes, 
								 target_node)
		self.assertEqual(len(list(noises.keys())), len(self.sorted_nodes))
		for node in self.sorted_nodes:
			self.assertEqual(noises[node].shape[0], 5)

	def test_data_dict_to_data_df(self):
		n_samples = 100
		target_node = 'X5'
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
											m_samples,
											target_node,
											self.sorted_nodes)

		ndata = pd.DataFrame(np.random.randint(0,100,size=(5, len(self.features))), columns=self.features)
		noises = generate_noises_from_data(ndata, 
								 noise_models, 
								 self.causal_graph, 
								 self.sorted_nodes, 
								 target_node)
		
		noise_df = data_dict_to_data_df(noises, self.sorted_nodes)
		self.assertEqual(noise_df.shape[0], 5)
		self.assertEqual(noise_df.shape[1], 5)
		self.assertNotEqual(noise_df.shape[0], len(self.features))

	def test_get_target_data_from_noise_arr(self):
		n_samples = 100
		target_node = 'X5'
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
											m_samples,
											target_node,
											self.sorted_nodes)

		ndata = pd.DataFrame(np.random.randint(0,100,size=(10, len(self.features))), columns=self.features)
		noise_samples = generate_noises_from_data(ndata, 
								 noise_models, 
								 self.causal_graph, 
								 self.sorted_nodes, 
								 target_node)
		
		
		noise_df = data_dict_to_data_df(noise_samples, self.sorted_nodes)
		noises = noise_df.to_numpy()

		target_data = get_target_data_from_noise_arr(noises, 
													 noise_models, 
													 self.sorted_nodes, 
													 self.causal_graph, 
													 target_node)
		self.assertEqual(target_data.shape[0], 10)

	def test_get_target_data_from_noise_arr2(self):
		n_samples = 100
		target_node = 'X5'
		m_samples = 0.75
		data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		noise_models = noise_model_fitting(data, 
											self.causal_graph, 
											m_samples,
											target_node,
											self.sorted_nodes)

		ndata = pd.DataFrame(np.random.randint(0,100,size=(10, len(self.features))), columns=self.features)
		noise_samples = generate_noises_from_data(ndata, 
								 noise_models, 
								 self.causal_graph, 
								 self.sorted_nodes, 
								 target_node)
		
		outliers = pd.DataFrame(np.random.randint(300,1000,size=(2,len(self.features))), 
								columns=self.features)
		
		outlier_noises = generate_noisedf_from_data(outliers, 
							 noise_models, 
							 self.causal_graph, 
							 self.sorted_nodes, 
							 target_node)
		
		target_data = get_target_data_from_noise_arr(outlier_noises.to_numpy(), 
													 noise_models, 
													 self.sorted_nodes, 
													 self.causal_graph, 
													 target_node)
		self.assertEqual(target_data.shape[0], 2)

if __name__ == '__main__': # pragma: no cover
	unittest.main()
