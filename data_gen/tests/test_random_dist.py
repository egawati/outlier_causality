import unittest
import pandas as pd
import numpy as np
import networkx as nx
import datetime

import scipy.stats as stats
import random

from causal_gen.random_dist import  RandomCausalDataGeneratorTS

from causal_gen.basic_ts import find_root_children_nodes
from causal_gen.basic_ts import generate_root_data

from causal_gen.basic_ts_with_outlier import merge_node_data_with_outliers

class TestRandomCausalDataGeneratorTS(unittest.TestCase):
	def setUp(self):
		self.features = ('X1', 'X2', 'X3', 'X4', 'X5')
		self.causal_graph = nx.DiGraph([('X1', 'X2'), 
                          ('X2', 'X3'),
                          ('X3', 'X4'),
                          ('X4', 'X5')])
		self.n_data = 10
		self.basic_time = datetime.datetime.now().timestamp()
		self.time_propagation = 1.0
		self.n_outliers = 1

		self.noise_dists = {
							    stats.norm: (),
							    stats.uniform: (),
							    stats.expon: (),
							    stats.beta: (random.uniform(0.5, 2.0), random.uniform(0.5, 2.0))
							}

	def test_generate_child_normal_data_rand_dist(self):
		out_gen =  RandomCausalDataGeneratorTS(self.causal_graph, 
													self.noise_dists,
													self.basic_time, 
													self.n_data, 
													self.time_propagation, 
													self.n_outliers, 
													outlier_root_cause_node='X1', 
													outlier_multiplier=3, 
													outlier_position=None)
		n_normal = 5
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		data_normal, ts_normal, root_cause = out_gen.generate_child_normal_data(n_normal, 
												start_ts=self.basic_time, 
												stop_ts=self.basic_time + n_normal * self.time_propagation, 
												time_propagation=self.time_propagation,
												distribution = distribution,
												dist_args = dist_args)
		self.assertEqual(data_normal.shape[0], n_normal)
		self.assertEqual(ts_normal.shape[0], n_normal)
		self.assertEqual(root_cause.shape[0], n_normal)
		self.assertEqual(ts_normal[0], self.basic_time)
		self.assertEqual(ts_normal[4], self.basic_time + 4 * self.time_propagation)

	def test_generate_root_data_with_outlier_rand_dist_1(self):
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		node_data = {}
		node = 'X1'
		start_ts = self.basic_time
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		outgen.generate_root_data_with_outlier(node, 
				                        node_data, 
				                        start_ts, 
				                        self.n_data, 
				                        self.time_propagation, 
				                        root_cause=False,
				                        distribution=distribution,
				                        dist_args=dist_args)
		self.assertEqual(node_data['X1']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][0],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][1],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][2],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][3],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][4],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][5],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][6],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][7],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][8],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][9],0)

	def test_generate_root_data_with_outlier_random_dist_2(self):
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		node_data = {}
		node = 'X1'
		start_ts = self.basic_time
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		outgen.generate_root_data_with_outlier(node, 
				                        node_data, 
				                        start_ts, 
				                        self.n_data, 
				                        self.time_propagation, 
				                        root_cause=True, 
				                        outlier_position=(5,),
				                        outlier_multiplier=3,
				                        distribution=distribution,
				                        dist_args=dist_args)
		self.assertEqual(node_data['X1']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][5],1)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][0],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][1],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][2],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][3],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][4],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][6],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][7],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][8],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][9],0)

	def test_generate_root_data_with_outlier_random_dist_3(self):
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		node_data = {}
		node = 'X1'
		start_ts = self.basic_time
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		outgen.generate_root_data_with_outlier(node, 
				                        node_data, 
				                        start_ts, 
				                        self.n_data, 
				                        self.time_propagation, 
				                        root_cause=True, 
				                        outlier_position=(2,8),
				                        outlier_multiplier=3,
				                        distribution = distribution,
				                        dist_args = dist_args)
		self.assertEqual(node_data['X1']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][0],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][1],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][2],1)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][3],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][4],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][5],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][6],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][7],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][8],1)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][9],0)

	def test_generate_root_data_with_outlier_random_dist_4(self):
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		node_data = {}
		node = 'X1'
		start_ts = self.basic_time
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		outgen.generate_root_data_with_outlier(node, 
				                        node_data, 
				                        start_ts, 
				                        self.n_data, 
				                        self.time_propagation, 
				                        root_cause=True, 
				                        outlier_position=(0,),
				                        outlier_multiplier=3,
				                        distribution=distribution,
				                        dist_args=dist_args)
		self.assertEqual(node_data['X1']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][0],1)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][1],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][2],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][3],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][4],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][5],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][6],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][7],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][8],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][9],0)

	def test_generate_root_data_with_outlier_random_dist_5(self):
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		node_data = {}
		node = 'X1'
		start_ts = self.basic_time
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		outgen.generate_root_data_with_outlier(node, 
				                        node_data, 
				                        start_ts, 
				                        self.n_data, 
				                        self.time_propagation, 
				                        root_cause=True, 
				                        outlier_position=(9,),
				                        outlier_multiplier=3,
				                        distribution=distribution,
				                        dist_args=dist_args)
		self.assertEqual(node_data['X1']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][0],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][1],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][2],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][3],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][4],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][5],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][6],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][7],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][8],0)
		self.assertEqual(node_data['X1']['data']['X1_root_cause'][9],1)

	def test_generate_child_data_with_outlier_random_dist_1(self):
		root, node_parents = find_root_children_nodes(self.causal_graph)
		node_data = {}
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		generate_root_data(node='X1', 
		                   node_data=node_data, 
		                   start_ts=self.basic_time, 
		                   n_data=self.n_data, 
		                   time_propagation=self.time_propagation,
		                   distribution = distribution,
		                   dist_args = dist_args)
		
		parents = node_parents['X2']
		node = 'X2'
		outgen.generate_child_data_with_outlier(node, 
	                                 parents, 
	                                 node_data, 
	                                 self.n_data, 
	                                 self.time_propagation, 
	                                 root_cause=False,
	                                 distribution=distribution,
	                                 dist_args=dist_args)
		self.assertEqual(node_data['X2']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][0],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][1],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][2],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][3],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][4],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][5],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][6],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][7],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][8],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][9],0)

	def test_generate_child_data_with_outlier_random_dist_2(self):
		root, node_parents = find_root_children_nodes(self.causal_graph)
		node_data = {}
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		generate_root_data(node='X1', 
		                   node_data=node_data, 
		                   start_ts=self.basic_time, 
		                   n_data=self.n_data, 
		                   time_propagation=self.time_propagation,
		                   distribution = distribution,
		                   dist_args = dist_args)
		
		parents = node_parents['X2']
		node = 'X2'
		outgen.generate_child_data_with_outlier(node, 
	                                 parents, 
	                                 node_data, 
	                                 self.n_data, 
	                                 self.time_propagation, 
	                                 root_cause=True, 
	                                 outlier_position=(5,),
	                                 outlier_multiplier=3,
	                                 distribution=distribution,
	                                 dist_args=dist_args)
		self.assertEqual(node_data['X2']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][0],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][1],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][2],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][3],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][4],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][5],1)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][6],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][7],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][8],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][9],0)

	def test_generate_child_data_with_outlier_random_dist_3(self):
		root, node_parents = find_root_children_nodes(self.causal_graph)
		node_data = {}
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		generate_root_data(node='X1', 
		                   node_data=node_data, 
		                   start_ts=self.basic_time, 
		                   n_data=self.n_data, 
		                   time_propagation=self.time_propagation,
		                   distribution = distribution,
		                   dist_args = dist_args)
		
		parents = node_parents['X2']
		node = 'X2'
		outgen.generate_child_data_with_outlier(node, 
	                                 parents, 
	                                 node_data, 
	                                 self.n_data, 
	                                 self.time_propagation, 
	                                 root_cause=True, 
	                                 outlier_position=(2,8),
	                                 outlier_multiplier=3,
	                                 distribution=distribution,
	                                 dist_args=dist_args)
		self.assertEqual(node_data['X2']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][0],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][1],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][2],1)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][3],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][4],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][5],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][6],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][7],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][8],1)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][9],0)

	def test_generate_child_data_with_outlier_random_dist_4(self):
		root, node_parents = find_root_children_nodes(self.causal_graph)
		node_data = {}
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		generate_root_data(node='X1', 
		                   node_data=node_data, 
		                   start_ts=self.basic_time, 
		                   n_data=self.n_data, 
		                   time_propagation=self.time_propagation,
		                   distribution = distribution,
		                   dist_args = dist_args)
		
		parents = node_parents['X2']
		node = 'X2'
		outgen.generate_child_data_with_outlier(node, 
	                                 parents, 
	                                 node_data, 
	                                 self.n_data, 
	                                 self.time_propagation, 
	                                 root_cause=True, 
	                                 outlier_position=(0,),
	                                 outlier_multiplier=3,
	                                 distribution=distribution,
	                                 dist_args=dist_args)
		self.assertEqual(node_data['X2']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][0],1)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][1],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][2],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][3],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][4],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][5],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][6],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][7],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][8],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][9],0)

	def test_generate_child_data_with_outlier_random_dist_5(self):
		root, node_parents = find_root_children_nodes(self.causal_graph)
		node_data = {}
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												self.n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		distribution, dist_args = random.choice(list(self.noise_dists.items()))
		generate_root_data(node='X1', 
		                   node_data=node_data, 
		                   start_ts=self.basic_time, 
		                   n_data=self.n_data, 
		                   time_propagation=self.time_propagation,
		                   distribution = distribution,
		                   dist_args = dist_args)
		
		parents = node_parents['X2']
		node = 'X2'
		outgen.generate_child_data_with_outlier(node, 
	                                 parents, 
	                                 node_data, 
	                                 self.n_data, 
	                                 self.time_propagation, 
	                                 root_cause=True, 
	                                 outlier_position=(9,),
	                                 outlier_multiplier=3,
	                                 distribution=distribution,
	                                 dist_args=dist_args)
		self.assertEqual(node_data['X2']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][0],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][1],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][2],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][3],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][4],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][5],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][6],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][7],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][8],0)
		self.assertEqual(node_data['X2']['data']['X2_root_cause'][9],1)

	def test_generate_data_with_outliers_random_dist_1(self):
		n_data = 20
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												n_data, 
												self.time_propagation, 
												self.n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=None)
		node_data = outgen.generate_data_with_outliers()
		self.assertEqual(node_data['X1']['data'].shape[0], 20)
		self.assertEqual(node_data['X2']['data'].shape[0], 20)
		self.assertEqual(node_data['X3']['data'].shape[0], 20)
		self.assertEqual(node_data['X4']['data'].shape[0], 20)
		self.assertEqual(node_data['X5']['data'].shape[0], 20)

		self.assertEqual(node_data['X1']['data']['ts'][1], node_data['X2']['data']['ts'][0])
		self.assertEqual(node_data['X2']['data']['ts'][1], node_data['X3']['data']['ts'][0])
		self.assertEqual(node_data['X3']['data']['ts'][1], node_data['X4']['data']['ts'][0])
		self.assertEqual(node_data['X4']['data']['ts'][1], node_data['X5']['data']['ts'][0])
		
		self.assertEqual(node_data['X1']['data']['ts'][4], node_data['X5']['data']['ts'][0])
		self.assertEqual(node_data['X2']['data']['ts'][3], node_data['X5']['data']['ts'][0])
		self.assertEqual(node_data['X3']['data']['ts'][2], node_data['X5']['data']['ts'][0])

		self.assertEqual(node_data['X1']['data']['ts'][4], node_data['X5']['data']['ts'][0])
		#self.assertEqual(node_data['X1']['data']['ts'][8], node_data['X5']['data']['ts'][4])

		self.assertIn(1, node_data['X1']['data']['X1_root_cause'].values)
		self.assertNotIn(1, node_data['X2']['data']['X2_root_cause'].values)
		self.assertNotIn(1, node_data['X3']['data']['X3_root_cause'].values)
		self.assertNotIn(1, node_data['X4']['data']['X4_root_cause'].values)
		self.assertNotIn(1, node_data['X5']['data']['X5_root_cause'].values)

	def test_generate_data_with_outliers_random_dist_2(self):
		n_data = 20
		n_outliers = 2
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												n_data, 
												self.time_propagation, 
												n_outliers, 
												outlier_root_cause_node='X2', 
												outlier_multiplier=3, 
												outlier_position=None)
		node_data = outgen.generate_data_with_outliers()
		self.assertEqual(node_data['X1']['data'].shape[0], 20)
		self.assertEqual(node_data['X2']['data'].shape[0], 20)
		self.assertEqual(node_data['X3']['data'].shape[0], 20)
		self.assertEqual(node_data['X4']['data'].shape[0], 20)
		self.assertEqual(node_data['X5']['data'].shape[0], 20)

		self.assertEqual(node_data['X1']['data']['ts'][1], node_data['X2']['data']['ts'][0])
		self.assertEqual(node_data['X2']['data']['ts'][1], node_data['X3']['data']['ts'][0])
		self.assertEqual(node_data['X3']['data']['ts'][1], node_data['X4']['data']['ts'][0])
		self.assertEqual(node_data['X4']['data']['ts'][1], node_data['X5']['data']['ts'][0])
		
		self.assertEqual(node_data['X1']['data']['ts'][4], node_data['X5']['data']['ts'][0])
		self.assertEqual(node_data['X1']['data']['ts'][8], node_data['X5']['data']['ts'][4])

		self.assertEqual(node_data['X2']['data']['ts'][3], node_data['X5']['data']['ts'][0])
		self.assertEqual(node_data['X3']['data']['ts'][2], node_data['X5']['data']['ts'][0])

		self.assertIn(1, node_data['X2']['data']['X2_root_cause'].values)
		self.assertNotIn(1, node_data['X1']['data']['X1_root_cause'].values)
		self.assertNotIn(1, node_data['X3']['data']['X3_root_cause'].values)
		self.assertNotIn(1, node_data['X4']['data']['X4_root_cause'].values)
		self.assertNotIn(1, node_data['X5']['data']['X5_root_cause'].values)

	def test_generate_data_with_outliers_random_dist_3(self):
		n_data = 20
		n_outliers = 2
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												n_data, 
												self.time_propagation, 
												n_outliers, 
												outlier_root_cause_node='X2', 
												outlier_multiplier=3, 
												outlier_position=(4,10))
		node_data = outgen.generate_data_with_outliers()
		self.assertEqual(node_data['X1']['data'].shape[0], 20)
		self.assertEqual(node_data['X2']['data'].shape[0], 20)
		self.assertEqual(node_data['X3']['data'].shape[0], 20)
		self.assertEqual(node_data['X4']['data'].shape[0], 20)
		self.assertEqual(node_data['X5']['data'].shape[0], 20)

		self.assertEqual(node_data['X1']['data']['ts'][1], node_data['X2']['data']['ts'][0])
		self.assertEqual(node_data['X2']['data']['ts'][1], node_data['X3']['data']['ts'][0])
		self.assertEqual(node_data['X3']['data']['ts'][1], node_data['X4']['data']['ts'][0])
		self.assertEqual(node_data['X4']['data']['ts'][1], node_data['X5']['data']['ts'][0])
		
		self.assertEqual(node_data['X1']['data']['ts'][4], node_data['X5']['data']['ts'][0])
		self.assertEqual(node_data['X1']['data']['ts'][8], node_data['X5']['data']['ts'][4])

		self.assertEqual(node_data['X2']['data']['ts'][3], node_data['X5']['data']['ts'][0])
		self.assertEqual(node_data['X3']['data']['ts'][2], node_data['X5']['data']['ts'][0])

		self.assertIn(1, node_data['X2']['data']['X2_root_cause'].values)
		self.assertNotIn(1, node_data['X1']['data']['X1_root_cause'].values)
		self.assertNotIn(1, node_data['X3']['data']['X3_root_cause'].values)
		self.assertNotIn(1, node_data['X4']['data']['X4_root_cause'].values)
		self.assertNotIn(1, node_data['X5']['data']['X5_root_cause'].values)

	def test_generate_data_with_outliers_random_dist_4(self):
		n_data = 20
		n_outliers = 2
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												n_data, 
												self.time_propagation, 
												n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=(4,10))

		node_data = outgen.generate_data_with_outliers()
		self.assertEqual(node_data['X1']['data'].shape[0], 20)
		self.assertEqual(node_data['X2']['data'].shape[0], 20)
		self.assertEqual(node_data['X3']['data'].shape[0], 20)
		self.assertEqual(node_data['X4']['data'].shape[0], 20)
		self.assertEqual(node_data['X5']['data'].shape[0], 20)

		for i in range(1,20):
			self.assertEqual(node_data['X1']['data']['ts'][i], node_data['X1']['data']['ts'][0] + i *self.time_propagation)

		self.assertEqual(node_data['X1']['data']['ts'][1], node_data['X2']['data']['ts'][0])
		self.assertEqual(node_data['X2']['data']['ts'][1], node_data['X3']['data']['ts'][0])
		self.assertEqual(node_data['X3']['data']['ts'][1], node_data['X4']['data']['ts'][0])
		self.assertEqual(node_data['X4']['data']['ts'][1], node_data['X5']['data']['ts'][0])
		
		self.assertEqual(node_data['X1']['data']['ts'][4], node_data['X5']['data']['ts'][0])
		self.assertEqual(node_data['X1']['data']['ts'][8], node_data['X5']['data']['ts'][4])

		self.assertEqual(node_data['X2']['data']['ts'][3], node_data['X5']['data']['ts'][0])
		self.assertEqual(node_data['X3']['data']['ts'][2], node_data['X5']['data']['ts'][0])

		self.assertIn(1, node_data['X1']['data']['X1_root_cause'].values)
		self.assertNotIn(1, node_data['X2']['data']['X2_root_cause'].values)
		self.assertNotIn(1, node_data['X3']['data']['X3_root_cause'].values)
		self.assertNotIn(1, node_data['X4']['data']['X4_root_cause'].values)
		self.assertNotIn(1, node_data['X5']['data']['X5_root_cause'].values)

		self.assertEqual(node_data['X5']['data']['ts'][0], node_data['X4']['data']['ts'][1])
		self.assertEqual(node_data['X5']['data']['ts'][0], node_data['X3']['data']['ts'][2])
		self.assertEqual(node_data['X5']['data']['ts'][0], node_data['X2']['data']['ts'][3])
		self.assertEqual(node_data['X5']['data']['ts'][0], node_data['X1']['data']['ts'][4])

		self.assertEqual(node_data['X5']['data']['ts'][1], node_data['X4']['data']['ts'][2])
		self.assertEqual(node_data['X5']['data']['ts'][1], node_data['X3']['data']['ts'][3])
		self.assertEqual(node_data['X5']['data']['ts'][1], node_data['X2']['data']['ts'][4])
		self.assertEqual(node_data['X5']['data']['ts'][1], node_data['X1']['data']['ts'][5])

		self.assertEqual(node_data['X5']['data']['ts'][10], node_data['X4']['data']['ts'][11])
		self.assertEqual(node_data['X5']['data']['ts'][10], node_data['X3']['data']['ts'][12])
		self.assertEqual(node_data['X5']['data']['ts'][10], node_data['X2']['data']['ts'][13])
		self.assertEqual(node_data['X5']['data']['ts'][10], node_data['X1']['data']['ts'][14])

	def test_merge_node_data_with_outliers_random_dist_1(self):
		n_data = 20
		n_outliers = 2
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												n_data, 
												self.time_propagation, 
												n_outliers, 
												outlier_root_cause_node='X1', 
												outlier_multiplier=3, 
												outlier_position=(4,10))

		node_data = outgen.generate_data_with_outliers()
		df = merge_node_data_with_outliers(node_data = node_data, 
										  causal_graph = self.causal_graph, 
										  target_node = 'X5',
										  time_propagation = self.time_propagation)
		indices = df.index[df['root_cause_gt'] == 'X1'].tolist()
		self.assertEqual(df.iloc[indices[0]]['ts'], node_data['X1']['data']['ts'][4] + 4*self.time_propagation)
		self.assertEqual(df.iloc[indices[1]]['ts'], node_data['X1']['data']['ts'][10] + 4*self.time_propagation)
		self.assertEqual(df.shape[0], 16)

	def test_merge_node_data_with_outliers_random_dist_2(self):
		n_data = 20
		n_outliers = 2
		outgen =  RandomCausalDataGeneratorTS(self.causal_graph, 
												self.noise_dists,
												self.basic_time, 
												n_data, 
												self.time_propagation, 
												n_outliers, 
												outlier_root_cause_node='X2', 
												outlier_multiplier=3, 
												outlier_position=(4,10))

		node_data = outgen.generate_data_with_outliers()
		df = merge_node_data_with_outliers(node_data = node_data, 
										  causal_graph = self.causal_graph, 
										  target_node = 'X5',
										  time_propagation = self.time_propagation)
		indices = df.index[df['root_cause_gt'] == 'X2'].tolist()
		self.assertEqual(df.iloc[indices[0]]['ts'], node_data['X2']['data']['ts'][4] + 3*self.time_propagation)
		self.assertEqual(df.iloc[indices[1]]['ts'], node_data['X2']['data']['ts'][10] + 3*self.time_propagation)
		self.assertEqual(df.shape[0], 16)
	