import unittest
import pandas as pd
import numpy as np
import networkx as nx
import datetime

from causal_gen.basic_ts import find_root_children_nodes
from causal_gen.basic_ts import generate_root_data
from causal_gen.basic_ts import generate_child_data
from causal_gen.basic_ts import generate_data
from causal_gen.basic_ts import merge_node_data

class TestBasicTest(unittest.TestCase):
	def setUp(self):
		self.features = ('X1', 'X2', 'X3', 'X4', 'X5')
		self.causal_graph = nx.DiGraph([('X1', 'X2'), 
			                          ('X2', 'X3'),
			                          ('X3', 'X4'),
			                          ('X4', 'X5')])
		self.n_data = 100
		self.basic_time = datetime.datetime.now().timestamp()
		self.time_propagation = 1.0

	def test_find_root_children_nodes(self):
		root, parents = find_root_children_nodes(self.causal_graph)
		self.assertEqual(len(root), 1)
		self.assertEqual(root[0], self.features[0])
		self.assertEqual(len(parents.keys()), len(self.features)-1)

	def test_generate_root_data(self):
		node_data = {}
		generate_root_data(node='X1', 
		                   node_data=node_data, 
		                   start_ts=self.basic_time, 
		                   n_data=self.n_data, 
		                   time_propagation=self.time_propagation)
		self.assertEqual(node_data['X1']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X1']['data'].shape[1], 2)
		self.assertEqual(node_data['X1']['data']['ts'][0], self.basic_time)
		self.assertEqual(node_data['X1']['start_ts'], self.basic_time)

	def test_generate_child_data(self):
		root, node_parents = find_root_children_nodes(self.causal_graph)
		node_data = {}
		generate_root_data(node='X1', 
		                   node_data=node_data, 
		                   start_ts=self.basic_time, 
		                   n_data=self.n_data, 
		                   time_propagation=self.time_propagation)
		
		parents = node_parents['X2']
		generate_child_data('X2', parents, node_data, self.n_data, self.time_propagation)
		self.assertEqual(node_data['X2']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X2']['data'].shape[1], 2)
		self.assertEqual(node_data['X2']['data']['ts'][0], self.basic_time + self.time_propagation)
		self.assertEqual(node_data['X2']['start_ts'], self.basic_time + self.time_propagation)

	def test_generate_data(self):
		node_data = generate_data(self.causal_graph, 
								  self.basic_time, 
								  self.n_data, 
								  self.time_propagation)
		self.assertEqual(node_data['X1']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X1']['data'].shape[1], 2)
		self.assertEqual(node_data['X1']['data']['ts'][0], self.basic_time)
		self.assertEqual(node_data['X1']['start_ts'], self.basic_time)

		self.assertEqual(node_data['X2']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X2']['data'].shape[1], 2)
		self.assertEqual(node_data['X2']['data']['ts'][0], self.basic_time + self.time_propagation)
		self.assertEqual(node_data['X2']['start_ts'], self.basic_time + self.time_propagation)

		self.assertEqual(node_data['X3']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X3']['data'].shape[1], 2)
		self.assertEqual(node_data['X3']['data']['ts'][0], self.basic_time + 2*self.time_propagation)
		self.assertEqual(node_data['X3']['start_ts'], self.basic_time + 2*self.time_propagation)

		self.assertEqual(node_data['X4']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X4']['data'].shape[1], 2)
		self.assertEqual(node_data['X4']['data']['ts'][0], self.basic_time + 3*self.time_propagation)
		self.assertEqual(node_data['X4']['start_ts'], self.basic_time + 3*self.time_propagation)

		self.assertEqual(node_data['X5']['data'].shape[0], self.n_data)
		self.assertEqual(node_data['X5']['data'].shape[1], 2)
		self.assertEqual(node_data['X5']['data']['ts'][0], self.basic_time + 4*self.time_propagation)
		self.assertEqual(node_data['X5']['start_ts'], self.basic_time + 4*self.time_propagation)

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

	def test_merge_node_data(self):
		node_data = generate_data(self.causal_graph, 
								  self.basic_time, 
								  self.n_data, 
								  self.time_propagation)
		df = merge_node_data(node_data, self.causal_graph)
		self.assertEqual(df.shape[0], self.n_data - len(self.features) + 1)
		self.assertEqual(df.shape[1], len(self.features) + 1)
		self.assertIn(self.features[0], df.columns)
		self.assertIn(self.features[1], df.columns)
		self.assertIn(self.features[2], df.columns)
		self.assertIn(self.features[3], df.columns)
		self.assertIn(self.features[4], df.columns)
		self.assertEqual(df['ts'][0], self.basic_time + 4*self.time_propagation)