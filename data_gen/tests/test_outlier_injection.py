import unittest
import pandas as pd
import numpy as np
import networkx as nx
import datetime

from causal_gen.outlier_injection import get_root_parents_node
from causal_gen.outlier_injection import get_lagged_values
from causal_gen.outlier_injection import get_original_df_stats
from causal_gen.outlier_injection import get_the_longest_path_from_root

from causal_gen.outlier_injection import learn_fcm_of_timeseries

from causal_gen.outlier_injection import inject_an_outlier
from causal_gen.outlier_injection import inject_n_outliers

class TestOutlierInjection(unittest.TestCase):
	def test_get_root_parents_node(self):
		causal_graph = nx.DiGraph([('X1','X2'),
                          ('X2','X3'),
                          ('X3','X4'),
                          ('X5','X4')])
		root, node_parents = get_root_parents_node(causal_graph)
		self.assertEqual(len(root), 2)
		self.assertIn('X1', root)
		self.assertIn('X5', root)

		self.assertEqual(len(node_parents), 3)
		self.assertIn('X1', node_parents['X2'])
		self.assertIn('X2', node_parents['X3'])
		self.assertIn('X3', node_parents['X4'])
		self.assertIn('X5', node_parents['X4'])

	def test_get_lagged_values(self):
		data = {
		    'Y': [1, 2, 3, 4, 5, 6, 7, 8, 9],
		    'X': [0, 1, 2, 3, 4, 5, 6, 7, 8]
		}

		df = pd.DataFrame(data)
		df_lagged, lagged_vars = get_lagged_values(df, target_node = 'Y', parents = ['X'], lag=1)
		self.assertEqual(df_lagged.shape[0],df.shape[0])
		self.assertEqual(len(lagged_vars), 1)
		self.assertIn('X_lag', lagged_vars)

	def test_get_original_df_stats(self):
		data = {
		    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
		    'X2': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], ## 2 * X1(t-1)
		    'X3': [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], ## X2(t-1) + 1
		    'X5': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
		    'X4': [1, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42] ## X3(t-1) + X5(t-1)
		}
		df = pd.DataFrame(data)
		original_df_stats = get_original_df_stats(df)
		self.assertIsInstance(original_df_stats, dict)
		self.assertEqual(len(original_df_stats), 4)
		self.assertIn('mean', original_df_stats)
		self.assertIn('std', original_df_stats)
		self.assertIn('min', original_df_stats)
		self.assertIn('max', original_df_stats)

		self.assertIsInstance(original_df_stats, dict)
		self.assertIsInstance(original_df_stats['mean'], dict)
		self.assertIsInstance(original_df_stats['std'], dict)
		self.assertIsInstance(original_df_stats['min'], dict)
		self.assertIsInstance(original_df_stats['max'], dict)

		self.assertIn('X1', original_df_stats['mean'])
		self.assertIn('X2', original_df_stats['mean'])
		self.assertIn('X3', original_df_stats['mean'])
		self.assertIn('X4', original_df_stats['mean'])
		self.assertIn('X5', original_df_stats['mean'])

	def test_learn_fcm_of_timeseries(self):
		causal_graph = nx.DiGraph([('X1','X2'),
		                          ('X2','X3'),
		                          ('X3','X4'),
		                          ('X5','X4')])

		data = {
		    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
		    'X2': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], ## 2 * X1(t-1)
		    'X3': [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], ## X2(t-1) + 1
		    'X5': [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
		    'X4': [1, 2, 3, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45] ## X3(t-1) + X5(t-1)
		}

		df = pd.DataFrame(data)
		fcm = learn_fcm_of_timeseries(df, causal_graph, lag=1)
		self.assertIsInstance(fcm, dict)
		self.assertEqual(len(fcm), 5)
		self.assertEqual(fcm['X1'], None)
		self.assertEqual(fcm['X5'], None)

	def test_inject_an_outlier(self):
		causal_graph = nx.DiGraph([('X1','X2'),
		                  ('X2','X3'),
		                  ('X3','X4'),
		                  ('X5','X4')])

		data = {
		'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
		'X2': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], ## 2 * X1(t-1)
		'X3': [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], ## X2(t-1) + 1
		'X5': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
		'X4': [1, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42] ## X3(t-1) + X5(t-1)
		}
		df = pd.DataFrame(data)
		original_df = df.copy()

		fcm = learn_fcm_of_timeseries(df, causal_graph, lag=1)
		root, node_parents = get_root_parents_node(causal_graph)
		original_df_stats = get_original_df_stats(df)
		inject_an_outlier(df, 
		                  original_df_stats,
		                  fcm, 
		                  causal_graph,
		                  node_parents,
		                  target_node='X4',
		                  root_cause='X1',
		                  target_node_position=10,
		                  multiplier = 5)

		self.assertEqual(df.at[6, 'X1'], original_df.at[6, 'X1'])
		self.assertEqual(df.at[7, 'X2'], original_df.at[7, 'X2'])
		self.assertEqual(df.at[8, 'X3'], original_df.at[8, 'X3'])
		self.assertEqual(df.at[9, 'X4'], original_df.at[9, 'X4'])

		self.assertNotEqual(df.at[7, 'X1'], original_df.at[7, 'X1'])
		self.assertNotEqual(df.at[8, 'X2'], original_df.at[8, 'X2'])
		self.assertNotEqual(df.at[9, 'X3'], original_df.at[9, 'X3'])
		self.assertNotEqual(df.at[10, 'X4'], original_df.at[10, 'X4'])
		
		self.assertEqual(df.at[8, 'X1'], original_df.at[8, 'X1'])
		self.assertEqual(df.at[9, 'X2'], original_df.at[9, 'X2'])
		self.assertEqual(df.at[10, 'X3'], original_df.at[10, 'X3'])
		self.assertEqual(df.at[11, 'X4'], original_df.at[11, 'X4'])

	def test_get_the_longest_path_from_root(self):
		causal_graph = nx.DiGraph([('X1','X2'),
		                  ('X2','X3'),
		                  ('X3','X4'),
		                  ('X5','X4')])
		roots, node_parents = get_root_parents_node(causal_graph)
		the_longest_path, the_root = get_the_longest_path_from_root(causal_graph, roots, target_node='X4')
		self.assertEqual(len(the_longest_path), 4)
		self.assertEqual(the_root,'X1')


	def test_inject_n_outliers(self):
		features = ('X1', 'X2', 'X3', 'X4', 'X5')
		causal_graph = nx.DiGraph([('X1','X2'),
		                          ('X2','X3'),
		                          ('X3','X4'),
		                          ('X5','X4')])

		data = {
		    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
		    'X2': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], ## 2 * X1(t-1)
		    'X3': [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], ## X2(t-1) + 1
		    'X5': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 
		    'X4': [1, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42] ## X3(t-1) + X5(t-1)
		}
		df = pd.DataFrame(data)
		
		target_outlier_positions, root_causes = inject_n_outliers(df,
								                  causal_graph,
								                  target_node='X4',
								                  n_outliers=2,
								                  multiplier = 5,
								                  lag = 1)
		self.assertEqual(len(target_outlier_positions), 2)
		
		self.assertEqual(df.at[target_outlier_positions[0],'label'], 1)
		self.assertEqual(df.at[target_outlier_positions[0],'root_cause_gt'], root_causes[0])

		self.assertEqual(df.loc[target_outlier_positions[1],'label'], 1)
		self.assertEqual(df.loc[target_outlier_positions[1],'root_cause_gt'], root_causes[1])
		


		

