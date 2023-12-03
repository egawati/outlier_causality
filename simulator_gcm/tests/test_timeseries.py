import unittest
import datetime
import pandas as pd
import numpy as np
import networkx as nx

from gcm_simulator.timeseries import process_timeseries

class TestTimeseries(unittest.TestCase):
	def test_process_timeseries(self):
		edges = [('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
		features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		target_node = 'X5'
		dag = nx.DiGraph(edges)
		
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 12
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(features))), columns=features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		out_index = (5,)
		out_event_ts = [event_ts[i] for i in out_index]

		data, outlier_data = process_timeseries(df, event_ts, out_event_ts, dag, target_node)
		
		self.assertEqual(data.shape[0], n_samples-len(edges))
		self.assertEqual(df['X0'][0], data['X0'][0])
		self.assertEqual(df['X1'][1], data['X1'][0])
		self.assertEqual(df['X2'][2], data['X2'][0])
		self.assertEqual(df['X3'][3], data['X3'][0])
		self.assertEqual(df['X4'][4], data['X4'][0])
		self.assertEqual(df['X5'][5], data['X5'][0])

		self.assertEqual(outlier_data[0]['X0'][0], data['X0'][0])
		self.assertEqual(outlier_data[0]['X1'][0], data['X1'][0])
		self.assertEqual(outlier_data[0]['X5'][0], data['X5'][0])
		self.assertEqual(outlier_data[0]['X5'][0], df['X5'][5])

	def test_process_timeseries_2(self):
		edges = [('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
		features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		target_node = 'X5'
		dag = nx.DiGraph(edges)
		
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 12
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(features))), columns=features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		out_index = (5,8)
		out_event_ts = [event_ts[i] for i in out_index]

		data, outlier_data = process_timeseries(df, event_ts, out_event_ts, dag, target_node)
		self.assertEqual(outlier_data[0]['X0'][0], data['X0'][0])
		self.assertEqual(outlier_data[0]['X1'][0], data['X1'][0])
		self.assertEqual(outlier_data[0]['X5'][0], data['X5'][0])
		self.assertEqual(outlier_data[0]['X5'][0], df['X5'][5])

		self.assertEqual(outlier_data[1]['X0'][0], data['X0'][3])
		self.assertEqual(outlier_data[1]['X1'][0], data['X1'][3])
		self.assertEqual(outlier_data[1]['X5'][0], data['X5'][3])
		self.assertEqual(outlier_data[1]['X5'][0], df['X5'][8])

	def test_process_timeseries_3(self):
		edges = [('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
		features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		target_node = 'X5'
		dag = nx.DiGraph(edges)
		
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 12
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(features))), columns=features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		out_index = (5,8)
		out_event_ts = [event_ts[i] for i in out_index]

		data, outlier_data = process_timeseries(df, event_ts, out_event_ts, dag, target_node)

	def test_process_timeseries_4(self):
		edges = [('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
		features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		target_node = 'X5'
		dag = nx.DiGraph(edges)
		
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 12
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(features))), columns=features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		out_index = (1,)
		out_event_ts = [event_ts[i] for i in out_index]

		data, outlier_data = process_timeseries(df, event_ts, out_event_ts, dag, target_node)

	def test_process_timeseries_5(self):
		num_nodes = 10
		features = edges = [f'X{i}' for i in range(1, num_nodes +1)]
		edges = [('X1', 'X2'), 
				('X2', 'X3'), 
				('X3', 'X4'), 
				('X4', 'X10'), 
				('X2', 'X6'),
				('X6', 'X7'),
				('X7', 'X8'),
				('X9', 'X5'),
				('X5', 'X10'),
				]
		target_node = 'X10'
		dag = nx.DiGraph(edges)
		
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 600
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(features))), columns=features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		out_index = (598,)
		out_event_ts = [event_ts[i] for i in out_index]

		data, outlier_data = process_timeseries(df, event_ts, out_event_ts, dag, target_node)
		data = data.dropna()
		self.assertEqual(data.shape[0], n_samples-4)
		