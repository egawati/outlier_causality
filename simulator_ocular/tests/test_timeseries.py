import unittest
import datetime
import pandas as pd
import numpy as np
import networkx as nx

from ocular_simulator.timeseries import process_timeseries
from ocular_simulator.timeseries import process_timeseries_data

class TestTimeseries(unittest.TestCase):
	def test_process_timeseries_data(self):
		num_nodes = 10
		features = nodes = [f'X{i}' for i in range(1, num_nodes +1)]
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
		
		data, target_level = process_timeseries_data(df, dag, target_node)
		self.assertEqual(data.shape[0], n_samples-target_level)
		self.assertEqual(target_level, 4)


	def test_process_timeseries_data2(self):
		num_nodes = 15
		features = nodes = [f'X{i}' for i in range(1, num_nodes +1)]
		edges = [('X1', 'X2'),
				('X2', 'X3'),
				('X2', 'X4'),
				('X3', 'X5'),
				('X3', 'X6'),
				('X4', 'X7'),
				('X4', 'X8'),
				('X5', 'X9'),
				('X10', 'X11'),
				('X11', 'X12'),
				('X12', 'X13'),
				('X9', 'X13'),
				('X13', 'X14'),
				('X14', 'X15'),
				]
		target_node = 'X15'
		dag = nx.DiGraph(edges)
		
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 27
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(features))), columns=features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		data, target_level = process_timeseries_data(df, dag, target_node)
		print(f'target_level is {target_level}')
		self.assertEqual(data.shape[0], n_samples-target_level)
		self.assertEqual(target_level, 7)

	def test_process_timeseries(self):
		num_nodes = 10
		features = nodes = [f'X{i}' for i in range(1, num_nodes +1)]
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
		
		self.assertEqual(data.shape[0], 596)
		self.assertEqual(outlier_data[0]['event_ts'], out_event_ts[0])
