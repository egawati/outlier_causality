import unittest
import datetime
import pandas as pd
import numpy as np
import networkx as nx

from dycause_simulator.timeseries import process_timeseries

class TestTimeseries(unittest.TestCase):
	def setUp(self):
		self.edges = [('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.target_node = 'X5'
		self.dag = nx.DiGraph(self.edges)

	def test_process_timeseries(self):
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 12
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		out_index = (5,)
		out_event_ts = [event_ts[i] for i in out_index]

		outliers_start_time = process_timeseries(event_ts, out_event_ts, self.dag, self.target_node)
		self.assertEqual(len(outliers_start_time), 1)
		self.assertEqual(outliers_start_time[0]['X0'], basic_time)

	def test_process_timeseries2(self):
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 12
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		out_index = (7,)
		out_event_ts = [event_ts[i] for i in out_index]

		outliers_start_time = process_timeseries(event_ts, out_event_ts, self.dag, self.target_node)
		self.assertEqual(len(outliers_start_time), 1)
		self.assertEqual(outliers_start_time[0]['X0'], basic_time+2)

	def test_process_timeseries3(self):
		basic_time = datetime.datetime.now().timestamp()
		n_samples = 12
		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		event_ts = [basic_time + i for i in range(n_samples)]
		
		out_index = (11,)
		out_event_ts = [event_ts[i] for i in out_index]

		outliers_start_time = process_timeseries(event_ts, out_event_ts, self.dag, self.target_node)
		self.assertEqual(len(outliers_start_time), 1)
		self.assertEqual(outliers_start_time[0]['X0'], basic_time+6)
	