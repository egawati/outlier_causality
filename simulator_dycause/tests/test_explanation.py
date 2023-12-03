import networkx as nx
from dowhy import gcm

import unittest
import datetime
import pandas as pd
import numpy as np
from scipy.stats import halfnorm

from causal_gen.basic_ts_with_outlier import generate_data_with_outliers
from causal_gen.basic_ts_with_outlier import merge_node_data_with_outliers

from dycause_simulator.explanation import find_root_causes
from dycause_simulator.explanation import run_dycause_explain_outlier
from dycause_simulator.explanation import ExplainerDyCauseMP

from ocular_simulator.sliding_window import SlidingWindow 

import random

from multiprocessing import Process
from multiprocessing import Manager

class TestExplanation(unittest.TestCase):
	def setUp(self):
		"""
		When a setUp() method is defined, the test runner will run that method prior to each test. 
		"""
		self.features = ('X1', 'X2', 'X3', 'X4', 'X5')
		self.causal_graph = nx.DiGraph([('X1', 'X2'), 
                          ('X2', 'X3'),
                          ('X3', 'X4'),
                          ('X4', 'X5')])
		self.basic_time = datetime.datetime.now().timestamp()
		self.time_propagation = 1.0
		self.target_node = 'X5'
		
		node_data = generate_data_with_outliers(causal_graph=self.causal_graph, 
												basic_time=self.basic_time, 
												n_data=500, 
												time_propagation=self.time_propagation, 
												n_outliers=2, 
												outlier_root_cause_node='X2',
												outlier_multiplier=3,
												outlier_position=(400,))
		
		self.df = merge_node_data_with_outliers(node_data = node_data, 
										  causal_graph = self.causal_graph, 
										  target_node = self.target_node,
										  time_propagation = self.time_propagation)
		self.data = self.df[list(self.features)]

		self.slide_size = 4
		self.window_size = 12
		self.max_slide = self.window_size/self.slide_size

		self.manager = Manager()

	def test_find_root_causes(self):
		outlier_index = 400
		root_causes = find_root_causes(data=self.data.to_numpy(), 
							data_head=self.features, 
							anomaly_start_index=outlier_index,
							target_node = self.target_node,
							causal_graph = self.causal_graph,
							before_length=1,
							after_length=1,
							step = 30,
							lag = 9,
							topk_path = 6,
							auto_threshold_ratio = 0.8,
							mean_method="arithmetic",
							max_path_length=None,
							num_sel_node=1,
					)
		self.assertIsInstance(root_causes, list)
		self.assertNotEqual(len(root_causes), 0)
		self.assertNotEqual(len(root_causes[0]), 0)

	def test_explainer_dycause(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 1000

		
		data_list = self.data.values.tolist()
		data = list()
		rate = 0.008
		last_index = 0

		self.basic_time = datetime.datetime.now().timestamp()
		
		for i, datum in enumerate(data_list):
			data.append({'values' : datum, 'event_ts' : self.basic_time + i*rate, 'index':i})
			last_index = i+1
		
		slide = {}
		slide['data_list'] = data
		swindow.add_new_slide_with_number(slide=slide, start_ts=self.basic_time, slide_number=slide_number)
		ex_window_queue.put(swindow)
		ex_window_queue.put(None)
		
		out_index = 400
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)

		run_dycause_explain_outlier(window_size=self.window_size, 
						   slide_size=self.slide_size, 
						   slide_number=slide_number,
						   causal_graph=self.causal_graph, 
						   ex_window_queue=ex_window_queue,
						   explainer_queue=explainer_queue,
						   outlier_queue=outlier_queue,
						   start_ts=self.basic_time,
						   features=self.features,
						   target_node=self.target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   step = 1,
						   lag = 0.1,
						   topk_path = 6,
						   auto_threshold_ratio = 0.8,
						   mean_method="arithmetic",
						   max_path_length=None,
						   num_sel_node=1
						   )
		self.assertEqual(explainer_queue.empty(),False)
		self.assertIsInstance(explainer_queue.get(), tuple)
		self.assertIsNone(explainer_queue.get())


		
if __name__ == '__main__': # pragma: no cover
    unittest.main()