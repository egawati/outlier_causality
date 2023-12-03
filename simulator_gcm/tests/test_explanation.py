import networkx as nx
from dowhy import gcm

import unittest
import datetime
import pandas as pd
import numpy as np
from scipy.stats import halfnorm

from gcm_simulator.explanation import run_gcm_explain_outlier
from gcm_simulator.explanation import ExplainerGCMMP

from ocular_simulator.sliding_window import SlidingWindow 

import random

from multiprocessing import Process
from multiprocessing import Manager

class TestExplanation(unittest.TestCase):
	def setUp(self):
		"""
		When a setUp() method is defined, the test runner will run that method prior to each test. 
		"""
		nodes = [('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.target_node = 'X5'
		causal_graph = nx.DiGraph(nodes)
		self.causal_model = gcm.StructuralCausalModel(causal_graph)
		for node in causal_graph.nodes:
		    if len(list(causal_graph.predecessors(node))) > 0: 
		        self.causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
		    else:
		        ### when the node has no parent
		        self.causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(halfnorm))
		
		
		self.slide_size = 2
		self.window_size = 4
		self.max_slide = self.window_size/self.slide_size

		self.manager = Manager()

		self.slide_size = 2
		self.window_size = 4
		self.msg_size = 1024
		self.msg_format = 'utf-8'
		self.max_slide = self.window_size/self.slide_size

		self.basic_time = datetime.datetime.now().timestamp()

	def test_run_gcm_explain_outlier_4(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 24

		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		data_list = df.values.tolist()
		data = list()
		rate = 0.5
		last_index = 0
		for i, datum in enumerate(data_list):
			data.append({'values' : datum, 'event_ts' : self.basic_time + i*rate, 'index':i})
			last_index = i+1
		slide = {}
		slide['data_list'] = data
		swindow.add_new_slide_with_number(slide=slide, start_ts=self.basic_time, slide_number=slide_number)
		ex_window_queue.put(swindow)
		ex_window_queue.put(None)

		out_index = 7
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)

		m_samples = 0.75

		run_gcm_explain_outlier(self.window_size, 
								self.slide_size, 
								slide_number,
								self.causal_model, 
								ex_window_queue,
								explainer_queue,
								outlier_queue,
								start_ts=self.basic_time,
								features=self.features,
								target_node=self.target_node)
		self.assertEqual(explainer_queue.empty(),False)
		self.assertIsInstance(explainer_queue.get(), tuple)
		self.assertIsNone(explainer_queue.get())

	def test_run_gcm_explain_outlier_process(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 12

		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		data_list = df.values.tolist()
		data = list()
		rate = 0.5
		last_index = 0
		for i, datum in enumerate(data_list):
			data.append({'values' : datum, 'event_ts' : self.basic_time + i*rate, 'index':i})
			last_index = i+1
		slide = {}
		slide['data_list'] = data
		swindow.add_new_slide_with_number(slide=slide, start_ts=self.basic_time, slide_number=slide_number)
		ex_window_queue.put(swindow)
		ex_window_queue.put(None)

		out_index = 7
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)

		m_samples = 0.75

		explainer = Process(target = run_gcm_explain_outlier, args =(self.window_size, 
																	self.slide_size, 
																	slide_number,
																	self.causal_model, 
																	ex_window_queue,
																	explainer_queue,
																	outlier_queue,
																	self.basic_time,
																	self.features,
																	self.target_node))
		explainer.start()
		explainer.join()
		self.assertEqual(explainer_queue.empty(),False)
		self.assertIsInstance(explainer_queue.get(), tuple)
		self.assertIsNone(explainer_queue.get())

	def test_explainer_gcmmp(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 12

		df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		data_list = df.values.tolist()
		data = list()
		rate = 0.5
		last_index = 0
		for i, datum in enumerate(data_list):
			data.append({'values' : datum, 'event_ts' : self.basic_time + i*rate, 'index':i})
			last_index = i+1
		slide = {}
		slide['data_list'] = data
		swindow.add_new_slide_with_number(slide=slide, start_ts=self.basic_time, slide_number=slide_number)
		ex_window_queue.put(swindow)
		ex_window_queue.put(None)
		
		out_index = 7
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)

		m_samples = 0.75

		explainer = ExplainerGCMMP(self.window_size, 
								self.slide_size, 
								slide_number,
								self.causal_model, 
								ex_window_queue,
								explainer_queue,
								outlier_queue,
								self.basic_time,
								self.features,
								self.target_node)
		explainer.start()
		explainer.join()
		self.assertEqual(explainer_queue.empty(),False)
		median_attribs_list, uncertainty_attribs_list, outlier_index = explainer_queue.get()
		self.assertEqual(outlier_index,out_index)
		self.assertIsNone(explainer_queue.get())
		
if __name__ == '__main__': # pragma: no cover
    unittest.main()