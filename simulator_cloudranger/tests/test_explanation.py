import networkx as nx
from dowhy import gcm

import unittest
import datetime
import pandas as pd
import numpy as np
from scipy.stats import halfnorm

from cloudranger_simulator.explanation import run_cloudranger_explain_outlier
from cloudranger_simulator.explanation import ExplainercloudrangerMP

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
		self.causal_graph = nx.DiGraph(nodes)
		
		self.slide_size = 8
		self.window_size = 16
		self.max_slide = self.window_size/self.slide_size

		self.manager = Manager()

		self.msg_size = 1024
		self.msg_format = 'utf-8'

		self.basic_time = datetime.datetime.now().timestamp()

	def test_explainer_cloudranger(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 1000

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
		
		out_index = 99
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)
		
		run_cloudranger_explain_outlier(self.window_size, 
						   self.slide_size, 
						   slide_number,
						   self.causal_graph, 
						   ex_window_queue,
						   explainer_queue,
						   outlier_queue,
						   start_ts=self.basic_time,
						   features=self.features,
						   target_node=self.target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   gamma_max=1,
						   sig_threshold=0.05
						   )

		root_causes, outlier_index = explainer_queue.get()
		self.assertEqual(outlier_index, out_index)
		self.assertIsInstance(root_causes[0],dict)
		self.assertIn('roots',root_causes[0][0])

	def test_explainer_cloudranger2(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 100

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
		
		out_index = 55
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)
		
		run_cloudranger_explain_outlier(self.window_size, 
						   self.slide_size, 
						   slide_number,
						   self.causal_graph, 
						   ex_window_queue,
						   explainer_queue,
						   outlier_queue,
						   start_ts=self.basic_time,
						   features=self.features,
						   target_node=self.target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   gamma_max=1,
						   sig_threshold=0.05
						   )

		root_causes, outlier_index = explainer_queue.get()
		self.assertEqual(outlier_index, out_index)
		self.assertIsInstance(root_causes[0],dict)
		self.assertIn('roots',root_causes[0][0])

	def test_explainer_cloudranger3(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 100

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
		
		out_index = 18
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)
		
		run_cloudranger_explain_outlier(self.window_size, 
						   self.slide_size, 
						   slide_number,
						   self.causal_graph, 
						   ex_window_queue,
						   explainer_queue,
						   outlier_queue,
						   start_ts=self.basic_time,
						   features=self.features,
						   target_node=self.target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   gamma_max=1,
						   sig_threshold=0.05
						   )

		root_causes, outlier_index = explainer_queue.get()
		self.assertEqual(outlier_index, out_index)
		self.assertIsInstance(root_causes[0],dict)
		self.assertIn('roots',root_causes[0][0])

	def test_explainer_cloudranger4(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 100

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
		
		out_index = 18
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)
		
		explainer = ExplainercloudrangerMP(self.window_size, 
						   self.slide_size, 
						   slide_number,
						   self.causal_graph, 
						   ex_window_queue,
						   explainer_queue,
						   outlier_queue,
						   start_ts=self.basic_time,
						   features=self.features,
						   target_node=self.target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   gamma_max=1,
						   sig_threshold=0.05
						   )
		explainer.start()
		explainer.join()

		root_causes, outlier_index = explainer_queue.get()
		self.assertEqual(outlier_index, out_index)
		self.assertIsInstance(root_causes[0],dict)
		self.assertIn('roots',root_causes[0][0])

	def test_explainer_cloudranger5(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 100

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
		
		explainer = ExplainercloudrangerMP(self.window_size, 
						   self.slide_size, 
						   slide_number,
						   self.causal_graph, 
						   ex_window_queue,
						   explainer_queue,
						   outlier_queue,
						   start_ts=self.basic_time,
						   features=self.features,
						   target_node=self.target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   gamma_max=1,
						   sig_threshold=0.05
						   )
		explainer.start()
		explainer.join()

		root_causes, outlier_index = explainer_queue.get()
		print(f'root_causes {root_causes}')
		self.assertEqual(outlier_index, out_index)
		self.assertIsInstance(root_causes[0],dict)
		self.assertIn('roots',root_causes[0][0])

	def test_explainer_cloudranger5(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 100

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
		
		out_index = 0
		outliers = {'values' : data[out_index]['values'], 'event_ts': (self.basic_time + out_index*rate,), 'index':out_index}
		outlier_queue.put((outliers,slide_number))
		outlier_queue.put(None)
		
		explainer = ExplainercloudrangerMP(self.window_size, 
						   self.slide_size, 
						   slide_number,
						   self.causal_graph, 
						   ex_window_queue,
						   explainer_queue,
						   outlier_queue,
						   start_ts=self.basic_time,
						   features=self.features,
						   target_node=self.target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   gamma_max=1,
						   sig_threshold=0.05
						   )
		explainer.start()
		explainer.join()

		root_causes, outlier_index = explainer_queue.get()
		print(f'root_causes {root_causes}')
		self.assertEqual(outlier_index, out_index)
		self.assertIsInstance(root_causes[0],dict)
		self.assertIn('roots',root_causes[0][0])

		
if __name__ == '__main__': # pragma: no cover
    unittest.main()