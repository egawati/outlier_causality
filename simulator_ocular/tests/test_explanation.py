import unittest
import datetime
import pandas as pd
import numpy as np


from ocular_simulator.sliding_window import SlidingWindow
from ocular_simulator.explanation import run_explain_outlier
from ocular_simulator.explanation import ExplainerMP

from ocular.causal_model import dag
from ocular.initialization import scm_initialization
from ocular.concept_drift_detector import compute_model_rmse


import random

from multiprocessing import Process
from multiprocessing import Manager

class TestExplanation(unittest.TestCase):
	def setUp(self):
		"""
		When a setUp() method is defined, the test runner will run that method prior to each test. 
		"""
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.causal_graph = dag.CausalGraph(nodes, self.features)
		self.target_node = 'X5'
		
		self.num_noise_samples = 1500
		self.m_samples = 0.75
		self.error_threshold_change = 0.1
		self.nslide_used = 1
		self.shapley_config = None
		self.attribute_mean_deviation = False
		self.n_jobs= -1
		
		n_samples = 32
		np.random.seed(0)
		init_data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		fm_types = {node : 'LinearModel' for node in self.features}
		noise_types = {node : 'AdditiveNoise' for node in self.features}

		
		self.snoise_models, self.snoise_samples, self.soutlier_scorers, self.sorted_nodes = scm_initialization(init_data, 
                       causal_graph=self.causal_graph, 
                       fm_types=fm_types, 
                       noise_types=noise_types, 
                       m_samples=self.m_samples, 
                       target_node = self.target_node, 
                       outlier_scorer='default',
                       num_noise_samples = 1500)

		model_rmse = compute_model_rmse(init_data, self.snoise_models[0], self.causal_graph)
		self.smodel_rmse = {}
		self.smodel_rmse[0] = model_rmse

		self.manager = Manager()

		self.slide_size = 4
		self.window_size = 8
		self.msg_size = 1024
		self.msg_format = 'utf-8'
		self.max_fcm_number = self.window_size//self.slide_size + 1

		self.basic_time = datetime.datetime.now().timestamp()

		self.active_fcms = {0 : {'start_ts' : self.basic_time - self.window_size, 'end_ts': self.basic_time}}
		

	def test_run_explain_outlier(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 32

		np.random.seed(13)
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

		print(f'at test the address of self.active_fcms is {hex(id(self.active_fcms))}\n')
		
		run_explain_outlier(self.window_size, 
					    self.slide_size, 
					    slide_number,
					    outlier_queue, 
					    ex_window_queue, 
					    explainer_queue,
						self.active_fcms, 
						self.snoise_models,
						self.snoise_samples,
						self.soutlier_scorers, 
						self.smodel_rmse,
						self.causal_graph,                              
						self.sorted_nodes,
						self.target_node,
						self.m_samples, 
						self.max_fcm_number,
						self.nslide_used,
						self.num_noise_samples,
						self.error_threshold_change,
						self.shapley_config,
						self.attribute_mean_deviation,
						self.n_jobs)

		print(f'at test the address of self.active_fcms is {hex(id(self.active_fcms))}\n')
		## need to remember that active_slide, models, outlier_scorers, noise_samples 
		## are modified within parallelized functions 
		## therefore the change won't be seen here since the updated version of those objects are in different memory address
		self.assertNotEqual(2, len(self.active_fcms.keys()))

		contributions, outlier_index = explainer_queue.get()
		self.assertEqual(len(contributions), 1)

	def test_explain_outlier_mp(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 32

		np.random.seed(13)
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

		explainer = Process(target=run_explain_outlier, 
							args=(self.window_size, 
								    self.slide_size, 
								    slide_number,
								    outlier_queue, 
								    ex_window_queue, 
								    explainer_queue,
									self.active_fcms, 
									self.snoise_models,
									self.snoise_samples,
									self.soutlier_scorers, 
									self.smodel_rmse,
									self.causal_graph,                              
									self.sorted_nodes,
									self.target_node,
									self.m_samples, 
									self.max_fcm_number,
									self.nslide_used,
									self.num_noise_samples,
									self.error_threshold_change,
									self.shapley_config,
									self.attribute_mean_deviation,
									self.n_jobs))
		explainer.start()
		explainer.join()

		print(f'at test the address of self.active_fcms is {hex(id(self.active_fcms))}\n')
		## need to remember that active_slide, models, outlier_scorers, noise_samples 
		## are modified within parallelized functions 
		## therefore the change won't be seen here since the updated version of those objects are in different memory address
		self.assertNotEqual(2, len(self.active_fcms.keys()))

		contributions, outlier_index = explainer_queue.get()
		self.assertEqual(len(contributions), 1)

	def test_explainer_mp(self):
		outlier_queue = self.manager.Queue()
		ex_window_queue = self.manager.Queue()
		explainer_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 32

		np.random.seed(13)
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

		explainer = ExplainerMP(self.window_size, 
							    self.slide_size, 
							    slide_number,
							    outlier_queue, 
							    ex_window_queue, 
							    explainer_queue,
								self.active_fcms, 
								self.snoise_models,
								self.snoise_samples,
								self.soutlier_scorers, 
								self.smodel_rmse,
								self.causal_graph,                              
								self.sorted_nodes,
								self.target_node,
								self.m_samples, 
								self.max_fcm_number,
								self.nslide_used,
								self.num_noise_samples,
								self.error_threshold_change,
								self.shapley_config,
								self.attribute_mean_deviation,
								self.n_jobs)
		explainer.start()
		explainer.join()

		print(f'at test the address of self.active_fcms is {hex(id(self.active_fcms))}\n')
		## need to remember that active_slide, models, outlier_scorers, noise_samples 
		## are modified within parallelized functions 
		## therefore the change won't be seen here since the updated version of those objects are in different memory address
		self.assertNotEqual(2, len(explainer.active_fcms.keys()))

		contributions, outlier_index = explainer_queue.get()
		self.assertEqual(len(contributions), 1)
		
if __name__ == '__main__': # pragma: no cover
    unittest.main()