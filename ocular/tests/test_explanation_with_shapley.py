import unittest
import datetime
import pandas as pd
import numpy as np

from ocular.causal_model import dag

from ocular.initialization import scm_initialization

from ocular.concept_drift_detector import compute_model_rmse

from ocular.explanation_with_shapley import explain_outlier_with_shapley

from ocular_simulator.sliding_window import SlidingWindow # pragma: no cover

class TestExplanationWithShapley(unittest.TestCase):
	def setUp(self):
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.causal_graph = dag.CausalGraph(nodes, self.features)
		self.target_node = 'X5'
		
		n_samples = 100
		np.random.seed(0)
		init_data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		fm_types = {node : 'LinearModel' for node in self.features}
		noise_types = {node : 'AdditiveNoise' for node in self.features}

		self.m_samples = 0.75

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

		basic_time = datetime.datetime.now().timestamp()
		self.slide_size = 2 #in second
		self.window_size = 4 #in second
		self.max_fcm_number = self.window_size//self.slide_size + 1
		self.start_ts = basic_time

		self.models_error_threshold = {node : 0.5 for node in self.features}

		self.basic_time = datetime.datetime.now().timestamp()

		self.active_fcms = {0 : {'start_ts' : self.basic_time - self.slide_size, 'end_ts' : self.basic_time}}
		
		self.assertIn('X0', self.snoise_models[0])
		self.assertIn('X1', self.snoise_models[0])
		self.assertIn('X2', self.snoise_models[0])
		self.assertIn('X3', self.snoise_models[0])
		self.assertIn('X5', self.snoise_models[0])

		self.assertIn(0, self.smodel_rmse)

	def test_explain_outlier_with_shapley(self):
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 100
		np.random.seed(15)
		slide_number = 1
		data1 = pd.DataFrame(np.random.randint(-100,-10,size=(n_samples, len(self.features))), columns=self.features)
		slide = {'values' : data1, 'event_ts' : self.basic_time}
		swindow.add_new_slide_with_number(slide=slide, start_ts=self.basic_time, slide_number=slide_number)

		outlier1 = {'values' : pd.DataFrame(np.random.randint(1000,1005,size=(1, len(self.features))), columns=self.features),
		           'event_ts' : self.basic_time + 0.5}

		outlier2 = {'values' : pd.DataFrame(np.random.randint(899,999,size=(1, len(self.features))), columns=self.features),
		           'event_ts' : self.basic_time + 1}
		
		outliers = [outlier1, outlier2]

		results = explain_outlier_with_shapley(data=swindow.window[slide_number]['values'], 
								 start_ts=self.start_ts,
							     slide_size=self.slide_size,
							     slide_number = slide_number,
								 outliers=outliers, 
								 active_fcms=self.active_fcms, 
                                 snoise_models=self.snoise_models,
                                 snoise_samples=self.snoise_samples,
                                 soutlier_scorers=self.soutlier_scorers, 
                                 smodel_rmse=self.smodel_rmse,
                                 causal_graph=self.causal_graph,                              
                                 sorted_nodes=self.sorted_nodes,
                                 target_node=self.target_node,
                                 m_samples=self.m_samples, 
                                 max_fcm_number=self.max_fcm_number,
							     num_noise_samples = 1500,
							     error_threshold_change=0.1,
                                 shapley_config=None,
                                 attribute_mean_deviation=False,
                                 n_jobs=-1)
		contributions, active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = results
		self.assertEqual(len(contributions),2)

		self.assertIn(0, active_fcms)
		self.assertIn(1, active_fcms)

		self.assertIn(0,snoise_models)
		self.assertIn(1,snoise_models)

		self.assertIn(0,snoise_samples)
		self.assertIn(1,snoise_samples)

		self.assertIn(0, soutlier_scorers)
		self.assertIn(1, soutlier_scorers)

		self.assertIn(0, smodel_rmse)
		self.assertIn(1, smodel_rmse)

	def test_explain_outlier_with_shapley2(self):
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		n_samples = 100
		np.random.seed(0)
		slide_number = 1
		data1 = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		slide = {'values' : data1, 'event_ts' : self.basic_time}
		swindow.add_new_slide_with_number(slide=slide, start_ts=self.basic_time, slide_number=slide_number)

		outlier1 = {'values' : pd.DataFrame(np.random.randint(1000,1005,size=(1, len(self.features))), columns=self.features),
		           'event_ts' : self.basic_time + 0.5}

		outlier2 = {'values' : pd.DataFrame(np.random.randint(899,999,size=(1, len(self.features))), columns=self.features),
		           'event_ts' : self.basic_time + 1}
		
		outliers = [outlier1, outlier2]

		results = explain_outlier_with_shapley(data=swindow.window[slide_number]['values'],
								 start_ts=self.start_ts,
							     slide_size=self.slide_size,
							     slide_number = slide_number,
								 outliers=outliers, 
								 active_fcms=self.active_fcms, 
                                 snoise_models=self.snoise_models,
                                 snoise_samples=self.snoise_samples,
                                 soutlier_scorers=self.soutlier_scorers, 
                                 smodel_rmse=self.smodel_rmse,
                                 causal_graph=self.causal_graph,                              
                                 sorted_nodes=self.sorted_nodes,
                                 target_node=self.target_node,
                                 m_samples=self.m_samples, 
                                 max_fcm_number=self.max_fcm_number,
							     num_noise_samples = 1500,
							     error_threshold_change=0.1,
                                 shapley_config=None,
                                 attribute_mean_deviation=False,
                                 n_jobs=-1)
		contributions, active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = results
		self.assertEqual(len(contributions),2)

		self.assertIn(0, active_fcms)
		self.assertNotIn(1, active_fcms)
		self.assertEqual(active_fcms[0]['end_ts'],self.slide_size+self.start_ts)

		self.assertIn(0,snoise_models)
		self.assertNotIn(1,snoise_models)

		self.assertIn(0,snoise_samples)
		self.assertNotIn(1,snoise_samples)

		self.assertIn(0, soutlier_scorers)
		self.assertNotIn(1, soutlier_scorers)

		self.assertIn(0, smodel_rmse)
		self.assertNotIn(1, smodel_rmse)


