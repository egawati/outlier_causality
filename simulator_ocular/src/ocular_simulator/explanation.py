import time
import datetime
import numpy as np
import pandas as pd

from ocular.explanation_with_shapley import explain_outlier_with_shapley

from ocular.fcm_generation import noise_model_and_samples_generation

from ocular_simulator.utils import transform_data_list_to_numpy
from ocular_simulator.timeseries import process_timeseries

from multiprocessing import Process

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def get_data_from_swindow(nslide_used, swindow):
	"""
	The idea is we can use data in the whole window or only the n recent slide of the window
	"""
	slide_nums = sorted(swindow.window.keys(), reverse=True)
	if not nslide_used:
		nslide_used = len(slide_nums)
	elif len(slide_nums) < nslide_used:
		nslide_used = len(slide_nums)
	print(f'nslide_used is {nslide_used}')
	used_slide_nums = slide_nums[:nslide_used]
	used_slide_nums = sorted(used_slide_nums)
	normal_data = None
	event_ts = None
	for slide_num in used_slide_nums:
		data_list = swindow.window[slide_num]['data_list']
		if normal_data is None:
			normal_data, event_ts, _, index = transform_data_list_to_numpy(data_list)
		else:
			arr_temp, event_ts_slide, _, index = transform_data_list_to_numpy(data_list)
			normal_data = np.vstack((normal_data, arr_temp))
			event_ts = np.hstack((event_ts, event_ts_slide))
	return normal_data, event_ts

def run_explain_outlier(window_size, 
					    slide_size, 
					    slide_number,
					    outlier_queue, 
					    ex_window_queue, 
					    explainer_queue,
						active_fcms, 
						snoise_models,
						snoise_samples,
						soutlier_scorers, 
						smodel_rmse,
						causal_graph,                              
						sorted_nodes,
						target_node,
						m_samples, 
						max_fcm_number,
						nslide_used = 1,
						num_noise_samples = 1500,
						error_threshold_change=0.1,
						shapley_config=None,
						attribute_mean_deviation=False,
						n_jobs=-1,
						est_time_queue = None,
						outlier_scorer = 'default',
						dist_type=None):
	"""
	window_size : int (in seconds)
   	slide_size : int (in seconds)
   	slide_number : int
   	causal_graph : ocular.causal_model.dag.CausalGraph
	"""
	first_time = True
	while(True):
		try:
			swindow = ex_window_queue.get()
			outlier_info = outlier_queue.get() 
		except Exception as e:
			pass
		else:
			if first_time:
				end_ts = datetime.datetime.now().timestamp()
				start_ts = end_ts - window_size
				first_time = False
			
			if outlier_info is None and swindow is None:
				explainer_queue.put(None)
				break
			else:
				normal_data = None
				event_ts = None

				## we want to update models and noise_samples based on the latest slide number
				## the same thing with outlier_noise 
				## yet we want to compute outlier score and find the root causes of an outlier 
				## based on the noise from the previous slide number (latest_slide_num - 1)
				latest_slide_num = max(swindow.window.keys())	
				
				normal_data, event_ts = get_data_from_swindow(nslide_used, swindow)

				if isinstance(normal_data, np.ndarray):
					normal_data = pd.DataFrame(normal_data, columns=causal_graph.features)

				normal_data.to_csv(f'test_nslide3_{slide_number}.csv', index=False)
				#outliers is a dictionary with keys : values, event_ts, and index
				outliers, outlier_slide_number = outlier_info 
				
				outlier_index = outliers['index'] ## np.array
				out_event_ts = outliers['event_ts'] ## list

				normal_data, outlier_data = process_timeseries(normal_data, event_ts.tolist(), out_event_ts, causal_graph.dag, target_node)
				normal_data = normal_data.dropna()

				print(f'at simulator explanation outlier_scorer is {outlier_scorer}')
				
				if normal_data.shape[0] > 0 and len(out_event_ts) > 0:
					print("THERE IS outliers in the slide")
					start = time.perf_counter()
					results = explain_outlier_with_shapley(data=normal_data, 
								 start_ts=start_ts,
							     slide_size= slide_size,
							     slide_number = slide_number,
								 outliers=outlier_data, 
								 active_fcms=active_fcms, 
                                 snoise_models=snoise_models,
                                 snoise_samples=snoise_samples,
                                 soutlier_scorers=soutlier_scorers, 
                                 smodel_rmse=smodel_rmse,
                                 causal_graph=causal_graph,                              
                                 sorted_nodes=sorted_nodes,
                                 target_node=target_node,
                                 m_samples=m_samples, 
                                 max_fcm_number=max_fcm_number,
							     num_noise_samples = num_noise_samples,
							     error_threshold_change= error_threshold_change,
                                 shapley_config=shapley_config,
                                 attribute_mean_deviation=attribute_mean_deviation,
                                 n_jobs=n_jobs,
                                 outlier_scorer = outlier_scorer,
                                 dist_type=dist_type)
					contributions, active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = results
					explainer_queue.put((contributions, outlier_index))
					end = time.perf_counter()
					if est_time_queue is not None:
						est_time_queue.put(end-start)
				elif normal_data.shape[0] > 0:
					print("NO outliers in the slide")
					ds_models = noise_model_and_samples_generation(data=normal_data, 
									   causal_graph=causal_graph, 
									   m_samples=m_samples, 
									   target_node=target_node, 
									   sorted_nodes=sorted_nodes,
									   active_fcms=active_fcms,
									   max_fcm_number=max_fcm_number,
									   snoise_models=snoise_models,
									   smodel_rmse=smodel_rmse,
									   start_ts=start_ts,
									   slide_size=slide_size,
									   snoise_samples=snoise_samples,
									   soutlier_scorers=soutlier_scorers,
									   num_noise_samples = num_noise_samples,
									   error_threshold_change=error_threshold_change,
									   dist_type=dist_type
									   )
					active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = ds_models
			while True:
				if (end_ts + slide_size < datetime.datetime.now().timestamp()):
					break
			end_ts = datetime.datetime.now().timestamp()
			start_ts = end_ts - slide_size
			slide_number += 1
	logging.info(f'Exiting Explainer Process')


class ExplainerMP(Process):
	def __init__(self, window_size, 
					    slide_size, 
					    slide_number,
					    outlier_queue, 
					    ex_window_queue, 
					    explainer_queue,
						active_fcms, 
						snoise_models,
						snoise_samples,
						soutlier_scorers, 
						smodel_rmse,
						causal_graph,                              
						sorted_nodes,
						target_node,
						m_samples, 
						max_fcm_number,
						nslide_used = 1,
						num_noise_samples = 1500,
						error_threshold_change=0.1,
						shapley_config=None,
						attribute_mean_deviation=False,
						n_jobs=-1,
						est_time_queue=None,
						outlier_scorer = 'default',
						dist_type=None):
		Process.__init__(self)
		self.window_size = window_size
		self.slide_size = slide_size
		self.slide_number = slide_number
		self.outlier_queue = outlier_queue 
		self.ex_window_queue = ex_window_queue
		self.explainer_queue = explainer_queue
		self.active_fcms = active_fcms
		self.snoise_models = snoise_models
		self.snoise_samples = snoise_samples
		self.soutlier_scorers = soutlier_scorers
		self.smodel_rmse = smodel_rmse
		self.causal_graph = causal_graph                              
		self.sorted_nodes = sorted_nodes
		self.target_node = target_node
		self.m_samples = m_samples
		self.max_fcm_number = max_fcm_number
		self.nslide_used = nslide_used
		self.num_noise_samples = num_noise_samples
		self.error_threshold_change= error_threshold_change
		self.shapley_config= shapley_config
		self.attribute_mean_deviation = attribute_mean_deviation
		self.n_jobs= n_jobs
		self.est_time_queue = est_time_queue
		self.outlier_scorer = outlier_scorer
		self.dist_type = dist_type

	def run(self): # pragma: no cover
		run_explain_outlier(self.window_size, 
					    self.slide_size, 
					    self.slide_number,
					    self.outlier_queue, 
					    self.ex_window_queue, 
					    self.explainer_queue,
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
						self.n_jobs,
						self.est_time_queue,
						self.outlier_scorer,
						self.dist_type)