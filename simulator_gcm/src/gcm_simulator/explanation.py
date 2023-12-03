import time
import datetime
import pandas as pd
import numpy as np

import networkx as nx

from dowhy import gcm

from multiprocessing import Process

from ocular_simulator.utils import transform_data_list_to_numpy

from gcm_simulator.timeseries import process_timeseries

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

def run_gcm_explain_outlier(window_size, 
						   slide_size, 
						   slide_number,
						   causal_model, 
						   ex_window_queue,
						   explainer_queue,
						   outlier_queue,
						   start_ts,
						   features,
						   target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   shapley_config = None,
						   num_bootstrap_resamples=10,
						   ):
	first_time = True
	max_slide = int(window_size/slide_size) + 1
	print(f'num_bootstrap_resamples = {num_bootstrap_resamples}')
	while(True):
		try:
			swindow = ex_window_queue.get()
			outliers = outlier_queue.get()
		except Exception as e:
			pass
		else:
			if first_time:
				end_ts = datetime.datetime.now().timestamp()
				start_ts = end_ts - window_size
				first_time = False
			
			if outliers is None and swindow is None:
				explainer_queue.put(None)
				break
			else:
				normal_data, event_ts = get_data_from_swindow(nslide_used, swindow)
				if isinstance(normal_data, np.ndarray):
					normal_data = pd.DataFrame(normal_data, columns=features)
				
				outlier_index = outliers[0]['index'] ## np.array
				out_event_ts = outliers[0]['event_ts'] ## list
				
				if normal_data.shape[0] > 0 and len(out_event_ts) > 0:
					normal_data, outlier_data = process_timeseries(normal_data, 
														event_ts.tolist(), 
														out_event_ts, 
														causal_model.graph, 
														target_node)
					normal_data = normal_data.dropna()
					
					gcm.config.disable_progress_bars() # to disable print statements when computing Shapley values
					start = time.perf_counter()
					median_attribs_list = list()
					uncertainty_attribs_list = list()
					for i in range(len(outlier_data)):
						outlier = pd.DataFrame(outlier_data[i].values.reshape(1,-1), columns=features)
						median_attribs, uncertainty_attribs = gcm.confidence_intervals(
									    gcm.fit_and_compute(gcm.attribute_anomalies,
									                        causal_model,
									                        normal_data,
									                        target_node=target_node,
									                        anomaly_samples=outlier,
									                        shapley_config=shapley_config),
									    num_bootstrap_resamples=num_bootstrap_resamples)
						median_attribs_list.append(median_attribs)
						uncertainty_attribs_list.append(uncertainty_attribs)
					explainer_queue.put((median_attribs_list, uncertainty_attribs_list, outlier_index))
					end = time.perf_counter()
					if est_time_queue is not None:
						est_time_queue.put(end-start)
			while True:
				if (end_ts + slide_size < datetime.datetime.now().timestamp()):
					break
			end_ts = datetime.datetime.now().timestamp()
			start_ts = end_ts - slide_size
			slide_number += 1
	logging.info(f'Exiting Explainer Process')

class ExplainerGCMMP(Process):
	def __init__(self,
				window_size, 
			   	slide_size, 
			   	slide_number,
			   	causal_model, 
			   	ex_window_queue,
			   	explainer_queue,
			   	outlier_queue,
			   	start_ts,
			   	features,
			   	target_node,
			   	nslide_used=None, 
			   	est_time_queue=None,
			   	shapley_config=None,
			   	num_bootstrap_resamples=10):
		Process.__init__(self)
		self.window_size = window_size
		self.slide_size = slide_size
		self.slide_number = slide_number
		self.causal_model = causal_model
		self.ex_window_queue = ex_window_queue
		self.explainer_queue = explainer_queue
		self.outlier_queue = outlier_queue
		self.start_ts = start_ts
		self.features = features
		self.target_node = target_node
		self.nslide_used = nslide_used
		self.est_time_queue = est_time_queue
		self.shapley_config = shapley_config
		self.num_bootstrap_resamples=num_bootstrap_resamples


	def run(self): # pragma: no cover
		run_gcm_explain_outlier(self.window_size, 
							   self.slide_size, 
							   self.slide_number,
							   self.causal_model, 
							   self.ex_window_queue,
							   self.explainer_queue,
							   self.outlier_queue,
							   self.start_ts,
							   self.features,
							   self.target_node,
							   self.nslide_used,
							   self.est_time_queue,
							   self.shapley_config,
							   self.num_bootstrap_resamples)
		