import time
import datetime
import pandas as pd
import numpy as np

import networkx as nx


from multiprocessing import Process

from EasyRCA.easyrca import EasyRCA

from .timeseries import process_timeseries
from .timeseries import get_anomalies_start_time

from ocular_simulator.utils import transform_data_list_to_numpy


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

	used_slide_nums = slide_nums[:nslide_used]
	used_slide_nums = sorted(used_slide_nums)
	data = None
	event_ts = None
	for slide_num in used_slide_nums:
		data_list = swindow.window[slide_num]['data_list']
		if data is None:
			data, event_ts, _, index = transform_data_list_to_numpy(data_list)
		else:
			arr_temp, event_ts_slide, _, index = transform_data_list_to_numpy(data_list)
			data = np.vstack((data, arr_temp))
			event_ts = np.hstack((event_ts, event_ts_slide))
	return data, event_ts

def run_easyrca_explain_outlier(window_size, 
						   slide_size, 
						   slide_number,
						   causal_graph, 
						   ex_window_queue,
						   explainer_queue,
						   outlier_queue,
						   start_ts,
						   features,
						   target_node,
						   nslide_used = None,
						   est_time_queue = None,
						   gamma_max=1,
						   sig_threshold=0.05
						   ):
	first_time = True
	max_slide = int(window_size/slide_size) + 1
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
				data, event_ts = get_data_from_swindow(nslide_used, swindow)
				if isinstance(data, np.ndarray):
					data = pd.DataFrame(data, columns=features)
				
				outlier_index = outliers[0]['index'] ## np.array
				out_event_ts = outliers[0]['event_ts'] ## list
				
				if data.shape[0] > 0 and len(out_event_ts) > 0:
					outliers_start_time = process_timeseries(event_ts.tolist(), out_event_ts,  causal_graph, target_node)
					#outliers_start_time = get_anomalies_start_time(event_ts.tolist(), out_event_ts, features)
					start = time.perf_counter()
					root_causes = list()
					for outlier_start_time in outliers_start_time:
						print(f'gamma_max {gamma_max}')
						erca = EasyRCA(causal_graph, 
							           features, 
									   anomalies_start_time=outlier_start_time,
									   anomaly_length=1, 
									   gamma_max=gamma_max, 
									   sig_threshold=sig_threshold)
						erca.run(data)
						root_causes.append(erca.root_causes)
					print(f'root_causes = {root_causes}, outlier_index = {outlier_index}')
					explainer_queue.put((root_causes, outlier_index))
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

class ExplainerEasyRCAMP(Process):
	def __init__(self,
			    window_size, 
			    slide_size, 
			    slide_number,
			    causal_graph, 
			    ex_window_queue,
			    explainer_queue,
			    outlier_queue,
			    start_ts,
			    features,
			    target_node,
			    nslide_used = None,
			    est_time_queue = None,
			    gamma_max=1,
			    sig_threshold=0.05):
		Process.__init__(self)
		self.window_size = window_size
		self.slide_size = slide_size
		self.slide_number = slide_number
		self.causal_graph = causal_graph
		self.ex_window_queue = ex_window_queue
		self.explainer_queue = explainer_queue
		self.outlier_queue = outlier_queue
		self.start_ts = start_ts
		self.features = features
		self.target_node = target_node
		self.nslide_used = nslide_used
		self.est_time_queue = est_time_queue
		self.gamma_max = gamma_max
		self.sig_threshold = sig_threshold

	def run(self): # pragma: no cover
		run_easyrca_explain_outlier(self.window_size, 
							   self.slide_size, 
							   self.slide_number,
							   self.causal_graph,
							   self.ex_window_queue,
							   self.explainer_queue,
							   self.outlier_queue,
							   self.start_ts,
							   self.features,
							   self.target_node,
							   self.nslide_used,
							   self.est_time_queue,
							   self.gamma_max,
							   self.sig_threshold)
		