import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

import time
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict

import networkx as nx
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor

from .timeseries import process_timeseries

from ocular_simulator.utils import transform_data_list_to_numpy

from .dycause_lib.Granger_all_code import loop_granger
from .dycause_lib.causal_graph_build import get_segment_split
from .dycause_lib.causal_graph_build import get_ordered_intervals
from .dycause_lib.causal_graph_build import get_overlay_count
from .dycause_lib.causal_graph_build import normalize_by_row, normalize_by_column
from .dycause_lib.ranknode import analyze_root

def find_root_causes(data, 
					data_head, 
					target_node,
					anomaly_start_index,
					causal_graph,
					before_length=0,
					after_length=0,
					step = 1,
					lag = 0.1,
					topk_path = 6,
					auto_threshold_ratio = 0.8,
					mean_method="arithmetic",
					max_path_length=None,
					num_sel_node=1,
					):
	"""
	data = numpy.array of shape Nxd representing normal data, where d = number of variables
	data_head = list of features, list length = d
	anomalous_data = numpy array of shape 1xd, representing an outlier point
	entry_point = int, the index of the target vertex in d (assuming index start from 1)
	step = the minimal step size in Granger causal interval.
	lag = the maximum causal lag in Granger causality test.
	"""
	anomaly_score = 'Not calculated'
	# Select abnormal data
	# print(f'anomaly_start_index {anomaly_start_index}')
	# print(f'before_length {before_length}')
	# print(f'after_length')
	# print(f'data.shape[0]-1 {data.shape[0]-1}')
	local_data = data[
	    max(0, min(anomaly_start_index - before_length, data.shape[0]-1)) : \
	    max(0, min(anomaly_start_index + after_length, data.shape[0]-1)), \
	    :]

	local_length = local_data.shape[0]
	method = "fast_version_3"
	trip = -1
	simu_real = "simu"
	max_segment_len = local_length
	min_segment_len = step
	list_segment_split = get_segment_split(local_length, step)

	local_results = defaultdict(dict)

	def granger_process(x, y):
	    try:
	        ret = loop_granger(
	            local_data,
	            data_head,
	            dir_output,
	            data_head[x],
	            data_head[y],
	            significant_thres,
	            method,
	            trip,
	            lag,
	            step,
	            simu_real,
	            max_segment_len,
	            min_segment_len,
	            verbose=False,
	            return_result=True,
	        )
	    except Exception as e:
	        ret = (None, None, None, None, None)
	    return ret

	# region ThreadPoolExecuter version
	total_thread_num = [len(data_head) * (len(data_head) - 1)]
	thread_results = [0 for i in range(total_thread_num[0])]

	def thread_func(i, x, y):
	    thread_results[i] = granger_process(x, y)
	    if verbose:
	        pbar.update(1)
	    return

	executor = ThreadPoolExecutor(max_workers=3)
	i = 0
	for x_i in range(len(data_head)):
	    for y_i in range(len(data_head)):
	        if x_i == y_i:
	            continue
	        executor.submit(thread_func, i, x_i, y_i)
	        i = i + 1
	executor.shutdown(wait=True)
	i = 0
	for x_i in range(len(data_head)):
	    for y_i in range(len(data_head)):
	        if x_i == y_i:
	            continue
	        (
	            total_time,
	            time_granger,
	            time_adf,
	            array_results_YX,
	            array_results_XY,
	        ) = thread_results[i]
	        if array_results_YX is None and array_results_XY is None:
	            # No intervals found. Maybe loop_granger has a bug or there does not exist an valid interval.
	            ordered_intervals = []
	        else:
	            matrics = [array_results_YX, array_results_XY]
	            ordered_intervals = get_ordered_intervals(
	                matrics, significant_thres, list_segment_split
	            )
	        local_results["%s->%s" % (x_i, y_i)]["intervals"] = ordered_intervals
	        local_results["%s->%s" % (x_i, y_i)]["result_YX"] = array_results_YX
	        local_results["%s->%s" % (x_i, y_i)]["result_XY"] = array_results_XY
	        i = i + 1
	# endregion

	# region Construction impact graph using generated intervals
	# Generate dynamic causal curve between two services by overlaying intervals
	
	if ordered_intervals:
		histogram_sum = defaultdict(int)
		edge = []
		edge_weight = dict()

		for x_i in range(len(data_head)):
		    for y_i in range(len(data_head)):
		        if y_i == x_i:
		            continue
		        key = "{0}->{1}".format(x_i, y_i)
		        intervals = local_results[key]["intervals"]
		        overlay_counts = get_overlay_count(local_length, intervals)
		        histogram_sum[key] = sum(overlay_counts)

		# Make edges from 1 node using comparison and auto-threshold
		# verbose level >=2: print adaptive thresholding info
		#               >=3: create aggre-imgs plotting progress bar

		for x_i in range(len(data_head[:])):
		    bar_data = []
		    for y_i in range(len(data_head)):
		        key = "{0}->{1}".format(x_i, y_i)
		        bar_data.append(histogram_sum[key])
		    
		    bar_data_thres = np.max(bar_data) * auto_threshold_ratio
		    for y_i in range(len(data_head)):
		        if bar_data[y_i] >= bar_data_thres:
		            edge.append((x_i, y_i))
		            edge_weight[(x_i, y_i)] = bar_data[y_i]

		# Make the transition matrix with edge weight estimation
		transition_matrix = np.zeros([data.shape[1], data.shape[1]])
		for key, val in edge_weight.items():
		    x, y = key
		    transition_matrix[x, y] = val
		transition_matrix = normalize_by_column(transition_matrix)
	else:
		transition_matrix = nx.to_numpy_matrix(causal_graph, nodelist=data_head, dtype=int)

	# region backtrace root cause analysis
	entry_point = data_head.index(target_node) + 1
	topk_list = range(1, data.shape[1])
	prkS = [0] * len(topk_list)
	ranked_nodes, new_matrix = analyze_root(
	    transition_matrix,
	    entry_point,
	    local_data,
	    mean_method=mean_method,
	    max_path_length=max_path_length,
	    topk_path=topk_path,
	    prob_thres=0.2,
	    num_sel_node=num_sel_node,
	    use_new_matrix=False,
	    verbose=False,
	)

	root_causes = list()
	for node_val in ranked_nodes:
		node_index = node_val[0] - 1
		root_causes.append(data_head[node_index])
	return root_causes

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

def run_dycause_explain_outlier(window_size, 
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
						   step = 1,
						   lag = 0.1,
						   topk_path = 6,
						   auto_threshold_ratio = 0.8,
						   mean_method="arithmetic",
						   max_path_length=None,
						   num_sel_node=1
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
				normal_data, event_ts = get_data_from_swindow(nslide_used, swindow)
				if isinstance(normal_data, np.ndarray):
					normal_data = pd.DataFrame(normal_data, columns=features)
				
				outlier_index = outliers[0]['index'] ## np.array
				out_event_ts = outliers[0]['event_ts'] ## list
				
				if normal_data.shape[0] > 0 and len(out_event_ts) > 0:
					data, outlier_indices = process_timeseries(normal_data, event_ts.tolist(), out_event_ts, causal_graph, target_node)
					data_head = list(data.columns)
					start = time.perf_counter()
					root_causes = list()
					before_length=0
					after_length=0
					for out_index in outlier_indices:
						root_cause = find_root_causes(data.to_numpy(), 
										data_head, 
										target_node,
										anomaly_start_index=out_index,
										causal_graph=causal_graph,
										before_length=before_length,
										after_length=after_length,
										step=step,
										lag=lag,
										topk_path=topk_path,
										auto_threshold_ratio=auto_threshold_ratio,
										mean_method=mean_method,
										max_path_length=max_path_length,
										num_sel_node=num_sel_node,
										)
						root_causes.append(root_cause)
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

class ExplainerDyCauseMP(Process):
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
			    step = 1,
				lag = 0.1,
				topk_path = 6,
				auto_threshold_ratio = 0.8,
				mean_method="arithmetic",
				max_path_length=None,
				num_sel_node=1):
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
		self.step=step
		self.lag=lag
		self.topk_path=topk_path
		self.auto_threshold_ratio=auto_threshold_ratio
		self.mean_method=mean_method
		self.max_path_length=max_path_length
		self.num_sel_node=num_sel_node
		
	def run(self): # pragma: no cover
		run_dycause_explain_outlier(self.window_size, 
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
							   self.step,
							   self.lag,
							   self.topk_path,
							   self.auto_threshold_ratio,
							   self.mean_method,
							   self.max_path_length,
							   self.num_sel_node)
		