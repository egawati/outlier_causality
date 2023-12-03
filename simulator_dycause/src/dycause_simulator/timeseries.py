import pandas as pd
import numpy as np 
import networkx as nx

from ocular_simulator.timeseries import process_timeseries_data

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def process_timeseries(df, event_ts, out_event_ts, dag, target_node):
	"""
	df : pandas.DataFrame
	event_ts : a list or a tuple
	out_event_ts : a list or a tuple
	dag : nx.DiGraph
	target_node : string
	"""
	data, target_level = process_timeseries_data(df, dag, target_node)
	
	outlier_indices = list()
	for ts in out_event_ts:
		outlier_index = event_ts.index(ts)
		new_outlier_index = outlier_index - target_level
		if new_outlier_index >= 0 :
			outlier_indices.append(new_outlier_index)
		else:
			logging.warning('the outlier causal data happened in the previous sliding window')
			outlier_indices.append(outlier_index)
	data = data.dropna()
	return data, outlier_indices