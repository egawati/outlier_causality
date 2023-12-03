import pandas as pd
import numpy as np 
import networkx as nx

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def get_anomalies_start_time(event_ts, out_event_ts, nodes):
	"""
	event_ts : a list or a tuple
	out_event_ts : a list or a tuple
	dag : nx.DiGraph
	target_node : string
	"""
	outliers_start_time = list()
	for ts in out_event_ts:
		outlier_index = event_ts.index(ts)	
		outliers_start = {}
		for node in nodes:
			outliers_start[node] = outlier_index
		outliers_start_time.append(outliers_start)
	return outliers_start_time


def process_timeseries(event_ts, out_event_ts, dag, target_node):
	"""
	event_ts : a list or a tuple
	out_event_ts : a list or a tuple
	dag : nx.DiGraph
	target_node : string
	"""
	longest_path = nx.dag_longest_path(dag)
	all_nodes = set(list(dag.nodes()))

	target_level = None
	root_node = None
	
	for level, node in enumerate(longest_path):
		if node == target_node:
			target_level = level

	outliers_start_time = list()
	
	for ts in out_event_ts:
		outlier_index = event_ts.index(ts)	
		outliers_start = {}
		if target_level:
			for level, node in enumerate(longest_path):
				outlier_start_time_index = outlier_index - target_level - level
				outliers_start[node] = outlier_start_time_index 
		
		covered_nodes = set(list(outliers_start.keys()))
		uncovered = list(all_nodes - covered_nodes)
		if uncovered:
			for node in uncovered:
				try:
					shortest_path = nx.shortest_path(dag, source=node, target=target_node)
					outliers_start[node] = outlier_index - len(shortest_path) + 1
				except nx.NetworkXNoPath:
					print(f'no shortest_path for {node} to {target_node}')
					outliers_start[node] = outlier_index
		outliers_start_time.append(outliers_start)
	return outliers_start_time