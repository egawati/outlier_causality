import pandas as pd
import numpy as np 
import networkx as nx

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def get_target_level(data, df, path, node_levels, target_node):
	target_level = None
	for level, node in enumerate(path):
		if level == 0: ## the root node
			data[node] = df[node]
		else:
			data[node] = df[node].shift(-level)
		
		if node == target_node:
			target_level = level
		node_levels[node] = target_level
	
	return target_level, data

def process_timeseries_data(df, dag, target_node):
	"""
	df : pandas.DataFrame
	event_ts : a list or a tuple
	out_event_ts : a list or a tuple
	dag : nx.DiGraph
	target_node : string
	"""
	all_nodes = list(nx.topological_sort(dag))
	data = pd.DataFrame(columns=all_nodes)

	longest_path = nx.dag_longest_path(dag)
	target_level = None
	seen = dict()
	
	if target_node in longest_path:
		for level, node in enumerate(longest_path):
			if level == 0: ## the root node
				data[node] = df[node]
			else:
				data[node] = df[node].shift(-level)
			
			if node == target_node:
				target_level = level
			seen[node] = target_level

	# just in case the target_node is not in the longest path
	if target_level is None:
		for node in all_nodes:
			try:
				path = nx.shortest_path(dag, source=node, target=target_node)
				target_level, data = get_target_level(data, df, path, seen, target_node)
			except nx.exception.NetworkXNoPath:
				continue

	## now check the rest of the nodes that have not been seen
	unseen = list(set(all_nodes) - set(seen.keys()))

	for node in unseen:
		if node not in seen:
			try:
				path = nx.shortest_path(dag, source=node, target=target_node)
				len_path = len(path)
				for i, vertex in enumerate(path):
					if vertex not in seen:
						seen[vertex] = i
						diff = len_path - i - 1
						level = target_level - diff
						data[vertex] = df[vertex].shift(-level)
						#print(f'vertex {vertex} is at level {level}')
			except nx.exception.NetworkXNoPath:
				seen[node] = 0
				data[node] = df[node]
	data = data.dropna()
	return data, target_level

def process_timeseries_data2(df, dag, target_node):
	"""
	df : pandas.DataFrame
	event_ts : a list or a tuple
	out_event_ts : a list or a tuple
	dag : nx.DiGraph
	target_node : string
	"""
	nodes = list(df.columns)
	data = pd.DataFrame(columns=nodes)

	longest_path = nx.dag_longest_path(dag)
	target_level = None
	seen = dict()
	
	if target_node in longest_path:
		for level, node in enumerate(longest_path):
			if level == 0: ## the root node
				data[node] = df[node]
			else:
				data[node] = df[node].shift(-level)
			
			if node == target_node:
				target_level = level
			seen[node] = level
	
	## now check for nodes not in the longest path
	unseen = list(set(nodes) - set(seen.keys()))
	
	for node in unseen:
		if node not in seen:
			try:
				path = nx.shortest_path(dag, source=node, target=target_node)
				len_path = len(path)
				for i, vertex in enumerate(path):
					if vertex not in seen:
						seen[vertex] = i
						diff = len_path - i - 1
						level = target_level - diff
						data[vertex] = df[vertex].shift(-level)
						print(f'vertex {vertex} is at level {level}')
			except nx.exception.NetworkXNoPath:
				node_parent = dag.pred[node]
				parent = node_parent.keys()[0]
				parent_level = seen[parent]
				node_level = parent_level + 1 
				data[node] = df[node].shift(-node_level)
	data = data.dropna()
	return data, target_level

def process_timeseries(df, event_ts, out_event_ts, dag, target_node):
	"""
	df : pandas.DataFrame
	event_ts : a list or a tuple
	out_event_ts : a list or a tuple
	dag : nx.DiGraph
	target_node : string
	"""
	data, target_level = process_timeseries_data(df, dag, target_node)
	
	# outliers : list of outlier, an outlier is a dictionary {"values" : ..., "event_ts" : ...}
    #            e.g {"values" : np.array(...), "event_ts" : datetime.datetime} or
    #            {"values" : pd.DataFrame(), "event_ts" : datetime.datetime}
    # we use outlier_index for the evaluation purpose

	outliers = list()
	for ts in out_event_ts:
		outlier_index = event_ts.index(ts)
		new_outlier_index = outlier_index - target_level
		if new_outlier_index >= 0 :
			outliers.append({"values":data.iloc[[new_outlier_index]].reset_index(drop=True), "event_ts": ts})
		else:
			logging.warning('the outlier causal data happened in the previous sliding window')
			outliers.append({"values":data.iloc[[outlier_index]], "event_ts":ts})
	return data, outliers