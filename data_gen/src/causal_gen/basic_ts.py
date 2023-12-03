import numpy as np
import pandas as pd
from scipy.stats import truncexpon, halfnorm

def find_root_children_nodes(causal_graph):
	"""
	Inputs:
		causal_graph : networkx.classes.digraph.DiGraph
	Outputs:
		root : the list of root nodes
		parents : dictionary of nodes and their corresponding parents
	"""
	root = list()
	parents = dict()
	for node in causal_graph.nodes:
	    predecessors = list(causal_graph.predecessors(node))
	    if predecessors:
	        parents[node] = predecessors
	    else:
	        root.append(node)
	return root, parents

def generate_root_data(node, node_data, start_ts, n_data, time_propagation, distribution=None, dist_args=None):
	"""
	Generating data for the root nodes
	Inputs:
		node : string of node name
		node_data: dictionary
		start_ts: float of timestamp 
		n_data: int, number of data to generate 
		time_propagation: float, time needed to propagate value from an upstream node to downstream node (in seconds)
	Outputs
		updated node_data
	"""
	ts = np.arange(start=start_ts, 
	               stop=start_ts + n_data * time_propagation, 
	               step=time_propagation).reshape(-1,1)
	if distribution is None:
		data = truncexpon.rvs(size=n_data, b=3, scale=0.2).reshape(-1,1)
	else:
		data = distribution.rvs(*dist_args, size=n_data).reshape(-1,1)
	data_ts = np.hstack((data, ts))
	node_data[node] = {'data' : pd.DataFrame(data_ts, columns=(node, f'ts')), 
	                   'start_ts' : start_ts,}  

def generate_child_data(node, parents, node_data, n_data, time_propagation, distribution=None, dist_args=None):
	"""
	Generating data for the child nodes
	Inputs:
		node : string of node name
		parents : dictionary of the node and its parent nodes
		node_data: dictionary
		n_data: int, number of data to generate 
		time_propagation: float, time needed to propagate value from an upstream node to downstream node (in seconds)
	Outputs:
		updated node_data
	"""
	if distribution is None:
		data = halfnorm.rvs(size=n_data, loc=0.5, scale=0.2).reshape(-1,1)
	else:
		data = distribution.rvs(*dist_args, size=n_data).reshape(-1,1)
		
	parent_start_ts = list()

	for parent in parents:
	    if parent in node_data.keys():
	        parent_start_ts.append(node_data[parent]['start_ts'])
	    else:
	        print(f'parent {parent} of node {node} has no data')
	        
	start_ts = max(parent_start_ts) + time_propagation
	ts = np.arange(start=start_ts, 
	               stop=start_ts + n_data * time_propagation, 
	               step=time_propagation).reshape(-1,1)

	for parent in parents:
	    if parent in node_data.keys():
	        data += node_data[parent]['data'][parent].values.reshape(-1,1)
	    else:
	        print(f'parent {parent} of node {node} has no data')

	data_ts = np.hstack((data, ts))
	node_data[node] = {'data' : pd.DataFrame(data_ts, columns=(node, f'ts')), 
	                   'start_ts' : start_ts}

def generate_data(causal_graph, basic_time, n_data, time_propagation):
	"""
	Generating data based on causal graph, 
	a value of a child node at time t depends on the values of its parents at time t-1
	Inputs:
		causal_graph : networkx.classes.digraph.DiGraph
		basic_time : timestamp indicating when the root nodes start generating data
		n_data: int, number of data to generate 
		time_propagation: float, time needed to propagate value from an upstream node to downstream node (in seconds)
	Outputs:
		node_data : a dictionary of nodes, 
					e.g {'X1' : {'data' : pd.DataFrame, 
	                   			 'start_ts' : timestamp}
	                   	'X2' : {'data' : pd.DataFrame, 
	                   			'start_ts' : timestamp}}
	"""
	node_data = dict()
	root, node_parents = find_root_children_nodes(causal_graph)
	for node in causal_graph.nodes:
	    if node in root:
	        generate_root_data(node, node_data, basic_time, n_data, time_propagation)
	    else:
	        parents = node_parents[node]
	        generate_child_data(node, parents, node_data, n_data, time_propagation)
	return node_data

def merge_node_data(node_data, causal_graph):
	"""
	Generating data based on causal graph
	a value of a child node at time t depends on the values of its parents at time t-1
	Inputs:
		causal_graph : networkx.classes.digraph.DiGraph
		node_data : a dictionary of nodes, 
					e.g {'X1' : {'data' : pd.DataFrame, 
	                   			 'start_ts' : timestamp}
	                   	'X2' : {'data' : pd.DataFrame, 
	                   			'start_ts' : timestamp}}
	Outputs:
		df : pd.dataframe, columns : node names and timestamp
	"""
	first = True
	for node in causal_graph.nodes:
	    if first:
	        df = node_data[node]['data']
	        first = False
	    else:
	        df = pd.merge(df, node_data[node]['data'], on='ts')
	return df