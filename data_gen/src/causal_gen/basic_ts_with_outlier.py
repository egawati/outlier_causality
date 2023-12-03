import numpy as np
import pandas as pd
import random
from scipy.stats import truncexpon, halfnorm
import networkx as nx

from causal_gen.basic_ts import find_root_children_nodes
from causal_gen.basic_ts import generate_root_data
from causal_gen.basic_ts import generate_child_data

def merge_node_data_with_outliers(node_data, 
								  causal_graph, 
								  target_node,
								  time_propagation):
	df = node_data[target_node]['data']
	df['label'] = np.zeros(df.shape[0])
	root_cause_gt = np.zeros(df.shape[0])
	root_cause_gt = np.NaN
	df['root_cause_gt'] = root_cause_gt

	for node in causal_graph.nodes:
		if node == target_node:
			pass 
		else:
			node_df = node_data[node]['data']
			all_paths = tuple(nx.all_simple_paths(causal_graph, source=node, target=target_node))
			#print(f'all_paths are {all_paths} for source {node} and target {target_node}')
			if len(all_paths) > 0:
				path = all_paths[0]
				
				## time_diff_multiplier represents the distance from the node to the target node
				time_diff_multiplier = len(path) - 1
				
				## get indices where the node is set as the root cause of an outlier
				indices = node_df.index[node_df[f'{node}_root_cause'] == 1].tolist()
				
				## if the node is set to be the root cause of outliers
				if len(indices) > 0 :
					# print(f'all_paths are {all_paths} for source {node} and target {target_node}')
					# print(f'{indices} of len {len(indices)}')
					for index in indices:
						node_ts = node_df.iloc[index]['ts']
						df['root_cause_gt'] = np.where((df['ts'] == node_ts + time_diff_multiplier * time_propagation), 
													   node, df['root_cause_gt'])
						df['label'] = np.where((df['ts'] == node_ts + time_diff_multiplier * time_propagation), 
													   1, df['label'])
			df = pd.merge(df, node_data[node]['data'], on='ts')
	return df


def generate_data_with_outliers(causal_graph, 
								basic_time, 
								n_data, 
								time_propagation, 
								n_outliers, 
								outlier_root_cause_node,
								outlier_multiplier=3,
								outlier_position=None):
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
	if outlier_position is None:
		outlier_position = random.sample(range(len(causal_graph.nodes), n_data-len(causal_graph.nodes)), n_outliers)
		outlier_position = tuple(sorted(outlier_position))
	for node in causal_graph.nodes:
	    if node in root:
	    	if node == outlier_root_cause_node:
	    		generate_root_data_with_outlier(node=node, 
				                        node_data=node_data, 
				                        start_ts=basic_time, 
				                        n_data=n_data, 
				                        time_propagation=time_propagation, 
				                        root_cause=True,
				                        outlier_position=outlier_position,
				                        outlier_multiplier=outlier_multiplier)
	    	else:
	        	generate_root_data(node, node_data, basic_time, n_data, time_propagation)
	        	node_data[node]['data'][f'{node}_root_cause'] = np.zeros(n_data)
	    else:
	        parents = node_parents[node]
	        if node == outlier_root_cause_node:
	        	generate_child_data_with_outlier(node, 
	        						 parents, 
	                                 node_data, 
	                                 n_data, 
	                                 time_propagation, 
	                                 root_cause=True, 
	                                 outlier_position=outlier_position,
	                                 outlier_multiplier=outlier_multiplier)
	        else:
	        	generate_child_data(node, parents, node_data, n_data, time_propagation)
	        	node_data[node]['data'][f'{node}_root_cause'] = np.zeros(n_data)
	return node_data

def generate_root_data_with_outlier(node, 
                                   node_data, 
                                   start_ts, 
                                   n_data, 
                                   time_propagation, 
                                   root_cause=False, 
                                   outlier_position=(),
                                   outlier_multiplier=3):
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
    init_start_ts = start_ts
    print(f'init_start_ts {init_start_ts}')
    print(f'time_propagation {time_propagation}')
    if not root_cause:
        generate_root_data(node, node_data, start_ts, n_data, time_propagation)
        node_data[node]['data'][f'{node}_root_cause'] = np.zeros(n_data)
    else:
        total_outlier = len(outlier_position)
        last_pos = 0
        datas = list()
        for pos in outlier_position:
            n_normal = pos - last_pos
            stop_ts = start_ts + n_normal * time_propagation
            ts_normal = np.arange(start=start_ts, 
                                  stop= stop_ts, 
                                  step= time_propagation)
            data_normal = truncexpon.rvs(size=n_normal, b=3, scale=0.2).reshape(-1,1)
            root_cause = np.zeros(n_normal)
            
            ts_outlier = stop_ts
            data_outlier = outlier_multiplier + truncexpon.rvs(size=1, b=3, scale=0.2).reshape(-1,1)
            root_cause = np.append(root_cause, 1).reshape(-1,1)
            
            ts = np.append(ts_normal, ts_outlier).reshape(-1,1)
        
            data = np.vstack((data_normal, data_outlier))
            data = np.hstack((data, ts))
            data = np.hstack((data, root_cause))
            datas.append(data)
            
            start_ts = stop_ts + time_propagation
            last_pos = pos + 1
        if last_pos < n_data:
            n_normal = n_data - last_pos
            data_normal = truncexpon.rvs(size=n_normal, b=3, scale=0.2).reshape(-1,1)
            stop_ts = start_ts + n_normal * time_propagation
            ts_normal = np.arange(start=start_ts, 
                                  stop= stop_ts, 
                                  step=time_propagation).reshape(-1,1)
            data = np.hstack((data_normal, ts_normal))
            root_cause = np.zeros(n_normal).reshape(-1,1)
            data = np.hstack((data, root_cause))
            datas.append(data)
        
        all_data = None
        for data in datas:
            if all_data is None:
                all_data = data
            else:
                all_data = np.vstack((all_data, data))
        
        node_data[node] = {'data' : pd.DataFrame(all_data, columns=(node, f'ts', f'{node}_root_cause')), 
         				   'start_ts' : init_start_ts,}


def generate_child_normal_data(n_normal, 
	                           start_ts, 
	                           stop_ts, 
	                           time_propagation):
	data_normal = halfnorm.rvs(size= n_normal, loc=0.5, scale=0.2).reshape(-1,1)
	ts_normal = np.arange(start=start_ts, 
	                      stop =stop_ts, 
	                      step=time_propagation)
	root_cause = np.zeros(n_normal)
	return data_normal, ts_normal, root_cause

def generate_child_data_with_outlier(node, 
	                                 parents, 
	                                 node_data, 
	                                 n_data, 
	                                 time_propagation, 
	                                 root_cause=False, 
	                                 outlier_position=(),
	                                 outlier_multiplier=3):
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
	if not root_cause:
	    generate_child_data(node, parents, node_data, n_data, time_propagation)
	    node_data[node]['data'][f'{node}_root_cause'] = np.zeros(n_data)
	else:
	    last_pos = 0
	    datas = list()
	    
	    parent_start_ts = list()
	    for parent in parents:
	        parent_start_ts.append(node_data[parent]['start_ts'])
	    start_ts = max(parent_start_ts) + time_propagation
	    
	    for pos in outlier_position:
	        n_normal = pos - last_pos
	        stop_ts= start_ts + n_normal * time_propagation
	        data_normal, ts_normal, root_cause = generate_child_normal_data(n_normal, 
	                                                                   start_ts, 
	                                                                   stop_ts, 
	                                                                   time_propagation) 
	        data_outlier = outlier_multiplier + halfnorm.rvs(size= 1, loc=0.5, scale=0.2).reshape(-1,1)
	        ts_outlier = stop_ts
	        
	        root_cause = np.append(root_cause, 1).reshape(-1,1)
	        ts = np.append(ts_normal, ts_outlier).reshape(-1,1)
	    
	        data = np.vstack((data_normal, data_outlier))
	        data = np.hstack((data, ts))
	        data = np.hstack((data, root_cause))
	        datas.append(data)
	        start_ts = stop_ts + time_propagation
	        last_pos = pos + 1
	    
	    if last_pos < n_data:
	        n_normal = n_data - last_pos
	        stop_ts = start_ts + n_normal * time_propagation
	        data_normal, ts_normal, root_cause = generate_child_normal_data(n_normal, 
	                                                                   start_ts, 
	                                                                   stop_ts, 
	                                                                   time_propagation) 
	        
	        data = np.hstack((data_normal, ts_normal.reshape(-1,1)))
	        root_cause = np.zeros(n_normal).reshape(-1,1)
	        data = np.hstack((data, root_cause))
	        datas.append(data)
	    
	    all_data = None
	    for data in datas:
	        if all_data is None:
	            all_data = data
	        else:
	            all_data = np.vstack((all_data, data))

	    df = pd.DataFrame(all_data, columns=(node, f'ts', f'{node}_root_cause'))
	    
	    for parent in parents:
	        if parent in node_data.keys():
	            df[node] += node_data[parent]['data'][parent]
	        else:
	            print(f'parent {parent} of node {node} has no data')

	    node_data[node] = {'data' : df, 'start_ts' : max(parent_start_ts) + time_propagation}