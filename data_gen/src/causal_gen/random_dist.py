import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import networkx as nx

from causal_gen.basic_ts import find_root_children_nodes
from causal_gen.basic_ts import generate_root_data
from causal_gen.basic_ts import generate_child_data

class RandomCausalDataGeneratorTS():
	def __init__(self, causal_graph, 
				noise_dists,
				basic_time, 
				n_data, 
				time_propagation, 
				n_outliers, 
				outlier_root_cause_node, 
				outlier_multiplier=3, 
				outlier_position=None,
				seed=0):
		"""
		Inputs:
			causal_graph : networkx.classes.digraph.DiGraph
			noise_dists : a dictionary of continous dist type : its parameter
						{
			                stats.norm: (),
			                stats.uniform: (),
			                stats.expon: (),
			                stats.beta: (random.uniform(0.5, 2.0), random.uniform(0.5, 2.0))
			            }
			basic_time : timestamp indicating when the root nodes start generating data
			n_data: int, number of data to generate 
			time_propagation: float, time needed to propagate value from an upstream node to downstream node (in seconds)
			n_outliers: number of outlier
			outlier_root_cause_node : node chosen as a root cause 
			outlier_multiplier: the multiplier that set the outlier values far away from the normal values in the root cause node
			outlier_position : the position of the outliers in the data set, by default it set to None
			seed : random seed number
		"""
		self.causal_graph = causal_graph
		self.noise_dists = noise_dists
		self.basic_time = basic_time
		self.n_data = n_data
		self.time_propagation = time_propagation
		self.n_outliers = n_outliers
		self.outlier_root_cause_node = outlier_root_cause_node
		self.outlier_multiplier = outlier_multiplier
		self.outlier_position = outlier_position
		self.node_noise_dists = dict()
		self.seed = seed
		self.sorted_nodes = list(nx.topological_sort(causal_graph))

	def generate_data_with_outliers(self):
		"""
		Generating data based on causal graph, 
		a value of a child node at time t depends on the values of its parents at time t-1
		Outputs:
			node_data : a dictionary of nodes, 
						e.g {'X1' : {'data' : pd.DataFrame, 
		                   			 'start_ts' : timestamp}
		                   	'X2' : {'data' : pd.DataFrame, 
		                   			'start_ts' : timestamp}}
		"""
		node_data = dict()
		root, node_parents = find_root_children_nodes(self.causal_graph)
		random.seed(self.seed)
		if self.outlier_position is None:
			if self.n_outliers:
				outlier_position = random.sample(range(len(self.causal_graph.nodes), self.n_data-len(self.causal_graph.nodes)), self.n_outliers)
				outlier_position = tuple(sorted(outlier_position))
			else:
				outlier_position = ()
		else:
			outlier_position = self.outlier_position
		random.seed(self.seed)
		for node in self.sorted_nodes:
			distribution, dist_args = random.choice(list(self.noise_dists.items()))
			self.node_noise_dists[node] = distribution.name
			if node in root:
				if node == self.outlier_root_cause_node:
					self.generate_root_data_with_outlier(node=node, 
					                        node_data=node_data, 
					                        start_ts=self.basic_time, 
					                        n_data=self.n_data, 
					                        time_propagation=self.time_propagation, 
					                        root_cause=True,
					                        outlier_position=outlier_position,
					                        outlier_multiplier=self.outlier_multiplier,
					                        distribution=distribution,
					                        dist_args=dist_args)
				else:
					generate_root_data(node, node_data, self.basic_time, self.n_data, self.time_propagation, distribution, dist_args)
					node_data[node]['data'][f'{node}_root_cause'] = np.zeros(self.n_data)
			else:
				parents = node_parents[node]
				if node == self.outlier_root_cause_node:
					self.generate_child_data_with_outlier(node, 
		        						 parents, 
		                                 node_data, 
		                                 self.n_data, 
		                                 self.time_propagation, 
		                                 root_cause=True, 
		                                 outlier_position=outlier_position,
		                                 outlier_multiplier=self.outlier_multiplier,
		                                 distribution=distribution,
		                                 dist_args=dist_args)
				else:
					generate_child_data(node, parents, node_data, self.n_data, self.time_propagation, distribution, dist_args)
					node_data[node]['data'][f'{node}_root_cause'] = np.zeros(self.n_data)
		return node_data

	def generate_root_data_with_outlier(self, node, 
	                                   node_data, 
	                                   start_ts, 
	                                   n_data, 
	                                   time_propagation, 
	                                   root_cause=False, 
	                                   outlier_position=(),
	                                   outlier_multiplier=3,
	                                   distribution = None,
	                                   dist_args = None):
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
	    if not root_cause:
	        generate_root_data(node, node_data, start_ts, n_data, time_propagation,distribution, dist_args)
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
	            data_normal = distribution.rvs(*dist_args,size=n_normal).reshape(-1,1)
	            root_cause = np.zeros(n_normal)
	            
	            ts_outlier = stop_ts
	            data_outlier = outlier_multiplier + distribution.rvs(*dist_args, size=1).reshape(-1,1)
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
	            data_normal = distribution.rvs(*dist_args,size=n_normal).reshape(-1,1)
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


	def generate_child_normal_data(self, n_normal, 
		                           start_ts, 
		                           stop_ts, 
		                           time_propagation,
		                           distribution,
		                           dist_args):
		data_normal = distribution.rvs(*dist_args, size=n_normal).reshape(-1,1)
		ts_normal = np.arange(start=start_ts, 
		                      stop =stop_ts, 
		                      step=time_propagation)
		root_cause = np.zeros(n_normal)
		return data_normal, ts_normal, root_cause

	def generate_child_data_with_outlier(self, node, 
		                                 parents, 
		                                 node_data, 
		                                 n_data, 
		                                 time_propagation, 
		                                 root_cause=False, 
		                                 outlier_position=(),
		                                 outlier_multiplier=3,
		                                 distribution = None,
	                                   	 dist_args = None,):
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
		    generate_child_data(node, parents, node_data, n_data, time_propagation, distribution, dist_args)
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
		        data_normal, ts_normal, root_cause = self.generate_child_normal_data(n_normal, 
		                                                                   start_ts, 
		                                                                   stop_ts, 
		                                                                   time_propagation,
		                                                                   distribution,
		                                                                   dist_args) 
		        data_outlier = outlier_multiplier + distribution.rvs(*dist_args, size= 1).reshape(-1,1)
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
		        data_normal, ts_normal, root_cause = self.generate_child_normal_data(n_normal, 
		                                                                   start_ts, 
		                                                                   stop_ts, 
		                                                                   time_propagation,
		                                                                   distribution,
		                                                                   dist_args) 
		        
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