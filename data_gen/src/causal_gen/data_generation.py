import numpy as np
import pandas as pd
import random

import networkx as nx
import scipy.stats as stats
import datetime

from causal_gen.random_dist import  RandomCausalDataGeneratorTS
from causal_gen.random_causal_graph import generate_causal_graph_with_min_leaf_depth

from causal_gen.basic_ts_with_outlier import merge_node_data_with_outliers

def generate_normal_and_outlier_dataset(causal_graph,
										noise_dists,
										basic_time, 
										time_propagation,
										nsets, 
										ndata_perset,
										outlier_percentage_perset,
										outlier_multiplier,
										possible_root_causes,
										target_node,
										seed):
	"""
	this function is useful when we want to create dataset where outlier root causes can be varied
	"""
	
	node_data_list = list()
	root_causes = list()

	for i in range(nsets):
		if isinstance(ndata_perset, tuple):
			n_data = ndata_perset[i]
		else:
			n_data = ndata_perset

		if isinstance(outlier_percentage_perset, tuple):
			n_outliers = int(outlier_percentage_perset[i] * n_data)
		else:
			n_outliers = int(outlier_percentage_perset * n_data)

		random.seed(i+seed)
		root_cause = random.choice(possible_root_causes)

		outgen =  RandomCausalDataGeneratorTS(causal_graph=causal_graph, 
	                                     	  noise_dists=noise_dists,
	                                      	  basic_time=basic_time, 
	                                      	  n_data=n_data, 
	                                          time_propagation=time_propagation, 
	                                          n_outliers=n_outliers, 
	                                          outlier_root_cause_node=root_cause, 
	                                      	  outlier_multiplier=outlier_multiplier, 
	                                          outlier_position=None,
	                                          seed=seed)

		node_data = outgen.generate_data_with_outliers()
		# print(f'outgen.node_noise_dists {outgen.node_noise_dists}')
		
		node_data_list.append(node_data)
		basic_time = basic_time + (n_data * time_propagation)
		root_causes.append(root_cause)
	return node_data_list, root_causes, outgen.sorted_nodes


def merge_node_data_list(node_data_list, causal_graph, target_node, time_propagation, root_causes):
	all_df = None 
	for i, node_data in enumerate(node_data_list):
		# print(f'Working on merging set {i} where root_causes is {root_causes[i]}') 
		df = merge_node_data_with_outliers(node_data = node_data_list[i], 
	                                  	   causal_graph = causal_graph, 
	                                       target_node = target_node,
	                                       time_propagation = time_propagation)
		if all_df is None:
			all_df = df 
		else:
			all_df = pd.concat([all_df, df], ignore_index=True)
	return all_df


def generate_dataset_with_random_causal_graph(num_nodes=10, 
											  min_leaf_depth=5, 
	                                          random_seed=1,
	                                          noise_dists=None,
	                                          nsets=2, 
	                                          ndata_perset=(120,120),
	                                          outlier_percentage_perset=(0.1,0.1),
	                                          outlier_multiplier=3,
	                                          time_propagation=1.0
	                                          ):
	random_dag, root_leaf_paths = generate_causal_graph_with_min_leaf_depth(num_nodes=num_nodes, 
	                                                                        min_leaf_depth=min_leaf_depth, 
	                                                                        random_seed=random_seed)

	max_depth = root_leaf_paths[0][0]
	max_path = root_leaf_paths[0][1]

	possible_root_causes = max_path[:-2]
	target_node = max_path[-1]
	# print(f'target_node is {target_node}')

	if noise_dists is None:
	    noise_dists = {
		                stats.norm: (),
		                stats.uniform: (),
		                stats.expon: (),
		                stats.beta: (random.uniform(0.5, 2.0), random.uniform(0.5, 2.0))
		              }

	basic_time = datetime.datetime.now().timestamp()


	node_data_list, root_causes, sorted_nodes =  generate_normal_and_outlier_dataset(causal_graph=random_dag,
										noise_dists=noise_dists,
										basic_time=basic_time, 
										time_propagation=time_propagation,
										nsets=nsets, 
										ndata_perset=ndata_perset,
										outlier_percentage_perset=outlier_percentage_perset,
										outlier_multiplier=outlier_multiplier,
										possible_root_causes=possible_root_causes,
										target_node=target_node,
										seed=random_seed)

	all_df = merge_node_data_list(node_data_list = node_data_list, 
	                              causal_graph = random_dag, 
	                              target_node = target_node,
	                              time_propagation = time_propagation,
	                              root_causes = root_causes)

	return random_dag, sorted_nodes, target_node, all_df

if __name__ == '__main__':
    causal_graph, sorted_nodes, target_node, all_df = generate_dataset_with_random_causal_graph(num_nodes=10, 
											  min_leaf_depth=5, 
                                              random_seed=0,
                                              noise_dists=None,
                                              nsets=2, 
                                              ndata_perset=(120,120),
                                              outlier_percentage_perset=(0.1,0.1),
                                              outlier_multiplier=3,
                                              time_propagation=1.0
                                              )
