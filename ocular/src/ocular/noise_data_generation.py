import copy
import datetime
import numpy as np
import pandas as pd
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def get_target_data_from_noise_arr(noises, 
								   noise_models, 
								   sorted_nodes, 
								   causal_graph, 
								   target_node):
	"""
	noises : np.array
	sorted_nodes : list
	"""
	noise_df = pd.DataFrame(data=noises, columns=sorted_nodes)
	data = pd.DataFrame(np.empty((noise_df.shape[0], len(sorted_nodes))), columns=sorted_nodes)
	for node in sorted_nodes:
	    parents = causal_graph.parents[node]
	    if not parents:
	        data[node] = noise_df[node].to_numpy()
	    else:
	        data[node] = noise_models[node].evaluate(data[parents].to_numpy(), noise_df[node].to_numpy())
	return data[target_node]


def generate_noisedf_from_data(data, 
							 noise_models, 
							 causal_graph, 
							 sorted_nodes, 
							 target_node):
	"""
	data : pd.DataFrame,
	noise_models : dict(),
	causal_graph : ocular.causal_model.dag.dag,
	all_ancestors_of_node : list()
	target_node : str,
	"""
	noise_df = pd.DataFrame(np.empty((data.shape[0], len(sorted_nodes))), 
						    columns=sorted_nodes)
	for node in sorted_nodes:
	    parents = causal_graph.parents[node]
	    ## if a node is a root
	    if not parents:
	        noise_df[node] = data[node].to_numpy()
	    else:
	    	noise_df[node] = noise_models[node].estimate_noise(data[node].to_numpy(), data[parents].to_numpy())
	    if node == target_node:
	        break
	return noise_df

def generate_noises_from_data(data, 
							 noise_models, 
							 causal_graph, 
							 sorted_nodes, 
							 target_node):
	"""
	data : pd.DataFrame,
	noise_models : dict(),
	causal_graph : ocular.causal_model.dag.dag,
	all_ancestors_of_node : list()
	target_node : str,
	"""
	noises = {}
	for node in sorted_nodes:
	    parents = causal_graph.parents[node]
	    ## if a node is a root
	    if not parents:
	        noises[node] = data[node].to_numpy()
	    else:
	    	noises[node] = noise_models[node].estimate_noise(data[node].to_numpy(), data[parents].to_numpy())
	return noises

def data_dict_to_data_df(data_dict, sorted_nodes):
	data_df = pd.DataFrame(columns=sorted_nodes)
	for node in sorted_nodes:
		data_df[node] = data_dict[node].reshape(-1)
	return data_df

def generate_noise_and_node_samples(noise_models, 
	        causal_graph, 
	        target_node, 
	        sorted_nodes, 
	        num_noise_samples = 1500):

	noise_samples = {}
	node_samples = {}

	for node in sorted_nodes:
	    parents = causal_graph.parents[node]
	    
	    ## if a node is a root
	    if not parents:
	        ## initialize model for root node and fit the distribution
	        noise = noise_models[node].draw_samples(num_noise_samples)
	        noise_samples[node] = noise
	        node_samples[node] = noise
	    else:
	        ## generate noise sample
	        noise = noise_models[node].draw_noise_samples(num_noise_samples)
	        noise_samples[node] = noise
	        
	        parent_node_samples = None 
	        for parent in parents:
	            if parent_node_samples is None:
	                parent_node_samples = node_samples[parent]
	            else:
	            	parent_node_samples = np.hstack((parent_node_samples, node_samples[parent]))

	        ## generate node sample
	        node_samples[node] = noise_models[node].evaluate(parent_node_samples, noise)
	return noise_samples, node_samples   

def generate_data_from_noise_samples(noise_samples, 
								     noise_models, 
								     causal_graph, 
								     target_node, 
								     sorted_nodes):
	"""
	we will need node_samples[target_node] when we generate outlier score for ShapleyValue
	"""
	node_samples = {}
	for node in sorted_nodes:
	    parents = causal_graph.parents[node]
	    if not parents:
	        node_samples[node] = noise_samples[node]
	    else:
	        noise = noise_samples[node]
	        parent_node_samples = None 
	        for parent in parents:
	            if parent_node_samples is None:
	                parent_node_samples = node_samples[parent]
	            else:
	            	parent_node_samples = np.hstack((parent_node_samples, node_samples[parent]))
	        ## generate node sample
	        node_samples[node] = noise_models[node].evaluate(parent_node_samples, noise)

	    if node == target_node:
	        break
	return node_samples

def generate_data_from_noise(noise_samples, 
							 noise_models, 
							 causal_graph, 
							 target_node, 
							 sorted_nodes):
	"""
	we will need node_samples[target_node] when we generate outlier score for ShapleyValue
	"""
	node_samples = {}
	for node in sorted_nodes:
	    parents = causal_graph.parents[node]
	    if not parents:
	        node_samples[node] = noise_samples[node]
	    else:
	        noise = noise_samples[node]
	        parent_node_samples = None 
	        for parent in parents:
	            if parent_node_samples is None:
	                parent_node_samples = node_samples[parent]
	            else:
	            	parent_node_samples = np.hstack((parent_node_samples, node_samples[parent]))
	        ## generate node sample
	        node_samples[node] = noise_models[node].evaluate(parent_node_samples, noise)
	return node_samples

