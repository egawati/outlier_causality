import copy
import datetime
import numpy as np

from .concept_drift_detector import compute_model_rmse
from .concept_drift_detector import check_concept_drift
from .concept_drift_detector import check_concept_drift_target_node

from .noise_data_generation import generate_data_from_noise_samples

from .noise_data_generation import generate_noise_and_node_samples

from sklearn.metrics import mean_squared_error

from dowhy.gcm import MedianCDFQuantileScorer

from dowhy import gcm

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def noise_model_and_samples_generation(data, 
									   causal_graph, 
									   m_samples, 
									   target_node, 
									   sorted_nodes,
									   active_fcms,
									   max_fcm_number,
									   snoise_models,
									   smodel_rmse,
									   start_ts,
									   slide_size,
									   snoise_samples,
									   soutlier_scorers,
									   num_noise_samples = 1500,
									   error_threshold_change=0.1,
									   outlier_scorer = 'default',
									   dist_type = None,
									   ):
	latest_fcm_number = max(active_fcms.keys())
	latest_noise_models = snoise_models[latest_fcm_number]
	
	latest_model_rmse = smodel_rmse[latest_fcm_number]
	drift = check_concept_drift_target_node(data=data, 
							target_node_model=snoise_models[latest_fcm_number][target_node], 
							causal_graph=causal_graph, 
							prev_target_node_rmse=smodel_rmse[latest_fcm_number][target_node], 
							target_node=target_node,
							error_threshold_change=error_threshold_change)

	if not drift:
		active_fcms[latest_fcm_number]['end_ts'] = start_ts + slide_size
		print('NO DRIFT')
	else:
		print('DRIFT HAPPENS')
		active_fcms[latest_fcm_number+1] = {'start_ts' : start_ts, 'end_ts' : start_ts + slide_size}
		noise_models = noise_model_fitting(data=data, 
										    causal_graph=causal_graph, 
										    m_samples=m_samples, 
										    target_node=target_node, 
										    sorted_nodes=sorted_nodes,
										    dist_type=dist_type)
		snoise_models[latest_fcm_number+1] = noise_models
		noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
                                                causal_graph, 
                                                target_node, 
                                                sorted_nodes, 
                                                num_noise_samples)
		snoise_samples[latest_fcm_number+1] = noise_samples

		if outlier_scorer == 'default':
			outlier_scorer = MedianCDFQuantileScorer()
		
		outlier_scorer.fit(node_samples[target_node])
		soutlier_scorers[latest_fcm_number+1] = outlier_scorer

		model_rmse = compute_model_rmse(data, noise_models, causal_graph)
		smodel_rmse[latest_fcm_number+1] = model_rmse

	if len(active_fcms) > max_fcm_number:
		oldest_fcm_number = min(active_fcms.keys())
		del active_fcms[oldest_fcm_number]
		del snoise_models[oldest_fcm_number]
		del soutlier_scorers[oldest_fcm_number]
		del smodel_rmse[oldest_fcm_number]

	return active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse

def noise_model_generation(data, 
						   causal_graph, 
						   m_samples, 
						   target_node, 
						   sorted_nodes,
						   active_fcms,
						   max_fcm_number,
						   snoise_models,
						   smodel_rmse,
						   start_ts,
						   slide_size,
						   error_threshold_change=0.1,
						   ):
	latest_fcm_number = max(active_fcms.keys())
	latest_noise_models = snoise_models[latest_fcm_number]
	latest_model_rmse = smodel_rmse[latest_fcm_number]

	model_rmse_change = check_concept_drift(data=data, 
											noise_models=latest_noise_models, 
											causal_graph=causal_graph, 
											prev_model_rmse=latest_model_rmse, 
											error_threshold_change=error_threshold_change)

	if m_samples <= 1:
	    m_samples = int(m_samples * data.shape[0])

	data_sample = data.iloc[np.random.choice(
				                data.shape[0],
				                m_samples,
				                replace=False,
				            )]
	noise_models = {}
	covered = set()
	for node in sorted_nodes:
		parents = causal_graph.parents[node]
		if parents and model_rmse_change[node] > error_threshold_change:
			noise_models[node] = gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor())
			noise_models[node].fit(X=data_sample[parents].to_numpy(), Y=data_sample[node].to_numpy())
			covered.add(node)
		elif not parents and model_rmse_change[node] is not None:
			noise_models[node] = model_rmse_change[node]

	## if nothing to update then active fcms will keep the latest_fcm_number
	## yet we need to update its corresponding 'end_ts'
	if not noise_models: 
		active_fcms[latest_fcm_number]['end_ts'] = start_ts + slide_size
	else:
		### noise_models is not empty, hence we need to update active_fcms
		uncovered = set(sorted_nodes) - covered
		for node in uncovered:
			noise_models[node] = copy.deepcopy(snoise_models[latest_fcm_number][node])
		new_fcm_number = latest_fcm_number + 1

		active_fcms[new_fcm_number] = {'start_ts' : start_ts, 'end_ts' : start_ts + slide_size}
		snoise_models[new_fcm_number] = noise_models

	
	if (len(active_fcms.keys()) > max_fcm_number):
		oldest_fcm_number = min(active_fcms.keys())
		del active_fcms[oldest_fcm_number]
		del snoise_models[oldest_fcm_number]

	return active_fcms, snoise_models

			
def noise_model_fitting(data, causal_graph, m_samples, target_node, sorted_nodes, dist_type=None):
	"""
	data : pd.DataFrame
	"""
	models = {} #data structure used to store noise model of each node
	if m_samples <= 1:
	    m_samples = int(m_samples * data.shape[0])

	data_sample = data.iloc[np.random.choice(
				                data.shape[0],
				                m_samples,
				                replace=False,
				            )]
	
	## do it based on the sorted_nodes
	for node in sorted_nodes:
	    parents = causal_graph.parents[node]
	    ## if a node is a root
	    if not parents:
	        ## initialize model for root node and fit the distribution
	        if not dist_type:
	        	models[node] = gcm.ScipyDistribution()
	        else:
	        	models[node] = gcm.ScipyDistribution(dist_type)
	        X = data_sample[node].to_numpy()
	        print(X.shape)
	        models[node].fit(X=X)
	    else:
	        #logging.info(f'at node {node} with parents {parents}')
	        fm_model = gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor())
	        fm_model.fit(X=data_sample[parents].to_numpy(), Y=data_sample[node].to_numpy())
	        
	        ## set the first model
	        models[node] = fm_model 
	return models