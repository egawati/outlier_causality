import copy
import datetime
import numpy as np

from dowhy import gcm
from sklearn.metrics import mean_squared_error

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def compute_model_rmse(data, noise_models, causal_graph):
	model_rmse = {}
	for node in noise_models:
		parents = causal_graph.parents[node]
		if parents:
			fcm = noise_models[node]
			y_predict = fcm._prediction_model.predict(X=data[parents].to_numpy())
			model_rmse[node] = abs(mean_squared_error(data[node].to_numpy(), y_predict, squared=False))
		else:
			model_rmse[node] = noise_models[node]
	return model_rmse

def check_concept_drift_target_node(data, 
						target_node_model, 
						causal_graph, 
						prev_target_node_rmse, 
						target_node,
						error_threshold_change=0.1):

	parents = causal_graph.parents[target_node]
	fcm = target_node_model
	y_predict = fcm._prediction_model.predict(X=data[parents].to_numpy())
	new_rmse = mean_squared_error(data[target_node].to_numpy(), y_predict, squared=False)
	change = (new_rmse - prev_target_node_rmse)/prev_target_node_rmse
	target_node_rmse_change = abs(change)
	if target_node_rmse_change > error_threshold_change:
		return True 
	return False

def check_concept_drift(data, 
						noise_models, 
						causal_graph, 
						prev_model_rmse, 
						error_threshold_change=0.1):
	model_rmse_change = {}
	for node in noise_models:
		parents = causal_graph.parents[node]
		if parents:
			fcm = noise_models[node]
			y_predict = fcm._prediction_model.predict(X=data[parents].to_numpy())
			new_rmse = mean_squared_error(data[node].to_numpy(), y_predict, squared=False)
			change = (new_rmse - prev_model_rmse[node])/prev_model_rmse[node]
			model_rmse_change[node] = abs(change)
		else:
			new_model = gcm.ScipyDistribution()
			new_model.fit(data[node].to_numpy())
			rmse = None
			if set(new_model._parameters) == set(noise_models[node]._parameters):
				for param in new_model._parameters:
					change = abs((new_model._parameters[param] - noise_models[node]._parameters[param])/noise_models[node]._parameters[param])
					if change > error_threshold_change:
						rmse = new_model
						break
			model_rmse_change[node] = rmse
	return model_rmse_change




