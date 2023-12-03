import datetime	

from joblib import Parallel, delayed

from .root_causes_finder import find_outliers_root_causes_paralel

from .fcm_generation import noise_model_and_samples_generation

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def explain_outlier_with_shapley(data, 
								 start_ts,
							     slide_size,
							     slide_number,
								 outliers, 
								 active_fcms, 
                                 snoise_models,
                                 snoise_samples,
                                 soutlier_scorers, 
                                 smodel_rmse,
                                 causal_graph,                              
                                 sorted_nodes,
                                 target_node,
                                 m_samples, 
                                 max_fcm_number,
							     num_noise_samples = 1500,
							     error_threshold_change=0.1,
                                 shapley_config=None,
                                 attribute_mean_deviation=False,
                                 n_jobs=-1,
                                 outlier_scorer='default',
                                 dist_type=None):

	#data = S.window[slide_number]['values']
	results = Parallel(n_jobs=2)(
			(delayed(find_outliers_root_causes_paralel)(outliers, 
					                                     snoise_models,
					                                     snoise_samples,
					                                     soutlier_scorers, 
					                                     causal_graph, 
					                                     active_fcms, 
					                                     sorted_nodes,
					                                     target_node,
					                                     shapley_config,
					                                     attribute_mean_deviation,
					                                     n_jobs),
			 delayed(noise_model_and_samples_generation)(data, 
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
													   num_noise_samples,
													   error_threshold_change,
													   outlier_scorer,
													   dist_type)
			 ))

	if isinstance(results[0], dict):
		active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = results[1]
		contributions = results[0]
	else:
		active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse = results[0]
		contributions = results[1]
		
	return contributions, active_fcms, snoise_models, snoise_samples, soutlier_scorers, smodel_rmse