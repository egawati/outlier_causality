import numpy as np
import pandas as pd

from .nodes_permutation import permute_features

from .noise_data_generation import generate_data_from_noise_samples
from .noise_data_generation import get_target_data_from_noise_arr
from .noise_data_generation import data_dict_to_data_df

from .nodes_permutation import permute_features
from dowhy.gcm.shapley import ShapleyConfig, estimate_shapley_values
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



def node_outlier_contribution_scores(outlier_noises,
                                    noise_samples,
                                    outlier_scorer,
                                    attribute_mean_deviation,
                                    noise_models,
                                    causal_graph,
                                    sorted_nodes,
                                    target_node,
                                    shapley_config = None):
    """
    outlier_noises : np.array,
    noise_samples : dict(),
    outlier_scorer : outlier scoring function,
    attribute_mean_deviation: bool,
    sorted_nodes : list(),
    target_node : str,
    this function is similar to attribute_anomaly_scores in dowhy.gcm.anomaly.attribute_anomaly_scores
    Compute the contribution of each node in order to determine the root causes of each outlier found in the window
    The nodes' contribution is based on the Shapley values. 
    """
    n_outliers = outlier_noises.shape[0]
    if attribute_mean_deviation:
        data_from_noise = generate_data_from_noise_samples(noise_samples, 
                                                         noise_models, 
                                                         causal_graph, 
                                                         target_node, 
                                                         sorted_nodes)
        expectation_of_score = np.mean(outlier_scorer.score(data_from_noise[target_node]))
    else:
        outlier_data = get_target_data_from_noise_arr(outlier_noises, 
                                                     noise_models, 
                                                     sorted_nodes, 
                                                     causal_graph, 
                                                     target_node)
        outlier_scores = [outlier_scorer.score(outlier_data[i]) for i in range(n_outliers)]
    
    noise_df = data_dict_to_data_df(noise_samples, sorted_nodes)
    
    noises = noise_df.to_numpy()
    
    num_players = len(sorted_nodes)
    target_node_index = sorted_nodes.index(target_node)
    def set_function(subset: np.ndarray):
        feature_samples = permute_features(noises, np.arange(0, subset.shape[0])[subset == 0], True)
        result = np.zeros(n_outliers)
        for i in range(n_outliers):
            feature_samples[:, subset == 1] = outlier_noises[i, subset == 1]
            target_data = get_target_data_from_noise_arr(feature_samples, 
                                                     noise_models, 
                                                     sorted_nodes, 
                                                     causal_graph, 
                                                     target_node)

            if attribute_mean_deviation:
                result[i] = np.mean(outlier_scorer.score(target_data)) - expectation_of_score
            else:
                result[i] = np.log(_relative_frequency(outlier_scorer.score(target_data) >= outlier_scores[i]))
        return result

    return estimate_shapley_values(set_function, num_players, shapley_config)

def _relative_frequency(conditions: np.ndarray):
    """
    this function is the same as _relative_frequency in dowhy.gcm.anomaly
    len(conditions) = the total rows in feature_samples (or noise_samples)
    """
    return (np.sum(conditions) + 0.5) / (len(conditions) + 0.5)

### we need to fit outlier scoring function for each slide
### hence we will have another data structure to keep the outlier_scoring_function for each slide
### outlier_scoring_functions = {1: outlier_scoring_function, 2:outlier_scoring_function, ...}
def compute_it_score(outlier_noise, noise_samples, outlier_scoring_function):
	outlier_score = outlier_scoring_function.score(outlier_noise.flatten())
	samples_score = outlier_scoring_function.score(noise_samples.flatten())
	it_score = -np.log((np.sum(samples_score >= outlier_score) + 0.5)/ (noise_samples.shape[0] + 0.5))
	return it_score

### outlier scoring function provided by dowhy library e.g. MedianCDFQuantileScorer can only support univariate data
### TODO : create scoring function that deal with multivariate data