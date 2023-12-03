import numpy as np

from .utils import sampling_data
from .causal_model.scm import LinearCausalModel

from .fcm_generation import noise_model_fitting
from .noise_data_generation import generate_noise_and_node_samples

from scipy.stats import halfnorm
from dowhy import gcm
from dowhy.gcm import MedianCDFQuantileScorer

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def init_outlier_scorer(data, scorer):
    if scorer == 'default':
        scorer = MedianCDFQuantileScorer()
    scorer.fit(data)
    return scorer


def init_model(fm_type, target, predictor):
    fm_model = None
    if fm_type == 'LinearModel' or fm_type is None:
        fm_model = LinearCausalModel()
        fm_model.fit(X=predictor, Y=target)
    else:
        logging.warning("Not supported functional model")
    return fm_model

def init_linear_noise_samples(fm_model, predictor, target):
    noises = fm_model.predict(predictor) - target
    return noises.reshape(-1,1)

def scm_initialization(init_data, 
                       causal_graph, 
                       fm_types, 
                       noise_types, 
                       m_samples, 
                       target_node = 'X5', 
                       outlier_scorer='default',
                       num_noise_samples = 1500,
                       dist_type=None):
    """
    inputs:
        init_data : dataframe of offline dataset
        causal_graph : dag 
        fm_types : dictionary of functional model type of each node in the DAG
                   for example : {'X1' : 'LinearModel'}
        noise_types : dictionary of noise type of each node in the DAG
                   for example : {'X1' : 'AdditiveNoise'}
        m_samples : number of noise samples to generate
        outlier_scorer : outlier scoring function
    outputs:
        models : data structure of noise models of all nodes
                 dictionary of node : (dictionary of slide_number : scm_model)
                 e.g. {'X0':{0:scm_model, 1:scm_model}, 'X1':{0:scm_model, 1:scm_model}, ...}
        noise_samples : the noise samples  
                 dictionary of node : (dictionary of slide_number : noise)
                 e.g. {'X0':{0:noise, 1:noise}, 'X1':{0:noise, 1:noise}, ...}
        outlier_scorers : dictionary of node : (dictionary of slide_number : outlier scoring function)

    """
    print(f'-- outlier_scorer is {type(outlier_scorer)}')
    print(f'-- init_data.shape is {init_data.shape}')
    snoise_models = {} #data structure used to store noise models generated in each slide
    snoise_samples = {} #data structure used to store noise samples generated in  each slide
    soutlier_scorers = {} #data structure used to store outlier_scorer generated in each slide

    all_ancestors_of_node = causal_graph.ancestors[target_node]
    all_ancestors_of_node.update({target_node})

    sorted_nodes = [node for node in causal_graph.sorted_nodes if node in all_ancestors_of_node]
    noise_models = noise_model_fitting(init_data, 
                                       causal_graph, 
                                       m_samples,
                                       target_node, 
                                       sorted_nodes,
                                       dist_type)

        
    noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
                                                causal_graph, 
                                                target_node, 
                                                sorted_nodes, 
                                                num_noise_samples)
    if outlier_scorer == 'default':
        outlier_scorer = MedianCDFQuantileScorer()

    print(f'outlier_scorer is {type(outlier_scorer)}')
    outlier_scorer.fit(node_samples[target_node])
    soutlier_scorers[0] = outlier_scorer

    snoise_models[0] = noise_models
    snoise_samples[0] = noise_samples

    print(f'at scm_initialization {soutlier_scorers}')
    
    return snoise_models, snoise_samples, soutlier_scorers, sorted_nodes