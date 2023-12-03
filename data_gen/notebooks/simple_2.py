#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
import numpy as np
import pandas as pd

from dowhy import gcm
from dowhy.gcm import MedianCDFQuantileScorer

from ocular.causal_model import dag

from ocular.outlier_score import compute_it_score
from ocular.outlier_score import _relative_frequency
from ocular.outlier_score import node_outlier_contribution_scores

from ocular.noise_data_generation import data_dict_to_data_df

from ocular.noise_data_generation import generate_noisedf_from_data

from ocular.noise_data_generation import generate_noise_and_node_samples

from ocular.noise_data_generation import get_target_data_from_noise_arr

from ocular.model_generation import noise_model_fitting


# ## Data Generation Process

# In[2]:

gcm.config.disable_progress_bars()
nodes = [('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
features = ['X1', 'X2', 'X3', 'X4', 'X5']
causal_graph = dag.CausalGraph(nodes, features)
target_node = 'X5'

data = pd.read_csv('inlier.csv', sep=',')
m_samples = 0.75

all_ancestors_of_node = causal_graph.ancestors[target_node]
all_ancestors_of_node.update({target_node})
sorted_nodes = [node for node in causal_graph.sorted_nodes if node in all_ancestors_of_node]
print(f'sorted_nodes is {sorted_nodes}')
## first we need to generate noise_models
noise_models = noise_model_fitting(data, 
                                causal_graph, 
                                m_samples,
                                target_node, 
                                sorted_nodes)

for model in noise_models:
    print(noise_models[model])

num_noise_samples = 1500
## next we generate noise_samples and node_samples based on the generated noise_models
noise_samples, node_samples = generate_noise_and_node_samples(noise_models, 
                                causal_graph, 
                                target_node, 
                                sorted_nodes, 
                                num_noise_samples)

## now we can train outlier_scorer using node_samples
print(f'fitting outlier_scorer with node_samples[target_node]')
print(node_samples[target_node].shape)
print(node_samples[target_node].reshape(-1))
outlier_scorer = MedianCDFQuantileScorer()
outlier_scorer.fit(node_samples[target_node])


## suppose there is one outlier
outliers = pd.read_csv('outlier.csv', sep=',')

## outlier_noises can have less columns than outliers since we only care about the nodes that has path to target_node
outlier_noises = generate_noisedf_from_data(outliers, 
                     noise_models, 
                     causal_graph, 
                     sorted_nodes, 
                     target_node)
out_noises_arr = outlier_noises.to_numpy()
print(f'outliers is {outliers}')
print(f'outlier_noises is {outlier_noises}')
print(f'out_noises_arr is {out_noises_arr}')

results = node_outlier_contribution_scores(outlier_noises=out_noises_arr,
                            noise_samples=noise_samples,
                            outlier_scorer=outlier_scorer,
                            attribute_mean_deviation=False,
                            noise_models=noise_models,
                            causal_graph=causal_graph,
                            sorted_nodes=sorted_nodes,
                            target_node=target_node,
                            shapley_config = None)

print(f'results {results}')




