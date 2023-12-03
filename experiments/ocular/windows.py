import os
import time
import datetime
import pandas as pd
import networkx as nx
import json

from multiprocessing import set_start_method

from ocular.causal_model import dag

from ocular_simulator.main_mp import run_main_mp

from ocular_simulator.evaluator import get_and_save_result
from ocular_simulator.evaluator import process_explanation_time
from ocular_simulator.evaluator import save_max_out_attribs_result
from rc_metrics.performance import compute_and_save_performance

from scipy.stats import halfnorm
from dowhy.gcm.shapley import ShapleyApproximationMethods, ShapleyConfig

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def read_data(data_path, features, client_id, data_separator=','):
    df = pd.read_csv(data_path, sep=data_separator)
    df_main = df[features]
    df_label = df['label']
    df_root_cause_gt = df['root_cause_gt']
    data_list = df_main.values.tolist()
    data = list()
    for i, datum in enumerate(data_list):
        data.append({'values' : datum, 'label' : df_label[i], 'index' : i, 'client_id' : client_id})
    return data


def run_experiment(exp_name,
                  data_path,
                  init_data_path,
                  rel_path,
                  client_id,
                  nslide_used,
                  ip, 
                  port, 
                  slide_size, 
                  window_size,
                  rate,
                  msg_size, 
                  msg_format, 
                  n_streams, 
                  causal_graph,
                  features,
                  target_node,
                  fm_types,
                  noise_types,
                  m_samples,
                  num_noise_samples,
                  error_threshold_change,
                  dist_type,
                  shapley_config=None,
                  attribute_mean_deviation=False,
                  n_jobs=-1,
                  outlier_scorer_type='default',
                  data_separator=','):
    
    if nslide_used is None:
        nslide_used = window_size//slide_size

    data = read_data(data_path, features, client_id, data_separator)
    
    init_data = pd.read_csv(init_data_path, sep=data_separator)
    init_data = init_data[features]
    print(f'init_data.shape {init_data.shape}')
    
    explainer_queues, run_time, est_time_queues = run_main_mp(ip, port, 
                                    slide_size, window_size,
                                    msg_size, msg_format, n_streams, 
                                    init_data, fm_types, noise_types, 
                                    causal_graph, target_node, 
                                    rate, data, client_id, 
                                    m_samples=m_samples, 
                                    nslide_used = nslide_used,
                                    num_noise_samples = num_noise_samples,
                                    error_threshold_change=error_threshold_change,
                                    shapley_config=shapley_config,
                                    attribute_mean_deviation=attribute_mean_deviation,
                                    n_jobs=n_jobs,
                                    outlier_scorer_type=outlier_scorer_type,
                                    dist_type=dist_type)

    n_outliers_explained = get_and_save_result(explainer_queues, run_time, n_streams, 
                        exp_name, rel_path, features, nslide_used)
    result_path = save_max_out_attribs_result(exp_name, rel_path, nslide_used)
    average_explanation_time = process_explanation_time(est_time_queues, n_streams, n_outliers_explained)
    compute_and_save_performance(data_path, 
                                 result_path, 
                                 exp_name, 
                                 features, 
                                 data_separator, 
                                 nslide_used,
                                 average_explanation_time)

if __name__ == '__main__':
    cwd = os.getcwd()
    set_start_method("spawn")

    # shapley_config_str = "SUBSET_SAMPLING"
    # shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.SUBSET_SAMPLING)

    shapley_config_str = "AUTO"
    shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.AUTO)
    
    num_nodes = 6
    
    main_folder = 'windows'
    folder = 'branch'
    possible_root_causes = {3 : ('X1',),
                            4 : ('X1','X3',),
                            5 : ('X1', 'X3', 'X4'),
                            6 : ('X1', 'X4', 'X5'),
                            7 : ('X1', 'X4', 'X5'),
                            8 : ('X1', 'X4', 'X5'),
                            9 : ('X1', 'X4', 'X5'),
                            10 : ('X1', 'X4', 'X5'),
                            11 : ('X1', 'X4', 'X5'),
                            12 : ('X1', 'X4', 'X5')}

    # folder = 'straight'
    # possible_root_causes = {3 : ('X1', 'X2'),
    #                         4 : ('X1', 'X2', 'X3'),
    #                         5 : ('X1', 'X2', 'X3'),
    #                         6 : ('X1', 'X2', 'X3'),
    #                         7 : ('X1', 'X2', 'X3'),
    #                         8 : ('X1', 'X2', 'X3'),
    #                         9 : ('X1', 'X2', 'X3'),
    #                         10 : ('X1', 'X2', 'X3'),
    #                         11 : ('X1', 'X2', 'X3'),
    #                         12 : ('X1', 'X2', 'X3'),}
    
    ip = '127.0.0.1'
    port = 49775
    slide_size = 12 # in seconds
    #window_sizes = (12, 24, 36, 48, 60, 72, 84, 96)
    window_sizes = (24, 36, 48, 60, 72, 84, 96)
    rate = 0.05
    nslide_used = 2
    
    msg_size = 1024
    msg_format = 'utf-8'
    n_streams = 1
    client_id = 1

    m_samples = 1.0 
    num_noise_samples = 1500
    error_threshold_change=0.1

    attribute_mean_deviation=False
    n_jobs=-1
    outlier_scorer_type='default'
    data_separator=','

    target_node = f'X{num_nodes}'
    nodes = [f'X{i}' for i in range(1,num_nodes+1)]
    gt_root_causes = possible_root_causes[num_nodes]

    for window_size in window_sizes:
        for root_cause_gt in gt_root_causes:
            bname = f'{num_nodes}_nodes_{root_cause_gt}_{target_node}'
            rel_path = f'result/{main_folder}/{folder}/window_{window_size}'
            
            data_folder = f'../dataset/nodes/{num_nodes}_nodes/{folder}'
            data_path =  f'{data_folder}/{bname}.csv'
            
            init_data_path =  f'{data_folder}/init_{bname}.csv'
            
            metadata_path = f'{data_folder}/{bname}.json'
            metadata = None

            with open(metadata_path) as metadata_json:
                metadata = json.load(metadata_json)

            edges = metadata['edges']
            features = metadata['nodes']
            causal_graph = dag.CausalGraph(edges, features)

            fm_types = {node : 'LinearModel' for node in nodes}
            noise_types = {node : 'AdditiveNoise' for node in nodes}
            
            root_cause_metadata = metadata['root_causes']
            if root_cause_metadata == root_cause_gt:
                print(f'Experiment on {num_nodes} where root causes ground truth = {root_cause_gt} and target_node = {target_node}')
            else:
                break

            if num_nodes >= 50 and num_nodes < 100:
                msg_size = 2048
            elif num_nodes >= 100:
                msg_size = 4096

            run_experiment(exp_name=bname,
                      data_path=data_path,
                      init_data_path=init_data_path,
                      rel_path=rel_path,
                      client_id=client_id,
                      nslide_used=nslide_used,
                      ip=ip, 
                      port=port, 
                      slide_size=slide_size, 
                      window_size=window_size,
                      rate=rate,
                      msg_size=msg_size, 
                      msg_format=msg_format, 
                      n_streams=n_streams, 
                      causal_graph=causal_graph,
                      features=features,
                      target_node=target_node,
                      fm_types=fm_types,
                      noise_types=noise_types,
                      m_samples=m_samples,
                      num_noise_samples=num_noise_samples,
                      error_threshold_change=error_threshold_change,
                      dist_type=halfnorm,
                      shapley_config=shapley_config,
                      attribute_mean_deviation=attribute_mean_deviation,
                      n_jobs=n_jobs,
                      outlier_scorer_type=outlier_scorer_type,
                      data_separator=data_separator)
            time.sleep(30)

        