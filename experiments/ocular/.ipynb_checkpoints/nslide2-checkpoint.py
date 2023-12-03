import os
import datetime
import pandas as pd
import networkx as nx

from multiprocessing import set_start_method
from ocular_simulator.main_mp import run_main_mp

from ocular.causal_model import dag

from ocular_simulator.evaluator import get_and_save_result
from ocular_simulator.evaluator import process_explanation_time
from ocular_simulator.evaluator import save_max_out_attribs_result
from rc_metrics.performance import compute_and_save_performance

from scipy.stats import halfnorm

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


def varying_nslides(ip, 
                  port, 
                  slide_size, 
                  window_size,
                  msg_size, 
                  msg_format, 
                  n_streams, 
                  causal_graph,
                  features,
                  target_node,
                  data,
                  rate,
                  client_id,
                  nslide_used,
                  exp_name,
                  data_path,
                  rel_path,
                  init_data,
                  m_samples,
                  num_noise_samples,
                  error_threshold_change,
                  shapley_config=None,
                  attribute_mean_deviation=False,
                  n_jobs=-1,
                  outlier_scorer_type='default',
                  data_separator=','):
    print(f'RUNNING EXPERIMENTS where nslide_used={nslide_used}')
    fm_types = {node : 'LinearModel' for node in features}
    noise_types = {node : 'AdditiveNoise' for node in features}

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
                                    shapley_config=None,
                                    attribute_mean_deviation=False,
                                    n_jobs=-1,
                                    outlier_scorer_type='default',
                                    dist_type=halfnorm)

    get_and_save_result(explainer_queues, run_time, n_streams, 
                        exp_name, rel_path, features, nslide_used)
    result_path = save_max_out_attribs_result(exp_name, rel_path, nslide_used)
    average_explanation_time = process_explanation_time(est_time_queues, n_streams)
    print(f'average_explanation_time = {average_explanation_time}')
    compute_and_save_performance(data_path, 
                                 result_path, 
                                 exp_name, 
                                 features, 
                                 data_separator, 
                                 nslide_used,
                                 average_explanation_time)


def run_experiment(bname='rand_syn_5node_X2_X5', seed=0, rel_path='result/nslide_used', nslides=None):
    exp_name = f'{bname}_seed{seed}'
    ip = '127.0.0.1'
    port = 49775
    slide_size = 15 # in seconds
    window_size = 60 # in seconds
    
    msg_size = 1024
    msg_format = 'utf-8'
    n_streams = 1
    client_id = 1

    nodes = [('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    causal_graph = dag.CausalGraph(nodes, features)
    m_samples = 1.0 
    target_node = 'X5'
    num_noise_samples = 1500
    error_threshold_change=0.05

    for nslide_used in range(nslides, nslides+1):
        data_path =  f'../dataset/{exp_name}.csv'
        data_separator =','
        data = read_data(data_path, features, client_id, data_separator)

        init_data_path =  f'../dataset/init_{exp_name}.csv'
        init_data = pd.read_csv(init_data_path, sep=data_separator)
        init_data = init_data[features]
        varying_nslides(ip, 
                      port, 
                      slide_size, 
                      window_size,
                      msg_size, 
                      msg_format, 
                      n_streams, 
                      causal_graph,
                      features,
                      target_node,
                      data,
                      rate,
                      client_id,
                      nslide_used,
                      exp_name,
                      data_path,
                      rel_path,
                      init_data,
                      m_samples,
                      num_noise_samples,
                      error_threshold_change,
                      shapley_config=None,
                      attribute_mean_deviation=False,
                      n_jobs=-1,
                      outlier_scorer_type='default',
                      data_separator=data_separator)

if __name__ == '__main__':
    cwd = os.getcwd()
    set_start_method("spawn")
    rate = 0.1 # seconds
    seeds = (3, 47, 99, 101)
    bname = 'rand_syn_5node_X2_X5'
    rel_path = 'result/nslide_used2'
    nslides = 2
    for seed in seeds:
        print(f'DATASET {seed}')
        run_experiment(bname=bname, seed=seed, rel_path=rel_path, nslides=nslides)
    