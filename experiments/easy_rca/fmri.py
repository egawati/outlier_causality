import os
import time
import datetime
import pandas as pd
import networkx as nx
import json

from multiprocessing import set_start_method

from easyrca_simulator.main_mp import main_mp_easyrca_simulator

from easyrca_simulator.evaluator import get_and_save_result
from easyrca_simulator.evaluator import process_explanation_time

from rc_metrics.performance_multilabel import compute_and_save_performance_multilabel
#from rc_metrics.performance import compute_and_save_performance

import warnings
warnings.filterwarnings("ignore")

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
                  m_samples,
                  gamma_max,
                  sig_threshold,
                  data_separator=',',
                  ):
    
    if nslide_used is None:
        nslide_used = window_size//slide_size

    data = read_data(data_path, features, client_id, data_separator)
    
    start_ts = datetime.datetime.now().timestamp()

    explainer_queues, run_time, est_time_queues = main_mp_easyrca_simulator(ip, 
                                                  port, 
                                                  slide_size, 
                                                  window_size,
                                                  msg_size, 
                                                  msg_format, 
                                                  n_streams, 
                                                  causal_graph,
                                                  m_samples,
                                                  features,
                                                  target_node,
                                                  data,
                                                  rate,
                                                  start_ts,
                                                  client_id,
                                                  nslide_used,
                                                  gamma_max,
                                                  sig_threshold)

    result_path = get_and_save_result(explainer_queues, run_time, n_streams, 
                        exp_name, rel_path, features, nslide_used)
    average_explanation_time = process_explanation_time(est_time_queues, n_streams)
    compute_and_save_performance_multilabel(data_path, 
                                 result_path, 
                                 exp_name, 
                                 features, 
                                 data_separator, 
                                 nslide_used,
                                 average_explanation_time)

if __name__ == '__main__':
    cwd = os.getcwd()
    set_start_method("spawn")
     
    ip = '127.0.0.1'
    port = 49775
    slide_size = 12 # in seconds
    window_size = 48 # in seconds
    rate = 0.05
    nslide_used = 2
    
    msg_size = 1024
    msg_format = 'utf-8'
    n_streams = 1
    client_id = 1

    gamma_max=1
    sig_threshold=0.05

    outlier_scorer_type='default'
    data_separator=','

    m_samples = 1

    timeseries = [i for i in range(24, 29)]
    if 13 in timeseries:
        timeseries.remove(13)
    if 14 in timeseries:
        timeseries.remove(14)

    bfolder = 'I'
    data_folder = f'../dataset/real_dataset/fMRI/fMRI_with_outliers_{bfolder}'
    metadata_path = f'{data_folder}/metadata.json'
    metadata = None
    with open(metadata_path) as metadata_json:
        metadata = json.load(metadata_json)

    for dataset in timeseries:
        bname = f'timeseries{dataset}'
        data_path =  f'{data_folder}/{bname}.csv'
        rel_path = f'result/fmri/{bfolder}/{bname}'

        nodes = list(metadata[bname]['vertex_map'].values())
        nodes.sort()
        features = nodes
        edges = metadata[bname]['edges']
        target_node = metadata[bname]['target_node']

        causal_graph = nx.DiGraph()
        causal_graph.add_nodes_from(nodes)
        causal_graph.add_edges_from(edges)
        run_experiment(exp_name=bname,
                  data_path=data_path,
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
                  m_samples = m_samples,
                  gamma_max = gamma_max,
                  sig_threshold = sig_threshold,
                  data_separator=data_separator)
        time.sleep(30)

    