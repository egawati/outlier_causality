import os
import time
import datetime
import pandas as pd
import networkx as nx
import json

from multiprocessing import set_start_method

from cloudranger_simulator.main_mp import main_mp_cloudranger_simulator

from cloudranger_simulator.evaluator import get_and_save_result
from cloudranger_simulator.evaluator import process_explanation_time

from rc_metrics.performance_multilabel import compute_and_save_performance_multilabel
#from rc_metrics.performance import compute_and_save_performance

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

    explainer_queues, run_time, est_time_queues = main_mp_cloudranger_simulator(ip, 
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
    
    num_nodes = 6
    
    main_folder = 'rates'
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
    slide_size = 12
    window_size = 48 # in seconds
    rates = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)
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

    for rate in rates:
        nodes = [f'X{i}' for i in range(1,num_nodes+1)]
        gt_root_causes = possible_root_causes[num_nodes]
        for root_cause_gt in gt_root_causes:
            target_node = f'X{num_nodes}'

            bname = f'{num_nodes}_nodes_{root_cause_gt}_{target_node}'
            mrate = 10 * rate
            rel_path = f'result/{main_folder}/{folder}/rate_{mrate}'
            
            data_folder = f'../dataset/nodes/{num_nodes}_nodes/{folder}'
            data_path =  f'{data_folder}/{bname}.csv'
            
            metadata_path = f'{data_folder}/{bname}.json'
            metadata = None

            with open(metadata_path) as metadata_json:
                metadata = json.load(metadata_json)

            edges = metadata['edges']
            features = metadata['nodes']
            causal_graph = nx.DiGraph(edges)

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

    