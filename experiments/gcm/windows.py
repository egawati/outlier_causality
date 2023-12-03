import os
import datetime
import time
import pandas as pd
import networkx as nx
import json

from multiprocessing import set_start_method
from gcm_simulator.main_mp import main_mp_gcm_simulator
from gcm_simulator.evaluator import get_and_save_result
from gcm_simulator.evaluator import process_explanation_time
from gcm_simulator.evaluator import save_max_median_attribs_result
from rc_metrics.performance import compute_and_save_performance

from dowhy.gcm.shapley import ShapleyApproximationMethods, ShapleyConfig

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
                   rel_path,
                   features,
                   causal_graph,
                   target_node,
                   data_path,
                   data_separator=',',
                   m_samples = 1.0, # 1.0 means 100 %
                   slide_size = 15,
                   window_size = 60,
                   rate = 0.1, # seconds
                   nslides=None,
                   ip = '127.0.0.1',
                   port = 49775,
                   msg_size = 1024,
                   msg_format = 'utf-8',
                   n_streams = 1,
                   client_id = 1,
                   shapley_config = None,
                   num_bootstrap_resamples = 10
                   ):
    if nslides is None:
        nslides = window_size//slide_size
    
    data = read_data(data_path, features, client_id, data_separator)

    start_ts = datetime.datetime.now().timestamp()
    explainer_queues, run_time, est_time_queues = main_mp_gcm_simulator(ip, 
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
                                              nslides,
                                              shapley_config,
                                              num_bootstrap_resamples)

    n_outliers_explained = get_and_save_result(explainer_queues, run_time, n_streams, 
                        exp_name, rel_path, features, nslides)
    result_path = save_max_median_attribs_result(exp_name, rel_path, nslides)
    average_explanation_time = process_explanation_time(est_time_queues, n_streams, n_outliers_explained)
    compute_and_save_performance(data_path, 
                                 result_path, 
                                 exp_name, 
                                 features, 
                                 data_separator, 
                                 nslides,
                                 average_explanation_time)


if __name__ == '__main__':
    cwd = os.getcwd()
    set_start_method("spawn")
    
    # shapley_config_str = "EXACT_FAST"
    # shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT_FAST)

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
    
    n_jobs=-1
    
    data_separator=','

    num_bootstrap_resamples = 5
    target_node = f'X{num_nodes}'
    nodes = [f'X{i}' for i in range(1,num_nodes+1)]
    gt_root_causes = possible_root_causes[num_nodes]
    for window_size in window_sizes:
        for root_cause_gt in gt_root_causes:
            bname = f'{num_nodes}_nodes_{root_cause_gt}_{target_node}'
            rel_path = f'result/{main_folder}/{folder}/window_{window_size}'
            
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
                       rel_path=rel_path,
                       features=features,
                       causal_graph=causal_graph,
                       target_node=target_node,
                       data_path=data_path,
                       data_separator=data_separator,
                       m_samples = m_samples,
                       slide_size = slide_size,
                       window_size = window_size,
                       rate = rate,
                       nslides=nslide_used,
                       ip = ip,
                       port = port,
                       msg_size = msg_size,
                       msg_format = msg_format,
                       n_streams = n_streams,
                       client_id = client_id,
                       shapley_config = shapley_config,
                       num_bootstrap_resamples = num_bootstrap_resamples,
                       )
            
            time.sleep(30)
    


