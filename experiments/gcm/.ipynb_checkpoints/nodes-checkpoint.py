import os
import datetime
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
                   seed, 
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
                   shapley_config = None
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
                                              shapley_config)

    get_and_save_result(explainer_queues, run_time, n_streams, 
                        exp_name, rel_path, features, nslides)
    result_path = save_max_median_attribs_result(exp_name, rel_path, nslides)
    average_explanation_time = process_explanation_time(est_time_queues, n_streams)
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
    #n_nodes = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    n_nodes = (100,)

    
    # shapley_config_str = "EXACT_FAST"
    # shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT_FAST)

    shapley_config_str = "SUBSET_SAMPLING"
    shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.SUBSET_SAMPLING)

    
    msg_size = 1024
    
    for num_nodes in n_nodes:
        random_seed = num_nodes//2
        bname = f'{num_nodes}_nodes_seed_{random_seed}'
        print(f"Running {bname}")
        rel_path = f'result/nodes/{bname}/{shapley_config_str}'
        
        data_folder = '../dataset/RandomGraphs/'
        data_path =  f'{data_folder}/{bname}.csv'
        
        metadata_path = f'{data_folder}/{bname}.json'
        metadata = None

        with open(metadata_path) as metadata_json:
            metadata = json.load(metadata_json)

        edges = metadata['edges']
        features = metadata['nodes']
        causal_graph = nx.DiGraph(edges)
        target_node = metadata['target_node']

        if num_nodes >= 50 and num_nodes < 100:
            msg_size = 2048
        elif num_nodes >= 100:
            msg_size = 4096

        run_experiment(exp_name=bname,
                   seed=random_seed, 
                   rel_path=rel_path,
                   features=features,
                   causal_graph=causal_graph,
                   target_node=target_node,
                   data_path=data_path,
                   data_separator=',',
                   m_samples = 1.0, # 1.0 means 100 %
                   slide_size = 15,
                   window_size = 60,
                   rate = 0.1, # seconds
                   nslides=None,
                   ip = '127.0.0.1',
                   port = 49775,
                   msg_size = msg_size,
                   msg_format = 'utf-8',
                   n_streams = 1,
                   client_id = 1,
                   shapley_config = shapley_config,
                   )
    


