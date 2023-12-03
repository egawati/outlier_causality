import os
import datetime
import pandas as pd
import networkx as nx

from multiprocessing import set_start_method
from gcm_simulator.main_mp import main_mp_gcm_simulator
from gcm_simulator.evaluator import get_and_save_result
from gcm_simulator.evaluator import save_max_median_attribs_result
from rc_metrics.performance import compute_and_save_performance

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

if __name__ == '__main__':
    cwd = os.getcwd()
    set_start_method("spawn")
    exp_name = 'syn_5node_X1_X5_big'
    rel_path = 'result'
    
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
    causal_graph = nx.DiGraph(nodes)
    m_samples = 1
    target_node = 'X5'

    data_path =  f'../dataset/{exp_name}.csv'
    data_separator =','
    data = read_data(data_path, features, client_id, data_separator)

    start_ts = datetime.datetime.now().timestamp()
    rate = 0.1
    
    explainer_queues, run_time = main_mp_gcm_simulator(ip, 
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
                                              client_id)

    get_and_save_result(explainer_queues, run_time, n_streams, 
                        exp_name, rel_path, features)
    result_path = save_max_median_attribs_result(exp_name, rel_path)
    compute_and_save_performance(data_path, result_path, 
                                 exp_name, features, data_separator)


