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
    
    shapley_config_str = "AUTO"
    shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.AUTO)
    
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

    m_samples = 1.0 
    
    n_jobs=-1
    
    data_separator=','

    num_bootstrap_resamples = 5

    timeseries = [i for i in range(1, 29)]
    timeseries.remove(13)
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
        
        time.sleep(20)
    


