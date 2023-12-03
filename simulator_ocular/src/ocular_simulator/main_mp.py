import time
import datetime

from ocular_simulator.timeseries import process_timeseries_data

from ocular_simulator.stream_mp import run_simulator_mp
from ocular_simulator.sliding_window import SlidingWindow

from ocular.initialization import scm_initialization
from ocular.concept_drift_detector import compute_model_rmse


import threading
from ocular_simulator.server import TCPServer
from ocular_simulator.client import TCPClient

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def run_client(rate, data, start_ts, ip, port, msg_size, msg_format, disconnect_msg, client_id): # pragma: no cover
	logging.info("Started client thread")
	client = TCPClient(ip, port, msg_size, msg_format, disconnect_msg, client_id)
	client.client.settimeout(1)
	client.connect()
	time.sleep(10) ## need to wait before sending data so that server can be done with its configuration first
	event_start_time = start_ts
	logging.info("Client is sending send_message")
	client.send_message(event_start_time, rate, data)
	client.client.close()

def run_main_mp(ip, port, 
				slide_size, window_size,
				msg_size, msg_format, n_streams, 
				init_data, fm_types, noise_types, 
				causal_graph, target_node, 
				rate, data, client_id, 
				m_samples=0.75, 
				nslide_used = 1,
	            num_noise_samples = 1500,
	            error_threshold_change=0.1,
	            shapley_config=None,
	            attribute_mean_deviation=False,
	            n_jobs=-1,
				outlier_scorer_type='default',
				dist_type=None):
	
	logging.info("run main mp")
	max_fcm_number = window_size//slide_size + 1
	# initialize sliding window
	sliding_window = SlidingWindow(slide_size, window_size, unit='seconds')

	# initialize models, noise_samples, and outlier_scorers for explainer
	logging.info("initialize models, noise_samples, and outlier_scorers for explainer")
	init_data, _ = process_timeseries_data(df=init_data, 
											 dag=causal_graph.dag, 
											 target_node=target_node)
	snoise_models, snoise_samples, soutlier_scorers, sorted_nodes = scm_initialization(init_data, 
                       causal_graph=causal_graph, 
                       fm_types=fm_types, 
                       noise_types=noise_types, 
                       m_samples=m_samples, 
                       target_node = target_node,
                       outlier_scorer=outlier_scorer_type,
                       num_noise_samples = num_noise_samples,
                       dist_type=dist_type)

	model_rmse = compute_model_rmse(init_data, snoise_models[0], causal_graph)
	smodel_rmse = {}
	smodel_rmse[0] = model_rmse
	 
	# run TCP server

	logging.info(f"run TCP server")
	tcp_server = TCPServer(ip, port)

	## initialize client and run it in the background
	logging.info("initialize client and run it in the background")
	disconnect_msg = 'disconnect'

	start_ts = datetime.datetime.now().timestamp()
	client_thread = threading.Thread(target=run_client, args=(rate, data, start_ts, ip, port, msg_size, msg_format, disconnect_msg, client_id))
	client_thread.start()

	active_fcms = {0 : {'start_ts' : start_ts - window_size, 'end_ts': start_ts}}
	slide_number = 1

	print(f'at main dist_type is {dist_type}')

	explainer_queues, run_time, est_time_queues = run_simulator_mp(tcp_server, 
				                    msg_size, 
				                    msg_format, 
				                    n_streams,
				                    start_ts,
				                    sliding_window,
				                    window_size, 
				                    slide_size, 
				                    slide_number,
				                    active_fcms, 
				                    snoise_models,
				                    snoise_samples,
				                    soutlier_scorers, 
				                    smodel_rmse,
				                    causal_graph,                              
				                    sorted_nodes,
				                    target_node,
				                    m_samples, 
				                    max_fcm_number,
				                    nslide_used,
				                    num_noise_samples,
				                    error_threshold_change,
				                    shapley_config,
				                    attribute_mean_deviation,
				                    n_jobs,
				                    outlier_scorer_type,
				                    dist_type)
	client_thread.join()
	tcp_server.server.close()
	return explainer_queues, run_time, est_time_queues