from gcm_simulator.stream_mp import run_gcm_simulator_mp

from scipy.stats import halfnorm
from dowhy import gcm

from ocular_simulator.server import TCPServer
from ocular_simulator.sliding_window import SlidingWindow

import threading
from ocular_simulator.client import TCPClient

import time


def run_client(rate, data, start_ts, ip, port, msg_size, msg_format, disconnect_msg, client_id): # pragma: no cover
	print("Started client thread")
	client = TCPClient(ip, port, msg_size, msg_format, disconnect_msg, client_id)
	client.client.settimeout(1)
	client.connect()
	time.sleep(10) ## need to wait before sending data so that server can be done with its configuration first
	event_start_time = start_ts
	print("Client is sending send_message")
	client.send_message(event_start_time, rate, data)
	client.client.close()

def main_mp_gcm_simulator(ip, 
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
						  nslide_used=None,
						  shapley_config=None,
						  num_bootstrap_resamples=10):
	causal_model = gcm.StructuralCausalModel(causal_graph)
	for node in causal_graph.nodes:
	    if len(list(causal_graph.predecessors(node))) > 0: 
	        causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
	    else:
	        ### when the node has no parent
	        causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(halfnorm))
	
	slide_number = 1

	# initialize sliding window
	sliding_window = SlidingWindow(slide_size, window_size, unit='seconds')

	# initialize TCP server
	tcp_server = TCPServer(ip, port)

	## initialize client and run it in the background
	disconnect_msg = 'disconnect'
	client_thread = threading.Thread(target=run_client, args=(rate, data, start_ts, ip, port, msg_size, msg_format, disconnect_msg, client_id))
	client_thread.start()

	explainer_queues, run_time, est_time_queues = run_gcm_simulator_mp(tcp_server, 
				                         slide_size, 
				                         window_size, 
				                         start_ts, 
				                         msg_size, 
				                         msg_format, 
				                         n_streams,
				                         sliding_window,
				                         causal_model,
				                         m_samples,
				                         slide_number,
				                         features,
				                         target_node,
				                         nslide_used,
				                         shapley_config,
				                         num_bootstrap_resamples)
	## make sure to close client_thread and tcp_server
	client_thread.join()
	tcp_server.server.close()
	return explainer_queues, run_time, est_time_queues