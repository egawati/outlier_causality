from dycause_simulator.stream_mp import run_dycause_simulator_mp

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

def main_mp_dycause_simulator(ip, 
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
								step = 1,
								lag = 0.1,
								topk_path = 6,
								auto_threshold_ratio = 0.8,
								mean_method="arithmetic",
								max_path_length=None,
								num_sel_node=1):	
	slide_number = 1

	# initialize sliding window
	sliding_window = SlidingWindow(slide_size, window_size, unit='seconds')

	# initialize TCP server
	tcp_server = TCPServer(ip, port)

	## initialize client and run it in the background
	disconnect_msg = 'disconnect'
	client_thread = threading.Thread(target=run_client, args=(rate, data, start_ts, ip, port, msg_size, msg_format, disconnect_msg, client_id))
	client_thread.start()

	explainer_queues, run_time, est_time_queues = run_dycause_simulator_mp(tcp_server = tcp_server,
													slide_size = slide_size,
													window_size = window_size,
													start_ts = start_ts, 
													msg_size = msg_size,
													msg_format = msg_format,
													n_streams = n_streams,
													sliding_window = sliding_window,
													causal_graph = causal_graph,
													m_samples = m_samples,
													slide_number = slide_number,
													features = features,
													target_node = target_node,
													nslide_used=nslide_used,
													step=step,
													lag=lag,
													topk_path=topk_path,
													auto_threshold_ratio=auto_threshold_ratio,
													mean_method=mean_method,
													max_path_length=max_path_length,
													num_sel_node=num_sel_node)
	## make sure to close client_thread and tcp_server
	client_thread.join()
	tcp_server.server.close()
	return explainer_queues, run_time, est_time_queues