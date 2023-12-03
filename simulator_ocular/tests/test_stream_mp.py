import unittest

import socket
import threading
import time
import datetime
import pandas as pd
import numpy as np

from multiprocessing import Process

from ocular.causal_model import dag
from ocular.initialization import scm_initialization
from ocular.concept_drift_detector import compute_model_rmse

from ocular_simulator.client import TCPClient
from ocular_simulator.server import TCPServer
from ocular_simulator.sliding_window import SlidingWindow
from ocular_simulator.stream_mp import run_simulator_mp

import random

class TestSimulatorMP(unittest.TestCase):
	def setUp(self):
		"""
		When a setUp() method is defined, the test runner will run that method prior to each test. 
		"""
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.causal_graph = dag.CausalGraph(nodes, self.features)
		self.target_node = 'X5'
		
		self.num_noise_samples = 1500
		self.m_samples = 0.75 # make m_samples as percentage
		self.error_threshold_change = 0.1
		self.nslide_used = 1
		self.shapley_config = None
		self.attribute_mean_deviation = False
		self.n_jobs= -1
		self.n_streams = 1
		
		n_samples = 32
		np.random.seed(0)
		init_data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		
		fm_types = {node : 'LinearModel' for node in self.features}
		noise_types = {node : 'AdditiveNoise' for node in self.features}

		
		self.snoise_models, self.snoise_samples, self.soutlier_scorers, self.sorted_nodes = scm_initialization(init_data, 
                       causal_graph=self.causal_graph, 
                       fm_types=fm_types, 
                       noise_types=noise_types, 
                       m_samples=self.m_samples, 
                       target_node = self.target_node, 
                       outlier_scorer='default',
                       num_noise_samples = 1500)

		model_rmse = compute_model_rmse(init_data, self.snoise_models[0], self.causal_graph)
		self.smodel_rmse = {}
		self.smodel_rmse[0] = model_rmse

		self.slide_size = 16
		self.window_size = 32
		self.msg_size = 1024
		self.msg_format = 'utf-8'
		self.max_fcm_number = self.window_size//self.slide_size + 1
		self.basic_time = datetime.datetime.now().timestamp()
		self.slide_number = 0

		self.active_fcms = {0 : {'start_ts' : self.basic_time - self.window_size, 'end_ts': self.basic_time}}
		self.start_ts = self.basic_time
		
		self.ip = '127.0.0.1'
		self.port = random.randrange(49152, 65535)

		np.random.seed(1)
		self.df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		self.data_list = self.df.values.tolist()
		
		# initialize TCP server
		self.tcp_server = TCPServer(self.ip, self.port)
		self.assertIsInstance(self.tcp_server, TCPServer)
		self.assertIsInstance(self.tcp_server.server, socket.socket)

		time.sleep(1)

	def tearDown(self):
		"""
		When a tearDown() method is defined, the test runner will invoke that method after each test.
		"""
		self.tcp_server.server.close()

	def run_one_client_mp(self):
		client = TCPClient(self.ip, self.port, size=self.msg_size, 
						   format=self.msg_format, disconnect_msg = "disconnect", client_id = 1)
		client.client.settimeout(1)
		client.connect()
		time.sleep(2) ## need to wait before sending data so that server can be done with its configuration first
		rate = 0.5
		data = list()
		for i, datum in enumerate(self.data_list):
			if i != 4:
				data.append({'values':datum, 'client_id':1, 'event_ts':self.basic_time + i*rate, 'label':0, 'index':i})
			else:
				data.append({'values':datum, 'client_id':1 ,'event_ts':self.basic_time + i*rate, 'label':1, 'index':i})
		
		event_start_time = self.basic_time
		print("Sending Data")
		client.send_message(event_start_time, rate, data)
		client.client.close()


	def test_simulator_one_client_mp(self):
		sliding_window = SlidingWindow(self.slide_size, self.window_size, unit='seconds')

		# run client in the background
		client_thread = threading.Thread(target=self.run_one_client_mp)
		client_thread.start()

		
		self.slide_number += 1 ## at the beginning the slide number should be 1, slide_number = 0 is for init_data

		e_queues, running_time, est_time_queues = run_simulator_mp(self.tcp_server, 
				                    self.msg_size, 
				                    self.msg_format, 
				                    self.n_streams,
				                    self.start_ts,
				                    sliding_window,
				                    self.window_size, 
				                    self.slide_size, 
				                    self.slide_number,
				                    self.active_fcms, 
				                    self.snoise_models,
				                    self.snoise_samples,
				                    self.soutlier_scorers, 
				                    self.smodel_rmse,
				                    self.causal_graph,                              
				                    self.sorted_nodes,
				                    self.target_node,
				                    self.m_samples, 
				                    self.max_fcm_number,
				                    self.nslide_used,
				                    self.num_noise_samples,
				                    self.error_threshold_change,
				                    self.shapley_config,
				                    self.attribute_mean_deviation,
				                    self.n_jobs)
		
		
		# do not forget the close client_thread
		client_thread.join()
		self.assertEqual(e_queues[0].empty(), False)
		explanation, outlier_index =  e_queues[0].get()
		self.assertEqual(outlier_index[0], 4)
		outlier_ts = list(explanation.keys())[0]
		self.assertEqual(outlier_ts, self.basic_time + 4*0.5)
		self.tcp_server.server.close()	

if __name__ == '__main__': # pragma: no cover
    unittest.main()


