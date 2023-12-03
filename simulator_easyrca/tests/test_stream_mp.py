import unittest

import threading
import socket
import time
import datetime
import pandas as pd
import numpy as np

import networkx as nx
from dowhy import gcm
from scipy.stats import halfnorm

from ocular_simulator.client import TCPClient
from ocular_simulator.server import TCPServer
from ocular_simulator.server import ClientHandlerMP
from ocular_simulator.sliding_window import SlidingWindow

from easyrca_simulator.stream_mp import run_easyrca_simulator_mp

import random

from multiprocessing import Manager

class TestStreamMPEasyRCA(unittest.TestCase):
	def setUp(self):
		"""
		When a setUp() method is defined, the test runner will run that method prior to each test. 
		"""
		nodes = [('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4'), ('X4', 'X5')]
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.target_node = 'X5'
		self.causal_graph = nx.DiGraph(nodes)
		

		n_samples = 100
		self.df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		self.data_list = self.df.values.tolist()
		
		
		self.ip = '127.0.0.1'
		self.port = random.randrange(49152, 65535)
		self.slide_size = 4
		self.window_size = 12
		self.basic_time = datetime.datetime.now().timestamp()
		self.msg_size = 1024
		self.msg_format = 'utf-8'
		self.max_slide = self.window_size/self.slide_size

		
		# initialize TCP server
		self.tcp_server = TCPServer(self.ip, self.port)
		self.assertIsInstance(self.tcp_server, TCPServer)
		self.assertIsInstance(self.tcp_server.server, socket.socket)

		time.sleep(0.000001)

	def tearDown(self):
		"""
		When a tearDown() method is defined, the test runner will invoke that method after each test.
		"""
		self.tcp_server.server.close()

	def run_client(self): # pragma: no cover
		rate = 0.5
		data = list()
		for i, datum in enumerate(self.data_list):
			if i == 4 or i==24:
				data.append({'values':datum, 'client_id':1, 'event_ts':self.basic_time + i*rate, 'label':1, 'index':i})
			else:
				data.append({'values':datum, 'client_id':1 ,'event_ts':self.basic_time + i*rate, 'label':0, 'index':i})

		client = TCPClient(self.ip, self.port, self.msg_size, self.msg_format, disconnect_msg = "disconnect", client_id = 1)
		client.client.settimeout(1)
		client.connect()
		time.sleep(2) ## need to wait before sending data so that server can be done with its configuration first
		event_start_time = self.basic_time
		client.send_message(event_start_time, rate, data)
		client.client.close()

	def test_run_easyrca_stream_mp(self):
		sliding_window = SlidingWindow(self.slide_size, self.window_size, unit='seconds')

		# run client in the background
		client_thread = threading.Thread(target=self.run_client)
		client_thread.start()

		n_streams = 1
		explainer_queues, _, est_time_queues = run_easyrca_simulator_mp(tcp_server = self.tcp_server, 
					                         slide_size = self.slide_size, 
					                         window_size = self.window_size, 
					                         start_ts = self.basic_time, 
					                         msg_size = self.msg_size, 
					                         msg_format = self.msg_format, 
					                         n_streams = n_streams,
					                         sliding_window = sliding_window,
					                         causal_graph = self.causal_graph,
					                         m_samples = 0.75,
					                         slide_number = 1,
					                         features = self.features,
					                         target_node = self.target_node,
					                         nslide_used=None,
					                         gamma_max=1,
					                         sig_threshold=0.05)
		for i in range(n_streams):
			explainer_queue = explainer_queues[i]
			if not explainer_queue.empty():
				result = explainer_queue.get()
				if result is not None:
					root_causes, outlier_index = result
					print(f'root_causes = {root_causes}, outlier_index = {outlier_index}')
					#self.assertEqual(outlier_index, 4)
					self.assertIsInstance(root_causes[0],dict)
					self.assertIn('roots',root_causes[0][0])
				else:
					print(f'result is {result}')

		# do not forget the close client_thread
		client_thread.join()


		
if __name__ == '__main__': # pragma: no cover
    unittest.main()