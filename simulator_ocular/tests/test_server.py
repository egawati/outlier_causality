import unittest

import socket
import threading
import time
import datetime
import os
import pandas as pd
import numpy as np

from queue import Queue
from multiprocessing import Manager

from ocular_simulator.client import TCPClient
from ocular_simulator.server import TCPServer
from ocular_simulator.server import ClientHandlerThread
from ocular_simulator.server import ClientHandlerMP
from ocular_simulator.server import handle_client

import random

IP = '127.0.0.1'

class TestServer(unittest.TestCase):
	def setUp(self):
		"""
		When a setUp() method is defined, the test runner will run that method prior to each test. 
		"""
		self.data_queues = list()

		n_samples = 12
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		self.df = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(self.features))), columns=self.features)
		self.data_list = self.df.values.tolist()

		self.slide_size = 2 ## in seconds
		self.window_size = 4 ## in seconds
		self.window_threads = list()

		self.start_ts = datetime.datetime.now()
		self.basic_time = datetime.datetime.now().timestamp()

		self.queues = list()
		self.clients = list()
		self.sliders = list()

		self.msg_size = 1024
		self.msg_format = 'utf-8'

		self.n_streams = 1

		self.PORT = random.randrange(49152, 65535)


		self.tcp_server = TCPServer(IP, self.PORT, self.msg_size, self.msg_format)
		self.assertEqual(self.tcp_server.addr, (IP, self.PORT))
		self.assertEqual(self.tcp_server.size, self.msg_size)
		self.assertEqual(self.tcp_server.format, self.msg_format)
		self.assertIsInstance(self.tcp_server.server, socket.socket)
		self.assertIsInstance(self.tcp_server.start_ts, datetime.datetime)

		time.sleep(0.000001)


	def tearDown(self):
		"""
		When a tearDown() method is defined, the test runner will invoke that method after each test.
		"""
		self.tcp_server.server.close()

	def run_server_df(self): # pragma: no cover
		rate = 0.5
		data = list()
		for i, datum in enumerate(self.data_list):
			if i != 3:
				data.append({'values':datum, 'client_id':1, 'event_ts':self.basic_time + i*rate, 'label':0})
			else:
				data.append({'values':datum, 'client_id':1 ,'event_ts':self.basic_time + i*rate, 'label':1})

		client = TCPClient(IP, self.PORT, size=1024, format="utf-8", disconnect_msg = "disconnect", client_id = 1)
		client.client.settimeout(1)
		client.connect()
		time.sleep(2) ## need to wait before sending data so that server can be done with its configuration first
		event_start_time = self.basic_time
		client.send_message(event_start_time, rate, data)
		client.client.close()

	def test_server_one_df(self):
		# run client in the background
		client_thread = threading.Thread(target=self.run_server_df)
		client_thread.start()

		# server receiving connection from client
		conn, client_addr = self.tcp_server.server.accept()
		queue = Queue()
		client = ClientHandlerThread(conn, client_addr, queue, 
		                             self.slide_size, self.start_ts, 
		                             self.msg_size, self.msg_format)
		client.start()

		self.assertIsInstance(conn, socket.socket)
		# do not forget to end client_thread
		client_thread.join()
		self.assertEqual(queue.empty(), False)
	
	def run_one_client(self): 
		client = TCPClient(IP, self.PORT, size=1024, format="utf-8", disconnect_msg = "disconnect", client_id = 1)
		client.client.settimeout(1)
		client.connect()
		rate = 1
		data = ['one', 'two', 'three', 'four', 'five']
		time.sleep(0.5) ## need to wait before sending data so that server can be done with its configuration first
		event_start_time = datetime.datetime.now()
		client.send_message(event_start_time, rate, data)
		client.client.close()


	def test_server_one_client(self):
		# run client in the background
		client_thread = threading.Thread(target=self.run_one_client)
		client_thread.start()

		# server receiving connection from client
		conn, client_addr = self.tcp_server.server.accept()
		queue = Queue()
		client = ClientHandlerThread(conn, client_addr, queue, 
		                             self.slide_size, self.start_ts, 
		                             self.msg_size, self.msg_format)
		client.start()

		self.assertIsInstance(conn, socket.socket)

		server_data = queue.get()
		self.assertEqual(server_data['data_list'][0]['values'], 'one')
		self.assertEqual(server_data['data_list'][1]['values'], 'two')
		self.assertIn('values',server_data['data_list'][0].keys())
		self.assertIn('event_ts', server_data['data_list'][0].keys())
		self.assertIn('client_id', server_data['data_list'][0].keys())
		self.assertIn('values',server_data['data_list'][1].keys())
		self.assertIn('event_ts', server_data['data_list'][1].keys())
		self.assertIn('client_id', server_data['data_list'][1].keys())

		server_data = queue.get()
		self.assertEqual(server_data['data_list'][0]['values'], 'three')
		self.assertEqual(server_data['data_list'][1]['values'], 'four')

		# do not forget to end client_thread
		client_thread.join()

	def test_server_one_client_mp(self):
		# run client in the background
		client_thread = threading.Thread(target=self.run_one_client)
		client_thread.start()

		# server receiving connection from client
		conn, client_addr = self.tcp_server.server.accept()
		manager = Manager()
		queue = manager.Queue()
		client = ClientHandlerMP(conn, client_addr, queue, 
								 self.slide_size, self.start_ts, 
								 self.msg_size, self.msg_format)
		client.start()

		self.assertIsInstance(conn, socket.socket)

		server_data = queue.get()
		self.assertEqual(server_data['data_list'][0]['values'], 'one')
		self.assertEqual(server_data['data_list'][1]['values'], 'two')
		self.assertIn('values',server_data['data_list'][0].keys())
		self.assertIn('event_ts', server_data['data_list'][0].keys())
		self.assertIn('client_id', server_data['data_list'][0].keys())
		self.assertIn('values',server_data['data_list'][1].keys())
		self.assertIn('event_ts', server_data['data_list'][1].keys())
		self.assertIn('client_id', server_data['data_list'][1].keys())

		server_data = queue.get()
		self.assertEqual(server_data['data_list'][0]['values'], 'three')
		self.assertEqual(server_data['data_list'][1]['values'], 'four')

		# do not forget to end client_thread
		client_thread.join()

	def test_handle_client(self):
		# run client in the background
		client_thread = threading.Thread(target=self.run_one_client)
		client_thread.start()

		# server receiving connection from client
		conn, client_addr = self.tcp_server.server.accept()
		queue = Queue()
		self.assertIsInstance(conn, socket.socket)

		handler_tread = threading.Thread(target=handle_client, args=(conn, client_addr, queue, self.slide_size, self.start_ts, self.msg_size, self.msg_format))
		handler_tread.start()

		server_data = queue.get()
		self.assertEqual(server_data['data_list'][0]['values'], 'one')
		self.assertEqual(server_data['data_list'][1]['values'], 'two')
		self.assertIn('values',server_data['data_list'][0].keys())
		self.assertIn('event_ts', server_data['data_list'][0].keys())
		self.assertIn('client_id', server_data['data_list'][0].keys())
		self.assertIn('values',server_data['data_list'][1].keys())
		self.assertIn('event_ts', server_data['data_list'][1].keys())
		self.assertIn('client_id', server_data['data_list'][1].keys())

		client_thread.join()
		handler_tread.join()

	def run_one_client_dict(self): # pragma: no cover
		client = TCPClient(IP, self.PORT, size=1024, format="utf-8", disconnect_msg = "disconnect", client_id = 1)
		#client.client.settimeout(1)
		client.connect()
		rate = 1
		basic_time = datetime.datetime.now().timestamp
		data = [{'values':[1,10], 'client_id':1, 'event_ts' : basic_time, 'label': 0, 'index':0}, 
				{'values':[2,20], 'client_id':1, 'event_ts' : basic_time, 'label':0, 'index':1}]
		time.sleep(0.5) ## need to wait before sending data so that server can be done with its configuration first
		event_start_time = datetime.datetime.now()
		client.send_message(event_start_time, rate, data)
		client.client.close()

	def test_handle_client_dict(self):
		queue = Queue()

		# run client in the background
		client_thread = threading.Thread(target=self.run_one_client_dict)
		client_thread.start()

		# server receiving connection from client
		conn, client_addr = self.tcp_server.server.accept()
		
		handler_tread = threading.Thread(target=handle_client, args=(conn, client_addr, queue, self.slide_size, self.start_ts, self.msg_size, self.msg_format))
		handler_tread.start()

		server_data = queue.get()
		print(f'server_data {server_data}')
		self.assertEqual(server_data['data_list'][0]['values'], [1,10])
		self.assertEqual(server_data['data_list'][1]['values'], [2,20])
		self.assertIn('values',server_data['data_list'][0].keys())
		self.assertIn('event_ts', server_data['data_list'][0].keys())
		self.assertIn('client_id', server_data['data_list'][0].keys())
		self.assertIn('values',server_data['data_list'][1].keys())
		self.assertIn('event_ts', server_data['data_list'][1].keys())
		self.assertIn('client_id', server_data['data_list'][1].keys())

		client_thread.join()
		handler_tread.join()

if __name__ == '__main__': # pragma: no cover
    unittest.main()


