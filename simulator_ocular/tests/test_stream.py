import unittest

import socket
import threading
import time
import datetime

from queue import Queue

from ocular_simulator.client import TCPClient
from ocular_simulator.server import TCPServer
from ocular_simulator.stream import run_simulator


import random


class TestSimulator(unittest.TestCase):
	def setUp(self):
		"""
		When a setUp() method is defined, the test runner will run that method prior to each test. 
		"""
		self.ip = '127.0.0.1'
		self.port = random.randrange(49152, 65535)
		self.slide_size = 2
		self.window_size = 4
		self.basic_time = datetime.datetime.now()
		self.msg_size = 1024
		self.msg_format = 'utf-8'
		
		# initialize TCP server
		self.tcp_server = TCPServer(self.ip, self.port)
		self.assertIsInstance(self.tcp_server, TCPServer)
		self.assertIsInstance(self.tcp_server.server, socket.socket)

	def run_one_client(self):
		client = TCPClient(self.ip, self.port, size=self.msg_size, 
						   format=self.msg_format, disconnect_msg = "disconnect", client_id = 1)
		client.client.settimeout(1)
		client.connect()
		rate = 1
		data = ['one', 'two', 'three', 'four', 'five']
		event_start_time = self.basic_time
		client.send_message(event_start_time, rate, data)
		client.client.close()


	def test_simulator_one_client(self):
		self.assertIsInstance(self.tcp_server, TCPServer)
		self.assertIsInstance(self.tcp_server.server, socket.socket)
		
		# run client in the background
		client_thread = threading.Thread(target=self.run_one_client)
		client_thread.start()

		run_simulator(self.tcp_server,
					  self.slide_size, self.window_size, 
					  start_ts=self.basic_time, 
					  msg_size=self.msg_size, 
					  msg_format=self.msg_format, n_streams=1)

		# do not forget the close client_thread
		client_thread.join()

if __name__ == '__main__': # pragma: no cover
    unittest.main()


