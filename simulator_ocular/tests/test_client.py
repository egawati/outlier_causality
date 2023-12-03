import unittest

import socket
import threading
import time
import datetime
import os

from ocular_simulator.client import TCPClient
from ocular_simulator.client import run_client

import random
ip = '127.0.0.1'
port = random.randrange(49152, 65535)
rate = 0.1

class TestClient(unittest.TestCase):
	def run_fake_server(self):
		# Run a server to listen for a connection and then close it
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		
		server_address = (ip, port)
		
		# Bind the socket to the port
		sock.bind(server_address)
		
		# Listen for incoming connections
		sock.listen(1)
		connection, client_address = sock.accept()
		connection.recv(1024)
		time.sleep(rate)
		connection.recv(1024)
		connection.close()
		sock.close()

	def test_client(self):
		# Start fake server in background thread
		server_thread = threading.Thread(target=self.run_fake_server)
		server_thread.start()

		time.sleep(0.000001)  
		
		client = TCPClient(ip, port, size=1024, format="utf-8", disconnect_msg = "disconnect", client_id = 1)
		self.assertIsInstance(client.client, socket.socket)
		self.assertEqual(client.address, (ip, port))
		self.assertEqual(client.format, 'utf-8')
		self.assertEqual(client.client_id, 1)
		self.assertEqual(client.connected, False)
		self.assertEqual(client.disconnect_msg, "disconnect")
		
		client.client.settimeout(1)
		client.connect()
		self.assertEqual(client.connected, True)

		data = ['one', 'two']
		client.send_message(event_start_time=datetime.datetime.now(), rate=rate, data=data)
		self.assertEqual(client.connected, False)
		client.client.close()

		# Ensure server thread ends
		server_thread.join()

	def test_client_2(self):
		# Start fake server in background thread
		server_thread = threading.Thread(target=self.run_fake_server)
		server_thread.start()

		time.sleep(0.000001)  
		
		client = TCPClient(ip, port, size=1024, format="utf-8", disconnect_msg = "disconnect", client_id = 1)
		self.assertIsInstance(client.client, socket.socket)
		self.assertEqual(client.address, (ip, port))
		self.assertEqual(client.format, 'utf-8')
		self.assertEqual(client.client_id, 1)
		self.assertEqual(client.connected, False)
		self.assertEqual(client.disconnect_msg, "disconnect")
		
		client.client.settimeout(1)
		client.connect()
		self.assertEqual(client.connected, True)

		data = ['one', 'two']
		client.send_message(event_start_time=datetime.datetime.now().timestamp(), rate=rate, data=data)
		self.assertEqual(client.connected, False)
		client.client.close()

		# Ensure server thread ends
		server_thread.join()


	def test_client_dict(self):
		# Start fake server in background thread
		server_thread = threading.Thread(target=self.run_fake_server)
		server_thread.start()

		time.sleep(0.000001)  
		
		client = TCPClient(ip, port, size=1024, format="utf-8", disconnect_msg = "disconnect", client_id = 1)
		self.assertIsInstance(client.client, socket.socket)
		self.assertEqual(client.address, (ip, port))
		self.assertEqual(client.format, 'utf-8')
		self.assertEqual(client.client_id, 1)
		self.assertEqual(client.connected, False)
		self.assertEqual(client.disconnect_msg, "disconnect")
		
		client.client.settimeout(1)
		client.connect()
		self.assertEqual(client.connected, True)

		basic_time = datetime.datetime.now()

		data = [{'values':[1,10], 'client_id':1, 'event_ts' : basic_time.strftime('%Y-%m-%d %H:%M:%S'), 'label': 0}, 
				{'values':[2,20], 'client_id':1, 'event_ts' : basic_time.strftime('%Y-%m-%d %H:%M:%S'), 'label':0}]
		client.send_message(event_start_time=datetime.datetime.now(), rate=rate, data=data)
		
		self.assertEqual(client.connected, False)
		client.client.close()

		# Ensure server thread ends
		server_thread.join()

	def test_run_client(self):
		# Start fake server in background thread
		server_thread = threading.Thread(target=self.run_fake_server)
		server_thread.start()

		time.sleep(0.000001)  
		
		filepath = os.path.join(os.path.dirname(__file__), 'data.csv')
		self.assertIn('data.csv', filepath)
		run_client(IP=ip, PORT=port, client_id=1, date='2023-01-11', time='12:12:13', rate=rate, source=filepath)

		# Ensure server thread ends
		server_thread.join()

if __name__ == '__main__': # pragma: no cover
    unittest.main()
