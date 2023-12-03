import socket
import threading
import json

import queue
import datetime

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

from ocular_simulator.server import TCPServer
from ocular_simulator.server import ClientHandlerThread

from ocular_simulator.sliding_window import SlidingWindow
from ocular_simulator.sliding_window import SlidingWindowThread

def run_simulator(tcp_server, slide_size, window_size, start_ts, msg_size, msg_format, n_streams):
    """
    ip : string ip address e.g. '127.0.0.1'
    slide_size : int, slide size in seconds
    window_size : int, window size in seconds
    start_ts : datetime.datetime, marked when the server starts running
    """
    window_threads = list()
    queues = list()
    sw_queues = list()
    clients = list()
    sliders = list()

    ## initialize threads for each client
    for _ in range(n_streams):
        conn, client_addr = tcp_server.server.accept()
        queue1 = queue.Queue()
        queues.append(queue1)
        client = ClientHandlerThread(conn, client_addr, queues[-1], 
                                     slide_size, start_ts, 
                                     msg_size, msg_format)
        client.start()
        clients.append(client)

        ## create sliding window thread for each corresponding client thread
        sliding_window = SlidingWindow(slide_size, window_size, unit='seconds')
        sw_queue = queue.Queue()
        sw_queue.put(sliding_window)
        sw_queues.append(sw_queue)
        slider = SlidingWindowThread(queues[-1], sw_queues[-1], start_ts, slide_size)
        slider.start()
        sliders.append(slider)

        logging.info(f'Number of active client {len(clients)}')
        
    logging.info("finished initializing client thread")
    for i in range(n_streams):
        clients[i].join()
        sliders[i].join()
    logging.info("done join")
    tcp_server.server.close()
    logging.info("close server")