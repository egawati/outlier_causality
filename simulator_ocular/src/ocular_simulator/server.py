import socket
import json
import sys

import datetime

import threading

from multiprocessing import Process

from ocular_simulator.utils import process_data_list

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def handle_client(conn, addr, queue, slide_size, start_ts, size, format, handler_type='thread'):
    """
    Handling incoming message from each client
    ....
    Parameters:
    -----------
    conn        : socket.socket
    addr        : tuple of client's IP and PORT
    queue       : Queue()
    slide_size  : int
        time based buffer size in seconds
    size : int
        the maximum size of received message
    format : string
        the format of incoming message
    """
    logging.info(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    data_list = list() ## use as a buffer for Slide before sending it to the Sliding Window
    first_time = True
    if isinstance(start_ts, datetime.datetime):
        start_ts = start_ts.timestamp()
    end_ts = start_ts + slide_size
    client_id = None
    while connected:
        msg = conn.recv(size)
        # logging.info(f'msg received : \n{msg}')
        # logging.info(f'len msg = {len(msg)}')
        # logging.info(f'size = {sys.getsizeof(msg)} bytes')
        data = json.loads(msg.decode(format))
        if 'disconnect' in data:
            ## before disconneting send the last buffer/Slide to the queue
            data_dict = process_data_list(data_list, start_ts, end_ts)
            queue.put(data_dict) ## add the last buffer to the queue
            #logging.info(f'\n[{datetime.datetime.now()}] Last Slide : {data_dict}\n')

            ## need to put None into the queue for clean exit later
            queue.put(None) 
            logging.info(f'\nHanlder disconnect client {client_id}\n')
            connected = False
        else:
            if isinstance(data["event_ts"], str):
                data["event_ts"] = datetime.datetime.strptime(data["event_ts"], '%Y-%m-%d %H:%M:%S').timestamp()
            if first_time:
                # logging.info(f'starting time is {start_ts}')
                first_time = False
                # logging.info(f"next sliding is {end_ts}")
                # client_id = data["client_id"]
            
            if  data["event_ts"] < end_ts:
                data_list.append(data)
                #logging.info(f'append [{addr}] {data} into data list')
            else:
                ## send the current buffer (Slide) to the queue
                data_dict = process_data_list(data_list, start_ts, end_ts) 
                queue.put(data_dict) ## it is time to put the slide 
                # logging.info(f"\n[{datetime.datetime.now()}] Slide : {data_dict}\n")
                
                ## create a new buffer (new slide)
                start_ts = end_ts
                data_list = list()
                data_list.append(data)
                end_ts += slide_size
                # logging.info(f'append [{addr}] {data} into new data list')
                # logging.info(f"next sliding is {end_ts}")
    logging.info("closing connection from client")
    conn.close()


class ClientHandlerMP(Process): 
    """
    Client Handler using multiprocessing
    Incoming data from each client are handled by a process
    ...

    Attributes
    ----------
    conn : socket.socket
    addr : tuple
        tuple of (IP, PORT) of client
    queue : Queue
        used to pass Slide to the Sliding Window thread
    slide_size : int
        the size of a Slide in seconds
    start_ts : datetime.datetime
        datetime when the server starts
    size : int
        the maximum size received for each incoming message
    format : str
        the format of the message received
    
    """
    def __init__(self, conn, addr, queue, slide_size, start_ts, size=2048, 
                 msg_format='utf-8', handler_type='thread'): 
        Process.__init__(self) 
        self.conn = conn
        self.addr = addr
        self.queue = queue
        self.slide_size = slide_size
        self.start_ts = start_ts
        self.size = size
        self.msg_format = msg_format
        self.handler_type = handler_type
        logging.info(f"New process for handling client {self.addr}")
        
    def run(self): 
        handle_client(self.conn, self.addr, self.queue, 
                      self.slide_size, self.start_ts, 
                      self.size, self.msg_format, self.handler_type)

class ClientHandlerThread(threading.Thread): 
    """
    Client Handler using Thread
    Incoming data from each client are handled by a thread
    ...

    Attributes
    ----------
    conn : socket.socket
    addr : tuple
        tuple of (IP, PORT) of client
    queue : Queue
        used to pass Slide to the Sliding Window thread
    slide_size : int
        the size of a Slide in seconds
    start_ts : datetime.datetime
        datetime when the server starts
    size : int
        the maximum size received for each incoming message
    format : str
        the format of the message received
    
    """
    def __init__(self, conn, addr, queue, slide_size, start_ts, size=2048, msg_format='utf-8'): 
        threading.Thread.__init__(self) 
        self.conn = conn
        self.addr = addr
        self.queue = queue
        self.slide_size = slide_size
        self.start_ts = start_ts
        self.size = size
        self.msg_format = msg_format
        logging.info(f"New thread for client {self.addr}")
        
    def run(self): 
        handle_client(self.conn, self.addr, self.queue, 
                      self.slide_size, self.start_ts, 
                      self.size, self.msg_format)

class TCPServer():
    """
    TCP Server class
    ...

    Attributes
    ----------
    ip : str
        IP address
    port : int
        port number
    size : int
        the maximum size received for each incoming message
    format : str
        the format of the message received
    server : socket.socket
        server socket
    start_ts : datetime.datetime
        datetime when the server starts
    """
    def __init__(self, ip, port, 
                 size=1024, format="utf-8"):
        self.addr = (ip, port)
        self.size = size
        self.format = format 
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(self.addr)
        self.start_ts = datetime.datetime.now()
        self.server.listen()