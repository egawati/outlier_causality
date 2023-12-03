import sys
import socket
import time
import datetime

import json
import os

from .data import read_csv

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


class TCPClient:
    def __init__(self, ip, port, size=1024, format="utf-8", 
                 disconnect_msg = "disconnect", client_id = None):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = (ip, port)
        self.format = format
        self.disconnect_msg = disconnect_msg
        self.connected = False
        self.client_id = client_id

    def connect(self):
        while not self.connected:
            try:
                self.client.connect(self.address)
                self.connected = True
            except Exception as e: # pragma: no cover
                continue

    def send_message(self, event_start_time, rate,  data):
        i = 0 
        if isinstance(event_start_time, datetime.datetime):
            event_start_time = event_start_time.timestamp()
        while self.connected:
            if i < len(data):
                if isinstance(data[i], dict):
                    data[i]['event_ts'] = event_start_time
                    msg = json.dumps(data[i], default=str)
                else:
                    msg = json.dumps({"values": data[i], "event_ts": event_start_time, 
                                  "client_id":self.client_id}, default=str)
            else:
                msg = json.dumps({"disconnect" : self.disconnect_msg})
                self.connected = False
            self.client.send(msg.encode(self.format))
            i += 1
            event_start_time = event_start_time + rate
            time.sleep(rate)
        self.client.close()


def run_client(IP, PORT, client_id, date, time, rate, source):
    """
    IP  : IP address, '127.0.0.1'
    PORT : Port number, 5566
    client_id : client number, 1
    date : date, following the format %Y-%m-%d
    time : time, following the format %H:%M:%S
    rate : arrival rate of sending each message in seconds
    source : data source (filepath)
    """
    data = list()
    if '.csv' in source:
        data = read_csv(source)
    client = TCPClient(IP, PORT, client_id)
    client.connect()
    event_start_time = datetime.datetime.strptime(f'{date} {time}', '%Y-%m-%d %H:%M:%S')
    client.send_message(event_start_time, rate, data)


if __name__ == "__main__": # pragma: no cover
    if len(sys.argv) != 7:
        msg = ("Usage: client.py [date] [time] [client_id]")
        print(msg, file=sys.stderr)
        sys.exit(-1)
    run_client(IP=sys.argv[1], PORT=sys.argv[2], 
               client_id=sys.argv[3], date=sys.argv[4], 
               time=sys.argv[5], rate=sys.argv[6],
               source=sys.argv[7])