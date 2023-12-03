import pandas as pd
import numpy as np
import datetime

from multiprocessing import Process

from odds.labeled_data import get_outliers
from odds.labeled_data import get_outlier_list
from ocular_simulator.utils import transform_data_list_to_numpy

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def run_detector_with_labeled_data(o_window_queue, slide_size, window_size, outlier_queue, start_ts, features):
	first_time = True 
	end_ts = None
	if isinstance(start_ts, datetime.datetime):
		start_ts = start_ts.timestamp()
	while(True):
		try:
			swindow = o_window_queue.get(block=False)
		except Exception as e:
			pass
		else:
			if first_time:
				end_ts = start_ts + slide_size
				first_time = False
			
			if swindow is None:
				outlier_queue.put(None)
				break
			else:
				latest_slide_num = max(swindow.window.keys())
				data_list = swindow.window[latest_slide_num]['data_list']
				arr, event_ts, label, index = transform_data_list_to_numpy(data_list)
				arr_outlier, event_ts_outlier, index_outlier = get_outliers(arr, event_ts, label, index)
				outliers = {'values' : arr_outlier, 'event_ts' : event_ts_outlier.tolist(), 'index': index_outlier}
				outlier_queue.put((outliers, latest_slide_num))
			while True:
				if (end_ts + slide_size < datetime.datetime.now().timestamp()):
					break
			start_ts = end_ts
			end_ts = end_ts + slide_size
	logging.info(f'Exiting Detector Process')

class DetectorMP(Process):
	def __init__(self, o_window_queue, slide_size, window_size, outlier_queue, start_ts, features):
		Process.__init__(self)
		self.o_window_queue = o_window_queue
		self.slide_size = slide_size
		self.window_size = window_size
		self.outlier_queue = outlier_queue
		self.start_ts = start_ts
		self.features = features

	def run(self): # pragma: no cover
		run_detector_with_labeled_data(self.o_window_queue, 
									   self.slide_size, 
									   self.window_size, 
									   self.outlier_queue, 
									   self.start_ts,
									   self.features)