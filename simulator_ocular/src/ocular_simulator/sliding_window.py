import datetime
import sys
import os

from threading import Thread
from multiprocessing import Process

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def map_time_to_seconds(size, ori_unit = 'seconds'):
	if ori_unit == 'seconds':
		return size
	elif ori_unit == 'minutes':
		size = 60 * size 
	elif ori_unit == 'hours':
		size = 3600 * size
	elif ori_unit == 'days':
		size = 24 * 3600 * size
	else:
		logging.warning(f'Unhandled time unit {ori_unit}')
	return size 


class Slide:
	"""
	A class to represent a Slide in the time-based sliding window.
	...

	Attributes
	----------
	size : int
	    the size of the Slide
	unit : str
	    the unit of the Slide's size, 
	    for example 'minutes'
	start_ts : datetime.datetime
		datetime when the slide starts
	end_ts : datetime.datetime
		datetime when the slide ends
	data : list
		list of data belongs to the slide
	"""
	def __init__(self, start_ts, size, unit='seconds'):
		self.size = map_time_to_seconds(size, unit)
		self.unit = unit
		self.start_ts = start_ts
		self.end_ts = start_ts + datetime.timedelta(seconds=self.size)
		self.data = list()

	def add_incoming_datum(self, datum, event_time):
		"""
		Add incoming datum into the list of datum
		.....
		Parameters
		----------
		datum : Tuple
			is a d-dimensional object
		event_time: datetime.datetime
			datetime of the datum
		"""
		if self.start_ts <= event_time < self.end_ts:
			self.data.append(datum)


class SlidingWindow:
	def __init__(self, slide_size, window_size, unit='seconds'):
		self.slide_size = map_time_to_seconds(slide_size, unit)
		self.window_size = map_time_to_seconds(window_size, unit)
		self.max_slides = self.window_size//self.slide_size
		self.window = dict()
		self.oldest_slide_ts = None
		self.latest_slide_ts = None

	def add_new_slide(self, slide, start_ts):
		"""
		adding a new slide into the window 
		if the window already has maximum number of slides, 
		let the oldest slide expire (removed from the window)
		"""
		if len(self.window) == self.max_slides:
			del[self.window[self.oldest_slide_ts]]
		self.latest_slide_ts = start_ts
		self.window[self.latest_slide_ts] = slide
		self.oldest_slide_ts = min(self.window.keys())

	def add_new_slide_with_number(self, slide, start_ts, slide_number):
		"""
		adding a new slide into the window 
		if the window already has maximum number of slides, 
		let the oldest slide expire (removed from the window)
		"""
		if len(self.window) == self.max_slides:
			del[self.window[min(self.window.keys())]]
		self.window[slide_number] = slide
		self.latest_slide_ts = start_ts

def run_sliding_window(queue1, sliding_window_q, start_ts, slide_size, slide_number=1):
	connected = True
	end_ts = start_ts + datetime.timedelta(seconds=slide_size) 
	stream_id = None
	while connected:
		## get new slide from the queue
		try:
			logging.info(f"sliding window #{slide_number}")
			new_slide = queue1.get()
			if new_slide is None:
				sliding_window_q.put(None)
				connected = False
				# logging.info(f'end sliding window')
			else:
				sliding_window = sliding_window_q.get()
				if stream_id is None:
					stream_id = new_slide["data_list"][0]["client_id"]
				sliding_window.add_new_slide_with_number(new_slide, start_ts, slide_number)
				sliding_window_q.put(sliding_window)
				slide_number += 1
				start_ts = end_ts
				end_ts += datetime.timedelta(seconds=sliding_window.slide_size)
		except Exception as e: # pragma: no cover
			logging.error(f'Exception at Sliding Window Handler : {e}')
	logging.info(f'Exiting sliding window')

class SlidingWindowThread(Thread):
	def __init__(self, queue, sliding_window_q, start_ts, slide_size):
		Thread.__init__(self)
		self.queue = queue
		self.sliding_window_q = sliding_window_q
		self.start_ts = start_ts
		self.slide_size = slide_size

	def run(self): # pragma: no cover
		run_sliding_window(self.queue, self.sliding_window_q, self.start_ts, self.slide_size)

def run_sliding_window_mp(queue, sliding_window_q, start_ts, slide_size, detector_queue, explainer_queue, slide_number=1):
    connected = True
    if isinstance(start_ts, datetime.datetime):
    	start_ts = start_ts.timestamp()
    end_ts = start_ts + slide_size
    stream_id = None
    while connected:
        ## get new slide from the queue
        try:
        	logging.info(f"sliding window #{slide_number}")
        	logging.info(f"----------------------------------------")
        	new_slide = queue.get()
        except Exception as e: # pragma: no cover
            pass
        else:
            if new_slide is None:
            	# logging.info("No more incoming data")
            	sliding_window_q.put(None)
            	detector_queue.put(None)
            	explainer_queue.put(None)
            	connected = False
            else:
            	# logging.info(f'new_slide {new_slide}')
            	sliding_window = sliding_window_q.get()
            	if stream_id is None:
            		stream_id = new_slide["data_list"][0]["client_id"]
            	sliding_window.add_new_slide_with_number(new_slide, start_ts, slide_number)
            	sliding_window_q.put(sliding_window)
            	detector_queue.put(sliding_window)
            	explainer_queue.put(sliding_window)
            	# logging.info(f'Receiving new_slide from Stream {stream_id} at {end_ts}\n{new_slide}')
            	# logging.info(f'Sliding window latest timestamp is {sliding_window.latest_slide_ts}')
            	# logging.info(f'Sliding window oldest timestamp is {sliding_window.oldest_slide_ts}')
            	start_ts = end_ts
            	end_ts += slide_size
            	slide_number += 1

class SlidingWindowMP(Process):
	def __init__(self, queue, sliding_window_q, start_ts, slide_size, detector_queue, explainer_queue):
		Process.__init__(self)
		self.queue = queue
		self.sliding_window_q = sliding_window_q
		self.start_ts = start_ts
		self.slide_size = slide_size
		self.detector_queue = detector_queue
		self.explainer_queue = explainer_queue

	def run(self): # pragma: no cover
		run_sliding_window_mp(self.queue, 
							  self.sliding_window_q,
							  self.start_ts, 
							  self.slide_size,
							  self.detector_queue,
							  self.explainer_queue)