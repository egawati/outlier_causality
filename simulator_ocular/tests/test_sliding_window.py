import unittest

import time
import datetime

from queue import Queue
from multiprocessing import Manager

from multiprocessing.managers import BaseManager
from queue import LifoQueue

from ocular_simulator.sliding_window import SlidingWindow
from ocular_simulator.sliding_window import run_sliding_window
from ocular_simulator.sliding_window import run_sliding_window_mp
from ocular_simulator.sliding_window import SlidingWindowThread
from ocular_simulator.sliding_window import SlidingWindowMP
from ocular_simulator.sliding_window import map_time_to_seconds
from ocular_simulator.sliding_window import Slide

import random

IP = '127.0.0.1'
rate = 0.1

class MyManager(BaseManager): # pragma: no cover
    pass
MyManager.register('LifoQueue', LifoQueue) # pragma: no cover

class TestSlidingWindow(unittest.TestCase):
	def test_sliding_window(self):
		swindow = SlidingWindow(slide_size=2, window_size=4, unit='seconds')
		self.assertEqual(swindow.slide_size, 2)
		self.assertEqual(swindow.window_size, 4)
		self.assertEqual(swindow.max_slides, 2)
		self.assertIsInstance(swindow.window, dict)
		self.assertIsNone(swindow.oldest_slide_ts)
		self.assertIsNone(swindow.latest_slide_ts)

		basic_time = datetime.datetime.now()
		swindow.add_new_slide(slide=[], start_ts=basic_time)
		self.assertEqual(swindow.oldest_slide_ts, basic_time)
		self.assertEqual(swindow.latest_slide_ts, basic_time)

		swindow.add_new_slide(slide=[], start_ts=basic_time + datetime.timedelta(seconds=2))
		self.assertEqual(swindow.oldest_slide_ts, basic_time)
		self.assertEqual(swindow.latest_slide_ts, basic_time + datetime.timedelta(seconds=2))

		swindow.add_new_slide(slide=[], start_ts=basic_time + datetime.timedelta(seconds=4))
		self.assertEqual(swindow.oldest_slide_ts, basic_time + datetime.timedelta(seconds=2))
		self.assertEqual(swindow.latest_slide_ts, basic_time + datetime.timedelta(seconds=4))

	def test_sliding_window_with_slide_number(self):
		swindow = SlidingWindow(slide_size=2, window_size=4, unit='seconds')
		self.assertEqual(swindow.slide_size, 2)
		self.assertEqual(swindow.window_size, 4)
		self.assertEqual(swindow.max_slides, 2)
		self.assertIsInstance(swindow.window, dict)
		self.assertIsNone(swindow.oldest_slide_ts)
		self.assertIsNone(swindow.latest_slide_ts)

		basic_time = datetime.datetime.now()
		swindow.add_new_slide_with_number(slide={}, start_ts=basic_time, slide_number=1)
		self.assertEqual(len(swindow.window),1)
		self.assertIn(1, swindow.window.keys())

		swindow.add_new_slide_with_number(slide={}, start_ts=basic_time + datetime.timedelta(seconds=2), slide_number=2)
		self.assertEqual(len(swindow.window),2)
		self.assertIn(2, swindow.window.keys())

	def test_run_sliding_window(self):
		queue = Queue()
		sliding_window_q = Queue()

		swindow = SlidingWindow(slide_size=2, window_size=4, unit='seconds')
		sliding_window_q.put(swindow)

		self.assertEqual(swindow.slide_size, 2)
		self.assertEqual(swindow.window_size, 4)
		self.assertEqual(swindow.max_slides, 2)
		self.assertIsInstance(swindow.window, dict)
		self.assertIsNone(swindow.oldest_slide_ts)
		self.assertIsNone(swindow.latest_slide_ts)

		basic_time = datetime.datetime.now()

		slide = {}
		slide["data_list"] = [{'values':[1], 'client_id':1}, {}]
		slide["start_time"] = basic_time
		slide["end_time"] = basic_time + datetime.timedelta(seconds=2)

		queue.put(slide)
		queue.put(None)
		run_sliding_window(queue, sliding_window_q, start_ts=basic_time, slide_size=2)
		swindow = sliding_window_q.get()
		self.assertEqual(min(swindow.window.keys()), 1)
		self.assertEqual(swindow.latest_slide_ts, basic_time)


	def test_run_sliding_window_mp(self):
		manager = Manager()
		queue = manager.Queue()
		sliding_window_q = manager.Queue()
		
		l_manager = MyManager()
		l_manager.start()
		detector_queue = l_manager.LifoQueue()
		explainer_queue = l_manager.LifoQueue()

		swindow = SlidingWindow(slide_size=2, window_size=4, unit='seconds')
		sliding_window_q.put(swindow)

		self.assertEqual(swindow.slide_size, 2)
		self.assertEqual(swindow.window_size, 4)
		self.assertEqual(swindow.max_slides, 2)
		self.assertIsInstance(swindow.window, dict)
		self.assertIsNone(swindow.oldest_slide_ts)
		self.assertIsNone(swindow.latest_slide_ts)

		basic_time = datetime.datetime.now()

		slide = {}
		slide["data_list"] = [{'values':[1], 'client_id':1}, {}]
		slide["start_time"] = basic_time
		slide["end_time"] = basic_time + datetime.timedelta(seconds=2)

		queue.put(slide)
		queue.put(None)
		run_sliding_window_mp(queue, sliding_window_q, 
							  start_ts=basic_time, slide_size=2, 
							  detector_queue=detector_queue,
							  explainer_queue=explainer_queue)
		swindow = sliding_window_q.get()
		self.assertEqual(min(swindow.window.keys()), 1)
		self.assertEqual(datetime.datetime.fromtimestamp(swindow.latest_slide_ts), basic_time)

		ewindow = explainer_queue.get()
		ewindow = explainer_queue.get()
		self.assertEqual(min(ewindow.window.keys()), 1)
		self.assertEqual(datetime.datetime.fromtimestamp(ewindow.latest_slide_ts), basic_time)

		dwindow = detector_queue.get()
		dwindow = detector_queue.get()
		self.assertEqual(min(dwindow.window.keys()), 1)
		self.assertEqual(datetime.datetime.fromtimestamp(dwindow.latest_slide_ts), basic_time)

	def test_run_sliding_window_mp_2(self):
		manager = Manager()
		queue = manager.Queue()
		sliding_window_q = manager.Queue()
		
		l_manager = MyManager()
		l_manager.start()
		detector_queue = l_manager.LifoQueue()
		explainer_queue = l_manager.LifoQueue()

		swindow = SlidingWindow(slide_size=2, window_size=4, unit='seconds')
		sliding_window_q.put(swindow)

		self.assertEqual(swindow.slide_size, 2)
		self.assertEqual(swindow.window_size, 4)
		self.assertEqual(swindow.max_slides, 2)
		self.assertIsInstance(swindow.window, dict)
		self.assertIsNone(swindow.oldest_slide_ts)
		self.assertIsNone(swindow.latest_slide_ts)

		basic_time = datetime.datetime.now().timestamp()

		slide = {}
		slide["data_list"] = [{'values':[1], 'client_id':1}, {}]
		slide["start_time"] = basic_time
		slide["end_time"] = basic_time + 2

		queue.put(slide)
		queue.put(None)
		run_sliding_window_mp(queue, sliding_window_q, 
							  start_ts=basic_time, slide_size=2, 
							  detector_queue=detector_queue,
							  explainer_queue=explainer_queue)
		swindow = sliding_window_q.get()
		self.assertEqual(min(swindow.window.keys()), 1)
		self.assertEqual(swindow.latest_slide_ts, basic_time)

		ewindow = explainer_queue.get()
		ewindow = explainer_queue.get()
		self.assertEqual(min(ewindow.window.keys()), 1)
		self.assertEqual(ewindow.latest_slide_ts, basic_time)

		dwindow = detector_queue.get()
		dwindow = detector_queue.get()
		self.assertEqual(min(dwindow.window.keys()), 1)
		self.assertEqual(dwindow.latest_slide_ts, basic_time)
	
	
	def test_sliding_window_thread(self):
		queue = Queue()
		sliding_window_q = Queue()
		
		swindow = SlidingWindow(slide_size=2, window_size=4, unit='seconds')
		sliding_window_q.put(swindow)
		
		self.assertEqual(swindow.slide_size, 2)
		self.assertEqual(swindow.window_size, 4)
		self.assertEqual(swindow.max_slides, 2)
		self.assertIsInstance(swindow.window, dict)
		self.assertIsNone(swindow.oldest_slide_ts)
		self.assertIsNone(swindow.latest_slide_ts)

		basic_time = datetime.datetime.now()

		# Slide 1
		slide = {}
		slide["data_list"] = [{'values':1, 'client_id':1, 'event_ts' : basic_time}, 
							  {'values':2, 'client_id':1, 'event_ts' : basic_time + datetime.timedelta(seconds=1)},
							  {'values':2, 'client_id':1, 'event_ts' : basic_time + datetime.timedelta(seconds=1.9)},]
		slide["start_time"] = basic_time
		slide["end_time"] = basic_time + datetime.timedelta(seconds=2)
		queue.put(slide)
		self.assertEqual(queue.qsize(),1)

		# Slide 2
		slide = {}
		slide["data_list"] = [{'values':1, 'client_id':1, 'event_ts' : basic_time + datetime.timedelta(seconds=3)}, 
							  {'values':2, 'client_id':1, 'event_ts' : basic_time + datetime.timedelta(seconds=3.9)},]
		slide["start_time"] = basic_time + datetime.timedelta(seconds=2)
		slide["end_time"] = basic_time + datetime.timedelta(seconds=4)
		queue.put(slide)
		self.assertEqual(queue.qsize(),2)

		queue.put(None)
		self.assertEqual(queue.qsize(),3)

		swindow_thread = SlidingWindowThread(queue, sliding_window_q, start_ts=basic_time, slide_size=2)
		swindow_thread.start()
		swindow_thread.join()

		swindow = sliding_window_q.get()
		self.assertEqual(min(swindow.window.keys()), 1)
		self.assertEqual(swindow.latest_slide_ts, basic_time + datetime.timedelta(seconds=2))
		self.assertEqual(len(swindow.window.keys()), 2)

	def test_sliding_window_mp(self):
		queue = Manager().Queue()
		sliding_window_q = Manager().Queue()

		manager = MyManager()
		manager.start()
		detector_queue = manager.LifoQueue()
		explainer_queue = manager.LifoQueue()
		
		swindow = SlidingWindow(slide_size=2, window_size=4, unit='seconds')
		sliding_window_q.put(swindow)
		
		self.assertEqual(swindow.slide_size, 2)
		self.assertEqual(swindow.window_size, 4)
		self.assertEqual(swindow.max_slides, 2)
		self.assertIsInstance(swindow.window, dict)
		self.assertIsNone(swindow.oldest_slide_ts)
		self.assertIsNone(swindow.latest_slide_ts)

		basic_time = datetime.datetime.now()

		# Slide 1
		slide1 = {}
		slide1["data_list"] = [{'values':1, 'client_id':1, 'event_ts' : basic_time}, 
							  {'values':2, 'client_id':1, 'event_ts' : basic_time + datetime.timedelta(seconds=1)},
							  {'values':2, 'client_id':1, 'event_ts' : basic_time + datetime.timedelta(seconds=1.9)},]
		slide1["start_time"] = basic_time
		slide1["end_time"] = basic_time + datetime.timedelta(seconds=2)
		queue.put(slide1)
		self.assertEqual(queue.qsize(), 1)

		# Slide 2
		slide2 = {}
		slide2["data_list"] = [{'values':1, 'client_id':1, 'event_ts' : basic_time + datetime.timedelta(seconds=3)}, 
							  {'values':2, 'client_id':1, 'event_ts' : basic_time + datetime.timedelta(seconds=3.9)},]
		slide2["start_time"] = basic_time + datetime.timedelta(seconds=2)
		slide2["end_time"] = basic_time + datetime.timedelta(seconds=4)
		queue.put(slide2)
		self.assertEqual(queue.qsize(), 2)

		queue.put(None)
		self.assertEqual(queue.qsize(), 3)

		swindow_thread = SlidingWindowMP(queue, sliding_window_q, start_ts=basic_time, slide_size=2,
										 detector_queue=detector_queue, explainer_queue=explainer_queue)
		swindow_thread.start()
		swindow_thread.join()

		swindow = sliding_window_q.get()
		self.assertEqual(min(swindow.window.keys()), 1)
		self.assertEqual(len(swindow.window.keys()), 2)
		self.assertEqual(datetime.datetime.fromtimestamp(swindow.latest_slide_ts), basic_time + datetime.timedelta(seconds=2))

		e_window = explainer_queue.get()		
		e_window = explainer_queue.get()		
		self.assertEqual(min(e_window.window.keys()), 1)
		self.assertEqual(len(e_window.window.keys()), 2)
		self.assertEqual(datetime.datetime.fromtimestamp(e_window.latest_slide_ts), basic_time + datetime.timedelta(seconds=2))

		d_window = detector_queue.get()		
		d_window = detector_queue.get()		
		self.assertEqual(min(d_window.window.keys()), 1)
		self.assertEqual(len(d_window.window.keys()), 2)
		self.assertEqual(datetime.datetime.fromtimestamp(d_window.latest_slide_ts), basic_time + datetime.timedelta(seconds=2))

	def test_map_time_to_seconds(self):
		size = map_time_to_seconds(1, ori_unit = 'seconds')
		self.assertEqual(size,1)
		size = map_time_to_seconds(1, ori_unit = 'minutes')
		self.assertEqual(size,60)
		size = map_time_to_seconds(1, ori_unit = 'hours')
		self.assertEqual(size,3600)
		size = map_time_to_seconds(1, ori_unit = 'days')
		self.assertEqual(size,86400)
		size = map_time_to_seconds(1, ori_unit = 's')
		self.assertEqual(size,1)

	def test_slide(self):
		basic_time = datetime.datetime.now()
		slide = Slide(start_ts=basic_time, size=2, unit='seconds')
		self.assertEqual(slide.unit, 'seconds')
		self.assertEqual(slide.size, 2)
		self.assertEqual(slide.start_ts, basic_time)
		self.assertEqual(slide.end_ts, basic_time + datetime.timedelta(seconds=2))
		self.assertIsInstance(slide.data, list)

		datum = 1.5
		event_time = basic_time + datetime.timedelta(seconds=1)
		slide.add_incoming_datum(datum, event_time)
		self.assertEqual(slide.data[0], 1.5)


if __name__ == '__main__': # pragma: no cover
    unittest.main()

		
