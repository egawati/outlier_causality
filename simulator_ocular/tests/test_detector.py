import unittest
import datetime
import pandas as pd
import numpy as np

from ocular_simulator.sliding_window import SlidingWindow
from ocular_simulator.detector import run_detector_with_labeled_data
from ocular_simulator.detector import DetectorMP

from multiprocessing import Process
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
from queue import LifoQueue

class LifoManager(BaseManager): # pragma: no cover
    pass

LifoManager.register('LifoQueue', LifoQueue) # pragma: no cover


class TestDetector(unittest.TestCase):
	def setUp(self):
		"""
		When a setUp() method is defined, the test runner will run that method prior to each test. 
		"""
		self.features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']

		self.slide_size = 2
		self.window_size = 4
		self.max_slide = self.window_size/self.slide_size

		self.manager = Manager()
		self.lifo_manager = LifoManager()
		self.lifo_manager.start()
	
		self.basic_time = datetime.datetime.now()

		self.data_list = [{'values':[1,10, 1, 2, 3, 4], 'client_id':1, 'event_ts' : self.basic_time, 'label': 0, 'index':0}, 
						  {'values':[2,20, 1, 2, 3, 4], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=1), 'label':0, 'index':1}, 
						  {'values':[3,30, 1, 2, 3, 4], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2), 'label':0, 'index':2}, 
						  {'values':[30,3, 1, 2, 3, 4], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2), 'label':1, 'index':3}
						  ]

	def test_run_detector_with_labeled_data(self):
		outlier_queue = self.manager.Queue()
		o_window_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		slide = {}
		slide["data_list"] = self.data_list

		swindow.add_new_slide_with_number(slide=slide, 
										  start_ts=self.basic_time, 
										  slide_number=slide_number)
		o_window_queue.put(swindow)
		o_window_queue.put(None)
		self.assertEqual(o_window_queue.empty(), False)

		run_detector_with_labeled_data(o_window_queue, 
							 self.slide_size, 
							 self.window_size, 
							 outlier_queue, 
							 start_ts=self.basic_time,
							 features=self.features)
		
		outliers, outlier_slide_number = outlier_queue.get()
		self.assertIsInstance(outliers, dict)
		self.assertIn("values", outliers.keys())
		self.assertIn("event_ts", outliers.keys())
		self.assertIn("index", outliers.keys())
		self.assertEqual(outliers['values'][0][0], 30)
		self.assertEqual(outliers['values'][0][1], 3)
		self.assertEqual(outliers['values'][0][2], 1)
		self.assertEqual(outliers['values'][0][3], 2)
		self.assertEqual(outliers['values'][0][4], 3)
		self.assertEqual(outliers['values'][0][5], 4)
		self.assertEqual(outlier_slide_number, slide_number)

	def test_run_detector_with_labeled_data_mp(self):
		outlier_queue = self.manager.Queue()
		o_window_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		slide = {}
		slide["data_list"] = self.data_list

		swindow.add_new_slide_with_number(slide=slide, 
										  start_ts=self.basic_time, 
										  slide_number=slide_number)
		o_window_queue.put(swindow)
		o_window_queue.put(None)
		self.assertEqual(o_window_queue.empty(), False)

		detector = Process(target = run_detector_with_labeled_data, args = (o_window_queue, 
																 self.slide_size, 
																 self.window_size, 
																 outlier_queue, 
																 self.basic_time,
																 self.features))
		detector.start()
		detector.join()

		outliers, outlier_slide_number = outlier_queue.get()
		self.assertIsInstance(outliers, dict)
		self.assertIn("values", outliers.keys())
		self.assertIn("event_ts", outliers.keys())
		self.assertIn("index", outliers.keys())
		self.assertEqual(outliers['values'][0][0], 30)
		self.assertEqual(outliers['values'][0][1], 3)
		self.assertEqual(outliers['values'][0][2], 1)
		self.assertEqual(outliers['values'][0][3], 2)
		self.assertEqual(outliers['values'][0][4], 3)
		self.assertEqual(outliers['values'][0][5], 4)
		self.assertEqual(outlier_slide_number, slide_number)

	def test_run_detector_with_labeled_data_mp_2(self):
		outlier_queue = self.manager.Queue()
		o_window_queue = self.manager.Queue()
		basic_time = datetime.datetime.now().timestamp()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		slide = {}
		slide["data_list"] = self.data_list

		swindow.add_new_slide_with_number(slide=slide, 
										  start_ts=basic_time, 
										  slide_number=slide_number)
		o_window_queue.put(swindow)
		o_window_queue.put(None)
		self.assertEqual(o_window_queue.empty(), False)

		detector = Process(target = run_detector_with_labeled_data, args = (o_window_queue, 
																		 self.slide_size, 
																		 self.window_size, 
																		 outlier_queue, 
																		 basic_time,
																		 self.features))
		detector.start()
		detector.join()

		outliers, outlier_slide_number = outlier_queue.get()
		self.assertIsInstance(outliers, dict)
		self.assertIn("values", outliers.keys())
		self.assertIn("event_ts", outliers.keys())
		self.assertIn("index", outliers.keys())
		self.assertEqual(outliers['values'][0][0], 30)
		self.assertEqual(outliers['values'][0][1], 3)
		self.assertEqual(outliers['values'][0][2], 1)
		self.assertEqual(outliers['values'][0][3], 2)
		self.assertEqual(outliers['values'][0][4], 3)
		self.assertEqual(outliers['values'][0][5], 4)
		self.assertEqual(outlier_slide_number, slide_number)

	def test_run_detector_mp(self):
		outlier_queue = self.manager.Queue()
		o_window_queue = self.manager.Queue()

		slide_number = 1
		swindow = SlidingWindow(slide_size=self.slide_size, window_size=self.window_size, unit='seconds')
		slide = {}
		slide["data_list"] = self.data_list

		swindow.add_new_slide_with_number(slide=slide, 
										  start_ts=self.basic_time, 
										  slide_number=slide_number)
		o_window_queue.put(swindow)
		o_window_queue.put(None)
		self.assertEqual(o_window_queue.empty(), False)

		detector = DetectorMP(o_window_queue, 
							  self.slide_size,
							  self.window_size,
							  outlier_queue,
							  self.basic_time,
							  self.features)
		detector.start()
		detector.join()

		outliers, outlier_slide_number = outlier_queue.get()
		self.assertIsInstance(outliers, dict)
		self.assertIn("values", outliers.keys())
		self.assertIn("event_ts", outliers.keys())
		self.assertIn("index", outliers.keys())
		self.assertEqual(outliers['values'][0][0], 30)
		self.assertEqual(outliers['values'][0][1], 3)
		self.assertEqual(outliers['values'][0][2], 1)
		self.assertEqual(outliers['values'][0][3], 2)
		self.assertEqual(outliers['values'][0][4], 3)
		self.assertEqual(outliers['values'][0][5], 4)
		self.assertEqual(outlier_slide_number, slide_number)

if __name__ == '__main__': # pragma: no cover
    unittest.main()