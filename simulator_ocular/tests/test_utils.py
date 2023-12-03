import unittest
import pandas as pd
import numpy as np
import datetime

from ocular_simulator.utils import process_data_list
from ocular_simulator.utils import transform_data_list_to_numpy

class TestUtils(unittest.TestCase):
	def setUp(self):
		self.basic_time = datetime.datetime.now()
		self.data_list = [{'values':1, 'client_id':1, 'event_ts' : self.basic_time}, 
						  {'values':2, 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=1)},
						  {'values':3, 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2)}
						  ]

		self.data_list2 = [{'values':[1,10], 'client_id':1, 'event_ts' : self.basic_time}, 
						  {'values':[2,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=1)},
						  {'values':[3,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2)}
						  ]

		self.data_list3 = [{'values':[1,10], 'client_id':1, 'event_ts' : self.basic_time, 'label': 0, 'index':0}, 
						  {'values':[2,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=1), 'label':0, 'index':1},
						  {'values':[3,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2), 'label':0, 'index':2}
						  ]
		
	def test_process_data_list(self):
		data_dict = process_data_list(self.data_list, self.basic_time, self.basic_time + datetime.timedelta(seconds=2))
		self.assertIsInstance(data_dict, dict)
		self.assertEqual(data_dict['data_list'], self.data_list)
		self.assertEqual(data_dict['start_time'], self.basic_time)
		self.assertEqual(data_dict['end_time'],self.basic_time + datetime.timedelta(seconds=2))

	def test_transform_data_list_to_numpy(self):
		arr, event_ts, _, _ = transform_data_list_to_numpy(self.data_list)
		self.assertEqual(arr.shape[0], 3)
		self.assertEqual(arr.shape[1], 1)
		self.assertEqual(event_ts.shape[0],3)
		arr2, event_ts2, _, _ = transform_data_list_to_numpy(self.data_list2)
		self.assertEqual(arr2.shape[0], 3)
		self.assertEqual(arr2.shape[1], 2)
		self.assertEqual(event_ts2.shape[0],3)
		arr3, event_ts3, label3, index3 = transform_data_list_to_numpy(self.data_list3)
		self.assertEqual(arr3.shape[0], 3)
		self.assertEqual(arr3.shape[1], 2)
		self.assertEqual(event_ts3.shape[0],3)
		self.assertEqual(label3.shape[0],3)
		
if __name__ == '__main__': # pragma: no cover
    unittest.main()