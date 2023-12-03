import unittest
import pandas as pd
import numpy as np
import datetime

from odds.labeled_data import get_outliers
from odds.labeled_data import get_outlier_list


class TestLabeledData(unittest.TestCase):
	def setUp(self):
		self.basic_time = datetime.datetime.now()
		
	def test_get_outliers1(self):
		data_list = [{'values':[1,10], 'client_id':1, 'event_ts' : self.basic_time, 'label': 0, 'index':0}, 
					{'values':[2,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=1), 'label':0, 'index':1}, 
					{'values':[3,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2), 'label':0, 'index':2}, 
					{'values':[30,3], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=3), 'label':1, 'index':3}, 
					]
		df = pd.DataFrame(data_list)
		arr = np.array(df['values'].to_list())
		event_ts = df['event_ts'].values
		label = df['label'].values
		index = df['index'].values

		arr_1, event_ts_1, index_1 = get_outliers(arr, event_ts, label, index)
		self.assertEqual(arr_1.shape[0], 1)
		self.assertEqual(arr_1.shape[1], 2)
		
		self.assertEqual(event_ts_1.shape[0],1)
		self.assertEqual(event_ts_1[0], event_ts[3])

	def test_get_outlier_list_1(self):
		data_list = [{'values':[1,10], 'client_id':1, 'event_ts' : self.basic_time, 'label': 0, 'index':0}, 
					{'values':[2,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=1), 'label':0, 'index':1}, 
					{'values':[3,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2), 'label':0, 'index':2}, 
					{'values':[30,3], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=3), 'label':1, 'index':3}, 
					]
		df = pd.DataFrame(data_list)
		arr = np.array(df['values'].to_list())
		event_ts = df['event_ts'].values
		label = df['label'].values
		index = df['index'].values

		arr_1, event_ts_1, index_1 = get_outliers(arr, event_ts, label, index)

		features = ('A', 'B')
		outliers = get_outlier_list(arr_1, event_ts_1, features)

		self.assertEqual(len(outliers), 1)
		self.assertEqual(outliers[0]['values'][features[0]].to_numpy(), arr_1[0,0])
		self.assertEqual(outliers[0]['values'][features[1]].to_numpy(), arr_1[0,1])
		

	def test_get_outliers2(self):
		data_list = [{'values':[10,1], 'client_id':1, 'event_ts' : self.basic_time, 'label': 1, 'index':0}, 
					{'values':[2,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=1), 'label':0, 'index':1}, 
					{'values':[3,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2), 'label':0, 'index':2}, 
					{'values':[30,3], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=3), 'label':1, 'index':3}, 
					]
		df = pd.DataFrame(data_list)
		arr = np.array(df['values'].to_list())
		event_ts = df['event_ts'].values
		label = df['label'].values
		index = df['index'].values

		arr_1, event_ts_1, index_1 = get_outliers(arr, event_ts, label, index)

		self.assertEqual(arr_1.shape[0], 2)
		self.assertEqual(arr_1.shape[1], 2)

		self.assertEqual(event_ts_1.shape[0],2)
		self.assertEqual(event_ts_1[0], event_ts[0])
		self.assertEqual(event_ts_1[1], event_ts[3])

	def test_get_outlier_list_2(self):
		data_list = [{'values':[10,1], 'client_id':1, 'event_ts' : self.basic_time, 'label': 1, 'index':0}, 
					{'values':[2,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=1), 'label':0, 'index':1}, 
					{'values':[3,20], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=2), 'label':0, 'index':2}, 
					{'values':[30,3], 'client_id':1, 'event_ts' : self.basic_time + datetime.timedelta(seconds=3), 'label':1, 'index':3}, 
					]
		df = pd.DataFrame(data_list)
		arr = np.array(df['values'].to_list())
		event_ts = df['event_ts'].values
		label = df['label'].values
		index = df['index'].values

		arr_1, event_ts_1, index_1 = get_outliers(arr, event_ts, label, index)

		features = ('A', 'B')
		
		outliers = get_outlier_list(arr_1, event_ts_1, features)
		self.assertEqual(len(outliers), 2)
		self.assertEqual(outliers[0]['values'][features[0]].to_numpy(), arr_1[0,0])
		self.assertEqual(outliers[0]['values'][features[1]].to_numpy(), arr_1[0,1])
		self.assertEqual(outliers[1]['values'][features[0]].to_numpy(), arr_1[1,0])
		self.assertEqual(outliers[1]['values'][features[1]].to_numpy(), arr_1[1,1])
	