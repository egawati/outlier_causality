import unittest
import os
import pandas as pd
from ocular_simulator.data import read_csv

class TestData(unittest.TestCase):
	def test_read_csv(self):
		filepath = os.path.join(os.path.dirname(__file__), 'data.csv')
		self.assertIn('data.csv', filepath)
		df = pd.read_csv(filepath)
		data = read_csv(filepath)
		self.assertIsInstance(data, list)
		self.assertEqual(len(data), df.shape[0])
		self.assertIn('A', data[0].keys())
		self.assertIn('B', data[0].keys())
		self.assertIn('C', data[0].keys())
		self.assertIn('D', data[0].keys())
		self.assertIn(1, data[0].values())
		self.assertIn(2, data[0].values())
		self.assertIn(3, data[0].values())
		self.assertIn(4, data[0].values())
		
if __name__ == '__main__': # pragma: no cover
    unittest.main()