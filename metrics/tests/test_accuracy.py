import os
import unittest
import pandas as pd

from rc_metrics.accuracy import accuracy_metrics

from rc_metrics.performance import get_outlier_root_cause_gt
from rc_metrics.performance import get_root_cause_prediction_result
from rc_metrics.performance import combine_ground_truth_result_indices


class TestPerformance(unittest.TestCase):
	def setUp(self):
		self.data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
		self.features = ['X1', 'X2', 'X3', 'X4', 'X5']

	def test_accuracy_metrics_1(self):
		gt_dict = get_outlier_root_cause_gt(self.data_path, separator=',')
		result_path = os.path.join(os.path.dirname(__file__), 'result.json')
		pred_result = get_root_cause_prediction_result(result_path)
		pred_dict = pred_result[0]
		run_time = pred_result[1]
		df = combine_ground_truth_result_indices(gt_dict, pred_dict)
		metrics = accuracy_metrics(df['root_cause_gt'].values, df['root_cause_pred'].values, self.features)
		self.assertEqual(metrics['accuracy'], 0.5)
		self.assertEqual(metrics['f1'], 0.5)
		self.assertEqual(metrics['precision'], 0.5)
		self.assertEqual(metrics['recall'], 0.5)

	def test_accuracy_metrics_2(self):
		gt_dict = get_outlier_root_cause_gt(self.data_path, separator=',')
		result_path = os.path.join(os.path.dirname(__file__), 'result2.json')
		pred_result = get_root_cause_prediction_result(result_path)
		pred_dict = pred_result[0]
		run_time = pred_result[1]
		df = combine_ground_truth_result_indices(gt_dict, pred_dict)
		metrics = accuracy_metrics(df['root_cause_gt'].values, df['root_cause_pred'].values, self.features)
		self.assertEqual(metrics['accuracy'], 1)
		self.assertEqual(metrics['f1'], 1)
		self.assertEqual(metrics['precision'], 1)
		self.assertEqual(metrics['recall'], 1)

	def test_accuracy_metrics_3(self):
		gt_dict = get_outlier_root_cause_gt(self.data_path, separator=',')
		result_path = os.path.join(os.path.dirname(__file__), 'result3.json')
		pred_result = get_root_cause_prediction_result(result_path)
		pred_dict = pred_result[0]
		run_time = pred_result[1]
		df = combine_ground_truth_result_indices(gt_dict, pred_dict)
		metrics = accuracy_metrics(df['root_cause_gt'].values, df['root_cause_pred'].values, self.features)
		self.assertEqual(metrics['accuracy'], 0)
		self.assertEqual(metrics['f1'], 0)
		self.assertEqual(metrics['precision'], 0)
		self.assertEqual(metrics['recall'], 0)
	
if __name__ == '__main__': # pragma: no cover
    unittest.main()