import os
import unittest
import pandas as pd
from rc_metrics.performance import get_outlier_root_cause_gt
from rc_metrics.performance import get_root_cause_prediction_result
from rc_metrics.performance import combine_ground_truth_result_indices
from rc_metrics.performance import save_performance_result
from rc_metrics.performance import compute_and_save_performance
from rc_metrics.accuracy import accuracy_metrics

class TestPerformance(unittest.TestCase):
	def setUp(self):
		self.data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
		self.result_path = os.path.join(os.path.dirname(__file__), 'result.json')
		self.features = ['X1', 'X2', 'X3', 'X4', 'X5']
	
	def test_get_outlier_root_cause_gt(self):
		gt_dict = get_outlier_root_cause_gt(self.data_path, separator=',')
		self.assertIn('outlier_index', gt_dict.keys())
		self.assertIn('root_cause_gt', gt_dict.keys())
		self.assertIn(1, gt_dict['outlier_index'])
		self.assertIn(16, gt_dict['outlier_index'])
		self.assertIn('X1', gt_dict['root_cause_gt'])

	def test_get_root_cause_prediction_result(self):
		pred_result = get_root_cause_prediction_result(self.result_path)
		pred_dict = pred_result[0]
		run_time = pred_result[1]
		self.assertEqual(run_time, 10)
		self.assertIn(1, pred_dict.keys())
		self.assertIn(16, pred_dict.keys())
		self.assertIn('X1', pred_dict.values())
		self.assertIn('X5', pred_dict.values())

	def test_combine_ground_truth_result_indices(self):
		gt_dict = get_outlier_root_cause_gt(self.data_path, separator=',')
		pred_result = get_root_cause_prediction_result(self.result_path)
		pred_dict = pred_result[0]
		run_time = pred_result[1]
		df = combine_ground_truth_result_indices(gt_dict, pred_dict)
		self.assertIn('outlier_index', df.columns)
		self.assertIn('root_cause_gt', df.columns)
		self.assertIn('root_cause_pred', df.columns)
		self.assertIn(1, df['outlier_index'].tolist())
		self.assertIn(16, df['outlier_index'].tolist())
		self.assertIn('X1', df['root_cause_gt'].tolist())
		self.assertNotIn('X5', df['root_cause_gt'].tolist())
		self.assertIn('X1', df['root_cause_pred'].tolist())
		self.assertIn('X5', df['root_cause_pred'].tolist())

	def test_accuracy_metrics(self):
		gt_dict = get_outlier_root_cause_gt(self.data_path, separator=',')
		pred_result = get_root_cause_prediction_result(self.result_path)
		pred_dict = pred_result[0]
		run_time = pred_result[1]
		df = combine_ground_truth_result_indices(gt_dict, pred_dict)
		metrics = accuracy_metrics(df['root_cause_gt'].values, df['root_cause_pred'].values, self.features)
		self.assertEqual(metrics['accuracy'], 0.5)
		self.assertEqual(metrics['f1'], 0.5)
		self.assertEqual(metrics['precision'], 0.5)
		self.assertEqual(metrics['recall'], 0.5)

	def test_save_performance_result(self):
		gt_dict = get_outlier_root_cause_gt(self.data_path, separator=',')
		pred_result = get_root_cause_prediction_result(self.result_path)
		pred_dict = pred_result[0]
		run_time = pred_result[1]
		df = combine_ground_truth_result_indices(gt_dict, pred_dict)
		metrics = accuracy_metrics(df['root_cause_gt'].values, df['root_cause_pred'].values, self.features)
		performance = {'accuracy_metrics' : metrics, 
					   'run_time' : run_time}
		rel_path = os.path.dirname(__file__)
		save_performance_result(performance, 'test', rel_path)

		filepath = os.path.join(os.path.dirname(__file__), 'performance_test.json')
		check_filepath = os.path.exists(filepath)
		self.assertTrue(check_filepath)
		os.remove(filepath)

	def test_compute_and_save_performance(self):
		compute_and_save_performance(self.data_path, self.result_path, 
									'test', self.features, data_separator=',')
		filepath = os.path.join(os.path.dirname(__file__), 'performance_test.json')
		check_filepath = os.path.exists(filepath)
		self.assertTrue(check_filepath)
		os.remove(filepath)


if __name__ == '__main__': # pragma: no cover
    unittest.main()