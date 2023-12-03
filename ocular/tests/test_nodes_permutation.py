import unittest
import numpy as np

from ocular.nodes_permutation import permute_features

class TestNodesPermutation(unittest.TestCase):
	def test_permute_features1(self):
		feature_samples = np.random.rand(10,5)
		#anomaly_samples = np.array([[4.146301,  5.845693,  4.936191,  5.490878,  5.848454]])
		##subset = np.array([0, 0, 0, 0, 1])
		features_to_permute = [0, 1, 2, 3]

		new_feature_samples = permute_features(feature_samples, 
											   features_to_permute, 
											   randomize_features_jointly=False)
		## since we permutate column 0, 1, 2, and 3, those columns at new_feature_samples and feature_samples are not equal
		self.assertNotEqual(new_feature_samples[:,0].tolist(), feature_samples[:,0].tolist())
		self.assertNotEqual(new_feature_samples[:,1].tolist(), feature_samples[:,1].tolist())
		self.assertNotEqual(new_feature_samples[:,2].tolist(), feature_samples[:,2].tolist())
		self.assertNotEqual(new_feature_samples[:,3].tolist(), feature_samples[:,3].tolist())

		## since we do not permutate column 4, new_feature_samples and feature_samples will stay the same at column 4
		self.assertListEqual(new_feature_samples[:,4].tolist(), feature_samples[:,4].tolist())

		self.assertEqual(new_feature_samples.shape[0], feature_samples.shape[0])

	def test_permute_features2(self):
		feature_samples = np.random.rand(10,5)
		#anomaly_samples = np.array([[4.146301,  5.845693,  4.936191,  5.490878,  5.848454]])
		##subset = np.array([0, 0, 0, 0, 1])
		features_to_permute = [0, 1, 2, 3]

		new_feature_samples = permute_features(feature_samples, 
											   features_to_permute, 
											   randomize_features_jointly=True)
		## since we permutate column 0, 1, 2, and 3, those columns at new_feature_samples and feature_samples are not equal
		self.assertNotEqual(new_feature_samples[:,0].tolist(), feature_samples[:,0].tolist())
		self.assertNotEqual(new_feature_samples[:,1].tolist(), feature_samples[:,1].tolist())
		self.assertNotEqual(new_feature_samples[:,2].tolist(), feature_samples[:,2].tolist())
		self.assertNotEqual(new_feature_samples[:,3].tolist(), feature_samples[:,3].tolist())

		## since we do not permutate column 4, new_feature_samples and feature_samples will stay the same at column 4
		self.assertListEqual(new_feature_samples[:,4].tolist(), feature_samples[:,4].tolist())

		self.assertEqual(new_feature_samples.shape[0], feature_samples.shape[0])

	def test_permute_features3(self):
		feature_samples = np.random.rand(10,3)
		features_to_permute = [0, 2]

		new_feature_samples = permute_features(feature_samples, 
											   features_to_permute, 
											   randomize_features_jointly=True)
		## since we permutate column 0 and 2, those columns at new_feature_samples and feature_samples are not equal
		self.assertNotEqual(new_feature_samples[:,0].tolist(), feature_samples[:,0].tolist())
		self.assertNotEqual(new_feature_samples[:,2].tolist(), feature_samples[:,2].tolist())

		## since we do not permutate column 1, new_feature_samples and feature_samples will stay the same at column 4
		self.assertListEqual(new_feature_samples[:,1].tolist(), feature_samples[:,1].tolist())

		self.assertEqual(new_feature_samples.shape[0], feature_samples.shape[0])


if __name__ == '__main__': # pragma: no cover
	unittest.main()