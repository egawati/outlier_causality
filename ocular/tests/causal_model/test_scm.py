import unittest
from unittest.mock import patch

import numpy as np

from sklearn.linear_model import SGDRegressor

from ocular.causal_model import dag
from ocular.causal_model.scm import StructuralCausalModel
from ocular.causal_model.scm import LinearCausalModel


class TestLinearCausalModel(unittest.TestCase):
	def setUp(self):
		self.linear_model = LinearCausalModel()
		self.n_samples, self.n_features = 10, 5

	def test_linear_model_fit(self):
		rng = np.random.RandomState(0)
		X = rng.randn(self.n_samples, self.n_features)
		Y = rng.randn(self.n_samples)
		result = self.linear_model.fit(X,Y)
		self.assertEqual(None, result)

	def test_linear_model_partial_fit(self):
		rng = np.random.RandomState(0)
		X = rng.randn(self.n_samples, self.n_features)
		Y = rng.randn(self.n_samples)
		result = self.linear_model.partial_fit(X,Y)
		self.assertIsInstance(result, SGDRegressor)

		rng = np.random.RandomState(1)
		X = rng.randn(self.n_samples, self.n_features)
		Y = rng.randn(self.n_samples)
		result = self.linear_model.partial_fit(X,Y)
		self.assertIsInstance(result, SGDRegressor)

	def test_linear_model_predict(self):
		rng = np.random.RandomState(0)
		X = rng.randn(self.n_samples, self.n_features)
		Y = rng.randn(self.n_samples)
		self.linear_model.fit(X,Y)
		
		rng = np.random.RandomState(1)
		X_test = rng.randn(self.n_samples, self.n_features)
		result = self.linear_model.predict(X_test)
		self.assertEqual(result.shape[0], self.n_samples)

	@patch("ocular.causal_model.scm.StructuralCausalModel.__abstractmethods__", set())
	def test_returns_1(self):
		instance = StructuralCausalModel()
		rng = np.random.RandomState(0)
		X = rng.randn(self.n_samples, self.n_features)
		Y = rng.randn(self.n_samples)
		with self.assertRaises(NotImplementedError):
			instance.fit(X,Y)
		with self.assertRaises(NotImplementedError):
			instance.predict(X)
		with self.assertRaises(NotImplementedError):
			instance.partial_fit(X,Y)

if __name__ == '__main__': # pragma: no cover
	unittest.main()