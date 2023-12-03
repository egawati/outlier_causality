import unittest

from ocular.initialization import init_model
from ocular.initialization import init_outlier_scorer
from ocular.initialization import init_linear_noise_samples
from ocular.initialization import scm_initialization

import numpy as np
import pandas as pd

from ocular.causal_model.scm import LinearCausalModel
from ocular.causal_model import dag

from dowhy import gcm
from dowhy.gcm import MedianCDFQuantileScorer

class TestInitialization(unittest.TestCase):
	def test_init_model_one_predictor(self):
		fm_type = 'LinearModel'
		rng = np.random.RandomState(0)
		n_samples = 100
		n_features = 1
		predictor = rng.randn(n_samples, n_features)
		target = rng.randn(n_samples)
		fm_model = init_model(fm_type, target, predictor)
		self.assertIsInstance(fm_model, LinearCausalModel)

	def test_init_model_multi_predictor(self):
		fm_type = 'LinearModel'
		rng = np.random.RandomState(0)
		n_samples = 100
		n_features = 3
		predictor = rng.randn(n_samples, n_features)
		target = rng.randn(n_samples)
		fm_model = init_model(fm_type, target, predictor)
		self.assertIsInstance(fm_model, LinearCausalModel)

	def test_init_model_no_type(self):
		fm_type = None
		rng = np.random.RandomState(0)
		n_samples = 100
		n_features = 1
		predictor = rng.randn(n_samples, n_features)
		target = rng.randn(n_samples)
		fm_model = init_model(fm_type, target, predictor)
		self.assertIsInstance(fm_model, LinearCausalModel)

	def test_init_model_none(self):
		fm_type = 'AnythingElse'
		rng = np.random.RandomState(0)
		n_samples = 100
		n_features = 1
		predictor = rng.randn(n_samples, n_features)
		target = rng.randn(n_samples)
		fm_model = init_model(fm_type, target, predictor)
		self.assertIsNone(fm_model)

	def test_init_linear_noise_samples(self):
		fm_type = 'LinearModel'
		rng = np.random.RandomState(0)
		n_samples = 100
		n_features = 1
		predictor = rng.randn(n_samples, n_features)
		target = rng.randn(n_samples)
		fm_model = init_model(fm_type, target, predictor)
		self.assertIsInstance(fm_model, LinearCausalModel)

		rng = np.random.RandomState(1)
		n_samples_test = 10
		X  = rng.randn(n_samples_test, n_features)
		Y = rng.randn(n_samples_test)
		noises = init_linear_noise_samples(fm_model, X, Y)

		self.assertEqual(noises.shape[0], n_samples_test)
		self.assertEqual(noises.shape[0], Y.shape[0])
		self.assertEqual(noises.shape[0], X.shape[0])

	def test_scm_initialization(self):
		nodes = [('X0', 'X2'), ('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
		causal_graph = dag.CausalGraph(nodes, features)
		self.assertEqual(len(causal_graph.dag.nodes), len(features))
		self.assertIn('X1', causal_graph.dag.nodes)

		n_samples = 100
		init_data = pd.DataFrame(np.random.randint(0,100,size=(n_samples, len(features))), columns=features)
		self.assertEqual(init_data.shape[0], n_samples)
		self.assertEqual(init_data.shape[1], len(features))

		fm_types = {node : 'LinearModel' for node in features}
		noise_types = {node : 'AdditiveNoise' for node in features}

		m_samples = 0.75 ## use percentage
		num_noise_samples = 1500

		snoise_models, snoise_samples, soutlier_scorers, sorted_nodes = scm_initialization(init_data, 
																	causal_graph, 
																	fm_types, 
																	noise_types, 
																	m_samples,
																	outlier_scorer = 'default',
																	num_noise_samples = num_noise_samples)

		noise_models = snoise_models[0]
		self.assertIn('X0', noise_models.keys())
		self.assertIn('X1', noise_models.keys())
		self.assertIn('X2', noise_models.keys())
		self.assertIn('X3', noise_models.keys())
		self.assertNotIn('X4', noise_models.keys())
		self.assertIn('X5', noise_models.keys())

		self.assertIsInstance(noise_models['X2'], gcm.AdditiveNoiseModel)
		self.assertIsInstance(noise_models['X3'], gcm.AdditiveNoiseModel)
		self.assertIsInstance(noise_models['X5'], gcm.AdditiveNoiseModel)

		noise_samples = snoise_samples[0]
		self.assertIn('X0', noise_samples.keys())
		self.assertIn('X1', noise_samples.keys())
		self.assertIn('X2', noise_samples.keys())
		self.assertIn('X3', noise_samples.keys())
		self.assertIn('X5', noise_samples.keys())

		self.assertEqual(noise_samples['X0'].shape[0], num_noise_samples)
		self.assertEqual(noise_samples['X1'].shape[0], num_noise_samples)
		self.assertEqual(noise_samples['X2'].shape[0], num_noise_samples)
		self.assertEqual(noise_samples['X3'].shape[0], num_noise_samples)
		self.assertEqual(noise_samples['X5'].shape[0], num_noise_samples)


		self.assertIn(0, soutlier_scorers.keys())

	def test_init_outlier_scorer(self):
		data = np.random.rand(30,1)
		scorer = MedianCDFQuantileScorer()
		scorer = init_outlier_scorer(data, scorer)
		self.assertIsInstance(scorer, MedianCDFQuantileScorer)

	def test_init_outlier_scorer_default(self):
		data = np.random.rand(30,1)
		scorer = 'default'
		scorer = init_outlier_scorer(data, scorer)
		self.assertIsInstance(scorer, MedianCDFQuantileScorer)


if __name__ == '__main__': # pragma: no cover
	unittest.main()



