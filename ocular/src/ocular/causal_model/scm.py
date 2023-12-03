import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from abc import ABC, abstractmethod

import warnings

class StructuralCausalModel(ABC):
	"""
	Functional causal model for each node in the causal graph
	"""
	@abstractmethod
	def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
	    raise NotImplementedError

	@abstractmethod
	def predict(self, X: np.ndarray) -> np.ndarray:
	    raise NotImplementedError

	@abstractmethod
	def partial_fit(self, X: np.ndarray, Y: np.ndarray) -> None:
	    raise NotImplementedError


class LinearCausalModel(StructuralCausalModel):
	def __init__(self):
		# self._causal_mechanism = make_pipeline(StandardScaler(),
		# 									   SGDRegressor())

		self._causal_mechanism = SGDRegressor()

	def fit(self, X, Y):
		self._causal_mechanism.fit(X, Y)

	def partial_fit(self, X, Y):
		return self._causal_mechanism.partial_fit(X,Y)

	def predict(self, X):
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			return self._causal_mechanism.predict(X)


    