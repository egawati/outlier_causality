import pandas as pd
import numpy as np

def get_outliers(arr, event_ts, label, index):
	"""
	arr : np.array of shape (n_samples, n_features)
	event_ts : np.array of shape (n_samples,), event timestamp of each row in arr
	label : np.array of shape (n_samples,), label of each row in arr, 0 & 1 indicates an inlier & outlier respectively
	"""
	outlier_indexes = np.where(label == 1)[0]
	return arr[outlier_indexes, :], event_ts[outlier_indexes], index[outlier_indexes].tolist()

def get_outlier_list(arr, event_ts, features):
	"""
	arr : np.array of shape (n_samples, n_features)
	event_ts : np.array of shape (n_samples,), event timestamp of each row in arr
	label : np.array of shape (n_samples,), label of each row in arr, 0 & 1 indicates an inlier & outlier respectively
	"""
	outliers = list()
	for i in range(arr.shape[0]):
		outliers.append({'values':pd.DataFrame(np.row_stack([arr[i,:]]), columns=features), 'event_ts':event_ts[i]})
	return outliers