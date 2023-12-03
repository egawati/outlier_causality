import numpy as np
import pandas as pd


def transform_data_list_to_numpy(data_list):
    """
    data_list : list of data sent from client
                e.g [{'values':1, 'client_id':1, 'event_ts' : ...}, 
                     {'values':2, 'client_id':1, 'event_ts' : ...},
                     {'values':3, 'client_id':1, 'event_ts' : ...},]
                or:
                [{'values':[1, 2], 'client_id':1, 'event_ts' : ...}, 
                 {'values':[3, 4], 'client_id':1, 'event_ts' : ...},
                 {'values':[5, 6], 'client_id':1, 'event_ts' : ...},]
    return arr : np.array, e.g [[1], [2], [3]]
                           or [[1, 2], [3, 4], [5, 6]]
           event_ts : np.arrage, e.g[..., ..., ...]
           where ... is a numpy.datetime64
    """
    df = pd.DataFrame(data_list)
    if len(data_list)>0 and isinstance(data_list[0]['values'], list):
        arr = np.array(df['values'].to_list())
    else:
        arr = np.column_stack([df['values'].to_numpy()])
    return arr, df['event_ts'].values

def sampling_data(data, m_samples):
	"""
	data : pandas dataframe or numpy array
	m_samples : int, number of rows to be sampled from data
	return sample_data : pandas dataframe
	"""
	if isinstance(data, pd.DataFrame):
		return data.sample(n = m_samples)
	else:
		number_of_rows = data.shape[0]
		random_indices = np.random.choice(number_of_rows, size=m_samples, replace=False)
		return data[random_indices, :]
	

def get_outlier_event_timestamp(outlier):
	"""
	outlier is a dictionary {"values" : ..., "event_ts" : ...}
	"""
	return outlier['event_ts']

def get_slide_number_to_explain_outlier(outlier_ts, active_slides):
	"""
	use binary search to find the slide number to explain an outlier
	outlier_ts : outlier event timestamp 
	active_slides : dictionary {0:{"star_ts" : ..., "end_ts" : ...}, 
								1:{"star_ts" : ..., "end_ts" : ...},
								2:{"star_ts" : ..., "end_ts" : ...}, ...}
	"""
	explainer_slide_number = 0
	slide_numbers = sorted(active_slides.keys())
	first = min(active_slides.keys())
	last = max(active_slides.keys())
	found = False
	while(first <= last and not found):
		mid = (first + last) // 2
		if active_slides[mid]['start_ts'] <= outlier_ts < active_slides[mid]['end_ts']:
			explainer_slide_number = max(0, mid-1)
			found = True
		else:
			if outlier_ts > active_slides[mid]['end_ts']:
				first = mid + 1
			else:
				last = mid - 1
	if first > last:
		explainer_slide_number = max(active_slides.keys())
	return explainer_slide_number