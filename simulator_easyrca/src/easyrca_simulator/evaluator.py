import os
import json
import statistics

def get_and_save_result(explainer_queues, run_time, n_streams, exp_name, rel_path, features, nslide_used=None):
	result_dict = {}
	result_dict['run_time'] = run_time
	result_dict['root_cause'] = list()
	result_dict['outlier_index'] = list()
	for i in range(n_streams):
		explainer_queue = explainer_queues[i]
		while not explainer_queue.empty():
			result = explainer_queue.get()
			if result is not None:
				root_causes, outlier_index = result
				for i, root_cause in enumerate(root_causes):
					result_dict['root_cause'].append(tuple(root_cause[0]['roots']
													 + root_cause[0]['time_defying']
													 + root_cause[0]['structure_defying']
													 + root_cause[0]['param_defying']))
					result_dict['outlier_index'].append(outlier_index[i])

	cwd = os.getcwd()
	result_folder = os.path.join(cwd, rel_path)
	if not os.path.exists(result_folder):
	    os.makedirs(result_folder)

	if nslide_used is None:
		filepath = f'{result_folder}/{exp_name}.json'
	else:
		filepath = f'{result_folder}/{exp_name}_{nslide_used}.json'

	with open(filepath, "w") as outfile:
		json.dump(result_dict, outfile, indent = 4)

	return filepath


def process_explanation_time(est_time_queues, n_streams):
	explanation_time = list()
	for i in range(n_streams):
		est_time_queue = est_time_queues[i]
		while not est_time_queue.empty():
			result = est_time_queue.get()
			if result is not None:
				explanation_time.append(result)
	return statistics.fmean(explanation_time)