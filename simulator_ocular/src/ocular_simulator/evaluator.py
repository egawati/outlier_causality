import os
import json
import statistics

def check_result_list(result_list):
	"""we want to ensure that each node contributions value is set to be a float not a numpy array"""
	for item in result_list:
		for key in item:
			item[key] = item[key][0]

def get_and_save_result(explainer_queues, run_time, n_streams, exp_name, rel_path, features, nslide_used=None):
	result_dict = {}
	result_dict['run_time'] = run_time
	result_dict['out_attribs'] = list()
	result_dict['outlier_index'] = list()
	for i in range(n_streams):
		explainer_queue = explainer_queues[i]
		while not explainer_queue.empty():
			result = explainer_queue.get()
			if result is not None:
				outlier_rc = result[0]
				result_list = [outlier_rc[key] for key in sorted(outlier_rc.keys())]
				check_result_list(result_list)
				result_dict['out_attribs'].extend(result_list)
				outlier_index_list = result[1]
				result_dict['outlier_index'].extend(outlier_index_list)

	cwd = os.getcwd()
	result_folder = os.path.join(cwd, rel_path)
	if not os.path.exists(result_folder):
	    os.makedirs(result_folder)
	filepath = f'{result_folder}/{exp_name}.json'

	if nslide_used is None:
		filepath = f'{result_folder}/{exp_name}.json'
	else:
		filepath = f'{result_folder}/{exp_name}_{nslide_used}.json'

	with open(filepath, "w") as outfile:
		json.dump(result_dict, outfile, indent = 4)

	return len(result_dict['outlier_index'])


def process_explanation_time(est_time_queues, n_streams, n_outliers_explained=None):
	explanation_time = list()
	for i in range(n_streams):
		est_time_queue = est_time_queues[i]
		while not est_time_queue.empty():
			result = est_time_queue.get()
			if result is not None:
				explanation_time.append(result)
	print(f'n_outliers_explained = {n_outliers_explained}')
	if not n_outliers_explained:
		return statistics.fmean(explanation_time)
	else:
		return sum(explanation_time)/n_outliers_explained

def save_max_out_attribs_result(exp_name, rel_path, nslide_used=None):
	cwd = os.getcwd()
	folder = os.path.join(cwd, rel_path)
	
	if nslide_used is None:
		filepath = f'{folder}/{exp_name}.json'
		max_filepath = f'{folder}/max_{exp_name}.json'
	else:
		filepath = f'{folder}/{exp_name}_{nslide_used}.json'
		max_filepath = f'{folder}/max_{exp_name}_{nslide_used}.json'

	with open(filepath) as input_file:
		file_contents = input_file.read()
	gcm_dict = json.loads(file_contents)

	max_out_attribs_list = list()
	outlier_index_list = gcm_dict['outlier_index']
	out_attribs_list = gcm_dict['out_attribs']

	for out_attribs in out_attribs_list:
		max_median_attrib = max(out_attribs, key=out_attribs.get)
		max_out_attribs_list.append(max_median_attrib)

	max_dict = {}
	max_dict['root_cause'] = max_out_attribs_list
	max_dict['outlier_index'] = outlier_index_list
	max_dict['run_time'] = gcm_dict['run_time']
	
	with open(max_filepath, "w") as outfile:
		json.dump(max_dict, outfile, indent = 4)

	return max_filepath