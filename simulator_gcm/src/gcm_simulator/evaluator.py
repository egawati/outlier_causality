import os
import json
import statistics

def get_and_save_result(explainer_queues, run_time, n_streams, exp_name, rel_path, features, nslide_used=None):
	result_dict = {}
	result_dict['run_time'] = run_time
	result_dict['median_attribs'] = list()
	#result_dict['uncertainty_attribs'] = list()
	result_dict['outlier_index'] = list()
	for i in range(n_streams):
		explainer_queue = explainer_queues[i]
		while not explainer_queue.empty():
			result = explainer_queue.get()
			if result is not None:
				median_attribs_list = result[0]
				result_dict['median_attribs'].extend(median_attribs_list)
				
				uncertainty_attribs_list = result[1]
				#result_dict['uncertainty_attribs'].extend(uncertainty_attribs_list)
				
				outlier_index_list = result[2]
				result_dict['outlier_index'].extend(outlier_index_list)

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

	return len(result_dict['outlier_index'])


def process_explanation_time(est_time_queues, n_streams, n_outliers=None):
	explanation_time = list()
	for i in range(n_streams):
		est_time_queue = est_time_queues[i]
		while not est_time_queue.empty():
			result = est_time_queue.get()
			if result is not None:
				explanation_time.append(result)
	print(f'n_outliers = {n_outliers}')
	if not n_outliers:
		return statistics.fmean(explanation_time)
	else:
		return sum(explanation_time)/n_outliers

def save_max_median_attribs_result(exp_name, rel_path, nslide_used=None):
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

	max_median_attribs_list = list()
	outlier_index_list = gcm_dict['outlier_index']
	median_attribs_list = gcm_dict['median_attribs']

	for median_attribs in median_attribs_list:
		max_median_attrib = max(median_attribs, key=median_attribs.get)
		max_median_attribs_list.append(max_median_attrib)

	max_dict = {}
	max_dict['root_cause'] = max_median_attribs_list
	max_dict['outlier_index'] = outlier_index_list
	max_dict['run_time'] = gcm_dict['run_time']

	with open(max_filepath, "w") as outfile:
		json.dump(max_dict, outfile, indent = 4)

	return max_filepath