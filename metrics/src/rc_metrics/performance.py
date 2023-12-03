import os
import json
import pandas as pd
from rc_metrics.accuracy import accuracy_metrics

def save_performance_result(result_dict, exp_name, rel_path, nslide_used=None):
    cwd = os.getcwd()
    result_folder = os.path.join(cwd, rel_path)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if nslide_used is None:
        filepath = f'{result_folder}/performance_{exp_name}.json'
    else:
        filepath = f'{result_folder}/performance_{exp_name}_{nslide_used}.json'

    with open(filepath, "w") as outfile:
        json.dump(result_dict, outfile, indent = 4)

def get_outlier_root_cause_gt(data_path, separator):
    df = pd.read_csv(data_path, sep=separator)
    out_indices = df.index[df['label'] == 1].tolist()
    root_cause_gts = df.loc[out_indices, 'root_cause_gt'].tolist()
    gt_dict  = {'outlier_index' : out_indices, 'root_cause_gt': root_cause_gts}
    return gt_dict

def get_root_cause_prediction_result(filepath):
    with open(filepath) as input_file:
        file_contents = input_file.read()
    pred_dict = json.loads(file_contents)
    pred_out_indices = pred_dict['outlier_index'] ##  a list
    pred_root_causes = pred_dict['root_cause'] ## a list

    new_pred_dict = {}
    for i, out_index in enumerate(pred_out_indices):
        new_pred_dict[out_index] = pred_root_causes[i]

    return new_pred_dict, pred_dict['run_time']

def combine_ground_truth_result_indices(gt_dict, pred_dict):
    df = pd.DataFrame(gt_dict)
    df['root_cause_pred'] = ''
    df['root_cause_pred'] = df['outlier_index'].map(pred_dict)
    return df

def compute_and_save_performance(data_path, result_path, 
                                 exp_name, features, data_separator=',',
                                 nslide_used=None,
                                 average_explanation_time=None):
    gt_dict = get_outlier_root_cause_gt(data_path, separator=data_separator)
    pred_result = get_root_cause_prediction_result(result_path)
    pred_dict = pred_result[0]
    run_time = pred_result[1]
    df = combine_ground_truth_result_indices(gt_dict, pred_dict)
    metrics = accuracy_metrics(df['root_cause_gt'].values, df['root_cause_pred'].values, features)
    performance = {'accuracy_metrics' : metrics, 
                   'run_time' : run_time,
                   'average_explanation_time':average_explanation_time}
    rel_path = os.path.dirname(result_path)
    save_performance_result(performance, exp_name, rel_path, nslide_used)