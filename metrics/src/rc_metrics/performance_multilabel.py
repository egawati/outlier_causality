import os
import json
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from rc_metrics.performance import get_outlier_root_cause_gt
from rc_metrics.performance import get_root_cause_prediction_result
from rc_metrics.performance import combine_ground_truth_result_indices
from rc_metrics.performance import save_performance_result

from rc_metrics.accuracy import accuracy_metrics

from sklearn.metrics import f1_score

from sklearn.preprocessing import MultiLabelBinarizer

def compute_and_save_performance_multilabel(data_path, result_path, 
                                 exp_name, features, data_separator=',',
                                 nslide_used=None,
                                 average_explanation_time=None):
    gt_dict = get_outlier_root_cause_gt(data_path, separator=data_separator)
    gt_dict['root_cause_gt'] = [(gt, ) for gt in gt_dict['root_cause_gt']]

    pred_result = get_root_cause_prediction_result(result_path)
    print(f'features {features}')
    print(pred_result)
    
    pred_dict = pred_result[0]
    run_time = pred_result[1]
    
    df = combine_ground_truth_result_indices(gt_dict, pred_dict)

    mlb = MultiLabelBinarizer()
    mlb.fit([features])
    y_true_bin = mlb.transform(df['root_cause_gt'].values)
    print(df['root_cause_gt'].values)
    print(f'y_true_bin {y_true_bin}')
    y_pred_bin = mlb.transform(df['root_cause_pred'].values)
    print(df['root_cause_pred'].values)
    print(f'y_pred_bin {y_pred_bin}')

    
    f1 = f1_score(y_true_bin, y_pred_bin, average='micro')
    precision = precision_score(y_true_bin, y_pred_bin, average='micro')
    recall = recall_score(y_true_bin, y_pred_bin, average='micro')

    metrics = {'f1' : f1,
              'precision' : precision,
              'recall' : recall}

    performance = {'accuracy_metrics' : metrics, 
                   'run_time' : run_time,
                   'average_explanation_time':average_explanation_time}
    rel_path = os.path.dirname(result_path)
    save_performance_result(performance, exp_name, rel_path, nslide_used)