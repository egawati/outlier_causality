from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

def accuracy_metrics(ground_truth, prediction, labels, average='micro'):
    print(f'ground_truth {ground_truth}')
    print(f'prediction {prediction}')
    accuracy = accuracy_score(ground_truth, prediction)
    f1 = f1_score(ground_truth, prediction, labels=labels, average=average)
    precision = precision_score(ground_truth, prediction, labels=labels, average=average)
    recall = recall_score(ground_truth, prediction, labels=labels, average=average)
    result = {'accuracy': accuracy,
              'f1' : f1,
              'precision' : precision,
              'recall' : recall}
    return result