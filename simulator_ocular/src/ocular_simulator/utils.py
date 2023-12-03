import numpy as np
import pandas as pd

def process_data_list(data_list, start_time, end_time):
    """
    data_list : list of data sent from client
                e.g [{'values':1, 'client_id':1, 'event_ts' : ...}, 
                     {'values':2, 'client_id':1, 'event_ts' : ...},
                     {'values':2, 'client_id':1, 'event_ts' : ...},]
    start_time : datetime.datetime
    end_time : datetime.datetime
    """
    data_dict = {}
    data_dict["data_list"] = data_list
    data_dict["start_time"] = start_time
    data_dict["end_time"] = end_time
    return data_dict

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
    
    if 'label' in df.columns:
        return arr, df['event_ts'].values, df['label'].values, df['index'].values
    else:
        return arr, df['event_ts'].values, (), ()


