import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dowhy import gcm

from .utils import get_outlier_event_timestamp
from .utils import get_slide_number_to_explain_outlier

from .outlier_score import node_outlier_contribution_scores

from .noise_data_generation import generate_noisedf_from_data

def find_outlier_root_causes(outlier, 
                             snoise_models,
                             snoise_samples,
                             soutlier_scorer, 
                             causal_graph, 
                             active_fcms, 
                             sorted_nodes,
                             target_node,
                             shapley_config=None,
                             attribute_mean_deviation=False):
    """
    outliers : dictionary {"values" : ..., "event_ts" : ...}
               e.g {"values" : np.array(...), "event_ts" : datetime.datetime}
    snoise_models : dictionary of slide_number : {dictionary of node : fcm_model}
             e.g. {0:{X0:scm_model, X1:scm_model}, 1:{X0:scm_model, X1:scm_model}, ...}
    outlier_score_function : a class of ocular.outlier_score
    causal_graph : a ocular.causal_model.dag CausalGraph object
    active_fcms : active_fcms : dictionary {0:{"star_ts" : ..., "end_ts" : ...}, 
                                1:{"star_ts" : ..., "end_ts" : ...},
                                2:{"star_ts" : ..., "end_ts" : ...}, ...}
    """
    outlier_ts = get_outlier_event_timestamp(outlier)
    #print(f'active_fcms is {active_fcms}')
    out_prev_slide = get_slide_number_to_explain_outlier(outlier_ts, active_fcms)
    
    noise_models = snoise_models[out_prev_slide]
    noise_samples = snoise_samples[out_prev_slide]
    outlier_scorer = soutlier_scorer[out_prev_slide]

    outlier_df = pd.DataFrame(data=outlier['values'], columns=sorted_nodes)
    outlier_noise = generate_noisedf_from_data(outlier_df, 
                             noise_models, 
                             causal_graph, 
                             sorted_nodes, 
                             target_node)
    out_noises_arr = outlier_noise.to_numpy()
    gcm.config.disable_progress_bars()
    contributions = node_outlier_contribution_scores(out_noises_arr,
                                                     noise_samples,
                                                     outlier_scorer,
                                                     attribute_mean_deviation,
                                                     noise_models,
                                                     causal_graph,
                                                     sorted_nodes,
                                                     target_node,
                                                     shapley_config)

    contributions_dict = {sorted_nodes[i]: contributions[:,i] for i in range(len(sorted_nodes))}
    return outlier_ts, contributions_dict


def find_outliers_root_causes_paralel(outliers, 
                                     snoise_models,
                                     snoise_samples,
                                     soutlier_scorer, 
                                     causal_graph, 
                                     active_fcms, 
                                     sorted_nodes,
                                     target_node,
                                     shapley_config=None,
                                     attribute_mean_deviation=False,
                                     n_jobs=-1):
    """
    Currently we compute outlier score per node
    """
    
    contributions = Parallel(n_jobs=n_jobs)(
                        delayed(find_outlier_root_causes)
                               (outlier, 
                                snoise_models,
                                snoise_samples,
                                soutlier_scorer, 
                                causal_graph, 
                                active_fcms, 
                                sorted_nodes,
                                target_node,
                                shapley_config,
                                attribute_mean_deviation) 
                               for outlier in outliers
                        )
    ## need to ensure that contributions are ordered based on the outlier timestamp
    contributions = sorted(contributions, key = lambda x:(x[0]))
    contributions_hmap = {}
    for item in contributions:
        contributions_hmap[item[0]] = item[1]
    return contributions_hmap