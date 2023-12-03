import numpy as np
from typing import Callable, List, Optional, Union

def permute_features(
    feature_samples: np.ndarray, features_to_permute: Union[List[int], np.ndarray], randomize_features_jointly: bool
) -> np.ndarray:
    """
    this function is the same as permute_features in dowhy.gcm.stats
    """
    feature_samples = np.array(feature_samples) #copy feature_samples so that the original object remains unchanged
    if randomize_features_jointly:
        # Permute samples jointly. This still represents an interventional distribution.
        feature_samples[:, features_to_permute] = feature_samples[
            np.random.choice(feature_samples.shape[0], feature_samples.shape[0], replace=False)
        ][:, features_to_permute]
    else:
        # Permute samples independently.
        for feature in features_to_permute:
            np.random.shuffle(feature_samples[:, feature])
    return feature_samples