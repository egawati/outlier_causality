import numpy as np
import pandas as pd
import random
import networkx as nx
import math


from causal_gen.basic_ts import generate_data
from sklearn.linear_model import LinearRegression


def get_root_parents_node(causal_graph):
    node_parents = {}
    root = list()
    for node in causal_graph.nodes:
        predecessors = list(causal_graph.predecessors(node))
        if predecessors:
            node_parents[node] = predecessors
            ## since there could be a case, a value of a node at time t is caused by it previous own value
            if len(predecessors) == 1 and predecessors[0] == node:
                root.append(node)
        else:
            root.append(node)
    return root, node_parents


def get_lagged_values(df, target_node, parents, lag=1):
    df_lagged = df[[target_node]].copy()
    lagged_vars = list()
    for vertex in parents:
        vertex_lag = f'{vertex}_lag'
        lagged_vars.append(vertex_lag)
        if lag == 1:
            df_lagged[vertex_lag] = df[vertex].shift(lag).copy()
    return df_lagged, lagged_vars

def get_original_df_stats(df):
    original_df_stats = {}
    original_df_stats['mean'] = {}
    original_df_stats['std'] = {}
    original_df_stats['min'] = {}
    original_df_stats['max'] = {}
    for col in df.columns:
        original_df_stats['mean'][col] = df[col].mean()
        original_df_stats['std'][col] = df[col].std()
        original_df_stats['min'][col] = df[col].min()
        original_df_stats['max'][col] = df[col].max()
    return original_df_stats

def learn_fcm_of_timeseries(df, causal_graph, lag=1):
    fcm = dict()
    for node in causal_graph.nodes:
        predecessors = list(causal_graph.predecessors(node))
        if predecessors:
            df_lagged, lagged_vars = get_lagged_values(df = df, 
                                                        target_node = node, 
                                                        parents=predecessors, 
                                                        lag=lag)
            df_lagged = df_lagged.dropna()
            predictors = df_lagged[lagged_vars].values
            target = df_lagged[node]
            model = LinearRegression()
            model.fit(predictors, target)
            fcm[node] = (lagged_vars, model)
        else:
            fcm[node] = None
    return fcm

def inject_an_outlier(df,
                      original_df_stats,
                      fcm, 
                      causal_graph,
                      node_parents,
                      target_node,
                      root_cause,
                      target_node_position,
                      multiplier = 5):
    n_features = df.shape[1]
    has_path = nx.has_path(causal_graph, 
                           source=root_cause, 
                           target=target_node)
    if not has_path:
        print(f'No path from {root_cause} to {target_node}')
        return
    
    ## determine outlier value at the root cause node
    mean_root_cause = original_df_stats['mean'][root_cause]
    std_root_cause = original_df_stats['std'][root_cause]
    min_val = original_df_stats['min'][root_cause]
    max_val = original_df_stats['max'][root_cause]
    root_cause_outlier_val = mean_root_cause + (multiplier * std_root_cause) + random.uniform(min_val, max_val)
    
    ## path from the root cause to the target node
    path = nx.shortest_path(causal_graph, 
                            source=root_cause, 
                            target=target_node)
    distance = len(path) - 1
    
    path_vals = {}
    for node in path:
        node_pos_in_path_from_target = distance - path.index(node) 
        ## get position of the node relative to the target_node
        rel_node_position = target_node_position - node_pos_in_path_from_target 
        
        if node == root_cause:
            path_vals[node] = root_cause_outlier_val
            df.at[rel_node_position, node] = path_vals[node]
            continue
        
        parents = node_parents[node]
        X = {}
        for parent in parents: 
            parent_val = path_vals.get(parent)
            if parent_val:
                X[f'{parent}_lag'] = parent_val
            else: ## just in case the parent is not part of the path
                ## get parent position relative to the target node
                rel_parent_pos = rel_node_position - 1
                X[f'{parent}_lag'] = df.loc[rel_parent_pos, parent]
        # print(f'X is {X}')
        X_pd = pd.DataFrame([X])
        predictor_vars, node_fcm = fcm[node]
        # print(f'predictor_vars {predictor_vars}')
        X_array = X_pd[predictor_vars].values
        path_vals[node] = node_fcm.predict(X_array)
        df.at[rel_node_position, node] = path_vals[node]


def get_the_longest_path_from_root(causal_graph, roots, target_node):
    max_path = -math.inf 
    the_longest_path = None
    the_root = None
    for i, root in enumerate(roots):
        all_paths = list(nx.all_simple_paths(causal_graph, source=root, target=target_node))
        if all_paths:
            all_paths = sorted(all_paths, key=lambda path: len(path), reverse=True)
            longest_path = all_paths[0]
            if max_path < len(longest_path):
                max_path = len(longest_path)
                the_longest_path = longest_path
                the_root = root
    return the_longest_path, the_root

def inject_n_outliers(df,
                      causal_graph,
                      target_node,
                      n_outliers,
                      multiplier = 5,
                      lag = 1,
                      using_root=False):
    """
    Assuming that each column in a row/tuple in the dataframe df shares the same timestamp.
    """
    n_data = df.shape[0]
    if n_outliers < 1:
        n_outliers = int(n_outliers * n_data)
    
    target_outlier_positions = random.sample(range(len(causal_graph.nodes), n_data), 
                                     n_outliers)
    target_outlier_positions = tuple(sorted(target_outlier_positions))
    
    fcm = learn_fcm_of_timeseries(df, causal_graph, lag=lag)
    roots, node_parents = get_root_parents_node(causal_graph)
    original_df_stats = get_original_df_stats(df)

    df['label'] = np.zeros(df.shape[0])

    root_cause_gt = np.zeros(df.shape[0])
    root_cause_gt = np.NaN
    
    df['root_cause_gt'] = root_cause_gt

    ### since we want to randomize the root cause
    possible_root_causes, dag_root =  get_the_longest_path_from_root(causal_graph, 
                                                                     roots, 
                                                                     target_node)
    possible_root_causes.remove(target_node)
    if not using_root:
        possible_root_causes.remove(dag_root)

    root_causes = list()
    for target_outlier_position in target_outlier_positions:
        root_cause = random.choice(possible_root_causes)
        inject_an_outlier(df, 
                          original_df_stats,
                          fcm, 
                          causal_graph,
                          node_parents,
                          target_node,
                          root_cause,
                          target_node_position=target_outlier_position,
                          multiplier = multiplier)
        df.at[target_outlier_position, 'label'] = 1 
        df.at[target_outlier_position, 'root_cause_gt'] = root_cause
        root_causes.append(root_cause)
    return target_outlier_positions, root_causes

