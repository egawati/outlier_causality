import networkx as nx
import matplotlib.pyplot as plt

import random

def generate_random_dag(num_nodes,random_seed=0):
    graph = nx.DiGraph()

    random.seed(random_seed)
    node_list = [f'X{num_node}' for num_node in range(num_nodes)]

    graph.add_nodes_from(node_list)

    nodes = list(graph.nodes())
    random.shuffle(nodes)  # Randomize node order

    for i in range(1, num_nodes):
        # Connect each node to a random subset of previously added nodes
        random.shuffle(nodes[:i])  # Randomize subset order
        num_edges = random.randint(0, i - 1)  # Random number of edges
        selected_nodes = nodes[:num_edges]  # Select subset of nodes

        # Add edges from selected nodes to the current node
        for node in selected_nodes:
            graph.add_edge(node, nodes[i])

    return graph

def generate_random_dag_with_min_leaf_depth(n, min_leaf_depth, random_seed=0):
    graph = nx.DiGraph()

    random.seed(random_seed)
    node_list = [f'X{num_node}' for num_node in range(n)]

    # Add nodes to the graph
    graph.add_nodes_from(node_list)

    nodes = list(graph.nodes())
    root_node = nodes[0]
    
    for i in range(1, min_leaf_depth+1):
        graph.add_edge(nodes[i-1], nodes[i])

    selected_parents = nodes[:min_leaf_depth+1]

    for node in nodes[min_leaf_depth+1:]:
        A = node 
        B = random.choice(selected_parents)
        random_number = random.choice([0, 1])
        
        if random_number == 0:
            graph.add_edge(B, A)
        else:
            graph.add_edge(A, B)
            selected_parents.append(A)

    return graph

def find_leaf_nodes(graph):
    leaf_nodes = []
    for node in graph.nodes:
        if graph.in_degree(node) > 0 and graph.out_degree(node) == 0:
            leaf_nodes.append(node)
    return leaf_nodes

def find_depth_of_leaf_node(graph, leaf_node):
    root_nodes = list()
    for node in graph.nodes:
        if graph.in_degree(node) == 0 and graph.out_degree(node) > 0:
            root_nodes.append(node)
    
    max_depth = 0
    path = None
    for root in root_nodes:
        if nx.has_path(graph, root, leaf_node):
            depth = nx.shortest_path_length(graph, source=root, target=leaf_node)
            if max_depth < depth:
                max_depth = depth
                path = nx.shortest_path(graph, source=root, target=leaf_node)
    return max_depth, path




def generate_causal_graph_with_min_leaf_depth(num_nodes, min_leaf_depth, random_seed):
    random_dag = generate_random_dag_with_min_leaf_depth(n=num_nodes, 
                                                         min_leaf_depth=min_leaf_depth,
                                                         random_seed=random_seed)

    
    leaf_nodes = find_leaf_nodes(random_dag)

    root_leaf_paths = list()
    for leaf_node in leaf_nodes:
        max_depth, path = find_depth_of_leaf_node(random_dag, leaf_node)
        root_leaf_paths.append((max_depth, path))

    root_leaf_paths = sorted(root_leaf_paths, key=lambda x:-x[0])
    return random_dag, root_leaf_paths


if __name__ == '__main__':
    random_dag, root_leaf_paths = generate_causal_graph_with_min_leaf_depth(num_nodes=10, 
                                                                            min_leaf_depth=5, 
                                                                            random_seed=1)
    print(root_leaf_paths)
    random.seed(1)
    max_depth = root_leaf_paths[0][0]
    max_path = root_leaf_paths[0][1]
    print(max_path)
    root_cause_node_idx = random.choice(range(max_depth-2))
    
    root_cause_node = max_path[root_cause_node_idx]
    target_node = max_path[-1]
    print(f'root_cause_node = {root_cause_node}, target_node = {target_node}')

    pos = nx.spring_layout(random_dag)
    nx.draw(random_dag, pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=500, 
            edge_color='gray', 
            arrows=True)
    plt.show()