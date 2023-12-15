import pandas as pd
import networkx as nx
from dowhy import gcm
from scipy.stats import halfnorm
import numpy as np
import matplotlib.pyplot as plt


def main():
	
	## build DAG (Assumption 1)
	causal_graph = nx.DiGraph([('www', 'Website'),
	                           ('Auth Service', 'www'),
	                           ('API', 'www'),
	                           ('Customer DB', 'Auth Service'),
	                           ('Customer DB', 'API'),
	                           ('Product Service', 'API'),
	                           ('Auth Service', 'API'),
	                           ('Order Service', 'API'),
	                           ('Shipping Cost Service', 'Product Service'),
	                           ('Caching Service', 'Product Service'),
	                           ('Product DB', 'Caching Service'),
	                           ('Customer DB', 'Product Service'),
	                           ('Order DB', 'Order Service')])


	## set causal mechanism (SCM) (Assumption 2)
	causal_model = gcm.StructuralCausalModel(causal_graph)

	for node in causal_graph.nodes:
	    if len(list(causal_graph.predecessors(node))) > 0: 
	        causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
	    else:
	        ### when the node has no parent
	        causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(halfnorm))

	for node in causal_graph.nodes:
	    print(f"causal mechanism at node [{node}] = {causal_model.graph.nodes[node]}")


	## normal data
	normal_data = pd.read_csv("rca_microservice_architecture_latencies.csv")

	## load the outlier data
	outlier_data = pd.read_csv("rca_microservice_architecture_anomaly.csv")


	## Attributing an outlier latency at a target service to other services
	gcm.config.disable_progress_bars() # to disable print statements when computing Shapley valuesΩ
	median_attribs, uncertainty_attribs = gcm.confidence_intervals(
	    gcm.bootstrap_training_and_sampling(gcm.attribute_anomalies,
	                                        causal_model,
	                                        normal_data,
	                                        target_node='Website',
	                                        anomaly_samples=outlier_data),
	    num_bootstrap_resamples=1)

	## plot the contribution of each node (attributes)
	def bar_plot_with_uncertainty(median_attribs, uncertainty_attribs, ylabel='Attribution Score', figsize=(8, 3), bwidth=0.8, xticks=None, xticks_rotation=90):
	    fig, ax = plt.subplots(figsize=figsize)
	    yerr_plus = [uncertainty_attribs[node][1] - median_attribs[node] for node in median_attribs.keys()]
	    yerr_minus = [median_attribs[node] - uncertainty_attribs[node][0] for node in median_attribs.keys()]
	    plt.bar(median_attribs.keys(), median_attribs.values(), yerr=np.array([yerr_minus, yerr_plus]), ecolor='#1E88E5', color='#ff0d57', width=bwidth)
	    plt.xticks(rotation=xticks_rotation)
	    plt.ylabel(ylabel)
	    ax.spines['right'].set_visible(False)
	    ax.spines['top'].set_visible(False)
	    if xticks:
	        plt.xticks(list(median_attribs.keys()), xticks)
	    plt.show()

	bar_plot_with_uncertainty(median_attribs, uncertainty_attribs)

if __name__ == '__main__':
    main()