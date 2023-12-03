import networkx as nx

class CausalGraph():
	def __init__(self, nodes, features):
		"""
		nodes : list of node relationship
				e.g. [('X1', 'X2'), ('X2', 'X3'), ('X2', 'X4'), ('X3', 'X5')]
		features : tuples of features
				e.g. (X1, X2, X3, X4, X5)
		"""
		self.dag = nx.DiGraph(nodes)
		self.sorted_nodes = list(nx.topological_sort(self.dag))
		self.root = list(nx.topological_sort(self.dag))
		self.ancestors = dict()
		self.ancestors_findex = dict()
		self.parents = dict()
		for node in self.dag.nodes:
			self.parents[node] = list(self.dag.predecessors(node))
			self.ancestors[node] = nx.ancestors(self.dag, node)
			self.ancestors_findex[node] = list()
			for anode in self.ancestors[node]:
				index = features.index(anode)
				self.ancestors_findex[node].append(index)
		self.features = features

	def get_shortest_path(self, source_node, dest_node):
		try:
			return nx.shortest_path(self.dag, source_node, dest_node)
		except : 
			return None

	def get_node_parent(self, child_node):
		return list(self.dag.predecessors(child_node))

	def predecessors(self, child_node):
		return list(self.dag.predecessors(child_node))
