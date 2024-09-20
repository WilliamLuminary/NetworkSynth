import networkx as nx
import numpy as np

def create_graph(positions, sparse_matrix):
    G = nx.from_scipy_sparse_array(sparse_matrix)
    for i, pos in enumerate(positions):
        G.nodes[i]['pos'] = pos.astype(np.float64)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    return G

def group_nodes_by_degree(G):
    degree_count = dict(G.degree())
    nodes_by_degree = defaultdict(list)
    for node, degree in degree_count.items():
        nodes_by_degree[degree].append(node)
    return degree_count, nodes_by_degree

def calculate_degree_distribution(G):
    degree_count, nodes_by_degree = group_nodes_by_degree(G)
    total_nodes = sum(len(nodes) for nodes in nodes_by_degree.values())
    degree_distribution = calculate_degree_probabilities(nodes_by_degree, total_nodes)
    return degree_distribution

# Other graph-related functions here...