import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from fruits import pagerank


def calc_histogram(G, num_bins):
    # Convert the graph to a similarity matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    # todo need to fix our code and replace it with that
    similarity_matrix = adj_matrix.tolist()
    pagerank_values = pagerank(similarity_matrix, damping_factor=0.01)

    # pagerank = nx.pagerank(G)
    # pagerank_values = list(pagerank.values())

    bin_edges = np.linspace(min(pagerank_values), max(pagerank_values), num_bins + 1)
    plt.hist(pagerank_values, bins=bin_edges, edgecolor='black')  # Equal-length bins
    plt.title('Histogram of PageRank Values')
    plt.xlabel('PageRank')
    plt.ylabel('Frequency')
    plt.show()

    print("Adjacency Matrix:")
    print(adj_matrix)

    print("\nGraph Structure (Edges):")
    print(G.edges())


def louvain_community_detection(G):
    # Compute the best partition
    partition = community_louvain.best_partition(G)

    # Draw the graph with the partition
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('viridis')

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=500,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title("Louvain Community Detection")
    plt.show()

    return partition


def aggregate_graph(G, partition):
    # Creating a new graph where each node is a community
    community_graph = nx.Graph()

    # Mapping each community to a representative node
    community_representative = {}
    for node, community in partition.items():
        if community not in community_representative:
            community_representative[community] = node

    # Adding nodes with community sizes
    community_sizes = {}
    for node, community in partition.items():
        if community not in community_sizes:
            community_sizes[community] = 0
        community_sizes[community] += 1

    for community, size in community_sizes.items():
        community_graph.add_node(community_representative[community], size=size)

    # Add edges between communities
    for (node1, node2) in G.edges():
        community1 = partition[node1]
        community2 = partition[node2]
        if community1 != community2:
            rep1 = community_representative[community1]
            rep2 = community_representative[community2]
            if community_graph.has_edge(rep1, rep2):
                community_graph[rep1][rep2]['weight'] += 1
            else:
                community_graph.add_edge(rep1, rep2, weight=1)

    # Draw the aggregated community graph
    pos = nx.spring_layout(community_graph)

    plt.figure(figsize=(8, 6))
    plt.title("Aggregated Graph Based on Communities")
    sizes = [community_graph.nodes[node]['size'] * 100 for node in community_graph.nodes]
    nx.draw(community_graph, pos, with_labels=True, node_size=sizes, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(community_graph, pos, edge_labels=nx.get_edge_attributes(community_graph, 'weight'))

    plt.show()


def question_9():
    # section (a)
    edges = [(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (4, 5), (4, 6)]
    G1 = nx.Graph()
    G1.add_edges_from(edges)
    calc_histogram(G1, 3)

    # section (b)
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (5, 6), (5, 7), (6, 7), (10, 11), (11, 12), (2, 13), (2, 15)]
    G2 = nx.Graph()
    G2.add_edges_from(edges)
    calc_histogram(G2, 4)

    # section (c)
    partition1 = louvain_community_detection(G1)
    aggregate_graph(G1, partition1)
    partition2 = louvain_community_detection(G2)
    aggregate_graph(G2, partition2)


if __name__ == '__main__':
    question_9()
