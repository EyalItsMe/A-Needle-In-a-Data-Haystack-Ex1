import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fruits import pagerank


def question_9_a_1():
    # # Step 1: Create a Ring Graph
    # num_nodes = 10  # Use a smaller number of nodes for simplicity
    # G = nx.cycle_graph(num_nodes)

    # Step 1: Create a Random Geometric Graph
    num_nodes = 10  # Adjust number of nodes for more variability
    radius = 0.5
    G = nx.random_geometric_graph(num_nodes, radius)

    # Convert the graph to a similarity matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    similarity_matrix = adj_matrix.tolist()

    # Step 2: Compute PageRank Values using Custom Function
    pagerank_values = pagerank(similarity_matrix, max_iterations=10)

    # Step 4: Plot the Histogram
    plt.hist(pagerank_values, bins=10, color='blue')
    plt.title('PageRank Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.show()

    # Step 5: Present the Graph and Adjacency Matrix
    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, node_size=50, with_labels=False)
    plt.title('Generated Graph')
    plt.show()

    # Adjacency Matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    print("Adjacency Matrix:")
    print(adj_matrix)

if __name__ == '__main__':

    question_9_a_1()