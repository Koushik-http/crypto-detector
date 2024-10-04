import networkx as nx
import matplotlib.pyplot as plt

class NetworkVisualizer:
    def __init__(self, graph_data):
        """
        Initializes the NetworkVisualizer with the provided graph data.
        :param graph_data: A NetworkX graph object.
        """
        if not isinstance(graph_data, nx.Graph):
            raise TypeError("Graph data must be a NetworkX Graph object.")
        self.graph = graph_data

    def plot_graph(self):
        """
        Plots the network graph with nodes and edges.
        """
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
        plt.title('Cryptocurrency Transaction Network')
        plt.show()

    def plot_degree_distribution(self):
        """
        Plots the degree distribution of the network.
        """
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=30, color='skyblue', edgecolor='black')
        plt.title('Degree Distribution')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.grid()
        plt.tight_layout()
        plt.show()
