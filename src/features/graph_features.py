import networkx as nx
import pandas as pd
from typing import Dict

class GraphFeatureExtractor:
    def __init__(self, transaction_data: pd.DataFrame):
        self.transaction_data = transaction_data
        self.graph = self._create_transaction_graph()

    def _create_transaction_graph(self) -> nx.Graph:
        G = nx.Graph()
        for _, row in self.transaction_data.iterrows():
            G.add_edge(row['from'], row['to'], weight=row['amount'])
        return G

    def extract_node_features(self) -> Dict[str, Dict[str, float]]:
        features = {}
        for node in self.graph.nodes():
            features[node] = {
                'degree_centrality': nx.degree_centrality(self.graph)[node],
                'betweenness_centrality': nx.betweenness_centrality(self.graph)[node],
                'clustering_coefficient': nx.clustering(self.graph, node)
            }
        return features

    def extract_edge_features(self) -> Dict[tuple, Dict[str, float]]:
        features = {}
        for edge in self.graph.edges():
            features[edge] = {
                'edge_betweenness': nx.edge_betweenness_centrality(self.graph)[edge],
                'jaccard_coefficient': list(nx.jaccard_coefficient(self.graph, [edge]))[0][2]
            }
        return features

if __name__ == "__main__":
    # Load preprocessed transaction data
    transaction_data = pd.read_csv("data/processed/preprocessed_blockchain_data.csv")
    
    extractor = GraphFeatureExtractor(transaction_data)
    node_features = extractor.extract_node_features()
    edge_features = extractor.extract_edge_features()
    
    # Save extracted features
    pd.DataFrame.from_dict(node_features, orient='index').to_csv("data/processed/node_features.csv")
    pd.DataFrame.from_dict(edge_features, orient='index').to_csv("data/processed/edge_features.csv")