import networkx as nx


class TopologyGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_module(self, module_id):
        self.graph.add_node(module_id)

    def attach(self, m1, m2):
        self.graph.add_edge(m1, m2)

    def detach(self, m1, m2):
        if self.graph.has_edge(m1, m2):
            self.graph.remove_edge(m1, m2)

    def neighbors(self, module_id):
        return list(self.graph.neighbors(module_id))
