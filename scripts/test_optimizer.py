import pandas as pd
import rendez.optimizer

def simple_test():
    edges = pd.read_csv("test_data/test_edges.csv")
    nodes = pd.read_csv("test_data/test_nodes.csv")
    rendez.optimizer.optimize(nodes, edges)

if __name__ == "__main__":
    simple_test()
