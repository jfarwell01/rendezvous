import pandas as pd
import rendez.optimizer
import json


def test_small():
    edges = pd.read_csv("test_data/test_edges.csv")
    nodes = pd.read_csv("test_data/test_nodes.csv")
    start_nodes = {0}
    end_nodes = {5, 6}
    soln = rendez.optimizer.optimize(nodes, edges, start_nodes, end_nodes)
    print(json.dumps(soln, sort_keys=True, indent=4))
    assert_solution_valid(edges, soln["edges"], start_nodes, end_nodes)


def assert_solution_valid(edges, soln, start_nodes, end_nodes):
    sources = set([s[0] for s in soln])
    dests = set([s[1] for s in soln])
    # C1: Continuity Constraint
    assert len(soln) == 3, "Solution length incorrect"
    # C2: Starting Constraint
    assert (
        len(start_nodes.intersection(sources - dests)) == 1
    ), "Starting/Continuity constraint violated"
    # C3: Ending Constraint
    assert (
        len(end_nodes.intersection(dests - sources)) == 1
    ), "Ending/Continuity constraint violated"


if __name__ == "__main__":
    test_small()
