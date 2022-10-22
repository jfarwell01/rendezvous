import pandas as pd
import rendez.optimizer


def simple_test():
    edges = pd.read_csv("test_data/test_edges.csv")
    nodes = pd.read_csv("test_data/test_nodes.csv")
    soln = rendez.optimizer.optimize(nodes, edges)
    assert_solution_valid(edges, soln)


def assert_solution_valid(edges, soln):
    print("Proposed Solution:\n", soln)
    sources = set([s[0] for s in soln])
    dests = set([s[1] for s in soln])
    # Solution Length
    assert len(soln) == 3, "Solution length incorrect"
    # C1: Continuity Constraint
    assert sources - dests == {0}, "Continuity constraint violated"
    assert len(dests - sources) == 1, "Continuity constraint violated"
    # C2: Starting Constraint
    assert 0 in sources, "Starting constraint violated"
    # C3: Ending Constraint
    # assert sum([s[1]  for s in soln]) == 1, "Starting constraint violated"


if __name__ == "__main__":
    simple_test()
