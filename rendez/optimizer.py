import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

model = cp_model.CpModel()


def add_var(x):
    model.new


def optimize(nodes: pd.DataFrame, edges: pd.DataFrame):
    """
    Args:
        nodes: Dataframe of nodes and attributes
        edges: Dataframe of edges, 'source' and 'destination'

    """
    ending_nodes = [5, 6]  # TODO: do not hardcode
    # Create Edge Variables
    edge_vars = {}
    for source, destination in zip(edges["source"], edges["destination"]):
        edge_vars[source, destination] = model.NewBoolVar(
            str(source) + "_" + str(destination)
        )

    # C1: Continuity Constraint
    for i in range(1, 5):
        model.Add(
            cp_model.LinearExpr.Sum(
                [edge for key, edge in edge_vars.items() if key[0] == i]
            )
            == cp_model.LinearExpr.Sum(
                [edge for key, edge in edge_vars.items() if key[1] == i]
            )
        )
    # C2: Starting Constraint
    model.Add(
        cp_model.LinearExpr.Sum(
            [edge for key, edge in edge_vars.items() if key[0] == 0]
        )
        == 1
    )
    # C3: Ending Constraint
    model.Add(
        cp_model.LinearExpr.Sum(
            [edge for key, edge in edge_vars.items() if key[1] in ending_nodes]
        )
        == 1
    )

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Model sucessful")
        return get_selected_edges(solver, edge_vars)
    else:
        print(f"Model failed with error code {status}")
        return None


def get_selected_edges(solver: cp_model.CpSolver, edge_vars: dict):
    """
    Extracts a solution from the solver and its variables
    """
    return [key for key, val in edge_vars.items() if solver.Value(val)]
