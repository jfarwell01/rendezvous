import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

model = cp_model.CpModel()


def optimize(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    start_nodes: set,
    end_nodes: set,
    edge_objectives: set = set(),
    node_objectives: set = set(),
) -> dict:
    """
    Args:
        nodes: Dataframe of nodes and attributes
        edges: Dataframe of edges, 'source' and 'destination'
        starting_nodes: set of eligble starting nodes
        ending_nodes: set of eligible ending nodes
        edge_objectives: set of column names from nodes to minimize when selected
        node_objectives: set of column names from edges to minimize when selected
    Return:
        solution dict
        {
            "objective" : solution cost
            "edges" : edges selected
            "time" : wall clock time to solution
        }

    """
    # Create Edge Variables
    edge_vars = {}
    dist_consts = {}
    dist_vars = {}
    for source, dest, dist in zip(
        edges["source"], edges["destination"], edges["distance"]
    ):
        name = str(source) + "_" + str(dest)
        edge_var = model.NewBoolVar(name)
        dist_const = model.NewConstant(dist)
        edge_vars[source, dest] = edge_var
        dist_consts[source, dest] = dist_const
        dist_var = model.NewIntVar(0, 10000, name + "_dist")
        dist_vars[source, dest] = dist_var
        # Makes sure that the dist_var is calculating the distance traveled
        model.AddMultiplicationEquality(dist_var, [edge_var, dist_const])

    # C1: Continuity Constraint
    for i in set(nodes["id"]) - start_nodes - end_nodes:
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
            [edge for key, edge in edge_vars.items() if key[0] in start_nodes]
        )
        == 1
    )
    # C3: Ending Constraint
    model.Add(
        cp_model.LinearExpr.Sum(
            [edge for key, edge in edge_vars.items() if key[1] in end_nodes]
        )
        == 1
    )
    # Objective
    dist_traveled = sum(dist_vars.values())
    model.Minimize(dist_traveled)
    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Model sucessful")
        return get_soln_dict(solver, edge_vars)
    else:
        print(f"Model failed with error code {status}")
        return None


def get_soln_dict(solver, edge_vars):
    return {
        "objective": solver.ObjectiveValue(),
        "edges": get_selected_edges(solver, edge_vars),
        "time": solver.WallTime(),
    }


def get_selected_edges(solver: cp_model.CpSolver, edge_vars: dict):
    """
    Extracts a solution from the solver and its variables
    """
    return [key for key, val in edge_vars.items() if solver.Value(val)]
