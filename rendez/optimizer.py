import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from collections import defaultdict


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
    edge_vars = {}
    obj_consts = defaultdict(dict)
    obj_vars = defaultdict(dict)
    # Populate Boolean Edge Variables
    edges.apply(add_edge_vars, args=(model, edge_vars), axis=1)
    # Populate Objective Variables and Constants
    edges.apply(
        add_objective_vars,
        args=(model, edge_vars, edge_objectives, obj_consts, obj_vars),
        axis=1,
    )

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
    loss = sum([sum(obj_vars[key].values()) for key in obj_vars])
    model.Minimize(loss)
    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Model sucessful")
        return get_soln_dict(solver, edge_vars)
    else:
        print(f"Model failed with error code {status}")
        return None


def add_edge_vars(row, model, edge_vars):
    name = str(row["source"]) + "_" + str(row["destination"])
    edge_var = model.NewBoolVar(name)
    edge_vars[row["source"], row["destination"]] = edge_var


def add_objective_vars(
    row, model, edge_vars, edge_objectives: set, obj_consts: dict, obj_vars: dict
):
    """
    Creates a dictionary of objective_vars that contain the objective constants when the corresponding
    edge is selected and contain 0 when the edge is not selected
    """
    for edge_obj in edge_objectives:
        name = str(row["source"]) + "_" + str(row["destination"]) + "_" + edge_obj
        new_constant = model.NewConstant(row[edge_obj])
        obj_consts[edge_obj][row["source"], row["destination"]] = new_constant
        new_var = model.NewIntVar(-10000, 10000, name)
        obj_vars[edge_obj][row["source"], row["destination"]] = new_var
        model.AddMultiplicationEquality(
            new_var, [edge_vars[row["source"], row["destination"]], new_constant]
        )


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
    return [
        (int(key[0]), int(key[1]))
        for key, val in edge_vars.items()
        if solver.Value(val)
    ]
