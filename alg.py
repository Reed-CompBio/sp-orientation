import argparse
from collections.abc import Iterable
import itertools
from ortools.linear_solver import pywraplp
from pathlib import Path
from mixed import Direction, MixedGraph
from typing import cast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--interactome",
        help="Interactome file. No header, tab-delimited, source-target-weight-direction",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sources",
        help="File which denotes source nodes, with one node per line.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--targets",
        help="File which denotes source nodes, with one node per line.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output", help="Prefix for all output files.", type=str, required=True
    )

    return parser.parse_args()


def read_nodes(path: Path) -> Iterable[str]:
    lines = path.read_text().splitlines()
    return map(str.strip, lines)


def mixed_graph_from_path(path: Path) -> MixedGraph[str]:
    graph = MixedGraph[str]()

    with path.open() as interactome_r:
        for line in interactome_r:
            components = line.split("\t")
            if len(components) != 4:
                raise ValueError(
                    "Line with content '{line}' does not have four entries!"
                )
            source, target, weight, direction = components
            weight = float(weight)
            if weight < 0:
                raise ValueError(
                    f"Weight {weight} is negative! Weight should be a positive real."
                )
            graph.add_edge(source, target, Direction.from_str(direction), weight=weight)

    return graph

def prepare_orientation_variables(
    solver: pywraplp.Solver, graph: MixedGraph[str]
) -> dict[tuple[str, str], pywraplp.Variable]:
    orientations: dict[tuple[str, str], pywraplp.Variable] = {}
    # First, we need to define our orientation variables, where
    # As following the paper, 0 is reversed (v, u) and 1 is as usual (u, v).
    for source, target, _data, direction in graph.edges():
        direction_var: pywraplp.Variable = solver.IntVar(
            0, 1, f"{direction}-edge-{source}-{target}"
        )
        orientations[(source, target)] = direction_var

        direction_var: pywraplp.Variable = solver.IntVar(
            0, 1, f"{direction}-edge-{target}-{source}"
        )
        orientations[(target, source)] = direction_var

    return orientations


def prepare_orientation_constraints(
    solver: pywraplp.Solver,
    graph: MixedGraph[str],
    orientations: dict[tuple[str, str], pywraplp.Variable],
):
    # W restrict each undirected edge s.t. for (u, v), exactly one of (u, v), (v, u) is marked with
    # '1' as aforementioned in prepare_variables, specifically for undirected edges
    for source, target, _data, _direction in graph.edges(
        direction=Direction.UNDIRECTED
    ):
        orientation_constraint: pywraplp.Constraint = solver.Constraint(
            1, 1, f"single-orient-edge-{source}-{target}"
        )
        orientation_variable = orientations[(source, target)]
        orientation_constraint.SetCoefficient(orientation_variable, 1)

        orientation_variable_reverse = orientations[(source, target)]
        orientation_constraint.SetCoefficient(orientation_variable_reverse, 1)

    # Then, we need to make sure our directed edges are satisfied.
    for source, target, _data, _direction in graph.edges(direction=Direction.DIRECTED):
        orientation_constraint: pywraplp.Constraint = solver.Constraint(
            1, 1, f"directed-orient-edge-{source}-{target}"
        )
        orientation_variable = orientations[(source, target)]
        orientation_constraint.SetCoefficient(orientation_variable, 1)

        orientation_constraint_reversed: pywraplp.Constraint = solver.Constraint(
            0, 0, f"directed-orient-edge-{target}-{source}"
        )
        orientation_variable_reversed = orientations[(target, source)]
        orientation_constraint_reversed.SetCoefficient(orientation_variable_reversed, 1)

def prepare_closure_variables(
    solver: pywraplp.Solver, source_to_targets: dict[str, list[str]]
) -> dict[tuple[str, str], pywraplp.Variable]:
    # We define c_(s, t) as a {0, 1} variable that is satisfied when it is equal to 1.
    # These check if the shortest path from (s, t) is properly satisfied by the directionality ILP.
    closure_variables: dict[tuple[str, str], pywraplp.Variable] = {}
    for source, targets in source_to_targets.items():
        for target in targets:
            closure_var: pywraplp.Variable = solver.IntVar(0, 1, f"c-{source}-{target}")
            closure_variables[(source, target)] = closure_var

    return closure_variables

def prepare_flow(
    solver: pywraplp.Solver,
    graph: MixedGraph[str],
    source_to_targets: dict[str, list[str]],
    orientations: dict[tuple[str, str], pywraplp.Variable],
    closures: dict[tuple[str, str], pywraplp.Variable]
):
    for source, targets in source_to_targets.items():
        edges: list[tuple[str, str]] = []
        paths: list[list[str]] = []
        for target in targets:
            path = graph.dijkstra(source, target, "weight")
            paths.append(path)

        for path in paths:
            for i, s in enumerate(path):
                t = path[i + 1]
                edges.append((s, t))

        edge_variables: dict[str, dict[str, pywraplp.Variable]] = dict()
        edge_variables_reversed: dict[str, dict[str, pywraplp.Variable]] = dict()
        for s, t in edges:
            # (10) We first add edge flow s.t. it must be positive along
            # the shortest paths
            edge_variable = solver.NumVar(0, solver.Infinity, f"{source}-edge-{s}-{t}")
            solver.Add(0 < edge_variable)
            if not edge_variables[s] or not edge_variable[t]:
                edge_variables[s] = dict()
                edge_variables_reversed[t] = dict()
            edge_variables[s][t] = edge_variable
            edge_variables_reversed[t][s] = edge_variable

            # (11) then, we ensure that this flow respects all of the orientations we have
            solver.Add(edge_variable <= orientations[(s, t)])

            # (12) we restrict the amount of flow exiting this edge to be less than the amount
            # of flow entering it
            # We use cast here since Variable doesn't know it can be added. This also holds for (13).
            solver.Add(edge_variable <= sum(cast(int, i) for i in edge_variables_reversed[s].values()))


        # (13) restrict closure variables to require flow
        for target in targets:
            solver.Add(cast(int, closures[(source, target)]) <= sum(itertools.chain.from_iterable((cast(int, v) for t, v in t.items()) for s, t in edge_variables.items())))

def main():
    args = parse_args()
    interactome = mixed_graph_from_path(args.interactome)
    sources = read_nodes(args.sources)
    targets = read_nodes(args.targets)

    source_to_targets = dict((a, [b]) for a, b in itertools.product(sources, targets))

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("GLOP")

    orientations = prepare_orientation_variables(solver, interactome)
    prepare_orientation_constraints(solver, interactome, orientations)
    closures = prepare_closure_variables(solver, source_to_targets)
    prepare_flow(solver, interactome, source_to_targets, orientations, closures)

    solver.Solve()

    # TODO: output solver result

if __name__ == "__main__":
    main()
