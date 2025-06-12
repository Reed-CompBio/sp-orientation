"""
A barebones mixed graph library for the purpose of handling edges.
"""

from enum import Enum
import heapq
import itertools
import math
from typing import Any, Callable, Generic, Optional, Iterator, TypeVar

V = TypeVar("V")


class Direction(Enum):
    UNDIRECTED = 1
    DIRECTED = 2

    @staticmethod
    def from_str(direction: str) -> "Direction":
        if direction.lower() in ["d", "directed"]:
            return Direction.DIRECTED
        elif direction.lower() in ["u", "undirected"]:
            return Direction.UNDIRECTED
        else:
            raise ValueError(f"Direction '{direction}' is not a valid direction!")


class MixedGraph(Generic[V]):
    vertices: dict[V, dict[str, Any]]

    undirected: dict[V, dict[V, dict[str, Any]]]
    undirected_reverse: dict[V, dict[V, dict[str, Any]]]

    directed: dict[V, dict[V, dict[str, Any]]]
    directed_reverse: dict[V, dict[V, dict[str, Any]]]

    def __init__(self):
        self.undirected = dict()
        self.undirected_reverse = dict()
        self.directed = dict()
        self.directed_reverse = dict()

    def get_maps(
        self, direction: Direction
    ) -> tuple[dict[V, dict[V, dict[str, Any]]], dict[V, dict[V, dict[str, Any]]]]:
        """
        Gets the direction and direction_reversed maps associated with a particular direction.
        This is generally useful for any repeated addition/removal on mixed graphs.

        For a usage example inside MixedGraph, see `MixedGraph::add_edge`.
        """
        if direction == Direction.DIRECTED:
            return (self.directed, self.directed_reverse)
        elif direction == Direction.UNDIRECTED:
            return (self.undirected, self.undirected_reverse)
        else:
            raise ValueError(f"Direction {direction} is not a valid direction!")

    def add_edge(self, source: V, target: V, direction: Direction, **kwargs):
        self.vertices[source] = dict()
        self.vertices[target] = dict()

        adj, adj_reverse = self.get_maps(direction)
        adj[source] = adj[source] if source in adj else dict()
        adj[source][target] = kwargs

        adj_reverse[source] = adj_reverse[source] if source in adj_reverse else {}
        adj_reverse[source][target] = kwargs

    def add_vertex(self, vertex: V, data: Any):
        self.vertices[vertex] = data

    def edges(
        self, direction: Direction | None = None
    ) -> Iterator[tuple[V, V, dict[str, Any], Direction]]:
        undirected_iter = (
            (
                itertools.chain(
                    (source, target, target_data, Direction.UNDIRECTED)
                    for target, target_data in target_dict.items()
                )
                for source, target_dict in self.undirected.items()
            )
            if direction in [Direction.UNDIRECTED, None]
            else []
        )

        directed_iter = (
            (
                itertools.chain(
                    (source, target, target_data, Direction.DIRECTED)
                    for target, target_data in target_dict.items()
                )
                for source, target_dict in self.undirected.items()
            )
            if direction in [Direction.DIRECTED, None]
            else []
        )

        return itertools.chain.from_iterable(
            itertools.chain(undirected_iter, directed_iter)
        )

    def neighbors(
        self, vertex: V, direction: Optional[Direction] = None
    ) -> Iterator[V]:
        edges_directed = (
            self.directed[vertex] if direction in [Direction.DIRECTED, None] else []
        )
        edges_undirected = (
            self.undirected[vertex] if direction in [Direction.UNDIRECTED, None] else []
        )
        return itertools.chain(edges_directed, edges_undirected)

    def _get_edge(
        self, source: V, target: V, dict: dict[V, dict[V, dict[str, Any]]]
    ) -> Optional[dict[str, Any]]:
        edge = dict[source]
        if not edge:
            return None
        edge = edge[target]
        return edge

    def get_edge_data(
        self, source: V, target: V, direction: Optional[Direction] = None
    ) -> Optional[dict[str, Any]]:
        edge_directed = (
            self._get_edge(source, target, self.directed)
            if direction in [Direction.DIRECTED, None]
            else None
        )
        edge_undirected = (
            self._get_edge(source, target, self.undirected)
            if direction in [Direction.DIRECTED, None]
            else None
        )
        return edge_directed if edge_directed else edge_undirected

    def single_source_dijkstra(
        self,
        source: V,
        weight: str,
        target_filter: Optional[Callable[[set[V]], bool]] = None,
    ) -> tuple[dict[V, float], dict[V, Optional[V]]]:
        """
        Run dijkstra starting from a single source.

        @param target_filter: If this lambda returns true from a set of explored nodes,
        the algorithm will stop here.
        """
        # Note: heapq elements can be tuples, so we assign
        # the priorities on the lhs of the pair-tuple.
        distance: dict[V, float] = dict()
        previous: dict[V, Optional[V]] = dict()
        queue: list[tuple[float, int, V]] = []
        # networkx trick to avoid dual-comparing nodes
        c = itertools.count()

        distance[source] = 0
        heapq.heappush(queue, (0, next(c), source))

        for vertex in self.vertices:
            if vertex != source:
                previous[vertex] = None
                distance[vertex] = math.inf

        explored_nodes: set[V] = set()

        while not len(queue) == 0:
            d, _, source = heapq.heappop(queue)
            if source in distance:
                continue  # node was already searched
            distance[source] = d

            if target_filter:
                explored_nodes.add(source)
                if target_filter(explored_nodes):
                    return (distance, previous)

            for target in self.neighbors(source):
                st_data = self.get_edge_data(source, target)
                if not st_data or weight not in st_data:
                    raise RuntimeError(
                        f"Data from {source}-{target} does not have 'weight.' Instead, it is '{st_data}'."
                    )
                alt = distance[source] + st_data[weight]
                if alt < distance[source]:
                    previous[target] = source
                    distance[target] = alt
                    heapq.heappush(queue, (alt, next(c), target))

        return (distance, previous)

    def dijkstra(self, source: V, target: V, weight: str):
        return self.single_source_dijkstra(
            source, weight, lambda targets: target in targets
        )
