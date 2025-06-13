from mixed import Direction, MixedGraph


def test_simple_path():
    graph = MixedGraph[str]()

    graph.add_edge("A", "B", Direction.DIRECTED, weight=1)
    graph.add_edge("B", "C", Direction.UNDIRECTED, weight=1)
    graph.add_edge("C", "D", Direction.UNDIRECTED, weight=1)

    path = graph.dijkstra("A", "D", "weight")
    assert path == ["A", "B", "C", "D"]
