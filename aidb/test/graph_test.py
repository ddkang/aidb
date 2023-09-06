from aidb.utils.graph import Graph
import collections

# Test cases
def test_cycle_detection():
    # Single Node, No Edges
    graph = collections.defaultdict(list)
    g1 = Graph(graph)
    g1.addEdge(0, 0)
    assert g1.isCyclic() == True

    # Two Nodes, No Cycle
    graph = collections.defaultdict(list)
    g2 = Graph(graph)
    g2.addEdge(0, 1)
    assert g2.isCyclic() == False

    # Two Nodes, With Cycle
    graph = collections.defaultdict(list)
    g3 = Graph(graph)
    g3.addEdge(0, 1)
    g3.addEdge(1, 0)
    assert g3.isCyclic() == True

    # Three Nodes, No Cycle
    graph = collections.defaultdict(list)
    g4 = Graph(graph)
    g4.addEdge(0, 1)
    g4.addEdge(1, 2)
    assert g4.isCyclic() == False

    # Three Nodes, Cycle at the End
    graph = collections.defaultdict(list)
    g5 = Graph(graph)
    g5.addEdge(0, 1)
    g5.addEdge(1, 2)
    g5.addEdge(2, 1)
    assert g5.isCyclic() == True

    # Three Nodes, Cycle at the Start
    graph = collections.defaultdict(list)
    g6 = Graph(graph)
    g6.addEdge(0, 1)
    g6.addEdge(1, 0)
    g6.addEdge(0, 2)
    assert g6.isCyclic() == True

    # Multiple Nodes, No Cycle
    graph = collections.defaultdict(list)
    g7 = Graph(graph)
    for i in range(5):
        g7.addEdge(i, i+1)
    assert g7.isCyclic() == False

    # Multiple Nodes, Cycle in the Middle
    graph = collections.defaultdict(list)
    g8 = Graph(graph)
    for i in range(4):
        g8.addEdge(i, i+1)
    g8.addEdge(4, 2)
    assert g8.isCyclic() == True

    # Multiple Nodes, Multiple Cycles
    graph = collections.defaultdict(list)
    g9 = Graph(graph)
    g9.addEdge(0, 1)
    g9.addEdge(1, 2)
    g9.addEdge(2, 0)
    g9.addEdge(2, 3)
    g9.addEdge(3, 4)
    g9.addEdge(4, 5)
    g9.addEdge(5, 3)
    assert g9.isCyclic() == True

    # Self-loop
    graph = collections.defaultdict(list)
    g10 = Graph(graph)
    g10.addEdge(0, 0)
    assert g10.isCyclic() == True

    # Disconnected Components, No Cycle
    graph = collections.defaultdict(list)
    g11 = Graph(graph)
    g11.addEdge(0, 1)
    g11.addEdge(2, 3)
    assert g11.isCyclic() == False

    # Disconnected Components, One with Cycle
    graph = collections.defaultdict(list)
    g12 = Graph(graph)
    g12.addEdge(0, 1)
    g12.addEdge(2, 3)
    g12.addEdge(3, 4)
    g12.addEdge(4, 2)
    assert g12.isCyclic() == True

    print("All test cases passed!")

if __name__ == '__main__':
    test_cycle_detection()