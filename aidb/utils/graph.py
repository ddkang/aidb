from collections import defaultdict
class Graph:
    def __init__(self, graph):
        self.graph = graph

    # function used for tests, can be removed
    def addEdge(self, v, w):
        self.graph[v].append(w)

    def _isCyclicUtil(self, node, visited, rec_stack):
        visited[node] = True
        rec_stack[node] = True

        if node in self.graph:
            for neighbor in self.graph[node]:
                if visited[neighbor] == False:
                    if self._isCyclicUtil(neighbor, visited, rec_stack):
                        return True
                elif rec_stack[neighbor] == True:
                    return True

        rec_stack[node] = False
        return False

    def isCyclic(self):
        visited = defaultdict(bool)
        rec_stack = defaultdict(bool)
        for node in self.graph:
            if visited[node] == False:
                if self._isCyclicUtil(node, visited, rec_stack):
                    return True
        return False



