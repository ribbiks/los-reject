from collections import deque


def graph_bfs(graph, starting_node, node_whitelist=None, node_blacklist=None):
    queue = deque([starting_node])
    visited = {}
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited[node] = True
        for neighbor in graph[node]:
            if (node_whitelist is None or neighbor in node_whitelist) and (node_blacklist is None or neighbor not in node_blacklist) and neighbor not in visited:
                queue.append(neighbor)
    return sorted(visited.keys())


def find_articulation_points(graph):
    timer = 0
    disc = {n:-1 for n in graph.keys()}
    low = {n:-1 for n in graph.keys()}
    parent = {n:-1 for n in graph.keys()}
    ap = {n:False for n in graph.keys()}
    stack = []
    #
    for i in graph.keys():
        if disc[i] == -1:
            stack.append((i, 0))  # (node, child_count)
            while stack:
                u, child_count = stack[-1]
                if disc[u] == -1:
                    disc[u] = timer
                    low[u] = timer
                    timer += 1
                    children = 0
                #
                all_children_visited = True
                for v in graph[u]:
                    if disc[v] == -1:
                        if child_count == len(graph[u]):
                            stack.pop()
                            break
                        parent[v] = u
                        stack[-1] = (u, child_count + 1)
                        stack.append((v, 0))
                        all_children_visited = False
                        break
                    elif v != parent[u]:
                        low[u] = min(low[u], disc[v])
                #
                if all_children_visited:
                    if parent[u] != -1:
                        low[parent[u]] = min(low[parent[u]], low[u])
                        if low[u] >= disc[parent[u]] and parent[parent[u]] != -1:
                            ap[parent[u]] = True
                    stack.pop()
                    children = sum(1 for v in graph[u] if parent[v] == u)
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
    return [i for i in graph.keys() if ap[i]]
