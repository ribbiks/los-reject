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


def process_sector_graph(sect_graph):
    sectors_visited = {}
    sect_graphs = []
    for si in sect_graph.keys():
        if si not in sectors_visited:
            s_visited = graph_bfs(sect_graph, si)
            for myv in s_visited:
                sectors_visited[myv] = True
            sect_graphs.append((len(s_visited), s_visited))
    sect_graphs = [n[1] for n in sorted(sect_graphs)]
    #
    subgraph_by_sect = {}
    articulation_dat = {}
    sorted_sector_inds = []
    for sgi,sg in enumerate(sect_graphs):
        for si in sg:
            subgraph_by_sect[si] = sgi
        subgraph = {n:sect_graph[n] for n in sg}
        articulation_points = find_articulation_points(subgraph)
        ap_dat = []
        for ap in articulation_points:
            sectors_visited = {} # reusing this variable
            my_wings = []
            for si in subgraph[ap]:
                if si not in sectors_visited:
                    s_visited = graph_bfs(sect_graph, si, node_blacklist={ap:True})
                    for myv in s_visited:
                        sectors_visited[myv] = True
                    my_wings.append(s_visited)
            ap_dat.append((len(subgraph[ap]), ap, my_wings))
        ap_dat = sorted(ap_dat, reverse=True)
        added_to_sinds = {}
        for apd in ap_dat:
            articulation_dat[apd[1]] = [{n:True for n in slist} for slist in apd[2]]
            sorted_sector_inds.append(apd[1])
            added_to_sinds[apd[1]] = True
        for si in sg:
            if si not in added_to_sinds:
                sorted_sector_inds.append(si)
    #
    return (subgraph_by_sect, articulation_dat, sorted_sector_inds)
