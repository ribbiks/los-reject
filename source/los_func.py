import matplotlib.pyplot as mpl
import numpy as np
import time

from collections import deque

from source.wad_func import IS_INVISIBLE

EPSILON = 0.1


def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def segments_intersect(A, B, C, D):
    # do line segments AB and DC intersect?
    return bool(ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D))


def line_sees_point(line, point):
    d1 = line[1] - line[0]
    n1 = [-d1[1], d1[0]]
    mid1 = (line[0] + line[1])/2.0
    return bool(np.dot(n1, point - mid1) > -EPSILON)


def line_sees_points(line, points):
    d1 = line[1] - line[0]
    n1 = [-d1[1], d1[0]]
    mid1 = (line[0] + line[1])/2.0
    return [bool(np.dot(n1, point - mid1) > -EPSILON) for point in points]


# True = point is seen by all lines in list
def lines_sees_points(lines, points):
    n_list = []
    mid_list = []
    for line in lines:
        d1 = line[1] - line[0]
        n_list.append([-d1[1], d1[0]])
        mid_list.append((line[0] + line[1])/2.0)
    vis_out = [True for n in points]
    for pi,point in enumerate(points):
        for li in range(len(lines)):
            if np.dot(n_list[li], point - mid_list[li]) < -EPSILON:
                vis_out[pi] = False
                break
    return vis_out


def check_collinearity(l1, l2):
    (x1, y1) = l1[0]
    (x2, y2) = l2[0]
    (a, b, c) = (y1 - y2, x2 - x1, x1*y2 - y1*x2)
    for (x,y) in [l1[1], l2[1]]:
        if abs(a*x + b*y + c) > EPSILON:
            return False
    return True


def plot_line(line, kwargs={}, normal_kwargs={}, normal_len_frac=0.1):
    mpl.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], label='_nolegend_', **kwargs)
    if normal_kwargs:
        v = line[1] - line[0]
        n = np.array([-v[1], v[0]])
        p = (line[0] + line[1])/2.0
        nline = [p, p + normal_len_frac * n]
        mpl.plot([nline[0][0], nline[1][0]], [nline[0][1], nline[1][1]], label='_nolegend_', **normal_kwargs)


def orient_src_and_tgt(l_src, l_tgt, plotting=False):
    # if src is not facing tgt, flip src
    tvis = line_sees_points(l_src, l_tgt)
    if tvis[0] is False and tvis[1] is False:
        l_src = [l_src[1], l_src[0]]
    elif tvis[0] is False or tvis[1] is False:
        # if src only sees one point of tgt, check if tgt can see both src points (and if so, swap them)
        svis = line_sees_points(l_tgt, l_src)
        if svis[0] and svis[1]:
            (l_src, l_tgt) = ([l_tgt[0], l_tgt[1]], [l_src[0], l_src[1]])
        elif svis[0] is False and svis[1] is False:
            (l_src, l_tgt) = ([l_tgt[1], l_tgt[0]], [l_src[0], l_src[1]])
        else:
            # I think this can only happen if linedefs are intersecting, which shouldn't happen
            return None
    # if tgt is not facing src, flip tgt
    if line_sees_point(l_tgt, (l_src[0] + l_src[1])/2.0) is False:
        l_tgt = [l_tgt[1], l_tgt[0]]
    if plotting:
        plot_line(l_src, {'linewidth':2, 'color':'g', 'linestyle':'-'}, {'linewidth':1, 'color':'g', 'linestyle':'--'})
        plot_line(l_tgt, {'linewidth':2, 'color':'b', 'linestyle':'-'}, {'linewidth':1, 'color':'b', 'linestyle':'--'})
    return (l_src, l_tgt)


def get_src_tgt_edges(l_src, l_tgt, plotting=False):
    int1 = segments_intersect(l_src[0], l_tgt[0], l_src[1], l_tgt[1])
    int2 = segments_intersect(l_src[0], l_tgt[1], l_src[1], l_tgt[0])
    tgt_enclosed = False
    if int1 is False and int2:
        edge1 = [l_src[0], l_tgt[0]]
        edge2 = [l_src[1], l_tgt[1]]
    elif int1 and int2 is False:
        edge1 = [l_src[0], l_tgt[1]]
        edge2 = [l_src[1], l_tgt[0]]
    else:
        # both boundary sets are valid, we only need the tgt point furthest away from src
        v0 = l_tgt[0] - l_src[0]
        v1 = l_tgt[1] - l_src[0]
        len0 = v0[0]*v0[0] + v0[1]*v0[1]
        len1 = v1[0]*v1[0] + v1[1]*v1[1]
        if len0 > len1:
            edge1 = [l_src[0], l_tgt[0]]
            edge2 = [l_src[1], l_tgt[0]]
        else:
            edge1 = [l_src[0], l_tgt[1]]
            edge2 = [l_src[1], l_tgt[1]]
        tgt_enclosed = True
    # orient edges so that they face inward
    if line_sees_point(edge1, l_src[1]) is False:
        edge1 = [edge1[1], edge1[0]]
    if line_sees_point(edge2, l_src[0]) is False:
        edge2 = [edge2[1], edge2[0]]
    if plotting:
        plot_line(edge1, {'linewidth':1, 'color':'r', 'linestyle':'--'})
        plot_line(edge2, {'linewidth':1, 'color':'r', 'linestyle':'--'})
    return (edge1, edge2, tgt_enclosed)


def line_graph_bfs(graph, starting_node, node_whitelist):
    queue = deque([(starting_node, [])])
    visited = {}
    while queue:
        (node, path) = queue.popleft()
        visited[node] = True
        for neighbor in graph[node]:
            if neighbor in node_whitelist and neighbor not in path:
                queue.append((neighbor, path+[node]))
    return sorted(visited.keys())


def linedef_visibility(linedat_i, linedat_j, all_solid_lines, line_graph, reject_table, my_inds, make_plot, plot_prefix=''):
    [line_i, sectors_i] = linedat_i
    [line_j, sectors_j] = linedat_j
    #
    # check if these sectors have already been analyzed and found visible
    #
    already_visible = True
    for si in sectors_i:
        for sj in sectors_j:
            if reject_table[si,sj] == IS_INVISIBLE:
                already_visible = False
    if already_visible:
        return (False, 'already_vis', my_inds)
    #
    # check for shared coordinates
    #
    for vert1 in line_i:
        for vert2 in line_j:
            if vert1[0] == vert2[0] and vert1[1] == vert2[1]:
                return (True, 'shared_vertex', my_inds)
    #
    # check collinearity
    #
    if check_collinearity(line_i, line_j):
        return (False, 'collinear', my_inds)
    #
    # orient line pair, get sightline edges
    #
    if make_plot:
        fig = mpl.figure(0, figsize=(9,9))
    (l_src, l_tgt) = orient_src_and_tgt(line_i, line_j, make_plot)
    (edge1, edge2, tgt_enclosed) = get_src_tgt_edges(l_src, l_tgt, make_plot)
    enclosing_lines = [l_src, edge1, edge2]
    if tgt_enclosed is False:
        enclosing_lines.append(l_tgt)
    #
    # get all 1s lines in the sight area and check for intersections with sightline edges
    #
    solid_line_candidates = []
    x_min = min(l_src[0][0], l_src[1][0], l_tgt[0][0], l_tgt[1][0])
    x_max = max(l_src[0][0], l_src[1][0], l_tgt[0][0], l_tgt[1][0])
    y_min = min(l_src[0][1], l_src[1][1], l_tgt[0][1], l_tgt[1][1])
    y_max = max(l_src[0][1], l_src[1][1], l_tgt[0][1], l_tgt[1][1])
    for sli,solid_line in enumerate(all_solid_lines):
        if solid_line[0][0] < x_min and solid_line[1][0] < x_min:
            continue
        if solid_line[0][0] > x_max and solid_line[1][0] > x_max:
            continue
        if solid_line[0][1] < y_min and solid_line[1][1] < y_min:
            continue
        if solid_line[0][1] > y_max and solid_line[1][1] > y_max:
            continue
        solid_line_candidates.append(sli)
    #
    solid_lines_of_interest = {}
    e1_ints = {}
    e2_ints = {}
    seed_lines = {}
    trivial_blocked = False
    for sli in solid_line_candidates:
        solid_line = all_solid_lines[sli]
        int1 = segments_intersect(solid_line[0], solid_line[1], edge1[0], edge1[1])
        int2 = segments_intersect(solid_line[0], solid_line[1], edge2[0], edge2[1])
        if int1:
            e1_ints[sli] = True
            solid_lines_of_interest[sli] = True
            seed_lines[sli] = True
        if int2:
            e2_ints[sli] = True
            solid_lines_of_interest[sli] = True
            seed_lines[sli] = True
        if int1 and int2:
            trivial_blocked = True
            break
    if trivial_blocked:
        # a single 1s line blocks our entire sightline
        if make_plot:
            mpl.close(fig)
        return (False, 'trivial_block', my_inds)
    if len(e1_ints) == 0 or len(e2_ints) == 0:
        # at least one of our sightline edges unblocked
        if make_plot:
            mpl.close(fig)
        return (True, 'trivial_vis', my_inds)
    #
    # look for connected sets of 1s lines that collectively intersect both sightline edges
    #
    for sli in solid_line_candidates:
        solid_line = all_solid_lines[sli]
        if any(lines_sees_points(enclosing_lines, solid_line)):
            solid_lines_of_interest[sli] = True
    bfs_blocked = False
    lines_visited = {}
    for lk in solid_lines_of_interest.keys():
        if lk not in lines_visited:
            l_visited = line_graph_bfs(line_graph, lk, solid_lines_of_interest)
            (hit_e1, hit_e2) = (False, False)
            for myv in l_visited:
                lines_visited[myv] = True
                hit_e1 |= myv in e1_ints
                hit_e2 |= myv in e2_ints
            if hit_e1 and hit_e2:
                bfs_blocked = True
                break
    if bfs_blocked:
        if make_plot:
            mpl.close(fig)
        return (False, 'bfs_block', my_inds)
    #
    # ok, we actually have to do the serious visibility checks now
    # -- alternately, if we're running in fast-mode just give up and say they're visible
    #
    return (True, 'possibly_vis', my_inds)
    #
    # plotting
    #
    if make_plot:
        plot_fn = f'{plot_prefix}.{my_inds[0]}.{my_inds[1]}.png'
        for sli,solid_line in enumerate(all_solid_lines):
            if sli in solid_lines_of_interest:
                plot_line(solid_line, {'linewidth':1, 'color':'k', 'linestyle':'-'})
            else:
                plot_line(solid_line, {'linewidth':0.5, 'color':'k', 'linestyle':'--'})
        mpl.axis('scaled')
        mpl.savefig(plot_fn)
        mpl.close(fig)
