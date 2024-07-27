import matplotlib.pyplot as mpl
import numpy as np

from source.graph_func import graph_bfs
from source.wad_func import IS_INVISIBLE, IS_VISIBLE

EPSILON = 0.1
MAX_PAIRWISE_DIST = 16000


def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def segments_intersect(A, B, C, D):
    # do line segments AB and DC intersect?
    return bool(ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D))


def line_sees_point_old(line, point):
    d1 = line[1] - line[0]
    n1 = [-d1[1], d1[0]]
    mid1 = (line[0] + line[1])/2.0
    return bool(np.dot(n1, point - mid1) > -EPSILON)


def line_sees_points_old(line, points):
    d1 = line[1] - line[0]
    n1 = [-d1[1], d1[0]]
    mid1 = (line[0] + line[1])/2.0
    return [bool(np.dot(n1, point - mid1) > -EPSILON) for point in points]


def prepare_line(line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    return y2 - y1, x1 - x2, x2*y1 - x1*y2


def prepline_sees_point(prepared_line, point):
    a, b, c = prepared_line
    x, y = point
    return bool(a * x + b * y + c < EPSILON)


def line_sees_point(line, point):
    x1, y1 = line[0]
    x2, y2 = line[1]
    px, py = point
    return bool((y2 - y1) * (px - x1) - (x2 - x1) * (py - y1) < EPSILON)


def line_sees_points(line, points):
    prepared_line = prepare_line(line)
    return [prepline_sees_point(prepared_line, point) for point in points]


def lines_sees_points_old(lines, points):
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


def lines_sees_points(lines, points):
    lines = np.array(lines)
    points = np.array(points)
    # Precompute line coefficients (A, B, C) for Ax + By + C = 0
    line_coeffs = np.zeros((len(lines), 3))
    line_coeffs[:,0] = lines[:,1,1] - lines[:,0,1]  # A = y2 - y1
    line_coeffs[:,1] = lines[:,0,0] - lines[:,1,0]  # B = x1 - x2
    line_coeffs[:,2] = lines[:,1,0] * lines[:,0,1] - lines[:,0,0] * lines[:,1,1]  # C = x2*y1 - x1*y2
    # Vectorized visibility check
    visibility = np.dot(points, line_coeffs[:,:2].T) + line_coeffs[:,2] < EPSILON
    # A point is visible if it's visible from all lines
    return np.all(visibility, axis=1)


def check_collinearity(l1, l2):
    (x1, y1) = l1[0]
    (x2, y2) = l2[0]
    (a, b, c) = (y1 - y2, x2 - x1, x1*y2 - y1*x2)
    for (x,y) in [l1[1], l2[1]]:
        if abs(a*x + b*y + c) > EPSILON:
            return False
    return True


def distance_from_point_to_line_segment(point, line):
    v = line[1] - line[0]
    num = abs(v[1]*point[0] - v[0]*point[1] + line[1][0]*line[0][1] - line[1][1]*line[0][0])
    den = np.sqrt(v[0]*v[0] + v[1]*v[1])
    return num/den


def plot_line(line, kwargs={}, normal_kwargs={}, normal_len_frac=0.1):
    mpl.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], label='_nolegend_', **kwargs)
    if normal_kwargs:
        v = line[1] - line[0]
        n = np.array([-v[1], v[0]])
        p = (line[0] + line[1])/2.0
        nline = [p, p + normal_len_frac * n]
        mpl.plot([nline[0][0], nline[1][0]], [nline[0][1], nline[1][1]], label='_nolegend_', **normal_kwargs)


def orient_src_and_tgt(l_src, l_tgt):
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
    return (l_src, l_tgt)


def get_src_tgt_edges(l_src, l_tgt):
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
    return (edge1, edge2, tgt_enclosed)


def get_span_of_polylines(line_collection_list, my_edge, other_edge, solid_lines_of_interest, all_solid_lines, both_edges=False):
    out_spans = []
    for line_collection in line_collection_list:
        points_analyzed = {}
        closest_point = None
        closest_point_2 = None
        for sli in line_collection:
            for i in range(2):
                if solid_lines_of_interest[sli][i]:
                    point = all_solid_lines[sli][i]
                    pkey = (int(point[0]), int(point[1]))
                    if pkey in points_analyzed:
                        continue
                    points_analyzed[pkey] = True
                    d_to_my_edge = distance_from_point_to_line_segment(point, my_edge)
                    d_to_other_edge = distance_from_point_to_line_segment(point, other_edge)
                    if d_to_my_edge > EPSILON and d_to_other_edge > EPSILON:
                        d_frac = d_to_other_edge/(d_to_my_edge + d_to_other_edge)
                        if closest_point is None or d_frac < closest_point[0]:
                            closest_point = (d_frac, point)
                        if both_edges:
                            d_frac_2 = d_to_my_edge/(d_to_my_edge + d_to_other_edge)
                            if closest_point_2 is None or d_frac_2 < closest_point_2[0]:
                                closest_point_2 = (d_frac_2, point)
        if closest_point is not None:
            if both_edges is False:
                out_spans.append([my_edge[0], closest_point[1]])
                out_spans.append([my_edge[1], closest_point[1]])
            elif closest_point_2 is not None:
                out_spans.append([closest_point[1], closest_point_2[1]])
    return out_spans


def linedef_visibility(linedat_i, linedat_j, all_solid_lines, line_graph, plot_fn=''):
    [line_i, sectors_i] = linedat_i
    [line_j, sectors_j] = linedat_j
    #
    # check for shared coordinates
    #
    for vert1 in line_i:
        for vert2 in line_j:
            if vert1[0] == vert2[0] and vert1[1] == vert2[1]:
                return (True, 'shared_vertex')
    #
    # check distance (this is an inaccuracy for improved performance on huge maps)
    #
    deltas = [line_j[0]-line_i[0], line_j[0]-line_i[1], line_j[1]-line_i[0], line_j[1]-line_i[1]]
    dists = [n[0]*n[0] + n[1]*n[1] for n in deltas]
    if min(dists) > MAX_PAIRWISE_DIST*MAX_PAIRWISE_DIST:
        return (False, 'too_far')
    #
    # check collinearity
    #
    if check_collinearity(line_i, line_j):
        return (False, 'collinear')
    #
    # orient line pair, get sightline edges
    #
    (l_src, l_tgt) = orient_src_and_tgt(line_i, line_j)
    (edge1, edge2, tgt_enclosed) = get_src_tgt_edges(l_src, l_tgt)
    enclosing_lines = [l_src, edge1, edge2]
    if tgt_enclosed is False:
        enclosing_lines.append(l_tgt)
    #
    # get all 1s lines that are within our sight area
    #
    solid_line_candidates = []
    solid_line_points = []
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
        solid_line_points.extend(solid_line)
    if len(solid_line_candidates) == 0:
        return (True, 'no_1s_lines')
    solid_lines_of_interest = {}
    enclosed_solid_points = lines_sees_points(enclosing_lines, solid_line_points)
    for i in range(0,len(enclosed_solid_points),2):
        if enclosed_solid_points[i] or enclosed_solid_points[i+1]:
            solid_lines_of_interest[solid_line_candidates[i//2]] = (enclosed_solid_points[i], enclosed_solid_points[i+1])
    #
    # check 1s lines for intersections with sightline edges
    #
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
            seed_lines[sli] = True
        if int2:
            e2_ints[sli] = True
            seed_lines[sli] = True
        if int1 and int2:
            trivial_blocked = True
            break
    if trivial_blocked:
        # a single 1s line blocks our entire sightline
        return (False, 'trivial_block')
    if len(e1_ints) == 0 or len(e2_ints) == 0:
        # at least one of our sightline edges unblocked
        return (True, 'trivial_vis')
    #
    # look for connected sets of 1s lines that collectively intersect both sightline edges
    #
    lines_visited = {}
    lines_1s_e1 = []
    lines_1s_e2 = []
    lines_1s_iso = []
    for sli in solid_lines_of_interest.keys():
        if sli not in lines_visited:
            l_visited = graph_bfs(line_graph, sli, node_whitelist=solid_lines_of_interest)
            (hit_e1, hit_e2) = (False, False)
            for myv in l_visited:
                lines_visited[myv] = True
                hit_e1 |= myv in e1_ints
                hit_e2 |= myv in e2_ints
            if hit_e1 and hit_e2:
                # this collection of lines intersects both edges
                return (False, 'bfs_block')
            if hit_e1:
                lines_1s_e1.append(l_visited)
            elif hit_e2:
                lines_1s_e2.append(l_visited)
            else:
                lines_1s_iso.append(l_visited)
    #
    # collapse polylines into spanning lines, if opposite spanlines intersect then we're occluded
    #
    spanning_lines = [get_span_of_polylines(lines_1s_e1, edge1, edge2, solid_lines_of_interest, all_solid_lines),
                      get_span_of_polylines(lines_1s_e2, edge2, edge1, solid_lines_of_interest, all_solid_lines)]
    for span_e1 in spanning_lines[0]:
        for span_e2 in spanning_lines[1]:
            if segments_intersect(span_e1[0], span_e1[1], span_e2[0], span_e2[1]):
                return (False, 'span_intersect')
    #
    # if an isolated spanlines intersects both edge spanlines then we're occluded
    #
    spanning_lines.append(get_span_of_polylines(lines_1s_iso, edge1, edge2, solid_lines_of_interest, all_solid_lines, both_edges=True))
    for span_iso in spanning_lines[2]:
        (have_s1_int, have_s2_int) = (False, False)
        for span_e1 in spanning_lines[0]:
            if segments_intersect(span_iso[0], span_iso[1], span_e1[0], span_e1[1]):
                have_s1_int = True
                break
        for span_e2 in spanning_lines[1]:
            if segments_intersect(span_iso[0], span_iso[1], span_e2[0], span_e2[1]):
                have_s2_int = True
                break
        if have_s1_int and have_s2_int:
            return (False, 'span_intersect_iso')
    #
    # plotting
    #
    if plot_fn:
        fig = mpl.figure(0, figsize=(9,9))
        for sli,solid_line in enumerate(all_solid_lines):
            if sli in solid_lines_of_interest:
                plot_line(solid_line, {'linewidth':1, 'color':'k', 'linestyle':'-'})
            else:
                plot_line(solid_line, {'linewidth':0.5, 'color':'k', 'linestyle':'--'})
        plot_line(l_src, {'linewidth':2, 'color':'g', 'linestyle':'-'}, {'linewidth':1, 'color':'g', 'linestyle':'--'})
        plot_line(l_tgt, {'linewidth':2, 'color':'b', 'linestyle':'-'}, {'linewidth':1, 'color':'b', 'linestyle':'--'})
        plot_line(edge1, {'linewidth':1, 'color':'r', 'linestyle':'--'})
        plot_line(edge2, {'linewidth':1, 'color':'r', 'linestyle':'--'})
        for spanline in spanning_lines[0]:
            plot_line(spanline, {'linewidth':1, 'color':'r', 'linestyle':'-'})
        for spanline in spanning_lines[1]:
            plot_line(spanline, {'linewidth':1, 'color':'r', 'linestyle':'-'})
        for spanline in spanning_lines[2]:
            plot_line(spanline, {'linewidth':1, 'color':'r', 'linestyle':'-'})
        mpl.axis('scaled')
        mpl.savefig(plot_fn)
        mpl.close(fig)
    #
    # I tried so hard and got so far...
    #
    return (True, 'possibly_vis')


def linedef_visibility_parallel(all_2s_lines, li, n_portals, all_solid_lines, line_graph, reject_table, plot_prefix):
    reject_out = np.zeros((reject_table.shape[0], reject_table.shape[1]), dtype='bool') + IS_INVISIBLE
    for lj in range(li+1, n_portals):
        plot_fn = ''
        if plot_prefix:
            plot_fn = f'{plot_prefix}.{li}.{lj}.png'
        # check if these sectors have already been analyzed and found visible
        [line_i, sectors_i] = all_2s_lines[li]
        [line_j, sectors_j] = all_2s_lines[lj]
        already_visible = True
        for si in sectors_i:
            for sj in sectors_j:
                if reject_table[si,sj] == IS_INVISIBLE:
                    already_visible = False
        if already_visible:
            continue
        # visibility check
        (vis_bool, vis_type) = linedef_visibility(all_2s_lines[li],
                                                  all_2s_lines[lj],
                                                  all_solid_lines,
                                                  line_graph,
                                                  plot_fn)
        if vis_bool:
            for si in all_2s_lines[li][1]:
                for sj in all_2s_lines[lj][1]:
                    reject_table[si,sj] = IS_VISIBLE
                    reject_table[sj,si] = IS_VISIBLE
                    reject_out[si,sj] = IS_VISIBLE
                    reject_out[sj,si] = IS_VISIBLE
    return reject_out


def sector_visibility_parallel(i_si, sorted_sector_inds, subgraph_by_sect, portals_by_sect, all_2s_lines, all_solid_lines, line_graph, ap_dat_all, reject_table, known_blocked, linedef_rej, plot_prefix):
    reject_out = np.zeros((reject_table.shape[0], reject_table.shape[1]), dtype='bool') + IS_INVISIBLE
    known_out = np.zeros((known_blocked.shape[0], known_blocked.shape[1]), dtype='bool')
    linerej_out = np.zeros((linedef_rej.shape[0], linedef_rej.shape[1]), dtype='bool')
    for i_sj in range(i_si+1, len(sorted_sector_inds)):
        si = sorted_sector_inds[i_si]
        sj = sorted_sector_inds[i_sj]
        if subgraph_by_sect[si] == subgraph_by_sect[sj]: # different subgraphs can never see each other
            if reject_table[si,sj] == IS_VISIBLE:
                # already known to be visible
                continue
            if known_blocked[si,sj]:
                # already known to be blocked
                continue
            # pairwise los check of all 2s lines
            sectors_can_see = []
            #print('--', i_si, i_sj, len(portals_by_sect[si]), len(portals_by_sect[sj]))
            for li in portals_by_sect[si]:
                for lj in portals_by_sect[sj]:
                    if not linedef_rej[li,lj]:
                        plot_fn = ''
                        if plot_prefix:
                            plot_fn = f'{plot_prefix}.{si}.{sj}.{li}.{lj}.png'
                        (vis_bool, vis_type) = linedef_visibility(all_2s_lines[li], all_2s_lines[lj], all_solid_lines, line_graph, plot_fn)
                        if vis_bool is False:
                            linedef_rej[li,lj] = True
                            linedef_rej[lj,li] = True
                            linerej_out[li,lj] = True
                            linerej_out[lj,li] = True
                        else:
                            sectors_can_see = [all_2s_lines[li][1], all_2s_lines[lj][1]]
                    if sectors_can_see:
                        break
                if sectors_can_see:
                    break
            # if we are visible, mark reject table
            if sectors_can_see:
                for vsi in sectors_can_see[0]:
                    for vsj in sectors_can_see[1]:
                        reject_table[vsi,vsj] = IS_VISIBLE
                        reject_table[vsj,vsi] = IS_VISIBLE
                        reject_out[vsi,vsj] = IS_VISIBLE
                        reject_out[vsj,vsi] = IS_VISIBLE
            # otherwise, check articulation point info to see what other sectors we can now rule out as well
            else:
                for (s1,s2) in [(si,sj), (sj,si)]:
                    if s1 in ap_dat_all:
                        for wing in ap_dat_all[s1]:
                            if s2 not in wing:
                                for sk in wing.keys():
                                    known_blocked[s2,sk] = True
                                    known_blocked[sk,s2] = True
                                    known_out[s2,sk] = True
                                    known_out[sk,s2] = True
    return (reject_out, known_out, linerej_out)
