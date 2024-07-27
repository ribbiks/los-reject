#!/usr/bin/env python
import argparse
import numpy as np
import time

from concurrent.futures import as_completed, ProcessPoolExecutor

from source.graph_func import find_articulation_points, graph_bfs
from source.los_func   import linedef_visibility, linedef_visibility_parallel
from source.wad_func   import *


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='los-reject', formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('-i', type=str, required=True,  metavar='input.wad',  help="* Input WAD")
    parser.add_argument('-m', type=str, required=True,  metavar='MAP01',      help="* Map name")
    parser.add_argument('-r', type=str, required=True,  metavar='REJECT.lmp', help="* Output reject table lmp")
    parser.add_argument('-p', type=int, required=False, metavar='4',          help="Number of processes to use", default=4)
    parser.add_argument('--plot',       required=False, action='store_true',  help="Plotting! (for debugging)", default=False)
    args = parser.parse_args()
    #
    IN_WAD = args.i
    WHICH_MAP = args.m
    OUT_REJECT = args.r
    NUM_PROCESSES = args.p
    PLOTTING = args.plot

    plot_prefix = ''
    if PLOTTING:
        plot_prefix = OUT_REJECT

    map_data = get_map_lmps(IN_WAD, WHICH_MAP)
    if not map_data:
        print(f'Error: {WHICH_MAP} not found.')
        exit(1)
    line_list = get_linedefs(map_data)
    side_list = get_sidedefs(map_data)
    sect_list = get_sectors(map_data)
    vert_list = get_vertexes(map_data)

    (all_solid_lines, all_2s_lines, line_graph, sect_graph, portals_by_sect) = process_map_data(line_list, side_list, vert_list)
    n_linedefs = len(line_list)
    n_sectors = len(sect_list)
    n_portals = len(all_2s_lines)
    print(f'{n_sectors} sectors / {n_linedefs} lines / {n_portals} portals')
    del line_list
    del side_list
    del sect_list
    del vert_list

    # identify disconnected subgraphs and articulation points
    tt = time.perf_counter()
    sectors_visited = {}
    sect_graphs = []
    for si in range(n_sectors):
        if si not in sect_graph: # sector doesn't have any portals
            continue
        if si not in sectors_visited:
            s_visited = graph_bfs(sect_graph, si)
            for myv in s_visited:
                sectors_visited[myv] = True
            sect_graphs.append((len(s_visited), s_visited))
    sect_graphs = [n[1] for n in sorted(sect_graphs)]
    #
    subgraph_by_sect = {}
    ap_dat_all = {}
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
            ap_dat_all[apd[1]] = [{n:True for n in slist} for slist in apd[2]]
            sorted_sector_inds.append(apd[1])
            added_to_sinds[apd[1]] = True
        for si in sg:
            if si not in added_to_sinds:
                sorted_sector_inds.append(si)
    print(f'finished graph construction ({int(time.perf_counter() - tt)} sec)')

    reject_table = np.zeros((n_sectors, n_sectors), dtype='bool') + IS_INVISIBLE
    known_to_be_blocked = np.zeros((n_sectors, n_sectors), dtype='bool')
    linedef_vis = np.zeros((n_portals, n_portals), dtype='B')

    # mark all sectors visible to themselves
    for i in range(n_sectors):
        reject_table[i,i] = IS_VISIBLE

    # mark all sectors visible to their immediate neighbors
    for li in range(n_portals):
        [line_i, sectors_i] = all_2s_lines[li]
        reject_table[sectors_i[0],sectors_i[1]] = IS_VISIBLE
        reject_table[sectors_i[1],sectors_i[0]] = IS_VISIBLE

    tt = time.perf_counter()
    for i_si in range(len(sorted_sector_inds)):
        for i_sj in range(i_si+1, len(sorted_sector_inds)):
            si = sorted_sector_inds[i_si]
            sj = sorted_sector_inds[i_sj]
            if subgraph_by_sect[si] == subgraph_by_sect[sj]: # different subgraphs can never see each other
                if reject_table[si,sj] == IS_VISIBLE:
                    # already known to be visible
                    continue
                if known_to_be_blocked[si,sj]:
                    # already known to be blocked
                    continue
                # pairwise los check of all 2s lines
                sectors_can_see = False
                print('--', i_si, i_sj, len(portals_by_sect[si]), len(portals_by_sect[sj]))
                for li in portals_by_sect[si]:
                    for lj in portals_by_sect[sj]:
                        if linedef_vis[li,lj] == 0:
                            plot_fn = ''
                            if plot_prefix:
                                plot_fn = f'{plot_prefix}.{si}.{sj}.{li}.{lj}.png'
                            (vis_bool, vis_type) = linedef_visibility(all_2s_lines[li], all_2s_lines[lj], all_solid_lines, line_graph, reject_table, plot_fn)
                            if vis_bool is False:
                                linedef_vis[li,lj] |= 2
                                linedef_vis[lj,li] |= 2
                            else:
                                sectors_can_see = True
                                linedef_vis[li,lj] |= 4
                                linedef_vis[lj,li] |= 4
                        elif linedef_vis[li,lj] & 4:
                            sectors_can_see = True
                        if sectors_can_see:
                            break
                    if sectors_can_see:
                        break
                # if we are visible, mark reject table
                if sectors_can_see:
                    reject_table[si,sj] = IS_VISIBLE
                    reject_table[sj,si] = IS_VISIBLE
                # otherwise, check articulation point info to see what other sectors we can now rule out as well
                else:
                    if si in ap_dat_all:
                        for wing in ap_dat_all[si]:
                            if sj not in wing:
                                for sk in wing.keys():
                                    known_to_be_blocked[sj,sk] = True
                                    known_to_be_blocked[sk,sj] = True
                    if sj in ap_dat_all:
                        for wing in ap_dat_all[sj]:
                            if si not in wing:
                                for sk in wing.keys():
                                    known_to_be_blocked[si,sk] = True
                                    known_to_be_blocked[sk,si] = True
        print(f'{i_si+1} / {n_sectors} ({int(time.perf_counter() - tt)} sec)')
    write_reject(reject_table, OUT_REJECT)
    exit(1)

    # pairwise compare all 2s lines
    tt = time.perf_counter()
    vis_type_count = {}
    if NUM_PROCESSES <= 1:
        for li in range(n_portals):
            for lj in range(li+1, n_portals):
                plot_fn = ''
                if plot_prefix:
                    plot_fn = f'{plot_prefix}.{li}.{lj}.png'
                (vis_bool, vis_type) = linedef_visibility(all_2s_lines[li], all_2s_lines[lj], all_solid_lines, line_graph, reject_table, plot_fn)
                if vis_bool:
                    for si in all_2s_lines[li][1]:
                        for sj in all_2s_lines[lj][1]:
                            reject_table[si,sj] = IS_VISIBLE
                            reject_table[sj,si] = IS_VISIBLE
                if vis_type not in vis_type_count:
                    vis_type_count[vis_type] = 0
                vis_type_count[vis_type] += 1
            print(f'{li+1} / {n_portals} ({int(time.perf_counter() - tt)} sec)')
    #
    else:
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            futures = {executor.submit(linedef_visibility_parallel, all_2s_lines, li, n_portals, all_solid_lines, line_graph, reject_table, plot_prefix):li for li in range(n_portals)}
            for future in as_completed(futures):
                new_rej = future.result()
                li = futures.pop(future)
                reject_table &= new_rej
                print(f'{li} ({int(time.perf_counter() - tt)} sec)')

    for k in sorted(vis_type_count.keys()):
        print(vis_type_count[k], k)

    write_reject(reject_table, OUT_REJECT)


if __name__ == '__main__':
    main()
