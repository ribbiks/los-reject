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

    (all_solid_lines, all_2s_lines, line_graph, sect_graph) = process_map_data(line_list, side_list, vert_list)
    n_linedefs = len(line_list)
    n_sectors = len(sect_list)
    n_portals = len(all_2s_lines)
    print(f'{n_sectors} sectors / {n_linedefs} lines / {n_portals} portals')
    del line_list
    del side_list
    del sect_list
    del vert_list

    ##### identify disconnected subgraphs and articulation points
    ####sectors_visited = {}
    ####sect_graphs = []
    ####for si in range(n_sectors):
    ####    if si not in sectors_visited:
    ####        s_visited = graph_bfs(sect_graph, si)
    ####        for myv in s_visited:
    ####            sectors_visited[myv] = True
    ####        sect_graphs.append((len(s_visited), s_visited))
    ####sect_graphs = [n[1] for n in sorted(sect_graphs)]
    ####for sg in sect_graphs:
    ####    subgraph = {n:sect_graph[n] for n in sg}
    ####    articulation_points = find_articulation_points(subgraph)
    ####    print(len(subgraph), len(articulation_points))
    ####    for ap in articulation_points:
    ####        print(ap, subgraph[ap])
    ####exit(1)

    reject_table = np.zeros((n_sectors, n_sectors), dtype='bool') + IS_INVISIBLE

    # mark all sectors visible to themselves
    for i in range(n_sectors):
        reject_table[i,i] = IS_VISIBLE

    # mark all sectors visible to their immediate neighbors
    for li in range(n_portals):
        [line_i, sectors_i] = all_2s_lines[li]
        reject_table[sectors_i[0],sectors_i[1]] = IS_VISIBLE
        reject_table[sectors_i[1],sectors_i[0]] = IS_VISIBLE

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
