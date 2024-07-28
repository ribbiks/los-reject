#!/usr/bin/env python
import argparse
import time

from concurrent.futures import as_completed, ProcessPoolExecutor

from source.bitarray2d import BitArray2D
from source.graph_func import process_sector_graph
from source.los_func   import sector_visibility_parallel
from source.quadtree   import get_quadtree
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
    solid_lines_quadtree = get_quadtree(all_solid_lines)
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
    (subgraph_by_sect, articulation_dat, sorted_sector_inds) = process_sector_graph(sect_graph)
    print(f'finished graph construction ({int(time.perf_counter() - tt)} sec)')

    reject_table = BitArray2D(n_sectors, init_val=IS_INVISIBLE)
    known_blocked = BitArray2D(n_sectors)
    linedef_rej = BitArray2D(n_portals)

    # mark all sectors visible to themselves
    for i in range(n_sectors):
        reject_table[i,i] = IS_VISIBLE

    # mark all sectors visible to their immediate neighbors
    for li in range(n_portals):
        sectors_i = all_2s_lines[li].metadata
        reject_table[sectors_i[0],sectors_i[1]] = IS_VISIBLE
        reject_table[sectors_i[1],sectors_i[0]] = IS_VISIBLE

    tt = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {executor.submit(sector_visibility_parallel,
                                   i_si,
                                   sorted_sector_inds,
                                   subgraph_by_sect,
                                   portals_by_sect,
                                   all_2s_lines,
                                   solid_lines_quadtree,
                                   line_graph,
                                   articulation_dat,
                                   reject_table,
                                   known_blocked,
                                   linedef_rej,
                                   plot_prefix):i_si for i_si in range(len(sorted_sector_inds))}
        for future in as_completed(futures):
            (new_rej, new_known, new_linerej) = future.result()
            i_si = futures.pop(future)
            reject_table &= new_rej
            known_blocked |= new_known
            linedef_rej |= new_linerej
            print(f'{i_si+1} / {len(sorted_sector_inds)} ({int(time.perf_counter() - tt)} sec)')

    write_reject(reject_table.get_2d_array(), OUT_REJECT)


if __name__ == '__main__':
    main()
