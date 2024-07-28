#!/usr/bin/env python
import argparse
import time

from concurrent.futures import as_completed, ProcessPoolExecutor

from source.bitarray2d import BitArray2D
from source.graph_func import process_sector_graph
from source.los_func   import linedef_visibility, sector_visibility_parallel
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
    (subgraph_by_sect, articulation_dat, articulation_sinds, normal_sinds) = process_sector_graph(sect_graph)
    all_sinds = articulation_sinds + normal_sinds
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

    # process articulation sectors in parallel
    tt = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {executor.submit(sector_visibility_parallel,
                                   articulation_sinds[i_si],
                                   all_sinds,
                                   subgraph_by_sect,
                                   portals_by_sect,
                                   all_2s_lines,
                                   solid_lines_quadtree,
                                   line_graph,
                                   articulation_dat,
                                   reject_table,
                                   known_blocked,
                                   linedef_rej,
                                   plot_prefix):i_si for i_si in range(len(articulation_sinds))}
        for future in as_completed(futures):
            (new_rej, new_known, new_linerej) = future.result()
            i_si = futures.pop(future)
            reject_table &= new_rej
            known_blocked |= new_known
            linedef_rej |= new_linerej
            print(f'articulation sector {i_si+1} / {len(articulation_sinds)} ({int(time.perf_counter() - tt)} sec)')

    # process the rest of the sectors serially
    for i_si in range(len(normal_sinds)):
        for i_sj in range(len(all_sinds)):
            si = normal_sinds[i_si]
            sj = all_sinds[i_sj]
            if subgraph_by_sect[si] == subgraph_by_sect[sj]: # different subgraphs can never see each other
                if reject_table[si,sj] == IS_VISIBLE:
                    # already known to be visible
                    continue
                if known_blocked[si,sj]:
                    # already known to be blocked
                    continue
                # pairwise los check of all 2s lines
                sectors_can_see = []
                for li in portals_by_sect[si]:
                    for lj in portals_by_sect[sj]:
                        if not linedef_rej[li,lj]:
                            plot_fn = ''
                            if plot_prefix:
                                plot_fn = f'{plot_prefix}.{si}.{sj}.{li}.{lj}.png'
                            (vis_bool, vis_type) = linedef_visibility(all_2s_lines[li], all_2s_lines[lj], solid_lines_quadtree, line_graph, plot_fn)
                            if vis_bool is False:
                                linedef_rej[li,lj] = True
                                linedef_rej[lj,li] = True
                            else:
                                sectors_can_see = [all_2s_lines[li].metadata, all_2s_lines[lj].metadata]
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
                # otherwise, check articulation point info to see what other sectors we can now rule out as well
                else:
                    known_blocked[si,sj] = True
                    known_blocked[sj,si] = True
                    for (s1,s2) in [(si,sj), (sj,si)]:
                        if s1 in articulation_dat:
                            for wing in articulation_dat[s1]:
                                if s2 not in wing:
                                    for sk in wing.keys():
                                        known_blocked[s2,sk] = True
                                        known_blocked[sk,s2] = True
        print(f'normal sector {i_si+1} / {len(normal_sinds)} ({int(time.perf_counter() - tt)} sec)')

    write_reject(reject_table.get_2d_array(), OUT_REJECT)


if __name__ == '__main__':
    main()
