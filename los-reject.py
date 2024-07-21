#!/usr/bin/env python
import argparse
import numpy as np
import time

from source.los_func import linedef_visibility
from source.wad_func import *


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='pvs-reject', formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('-i', type=str, required=True,  metavar='input.wad',  help="* Input WAD")
    parser.add_argument('-m', type=str, required=True,  metavar='MAP01',      help="* Map name")
    parser.add_argument('-r', type=str, required=True,  metavar='REJECT.lmp', help="* Output reject table lmp")
    args = parser.parse_args()
    #
    IN_WAD = args.i
    WHICH_MAP = args.m
    OUT_REJECT = args.r
    PLOTTING = False
    plot_prefix = OUT_REJECT

    map_data = get_map_lmps(IN_WAD, WHICH_MAP)
    if not map_data:
        print(f'Error: {WHICH_MAP} not found.')
        exit(1)
    line_list = get_linedefs(map_data)
    side_list = get_sidedefs(map_data)
    sect_list = get_sectors(map_data)
    vert_list = get_vertexes(map_data)

    (all_solid_lines, all_2s_lines, line_graph) = get_lines_by_type(line_list, side_list, vert_list)
    n_linedefs = len(line_list)
    n_sectors = len(sect_list)
    n_portals = len(all_2s_lines)
    print(f'{n_sectors} sectors / {n_linedefs} lines / {n_portals} portals')
    del line_list
    del side_list
    del sect_list
    del vert_list

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
    for li in range(n_portals):
        for lj in range(li+1, n_portals):
            (vis_bool, vis_type, my_inds) = linedef_visibility(all_2s_lines[li],
                                                               all_2s_lines[lj],
                                                               all_solid_lines,
                                                               line_graph,
                                                               reject_table,
                                                               (li, lj),
                                                               PLOTTING,
                                                               plot_prefix)
            if vis_bool:
                for si in all_2s_lines[li][1]:
                    for sj in all_2s_lines[lj][1]:
                        reject_table[si,sj] = IS_VISIBLE
                        reject_table[sj,si] = IS_VISIBLE
            if vis_type not in vis_type_count:
                vis_type_count[vis_type] = 0
            vis_type_count[vis_type] += 1
        print(f'{li+1} / {n_portals} ({int(time.perf_counter() - tt)} sec)')

    for k in sorted(vis_type_count.keys()):
        print(vis_type_count[k], k)

    write_reject(reject_table, OUT_REJECT)


if __name__ == '__main__':
    main()
