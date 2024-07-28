import numpy as np
from collections import defaultdict
from struct import unpack

from source.geometry import LineSegment

IS_VISIBLE = 0
IS_INVISIBLE = 1


def null_pad(string, length=8):
    return string + chr(0) * (length - len(string))


MAP_LUMPS = ['THINGS', 'LINEDEFS', 'SIDEDEFS', 'VERTEXES', 'SEGS', 'SSECTORS', 'NODES', 'SECTORS', 'REJECT', 'BLOCKMAP']
MAP_LUMPS = {null_pad(n):True for n in MAP_LUMPS}


# reject lump --> 2d numpy array
def read_reject(fn):
    f = open(fn,'rb')
    fr = f.read()
    f.close()
    n_sectors = int(np.ceil(np.sqrt(8.0*len(fr)-7.0)))
    reject_size = len(fr)*8
    reject = np.zeros((reject_size))
    for i in range(len(fr)):
        reject[i*8 + 0] = (fr[i] & 1) >> 0
        reject[i*8 + 1] = (fr[i] & 2) >> 1
        reject[i*8 + 2] = (fr[i] & 4) >> 2
        reject[i*8 + 3] = (fr[i] & 8) >> 3
        reject[i*8 + 4] = (fr[i] & 16) >> 4
        reject[i*8 + 5] = (fr[i] & 32) >> 5
        reject[i*8 + 6] = (fr[i] & 64) >> 6
        reject[i*8 + 7] = (fr[i] & 128) >> 7
    reject = reject[:n_sectors*n_sectors] # ignore trailing junk
    return reject.reshape((n_sectors,n_sectors))


# 2d numpy array --> reject lump
def write_reject(reject, fn):
    bytes_out = []
    total_size = reject.shape[0]*reject.shape[1]
    reject = reject.reshape((total_size))
    reject = np.hstack((reject, np.zeros((8), dtype='bool')))
    for i in range(0,total_size,8):
        my_byte = 1*reject[i+0] + 2*reject[i+1] + 4*reject[i+2] + 8*reject[i+3] + 16*reject[i+4] + 32*reject[i+5] + 64*reject[i+6] + 128*reject[i+7]
        bytes_out.append(my_byte)
    bytes_out = bytes(bytes_out)
    with open(fn,'wb') as f:
        f.write(bytes_out)


def get_map_lmps(in_wad, which_map):
    which_map = null_pad(which_map)
    wad_data = []
    lmp_list = []
    with open(in_wad,'rb') as f:
        f.read(4) # PWAD
        (n_lmp, p_dir) = unpack('ii', f.read(8))
        f.seek(p_dir)
        for i in range(n_lmp):
            (lmp_pos, lmp_size) = unpack('ii', f.read(8))
            lmp_name = f.read(8)
            lmp_list.append((lmp_name, lmp_pos, lmp_size))
        for n in lmp_list:
            f.seek(n[1])
            wad_data.append((n[0], f.read(n[2])))
    #
    map_data = {}
    in_current_map = False
    for n in wad_data:
        lmp_name = n[0].decode("utf-8")
        if lmp_name == which_map:
            in_current_map = True
            MAP_LUMPS['GL_' + which_map[:5]] = True
        elif in_current_map and lmp_name in MAP_LUMPS:
            map_data[lmp_name] = n[1]
        else:
            in_current_map = False
    #
    return map_data


def get_linedefs(map_data):
    line_list = []
    line_data = map_data[null_pad('LINEDEFS')]
    for offset in range(0,len(line_data),14):
        (start_vertex, end_vertex, line_flags, line_special, line_tag, sidedef_front, sidedef_back) = unpack('HHhhhHH', line_data[offset:offset+14])
        line_list.append((start_vertex, end_vertex, line_flags, line_special, line_tag, sidedef_front, sidedef_back))
    return line_list


def get_sidedefs(map_data):
    side_list = []
    side_data = map_data[null_pad('SIDEDEFS')]
    for offset in range(0,len(side_data),30):
        (x_off, y_off) = unpack('hh', side_data[offset:offset+4])
        upper_tex = side_data[offset+4:offset+12]
        lower_tex = side_data[offset+12:offset+20]
        middle_tex = side_data[offset+20:offset+28]
        facing_sector = unpack('H', side_data[offset+28:offset+30])[0]
        side_list.append((x_off, y_off, upper_tex, lower_tex, middle_tex, facing_sector))
    return side_list


def get_sectors(map_data):
    sect_list = []
    sect_data = map_data[null_pad('SECTORS')]
    for offset in range(0,len(sect_data),26):
        (floor_height, ceil_height) = unpack('hh', sect_data[offset:offset+4])
        floor_flat = sect_data[offset+4:offset+12]
        ceil_flat  = sect_data[offset+12:offset+20]
        (sec_light, sec_special, sec_tag) = unpack('hhh', sect_data[offset+20:offset+26])
        sect_list.append((floor_height, ceil_height, floor_flat, ceil_flat, sec_light, sec_special, sec_tag))
    return sect_list


def get_vertexes(map_data):
    normal_verts = []
    vert_data = map_data[null_pad('VERTEXES')]
    for offset in range(0,len(vert_data),4):
        (x, y) = unpack('hh', vert_data[offset:offset+4])
        normal_verts.append([x, y])
    return normal_verts


def process_map_data(line_list, side_list, vert_list):
    all_solid_lines = []
    all_2s_lines = []
    line_graph = {}
    line_ind_by_vert = defaultdict(list)
    sect_graph = {}
    portals_by_sect = defaultdict(list)
    for li,line in enumerate(line_list):
        (start_vertex, end_vertex, line_flags, line_special, line_tag, sidedef_front, sidedef_back) = line
        vert1 = tuple(vert_list[start_vertex])
        vert2 = tuple(vert_list[end_vertex])
        my_sectors = []
        if sidedef_front != 0xffff:
            my_sectors.append(side_list[sidedef_front][5])
        if sidedef_back != 0xffff:
            my_sectors.append(side_list[sidedef_back][5])
        if len(my_sectors) == 1:
            all_solid_lines.append(LineSegment(vert1, vert2))
            solid_ind = len(all_solid_lines) - 1
            line_ind_by_vert[vert1].append(solid_ind)
            line_ind_by_vert[vert2].append(solid_ind)
            line_graph[solid_ind] = []
        elif len(my_sectors) == 2 and my_sectors[0] != my_sectors[1]:
            all_2s_lines.append(LineSegment(vert1, vert2, metadata=my_sectors))
            portals_by_sect[my_sectors[0]].append(len(all_2s_lines) - 1)
            portals_by_sect[my_sectors[1]].append(len(all_2s_lines) - 1)
            for si in [(0,1), (1,0)]:
                if my_sectors[si[0]] not in sect_graph:
                    sect_graph[my_sectors[si[0]]] = [my_sectors[si[1]]]
                elif my_sectors[si[1]] not in sect_graph[my_sectors[si[0]]]:
                    sect_graph[my_sectors[si[0]]].append(my_sectors[si[1]])
    for vert in line_ind_by_vert.keys():
        li_list = line_ind_by_vert[vert]
        for i in li_list:
            for j in li_list:
                if i != j:
                    line_graph[i].append(j)
    return (all_solid_lines, all_2s_lines, line_graph, sect_graph, portals_by_sect)
