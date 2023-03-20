import collections

import numpy as np

# CT scan raw image voxel data are in IRC (index, row, column) coordinate system
# where I is vertical, R is anterior/posterior, and C is left/right
irc_tuple = collections.namedtuple('irc_tuple', ['index', 'row', 'col'])
# annotation file are in XYZ coordinate system (mm), where X is left/right, Y is anterior/posterior, Z is vertical
xyz_tuple = collections.namedtuple('xyz_tuple', ['x', 'y', 'z'])

# function to convert from IRC to XYZ
def irc_to_xyz(coord_irc, origin_xyz, vx_size_xyz, direction):
    # convert from IRC to CRI, so it maps with XYZ, and convert to np
    cri = np.array(coord_irc)[::-1]

    # origin of the xyz
    origin = np.array(origin_xyz)
    # voxel size in xyz
    vx_size = np.array(vx_size_xyz)

    # convert CRI to XYZ, by first scaling CRI with voxel size, multiply by transformation matrix, then offset from origin
    coords_xyz = (direction @ (cri * vx_size)) + origin

    return xyz_tuple(*coords_xyz)

# function to convert from XYZ to IRC
def xyz_to_irc(coord_xyz, origin_xyz, vx_size_xyz, direction):
    # origin of the xyz
    origin = np.array(origin_xyz)
    # voxel size in xyz
    vx_size = np.array(vx_size_xyz)
    # coord in xyz
    coord = np.array(coord_xyz)

    # inverse of CRI to XYZ, which is XYZ to CRI
    cri = ((coord - origin) @ np.linalg.inv(direction)) / vx_size

    # rounding to int
    cri = np.round(cri)

    # return as IRC
    return irc_tuple(int(cri[2]), int(cri[1]), int(cri[0]))