import collections
import datetime
import time

import numpy as np

from util.loggingconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

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

# function to estimate the completion time for an iterator, e.g. estimate the dataloader batch iterator estimated time for one epoch
def iterate_with_estimate(iter, desc_str, start_idx=0):
    # total number of iterations
    iter_len = len(iter)

    # print progress 10 times
    print_iterval = iter_len // 10

    # begin iteration
    log.warning(f"{desc_str} ----/{iter_len}, starting")
    start_time = time.time()

    # for each iteration
    for (current_idx, item) in enumerate(iter):
        # return the iteration
        yield (current_idx, item)

        if current_idx != 0 and current_idx % print_iterval == 0:

            # calculate the estimated time to complete all iteration
            avg_time = (time.time() - start_time) / (current_idx - start_idx + 1)   # avg time elapsed for all previous iterations
            duration = avg_time * (iter_len - start_idx)                            # avg time * number of iterations
            remaining_duration = avg_time * (iter_len - current_idx)                # avg time * remaining number of iterations

            # convert to date time
            estimated_done = datetime.datetime.fromtimestamp(start_time + duration)
            estimated_remaining = datetime.timedelta(seconds=remaining_duration)

            log.info(f"{desc_str} {current_idx}/{iter_len}, estimated done at {str(estimated_done).rsplit('.', 1)[0]}, remaining time {str(estimated_remaining).rsplit('.', 1)[0]}")

            # if start_idx is not 0, i.e. skipping some elapsed time from calculating estimates
            if current_idx + 1 == start_idx:
                start_time = time.time()

    log.warning(f"{desc_str} total {iter_len} iters, done at {str(datetime.datetime.now()).rsplit('.', 1)[0]}")

