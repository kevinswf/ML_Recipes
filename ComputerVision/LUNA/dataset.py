from collections import namedtuple
import functools
import csv
import SimpleITK as sitk
import glob

import numpy as np

from util.util import xyz_tuple, xyz_to_irc


# a sample of data contains whether it's nodule, its diamter (mm), its CT uid, and its center (xyz) in the CT
nodule_candidate_info_tuple = namedtuple("NoduleCandidateInfoTuple", "is_nodule, diameter, series_uid, center")

# function to populate the data with samples of (is_nodule, diameter, uid, center)
@functools.lru_cache(maxsize=1)                         # cache since data file parsing could be slow, and we use this function often
def get_nodule_candidate_info_list():

    # get data from annotations.csv in the form of {uid: (center, diameter)}
    diamter_dict = {}
    with open('F/MLData/LUNA/annotations.csv', 'r') as f:
        # read each row, skip header
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center = tuple([float(value) for value in row[1:4]])
            annotation_diameter = float(row[4])
            
            diamter_dict.setdefault(series_uid, []).append((annotation_center, annotation_diameter))


    # get data from candidates.csv and construct data of (is_nodule, diameter, uid, center)   
    nodule_candidate_info_list = []
    with open('F/MLData/LUNA/candidates.csv', 'r') as f:
        # read each row, skip header
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            is_nodule = bool(int(row[4]))
            candidate_center = tuple([float(value) for value in row[1:4]])

            # the center in the annotation file and the candidate files are different, try to match them if close enough, so we can set each sample with its diameter
            candidate_diameter = 0.0    # if don't see a corresponding sample in the annotations, set the sample's diameter to 0
            for annotation in diamter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation

                # check if the center between the two files are close enough
                for i in range(3):
                    delta_mm = abs(candidate_center[i] - annotation_center_xyz[i])
                    if(delta_mm > annotation_diameter_mm / 4):
                        break                                       # not close, set this sample's diameter to 0 (only using the ordered diameter to split into train and val set, so could be ok)
                else:
                    candidate_diameter = annotation_diameter_mm     # close, set the sample's diameter from the annotation file
                    break

            nodule_candidate_info_list.append(nodule_candidate_info_tuple(
                is_nodule,
                candidate_diameter,
                series_uid,
                candidate_center
            ))

    # sort from is_nodule and diameter first
    nodule_candidate_info_list.sort(reverse=True)
    return nodule_candidate_info_list


class CtScan:

    def __init__(self, series_uid):
        # reads a CT scan metadata and raw image file, and convert to np array
        mhd_path = glob.glob(f'F:/MLData/LUNA/subset0/{series_uid}.mhd')[0]
        ct_mhd = sitk.ReadImage(mhd_path)                                       # sitk reads both metadata and .raw file
        ct_np = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)      # 3D array of CT scan

        # clip the CT scan HU values between -1000 (think empty air) to 1000 (think bone), as to remove outliers
        ct_np.clip(-1000, 1000, ct_np)

        self.series_uid = series_uid
        self.ct_np = ct_np

        self.origin_xyz = xyz_tuple(*ct_mhd.GetOrigin())                # get annotation coord origin from metadata
        self.vx_size_xyz = xyz_tuple(*ct_mhd.GetSpacing())              # get CT scan voxel size (in mm) from metadata
        self.direction = np.array(ct_mhd.GetDirection()).reshape(3, 3)  # get the transformation matrix

    # function to extract some CT voxels around the specified candidate center
    def get_raw_candidate(self, center_xyz, width_irc):
        # convert the center from XYZ to IRC
        center_irc = xyz_to_irc(center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction)

        slice_list = []
        # iterate over each axis
        for axis, center in enumerate(center_irc):
            # check that the center is within bound
            assert center >=0 and center < self.ct_np.shape[axis], repr('candidate center out of bound error')

            # set the start and end index of voxels to include based on the specified width
            start_idx = int(round(center - width_irc[axis]/2))
            end_idx = int(start_idx + width_irc[axis])

            # cap start_idx = 0, if < 0
            if start_idx < 0:
                start_idx = 0
                end_idx = int(width_irc[axis])
            # cap end_idx to bound
            if end_idx > self.ct_np.shape[axis]:
                end_idx = self.ct_np.shape[axis]
                start_idx = int(end_idx - width_irc[axis])

            # include these voxels
            slice_list.append(slice(start_idx, end_idx))

        ct_chunk = self.ct_np[tuple(slice_list)]
        # return the subset of voxels around the center, and center
        return ct_chunk, center_irc

