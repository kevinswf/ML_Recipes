from collections import namedtuple
import functools
import csv


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