import os
from glob import glob
from natsort import natsorted

import argparse

from labelmergeandsplit.merging_utils import (get_label_to_channel_mapping,
                                              get_label_support,
                                              get_distance_matrix_from_input_label_files)

# read arguments from command line
parser = argparse.ArgumentParser(description='Run label_merge_and_split distance matrix calculation on the bratsunstr dataset labels')
# read --taskid argument from command line
parser.add_argument('--taskid', type=int, help='taskid', required=True)
args = parser.parse_args()

taskid = args.taskid


label_paths = natsorted(glob(os.path.join("../../data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
label_to_channel_map = get_label_to_channel_mapping(label_paths)
output_fpaths = [os.path.join("../../data/task2153_mind/output/distance_matrices/", os.path.basename(p).
                              replace(".nii.gz", "_distance_matrix.npy")) for p in label_paths]
                              
                              
label_paths = [os.path.abspath(p) for p in label_paths]
output_fpaths = [os.path.abspath(p) for p in output_fpaths]

for p in output_fpaths:
    os.makedirs(os.path.dirname(p), exist_ok=True)

result = get_distance_matrix_from_input_label_files(label_paths,
                                                    label_to_channel_map,
                                                    output_fpaths,
                                                    overwrite=True,
                                                    debug=True)

