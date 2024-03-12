import unittest

from config.config import PROJ_ROOT

from natsort import natsorted
from glob import glob
import os

from src.merging_utils import (get_label_to_channel_mapping, get_label_support, get_average_volume_ratio_matrix,
                               get_distance_matrix, get_adjacency_matrix, get_orig_to_merged_label_map)


class TestGetAdjacencyMatrix(unittest.TestCase):
    def test_get_adjacency_matrix(self):
        label_paths = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        label_to_channel_map = get_label_to_channel_mapping(label_paths)
        label_support = get_label_support(label_paths, label_to_channel_map)

        # use only the first labels
        nb_test_labels = 20
        label_support = label_support[:nb_test_labels, ...]
        label_to_channel_map = {label: channel for label, channel in label_to_channel_map.items() if
                                channel < nb_test_labels}

        # calculate the distance matrix
        distance_matrix = get_distance_matrix(label_support)
        # calculate the average volume ratio matrix
        average_volume_ratio_matrix = get_average_volume_ratio_matrix(label_support)
        # calculate the adjacency matrix
        adjacency_matrix = get_adjacency_matrix(distance_matrix, 1.0, average_volume_ratio_matrix, 3.5)

        # run the graph coloring algorithm to get the original to merged label map
        orig_to_merged_label_map = get_orig_to_merged_label_map(adjacency_matrix, label_to_channel_map, dont_merge_labels=[0,])

        # check the merged labels
        merged_labels = list(orig_to_merged_label_map.values())

        # number of merged labels
        num_merged_labels = len(set(merged_labels))
        print(f"Number of label groups: {num_merged_labels}")

        # check that the merged labels are consecutive
        self.assertEqual(sorted(list(set(merged_labels))), list(range(num_merged_labels)))

        # only the background label should map to 0
        for k, v in orig_to_merged_label_map.items():
            if k == 0:
                self.assertEqual(v, 0)
            else:
                self.assertNotEqual(v, 0)





