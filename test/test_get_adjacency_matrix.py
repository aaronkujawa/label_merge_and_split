import unittest

from config.config import PROJ_ROOT

from natsort import natsorted
from glob import glob
import os
from matplotlib import pyplot as plt

from src.merging_utils import (get_label_to_channel_mapping, get_label_support, get_average_volume_ratio_matrix,
                               get_distance_matrix, get_adjacency_matrix)


class TestGetAdjacencyMatrix(unittest.TestCase):
    def test_get_adjacency_matrix(self):
        label_paths = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        label_to_channel_map = get_label_to_channel_mapping(label_paths)
        label_support = get_label_support(label_paths, label_to_channel_map)

        # use only the first 10 labels
        label_support = label_support[:10, ...]

        # calculate the distance matrix
        distance_matrix = get_distance_matrix(label_support)
        # calculate the average volume ratio matrix
        average_volume_ratio_matrix = get_average_volume_ratio_matrix(label_support)

        result = get_adjacency_matrix(distance_matrix, 1.0, average_volume_ratio_matrix, 3.5)

        # plot the result
        plt.imshow(result)
        plt.colorbar()
        plt.title("Adjacency matrix")
        plt.show()
        self.assertEqual(result.shape, (len(label_support), len(label_support)))
        self.assertTrue((result >= 0).all() and (result <= 1).all())
        self.assertTrue((result == result.T).all())

