import unittest

from config.config import PROJ_ROOT

from natsort import natsorted
from glob import glob
import os
from matplotlib import pyplot as plt

from labelmergeandsplit.merging_utils import get_label_to_channel_mapping, get_label_support, get_distance_matrix


class TestGetDistanceMatrix(unittest.TestCase):
    def test_get_distance_matrix(self):
        label_paths = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        label_to_channel_map = get_label_to_channel_mapping(label_paths)
        label_support = get_label_support(label_paths, label_to_channel_map)

        # for this test, we will use the first 10 labels
        label_support = label_support[:10, ...]

        result = get_distance_matrix(label_support)

        # plot the result
        plt.imshow(result)
        plt.colorbar()
        plt.title("Distance matrix for the first 10 labels")
        plt.show()

