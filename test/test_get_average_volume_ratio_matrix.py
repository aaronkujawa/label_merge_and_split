import unittest

from config.config import PROJ_ROOT

from natsort import natsorted
from glob import glob
import os
from matplotlib import pyplot as plt

from src.merging_utils import get_label_to_channel_mapping, get_label_support, get_average_volume_ratio_matrix


class TestGetAverageVolumeRatioMatrix(unittest.TestCase):
    def test_get_average_volume_ratio_matrix(self):
        label_paths = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        label_to_channel_map = get_label_to_channel_mapping(label_paths)
        label_support = get_label_support(label_paths, label_to_channel_map)

        # calculate the distance matrix
        result = get_average_volume_ratio_matrix(label_support)

        # plot the result
        plt.imshow(result)
        plt.colorbar()
        plt.clim(0, 40)
        plt.title("Average volume ratio matrix")
        plt.show()