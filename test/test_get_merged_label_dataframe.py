import unittest

import pandas as pd

from config.config import PROJ_ROOT

from natsort import natsorted
from glob import glob
import os
from matplotlib import pyplot as plt

from src.merging_utils import (get_label_to_channel_mapping, get_label_support, get_average_volume_ratio_matrix,
                               get_distance_matrix, get_adjacency_matrix, get_orig_to_merged_label_map,
                               get_merged_label_dataframe)


class TestGetAdjacencyMatrix(unittest.TestCase):
    def test_get_adjacency_matrix(self):
        label_paths = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        label_to_name_csv_path = os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labels.csv")

        # get the label dataframe
        label_df = get_merged_label_dataframe(label_paths, label_to_name_csv_path, debug=True)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(label_df)

        #self.assertTrue(len(label_df) == nb_test_labels)
