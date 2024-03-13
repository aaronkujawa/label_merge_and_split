import unittest

import pandas as pd

from config.config import PROJ_ROOT

from natsort import natsorted
from glob import glob
import os

from src.merging_utils import get_merged_label_dataframe


class TestGetAdjacencyMatrix(unittest.TestCase):
    def test_get_adjacency_matrix(self):
        label_paths = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        label_to_name_csv_path = os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labels.csv")

        save = True
        if save:
            output_dir = os.path.join(PROJ_ROOT, "data/task2153_mind/output")

        # get the label dataframe
        label_df = get_merged_label_dataframe(label_paths, label_to_name_csv_path, output_dir=output_dir, debug=True)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(label_df)
