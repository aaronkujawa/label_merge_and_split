import unittest

import pandas as pd
import torch

from config.config import PROJ_ROOT
import os
from labelmergeandsplit import get_training_prior, load_label_support

from utils.timing import tic, toc
from utils.plot_matrix_slices import plot_matrix_slices


class TestGetTrainingPrior(unittest.TestCase):
    def test_get_training_prior(self):
        label_support_path = os.path.join(PROJ_ROOT, "data/task2153_mind/output/label_support.pt.npz")
        merged_labels_csv_path = os.path.join(PROJ_ROOT, "data/task2153_mind/output/merged_labels.csv")

        assert (os.path.isfile(label_support_path))
        assert (os.path.isfile(merged_labels_csv_path))

        merged_prior = True

        # map the channel to the original or merged label
        merged_labels_dataframe = pd.read_csv(merged_labels_csv_path, index_col='channel')
        if merged_prior:
            channel_to_label_mapping = merged_labels_dataframe['merged_label'].to_dict()
        else:
            channel_to_label_mapping = merged_labels_dataframe['label'].to_dict()

        label_support = load_label_support(label_support_path)
        tic()
        training_prior = get_training_prior(label_support, channel_to_label_mapping)
        toc("get_training_prior")

        plot_matrix_slices(training_prior)

