import unittest

import torch

from config.config import PROJ_ROOT
import os
from labelmergeandsplit.merging_utils import get_training_prior, load_label_support

from utils.timing import tic, toc
from utils.plot_matrix_slices import plot_matrix_slices


class TestGetTrainingPrior(unittest.TestCase):
    def test_get_training_prior(self):
        label_support_path = os.path.join(PROJ_ROOT, "data/task2153_mind/output/label_support.pt.gz")
        merged_labels_csv_path = os.path.join(PROJ_ROOT, "data/task2153_mind/output/merged_labels.csv")

        assert (os.path.isfile(label_support_path))
        assert (os.path.isfile(merged_labels_csv_path))


        label_support = load_label_support(label_support_path)
        tic()
        training_prior = get_training_prior(label_support, merged_labels_csv_path, merged_prior=True)
        toc("get_training_prior")

        plot_matrix_slices(training_prior)

