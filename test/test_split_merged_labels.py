import unittest
from config.config import PROJ_ROOT
import os
from src.splitting_utils import get_fuzzy_prior_fudged, split_merged_labels
from natsort import natsorted
from glob import glob

from utils.plot_matrix_slices import plot_matrix_slices


class TestSplitMergedLabels(unittest.TestCase):
    def test_get_fuzzy_prior_fudged(self):
        label_support_path = os.path.join(PROJ_ROOT, "data/task2153_mind/output/label_support.pt")
        label_paths_in = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/output/predictions/merged/*.nii.gz")))
        label_paths_out = [os.path.join(PROJ_ROOT, "data/task2153_mind/output/predictions/split",
                                        os.path.basename(p).replace("merged.nii.gz", ".nii.gz")) for p in label_paths_in]
        merged_labels_csv_path = os.path.join(PROJ_ROOT, "data/task2153_mind/output/merged_labels.csv")

        fuzzy_prior_fudged = get_fuzzy_prior_fudged(label_support_path)

        split_merged_labels(label_paths_in, label_paths_out, fuzzy_prior_fudged, merged_labels_csv_path)

        # check if the output files exist
        for p in label_paths_out:
            self.assertTrue(os.path.exists(p)), f"Expected output file {p} does not exist"

