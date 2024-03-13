import unittest
import os
from glob import glob
from natsort import natsorted

from labelmergeandsplit.merging_utils import merge_label_volumes
from config.config import PROJ_ROOT


class TestMergeLabelVolumes(unittest.TestCase):
    def test_merge_label_volumes(self):
        label_paths_in = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        label_paths_out = [os.path.join(PROJ_ROOT, "data/task2153_mind/output/predictions/merged",
                                        os.path.basename(label_path).replace(".nii.gz", "_predmerged.nii.gz"))
                           for label_path in label_paths_in]

        print(label_paths_in)
        print(label_paths_out)

        merged_labels_csv_path = os.path.join(PROJ_ROOT, "data/task2153_mind/output/merged_labels.csv")

        merge_label_volumes(label_paths_in, label_paths_out, merged_labels_csv_path)

        # check that output files exist
        self.assertTrue(all([os.path.exists(label_path) for label_path in label_paths_out]))
