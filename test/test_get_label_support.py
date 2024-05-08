import unittest

from config.config import PROJ_ROOT
from labelmergeandsplit.merging_utils import get_label_support, get_label_to_channel_mapping
import os
import nibabel as nib
import torch
from glob import glob
from natsort import natsorted
from utils.plot_matrix_slices import plot_matrix_slices


class TestGetLabelSupport(unittest.TestCase):
    def test_get_label_support(self):
        label_paths = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        label_to_channel_map = get_label_to_channel_mapping(label_paths)

        output_dir = os.path.join(PROJ_ROOT, "data/task2153_mind/output")
        result = get_label_support(label_paths,
                                   label_to_channel_map,
                                   save_path=os.path.join(output_dir, "label_support.pt"),
                                   compress=True)

        # check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor), "Result is not a tensor"

        # check that the result has the correct shape
        self.assertEqual(result.shape, (len(label_to_channel_map), *nib.load(label_paths[0]).shape), "Result has incorrect shape")

        # plot the result for first channel
        plot_matrix_slices(result[0, ...])
