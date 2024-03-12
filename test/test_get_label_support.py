import unittest

from config.config import PROJ_ROOT
from src.merging_utils import get_label_support, get_label_to_channel_mapping
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
        result = get_label_support(label_paths, label_to_channel_map)

        # check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor), "Result is not a tensor"

        # check that the result has the correct shape
        self.assertEqual(result.shape, (len(label_to_channel_map), *nib.load(label_paths[0]).shape), "Result has incorrect shape")

        # check that the result is not empty
        self.assertTrue(result.sum() > 0), "Result is empty"

        # check that the result is not all zeros
        self.assertTrue(result.sum() > 0), "Result is all zeros"

        # plot the result for first channel
        plot_matrix_slices(result[0, ...])
