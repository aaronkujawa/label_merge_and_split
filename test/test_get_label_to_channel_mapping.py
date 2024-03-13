import unittest
import os
from glob import glob
from natsort import natsorted

from src.merging_utils import get_label_to_channel_mapping
from config.config import PROJ_ROOT


class TestGetLabelToChannelMapping(unittest.TestCase):
    def test_get_label_to_channel_mapping(self):
        label_paths = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
        result = get_label_to_channel_mapping(label_paths)

        # check that the result is a dictionary
        self.assertIsInstance(result, dict), "Result is not a dictionary"

        # check that the result is not empty
        self.assertTrue(len(result) > 0), "Result is empty"

        # check that dictionary values are consecutive integers
        self.assertEqual(list(result.values()), list(range(len(result)))), "Values are not consecutive integers"

        # check that the dictionary keys are sorted
        self.assertEqual(list(result.keys()), sorted(list(result.keys()))), "Keys are not sorted"

        # check that the dictionary keys are unique
        self.assertEqual(len(result.keys()), len(set(result.keys()))), "Keys are not unique"

