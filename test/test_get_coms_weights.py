import unittest
import os
import numpy as np
import torch
import nibabel as nib
import pandas as pd

from labelmergeandsplit import map_labels_in_volume
from labelmergeandsplit.labelreg_utils import get_coms

from glob import glob
from natsort import natsorted

from labelmergeandsplit.labelreg_utils import optimize_affine_labelreg

from config.config import PROJ_ROOT


class TestGetComsWeights(unittest.TestCase):
    def test_get_coms_weights(self):

        tr_label_paths = natsorted(glob(os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'input', 'dataset', 'labelsTr','*.nii.gz')))

        print(tr_label_paths)

        merged_labels_csv_path = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'output', 'merged_labels.csv')

        channel_to_label_mapping = pd.read_csv(merged_labels_csv_path, index_col='channel').to_dict()['merged_label']
        merged_labels_df = pd.read_csv(merged_labels_csv_path, index_col='label')
        label_to_merged_label_mapping = merged_labels_df['merged_label'].to_dict()

        merged_labels = list(set(channel_to_label_mapping.values()))

        # get all coms for all label images and all labels
        coms = []
        for label_path in tr_label_paths:
            print(f"Processing {label_path}")
            label_img = torch.tensor(nib.load(label_path).get_fdata())

            merged_label_img = map_labels_in_volume(label_img, label_to_merged_label_mapping)


            coms.append(get_coms(merged_label_img, merged_labels))

        coms = torch.stack(coms)
        print(coms)


