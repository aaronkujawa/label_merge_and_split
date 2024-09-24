import unittest
from config.config import PROJ_ROOT
import os
from labelmergeandsplit.splitting_utils import get_fuzzy_prior_fudged, get_influence_regions
from labelmergeandsplit.labelreg_utils import affine_transform_influence_regions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TestAffineTransformInfluenceRegions(unittest.TestCase):
    def test_affine_transform_influence_regions(self):
        label_support_path = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'output', 'label_support.pt.npz')
        influence_regions_path = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'output', 'influence_regions.npy')
        merged_labels_csv_path = os.path.join(PROJ_ROOT, 'data', 'task2153_mind', 'output', 'merged_labels.csv')

        merged_labels_df = pd.read_csv(merged_labels_csv_path, index_col='label')
        label_to_merged_label_mapping = merged_labels_df['merged_label'].to_dict()

        if not os.path.exists(influence_regions_path):
            fuzzy_prior_fudged = get_fuzzy_prior_fudged(label_support_path)

            label_to_channel_mapping = merged_labels_df['channel'].to_dict()

            influence_regions = get_influence_regions(fuzzy_prior_fudged, label_to_merged_label_mapping,
                                                      label_to_channel_mapping)

            del fuzzy_prior_fudged
            # save the influence regions
            np.save(influence_regions_path, influence_regions)
        else:
            influence_regions = np.load(influence_regions_path, allow_pickle=True).item()


        R_est = np.array([
            [1.4, 0, 0, 2],
            [0, 0.8, 0, 1],
            [0, 0, 1, -5],
            [0, 0, 0, 1],
        ])

        # padding can prevent the cropping of foreground after transformation
        pad = 50
        transformed_influence_regions = affine_transform_influence_regions(influence_regions, R_est, pad_val=pad)

        # invert the transformation matrix to get back the original influence regions
        R_est_inv = np.linalg.inv(R_est)
        transformed_influence_regions_inv = affine_transform_influence_regions(transformed_influence_regions, R_est_inv)

        # remove the padding
        transformed_influence_regions = {k: v[pad:-pad, pad:-pad, pad:-pad] for k, v in
                                         transformed_influence_regions.items()}
        transformed_influence_regions_inv = {k: v[pad:-pad, pad:-pad, pad:-pad] for k, v in
                                             transformed_influence_regions_inv.items()}

        # plot one of the transformed influence regions and the original influence region and the restored influence
        # region

        # create subplots
        ir_idx_to_plot = 15
        slc = influence_regions[ir_idx_to_plot].shape[2] // 2
        fig, axs = plt.subplots(1, 3)
        # plot the original influence region
        axs[0].imshow(influence_regions[ir_idx_to_plot][:, :, slc].cpu().numpy())
        axs[0].set_title('Original \ninfluence region')
        # plot the transformed influence region
        axs[1].imshow(transformed_influence_regions[ir_idx_to_plot][:, :, slc].cpu().numpy())
        axs[1].set_title('Transformed \ninfluence region')
        # plot the restored influence region
        axs[2].imshow(transformed_influence_regions_inv[ir_idx_to_plot][:, :, slc].cpu().numpy())
        axs[2].set_title('Restored \ninfluence region')
        plt.show()

