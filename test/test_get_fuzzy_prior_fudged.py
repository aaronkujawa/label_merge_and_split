import unittest

from config.config import PROJ_ROOT
import os
from labelmergeandsplit.splitting_utils import get_fuzzy_prior_fudged
import nibabel as nib

from utils.plot_matrix_slices import plot_matrix_slices


class TestGetFuzzyPriorFudged(unittest.TestCase):
    def test_get_fuzzy_prior_fudged(self):
        label_support_path = os.path.join(PROJ_ROOT, "data/task2153_mind/output/label_support.pt")
        fuzzy_prior_fudged = get_fuzzy_prior_fudged(label_support_path)

        reference_space_image = nib.load(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/mind_000.nii.gz"))
        reference_affine = reference_space_image.affine

        plot_matrix_slices(fuzzy_prior_fudged[0].to("cpu").numpy(),
                           title="fuzzy_prior_fudged[0]",
                           save_nifti_path="tmp.nii.gz",
                           nifti_affine=reference_affine)


