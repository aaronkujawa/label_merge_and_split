import pandas as pd
import torch
import monai
import numpy as np
import nibabel as nib


def get_fuzzy_prior_fudged(label_support_path):
    # load the fuzzy prior, and smooth it so that pixels that haven't seen a certain structure in the fuzzy prior creation will
    # still get some probability if they are close to the structure
    if 'fuzzy_prior' not in locals():
        label_support = torch.load(label_support_path)

        fuzzy_prior = label_support/len(label_support)  # convert label_support to fuzzy prior
        fuzzy_prior = fuzzy_prior.to("cpu")

        # apply gaussian blur to fuzzy prior image, because otherwise, the probabilities decrease too quickly
        smooth = monai.transforms.GaussianSmooth(sigma=5.0, approx='erf')
        fuzzy_prior_fudged = torch.zeros_like(fuzzy_prior, device="cuda")

        for l in range(fuzzy_prior.shape[0]):
            # we don't want high probabilites to decrease because of the smoothing, so keep the maximum of smoothed and original
            fuzzy_prior_fudged[l, ...] = torch.maximum(smooth(fuzzy_prior[l, ...]).to("cuda"), fuzzy_prior[l, ...].to("cuda"))
        return fuzzy_prior_fudged


# relabel all pixels according to fudged prior labels
def split_merged_labels(label_paths_in, label_paths_out, fuzzy_prior_fudged, merged_labels_csv_path):

    for label_path_in, label_path_out in zip(label_paths_in, label_paths_out):
        in_nii = nib.load(label_path_in)
        in_data = torch.tensor(in_nii.get_fdata()).to("cuda")

        out_data = torch.zeros_like(in_data, dtype=torch.int, device="cuda")

        merged_labels_df = pd.read_csv(merged_labels_csv_path, index_col='label')

        for l_comb in merged_labels_df['merged_label'].unique():

            orig_labels = merged_labels_df[merged_labels_df['merged_label'] == l_comb].index
            channels = [merged_labels_df.loc[l, 'channel'] for l in orig_labels]

            # select only the priors of the old labels that correspond to the current new label l_comb
            influence_regions = torch.argmax(fuzzy_prior_fudged[channels], dim=0).type(torch.int)

            # convert the channels back to original labels
            orig_labels_argmax = torch.zeros_like(influence_regions)
            for orig, chan in zip(orig_labels, range(len(orig_labels))):
                orig_labels_argmax[influence_regions == chan] = orig

            # assign each pixel
            l_comb_selection = (in_data == l_comb)
            out_data[l_comb_selection] = orig_labels_argmax[l_comb_selection]

        out_nii = nib.Nifti1Image(out_data.cpu().numpy(), in_nii.affine)
        nib.save(out_nii, label_path_out)
