import pandas as pd
import torch
import monai
import nibabel as nib
from .merging_utils import load_label_support


def get_fuzzy_prior_fudged(label_support_path):
    """
    Get the fudged fuzzy prior from the label support
    :param label_support_path: path to the label support file
    :return: fudged fuzzy prior
    """
    if 'fuzzy_prior' not in locals():
        label_support = load_label_support(label_support_path)

        fuzzy_prior = label_support/len(label_support)  # convert label_support to fuzzy prior
        fuzzy_prior = fuzzy_prior.to("cpu")

        # apply gaussian blur to fuzzy prior image, because otherwise, the probabilities decrease too quickly
        smooth = monai.transforms.GaussianSmooth(sigma=5.0, approx='erf')
        fuzzy_prior_fudged = torch.zeros_like(fuzzy_prior, device="cuda")

        for l in range(fuzzy_prior.shape[0]):

            use_smoothing_strategy = True  # smoothing strategy is faster than distance transform when cuCIM is available

            if use_smoothing_strategy:
                smoothed_prior = smooth(fuzzy_prior[l]).to("cuda")
                prior = smoothed_prior
            else:  # alternative: apply distance transform to fuzzy prior (needs cuCIM installed to run on GPU, otherwise it's slow)
                edt_prior = -monai.transforms.DistanceTransformEDT()(fuzzy_prior[l].to("cuda")).to("cuda")
                prior = edt_prior

            # we don't want original probabilities to decrease because of the fudging, so keep the maximum of fudged
            # and original
            fuzzy_prior_fudged[l] = torch.maximum(prior, fuzzy_prior[l].to("cuda"))
        return fuzzy_prior_fudged


# relabel all pixels according to fudged prior labels
def split_merged_labels(label_paths_in, label_paths_out, fuzzy_prior_fudged, merged_labels_csv_path):
    """
    Split the merged labels according to influence regions based on the fudged prior.
    :param label_paths_in: list of paths to the merged label files
    :param label_paths_out: list of paths to the split label files
    :param fuzzy_prior_fudged: fudged fuzzy prior
    :param merged_labels_csv_path: path to the merged labels csv file
    :return: None
    """
    merged_labels_df = pd.read_csv(merged_labels_csv_path, index_col='label')

    unique_merged_labels = merged_labels_df['merged_label'].unique()
    # calculate the influence regions for each merged label's original labels
    influence_regions = {}
    for merged_label in unique_merged_labels:
        orig_labels = merged_labels_df[merged_labels_df['merged_label'] == merged_label].index
        channels = [merged_labels_df.loc[l, 'channel'] for l in orig_labels]

        # select only the priors of the old labels that correspond to the current merged_label
        influence_regions[merged_label] = torch.argmax(fuzzy_prior_fudged[channels], dim=0).type(torch.int)

    for label_path_in, label_path_out in zip(label_paths_in, label_paths_out):
        in_nii = nib.load(label_path_in)
        in_data = torch.tensor(in_nii.get_fdata()).to("cuda")

        out_data = torch.zeros_like(in_data, dtype=torch.int, device="cuda")

        for merged_label in unique_merged_labels:

            orig_labels = merged_labels_df[merged_labels_df['merged_label'] == merged_label].index
            channels = [merged_labels_df.loc[l, 'channel'] for l in orig_labels]

            # convert the channels back to original labels
            influence_region = influence_regions[merged_label]
            orig_labels_argmax = torch.zeros_like(influence_region)
            for orig, chan in zip(orig_labels, range(len(orig_labels))):
                orig_labels_argmax[influence_region == chan] = orig

            # assign each pixel
            l_comb_selection = (in_data == merged_label)
            out_data[l_comb_selection] = orig_labels_argmax[l_comb_selection]

        out_nii = nib.Nifti1Image(out_data.cpu().numpy(), in_nii.affine)
        nib.save(out_nii, label_path_out)
        print(f"Saved split label to {label_path_out}")
