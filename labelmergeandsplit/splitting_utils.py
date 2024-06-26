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

        label_support = label_support.to("cpu")
        fuzzy_prior = label_support #/len(label_support)  # convert label_support to fuzzy prior

        # apply gaussian blur to fuzzy prior image, because otherwise, the probabilities decrease too quickly
        smooth = monai.transforms.GaussianSmooth(sigma=10.0, approx='erf')
        fuzzy_prior_fudged = torch.zeros_like(fuzzy_prior, device="cuda")

        for l in range(fuzzy_prior.shape[0]):

            use_smoothing_strategy = False  # smoothing strategy is faster than distance transform when cuCIM is not available

            if use_smoothing_strategy:
                smoothed_prior = smooth(fuzzy_prior[l][None, :]).to("cuda").squeeze()
                prior = smoothed_prior
            else:  # alternative: apply distance transform to fuzzy prior (needs cuCIM installed to run on GPU, otherwise it's slow)
                bg_mask = fuzzy_prior[l].to("cuda") == 0
                edt_prior = -monai.transforms.DistanceTransformEDT()(bg_mask[None, :]).to("cuda").squeeze()

                prior = edt_prior

            # we don't want original probabilities to decrease because of the fudging, so where the original prior is
            # greater than 0 we set the fudged prior to the original prior
            fuzzy_prior_fudged[l] = torch.where(fuzzy_prior[l].to("cuda") > 0, fuzzy_prior[l].to("cuda"), prior)

            fuzzy_prior_fudged[l] = fuzzy_prior_fudged[l]/len(label_support)
        return fuzzy_prior_fudged


# relabel all pixels according to fudged prior labels


def get_influence_regions(fuzzy_prior_fudged, merged_label_mapping, channel_mapping):
    """
    Get the influence regions for each merged label set based on the fudged prior
    :param fuzzy_prior_fudged: fudged fuzzy prior
    :param merged_label_mapping: mapping from original labels to merged labels
    :param channel_mapping: mapping from original labels to channels in the fuzzy prior
    :return: influence regions with original labels. This is a dictionary, where the keys are the merged labels and each
    value is the corresponding influence region labelled with original labels
    """
    unique_merged_labels = list(set(merged_label_mapping.values()))
    # calculate the influence regions for each merged label's original labels
    influence_regions = {}
    for merged_label in unique_merged_labels:
        orig_labels = [l for l, m in merged_label_mapping.items() if m == merged_label]  # these labels are merged into the current merged_label
        channels = [channel_mapping[l] for l in orig_labels]  # these channels in the fuzzy prior correspond to the original labels

        # select only the priors of the old labels that correspond to the current merged_label
        influence_region_argmax_vals = torch.argmax(fuzzy_prior_fudged[channels], dim=0).type(torch.int)

        argmax_vals= range(len(orig_labels))  # these are the possible values returned from the argmax operation

        # convert the argmax_vals back to original labels
        influence_region_orig_labels = torch.zeros_like(influence_region_argmax_vals)
        for orig, argmax_val in zip(orig_labels, argmax_vals):
            influence_region_orig_labels[influence_region_argmax_vals == argmax_val] = orig

        influence_regions[merged_label] = influence_region_orig_labels

    return influence_regions


def split_merged_label(in_data, influence_regions):
    """
    Split the merged labels according to influence regions based on the influence regions.
    :param in_data: input data with merged labels
    :param influence_regions: influence regions with original labels. This is a dictionary, where the keys are the
    merged labels and each value is the corresponding influence region labelled with original labels
    :return: data with original split labels
    """
    out_data = torch.zeros_like(in_data, dtype=torch.int, device="cuda")

    for merged_label, influence_region in influence_regions.items():
        mask = (in_data == merged_label)

        # prepare influence_region for broadcasting (if in_data has batch or channel dimensions)
        while len(influence_region.shape) < len(mask.shape):
            influence_region = influence_region.unsqueeze(0)

        # broadcast to the shape of mask
        influence_region = influence_region.expand_as(mask)

        out_data[mask] = influence_region[mask]

    return out_data


def split_merged_labels_paths(label_paths_in, label_paths_out, fuzzy_prior_fudged, merged_labels_csv_path):
    """
    Split the merged labels according to influence regions based on the fudged prior for a list of label files.
    :param label_paths_in: list of paths to the merged label files
    :param label_paths_out: list of paths to the split label files
    :param fuzzy_prior_fudged: fudged fuzzy prior
    :param merged_labels_csv_path: path to the merged labels csv file
    :return: None
    """
    merged_labels_df = pd.read_csv(merged_labels_csv_path, index_col='label')
    merged_label_mapping = merged_labels_df['merged_label'].to_dict()
    channel_mapping = merged_labels_df['channel'].to_dict()

    influence_regions = get_influence_regions(fuzzy_prior_fudged, merged_label_mapping, channel_mapping)

    for label_path_in, label_path_out in zip(label_paths_in, label_paths_out):
        in_nii = nib.load(label_path_in)
        in_data = torch.tensor(in_nii.get_fdata()).to("cuda")

        out_data = split_merged_label(in_data, influence_regions)

        out_nii = nib.Nifti1Image(out_data.cpu().numpy(), in_nii.affine)
        nib.save(out_nii, label_path_out)
        print(f"Saved split label to {label_path_out}")
