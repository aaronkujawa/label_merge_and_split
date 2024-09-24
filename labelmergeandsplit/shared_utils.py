import torch
import numpy as np
import monai


def load_label_support(label_support_path, device="cuda"):
    """
    This function loads the label support from the label_support_path. The label support is a tensor that contains the
    label support for each label.
    :param label_support_path: path to the label support
    :param device: device to load the label support to
    :return: label_support: an array of shape (num_labels, *data_shape) that contains the label support for each label
    """
    if label_support_path.endswith(".npz"):
        print(f"Loading compressed label support from {label_support_path}")
        label_support = torch.from_numpy(np.load(label_support_path)["label_support"]).to(device)
    else:
        label_support = torch.load(label_support_path, map_location=device)

    return label_support


def get_fuzzy_prior_fudged(label_support_path, label_support=None):
    """
    Get the fudged fuzzy prior from the label support
    :param label_support_path: path to the label support file
    :param label_support: already load label support volume (default None), if defined label_support_path is not used
    :return: fudged fuzzy prior
    """
    if 'fuzzy_prior' not in locals():
        if label_support is None:
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

        argmax_vals = range(len(orig_labels))  # these are the possible values returned from the argmax operation

        # convert the argmax_vals back to original labels
        influence_region_orig_labels = torch.zeros_like(influence_region_argmax_vals)
        for orig, argmax_val in zip(orig_labels, argmax_vals):
            influence_region_orig_labels[influence_region_argmax_vals == argmax_val] = orig

        influence_regions[merged_label] = influence_region_orig_labels

    return influence_regions
