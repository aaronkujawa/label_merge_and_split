import pandas as pd
import torch
import nibabel as nib

from .shared_utils import get_influence_regions


def split_merged_label(in_data, influence_regions):
    """
    Split the merged labels according to influence regions based on the influence regions.
    :param in_data: input data with merged labels. Expected shape: (C, H, W [, D])
    :param influence_regions: influence regions with original labels. This is a dictionary, where the keys are the
    merged labels and each value is the corresponding influence region labelled with original labels. Expected shape of
    each influence region: (H, W [, D])
    :return: data with original split labels
    """

    assert(in_data.shape[1:] == influence_regions[list(influence_regions.keys())[0]].shape), \
        (f"Shape mismatch during label splitting: {in_data.shape[1:]} != "
         f"{influence_regions[list(influence_regions.keys())[0]].shape}. The spatial dimensions of the input data must "
         f"match the shape of the influence regions. You may need to register the input data to the reference space of "
         f"the influence regions and resample.")

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


def split_merged_labels_with_influence_regions(label_paths_in, label_paths_out, influence_regions):
    """

    :param label_paths_in:
    :param label_paths_out:
    :param influence_regions:
    :return:
    """
    for label_path_in, label_path_out in zip(label_paths_in, label_paths_out):
        in_nii = nib.load(label_path_in)
        in_data = torch.tensor(in_nii.get_fdata()).to("cuda")

        # input data is expected to be a tensor with shape (C, H, W [, D])
        in_data = in_data[None, ...]

        out_data = split_merged_label(in_data, influence_regions)

        out_nii = nib.Nifti1Image(out_data.cpu().numpy(), in_nii.affine)
        nib.save(out_nii, label_path_out)
        print(f"Saved split label to {label_path_out}")


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

    split_merged_labels_with_influence_regions(label_paths_in, label_paths_out, influence_regions)


def split_merged_labels_paths_from_influence_region(label_paths_in, label_paths_out, influence_regions_path):
    """
    Split the merged labels according to influence regions directly load from disk for a list of label files.
    Useful if get_merged_label_dataframe was run with save_influence_regions=True
    :param label_paths_in: list of paths to the merged label files
    :param label_paths_out: list of paths to the split label files
    :param influence_regions_path: path to the influence_region file
    :return: None
    """
    influence_regions_nii = nib.load(influence_regions_path)
    influence_regions_vol = influence_regions_nii.get_fdata()

    # transform the 4D volume to a dict
    influence_regions = {}
    for k in range(influence_regions_vol.shape[3]):
        influence_regions[k] = torch.tensor(influence_regions_vol[..., k]).to("cuda").type(torch.int)

    split_merged_labels_with_influence_regions(label_paths_in, label_paths_out, influence_regions)
