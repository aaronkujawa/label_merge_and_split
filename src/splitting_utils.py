import pandas as pd
import torch
import monai
import nibabel as nib


def get_fuzzy_prior_fudged(label_support_path):
    """
    Get the fudged fuzzy prior from the label support
    :param label_support_path: path to the label support file
    :return: fudged fuzzy prior
    """
    if 'fuzzy_prior' not in locals():
        label_support = torch.load(label_support_path)

        fuzzy_prior = label_support/len(label_support)  # convert label_support to fuzzy prior
        fuzzy_prior = fuzzy_prior.to("cpu")

        # apply gaussian blur to fuzzy prior image, because otherwise, the probabilities decrease too quickly
        smooth = monai.transforms.GaussianSmooth(sigma=5.0, approx='erf')
        fuzzy_prior_fudged = torch.zeros_like(fuzzy_prior, device="cuda")

        for l in range(fuzzy_prior.shape[0]):
            # we don't want high probabilites to decrease because of the smoothing, so keep the maximum of smoothed and original
            smoothed_prior = smooth(fuzzy_prior[l, ...]).to("cuda")

            # # alternative: apply distance transform to fuzzy prior
            # fuzzy_prior_l = fuzzy_prior[l, ...].to("cuda")
            # edt_prior = monai.transforms.DistanceTransformEDT()(fuzzy_prior_l).to("cuda")
            #
            # # scale values from 0 (furthest) to minimum fuzzy prior value (closest)
            # divisor = torch.max(edt_prior)/torch.min(fuzzy_prior_l)
            # edt_prior = (-edt_prior / divisor) + torch.min(fuzzy_prior_l)

            prior = smoothed_prior  # or edt_prior
            fuzzy_prior_fudged[l, ...] = torch.maximum(prior, fuzzy_prior[l, ...].to("cuda"))
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
        print(f"Saved split label to {label_path_out}")
