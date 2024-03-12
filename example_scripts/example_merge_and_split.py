import os
from glob import glob
from natsort import natsorted
import nibabel as nib

from src.merging_utils import get_merged_label_dataframe, merge_label_volumes
from src.splitting_utils import get_fuzzy_prior_fudged, split_merged_labels
from utils.plot_matrix_slices import plot_matrix_slices

from config.config import PROJ_ROOT

label_paths_in = natsorted(glob(os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labelsTr/*.nii.gz")))
label_to_name_csv_path = os.path.join(PROJ_ROOT, "data/task2153_mind/input/dataset/labels.csv")

merged_label_paths_out = [os.path.join(PROJ_ROOT, "data/task2153_mind/output/predictions/merged",
                                       os.path.basename(label_path).replace(".nii.gz", "_predmerged.nii.gz"))
                          for label_path in label_paths_in]

output_dir = os.path.join(PROJ_ROOT, "data/task2153_mind/output")

# get the merged label df
label_df = get_merged_label_dataframe(label_paths_in, label_to_name_csv_path, output_dir=output_dir)

# merge the label volumes
merged_labels_csv_path = os.path.join(output_dir, "merged_labels.csv")
merge_label_volumes(label_paths_in, merged_label_paths_out, merged_labels_csv_path)

# get the fudged fuzzy prior
label_support_path = os.path.join(output_dir, "label_support.pt")
fuzzy_prior_fudged = get_fuzzy_prior_fudged(label_support_path)

# split the merged labels
label_paths_merged_in = merged_label_paths_out
label_paths_split_out = [os.path.join(output_dir, "predictions/split",
                                      os.path.basename(p).replace("merged.nii.gz", ".nii.gz")) for p in label_paths_in]
split_merged_labels(label_paths_merged_in, label_paths_split_out, fuzzy_prior_fudged, merged_labels_csv_path)

# plot original, merged, and split labels for first file
plot_example_path = label_paths_in[0]
plot_merged_path = label_paths_merged_in[0]
plot_split_path = label_paths_split_out[0]

plot_matrix_slices(nib.load(plot_example_path).get_fdata(), title="Original")
plot_matrix_slices(nib.load(plot_merged_path).get_fdata(), title="Merged")
plot_matrix_slices(nib.load(plot_split_path).get_fdata(), title="Split")
