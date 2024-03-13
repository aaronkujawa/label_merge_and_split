from matplotlib import pyplot as plt
import numpy as np
import nibabel as nib
import os
import torch


def plot_matrix_slices(data, pixdim=[1,1,1], nibabel_header=None, lo_intensity_nanpercentile=5, up_intensity_nanpercentile=95, title="", save_png_path=None, save_nifti_path=None, nifti_affine=None, open_nifti_in_itksnap=False, suppress_show=False):

    if nifti_affine is not None:
        affine = nifti_affine
    elif hasattr(data, "affine"):
        affine = data.affine
    else:
        affine = None
        
    data = np.squeeze(data)
    
    if torch.is_tensor(data):
        data = data.cpu().numpy()

    if data.dtype == 'int64':
        data = data.astype('int32')
        
    s = data.shape

    if nibabel_header is not None:
        pixdim = nibabel_header['pixdim'][1:4]
        s = nibabel_header['dim'][1:4]

    h = 1  # len(filenames)

    fig = plt.figure()
    sub_idx = 1
    plt.subplot(h,3,sub_idx)
    slc = data[:, :, s[2]//2]
    plt.imshow(slc.T, cmap='gray', extent=[0,s[0],0,s[1]], aspect=pixdim[1]/pixdim[0], vmin=np.nanpercentile(slc, lo_intensity_nanpercentile), vmax=np.nanpercentile(slc, up_intensity_nanpercentile))
    plt.xticks([]); plt.yticks([])
    plt.gca().invert_yaxis()
    sub_idx+=1
    plt.subplot(h,3,sub_idx)
    slc = data[s[0]//2, :, :,]
    plt.imshow(slc.T, cmap='gray', extent=[0,s[1],0,s[2]], aspect=pixdim[2]/pixdim[1], vmin=np.nanpercentile(slc, lo_intensity_nanpercentile), vmax=np.nanpercentile(slc, up_intensity_nanpercentile))
    plt.xticks([]); plt.yticks([])
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=5)
    sub_idx+=1
    plt.subplot(h,3,sub_idx)
    slc = data[:, s[1]//2, :,]
    plt.imshow(slc.T, cmap='gray', extent=[0,s[0],0,s[2]], aspect=pixdim[2]/pixdim[0], vmin=np.nanpercentile(slc, lo_intensity_nanpercentile), vmax=np.nanpercentile(slc, up_intensity_nanpercentile))  
    plt.xticks([]); plt.yticks([])
    plt.gca().invert_yaxis()
    sub_idx+=1

    if save_png_path is not None:
        plt.savefig(save_png_path, facecolor='white', transparent=False, dpi=100)
    if save_nifti_path is not None:
        nii = nib.Nifti1Image(data, affine=affine)
        
        nib.save(nii, save_nifti_path)
        if open_nifti_in_itksnap:
            os.system("itksnap -g " + save_nifti_path + " &")
        
    if not suppress_show:
        plt.show()
        
    plt.cla()    
    plt.clf()
    plt.close(fig)
