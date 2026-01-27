from scipy import ndimage
from nilearn import image
import numpy as np
import nibabel as nb
from nibabel import processing

# mask = image.resample_to_img(
#     source_img="/Users/psadil/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz",
#     target_img="/Users/psadil/Documents/MATLAB/toolbox/gift/GroupICAT/icatb/icatb_templates/Neuromark_fMRI_2.0_modelorder-25.nii",
#     interpolation="nearest",
# )
# mask_eroded = ndimage.binary_erosion(
#     np.asarray(mask.get_fdata(), dtype=np.bool), iterations=2
# )

# image.new_img_like(mask, mask_eroded, affine=mask.affine).to_filename(
#     "derivatives/MNI152_T1_3mm_brain_mask_ero.nii"
# )

# image.resample_to_img(
#     source_img="/Users/psadil/fsl/data/standard/MNI152_T1_1mm.nii.gz",
#     target_img="/Users/psadil/git/manuscripts/denoise-comparison/derivatives/100206/abcd.nii",
#     interpolation="nearest",
# ).to_filename("derivatives/MNI152_T1_3mm.nii")


processing.resample_to_output(
    nb.nifti1.Nifti1Image.load(
        "/Users/psadil/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz"
    ),
    voxel_sizes=3,
    order=0,
).to_filename("derivatives/MNI152_T1_3mm_brain_mask_dil.nii")

# need to start with the dilated mask because the others have holes in the ventricles
mask = image.resample_to_img(
    source_img="/Users/psadil/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz",
    target_img="derivatives/MNI152_T1_3mm_brain_mask_dil.nii",
    interpolation="nearest",
)
# erode so that the edge artifacts from downsampling are excluded
mask_eroded = ndimage.binary_erosion(
    np.asarray(mask.get_fdata(), dtype=np.bool), iterations=2
)

image.new_img_like(mask, mask_eroded, affine=mask.affine).to_filename(
    "derivatives/MNI152_T1_3mm_brain_mask_ero.nii"
)

# used during gift reporting
processing.resample_to_output(
    nb.nifti1.Nifti1Image.load("/Users/psadil/fsl/data/standard/MNI152_T1_1mm.nii.gz"),
    voxel_sizes=3,
    order=0,
).to_filename("derivatives/MNI152_T1_3mm.nii")

# resample now so that gift doesn't try to do so automatically
image.resample_to_img(
    "/Users/psadil/Documents/MATLAB/toolbox/gift/GroupICAT/icatb/icatb_templates/Neuromark_fMRI_2.0_modelorder-25.nii",
    "derivatives/MNI152_T1_3mm.nii",
).to_filename("derivatives/Neuromark_fMRI_2.0_modelorder-25_resampled.nii")


image.resample_to_img(
    "/Users/psadil/Documents/MATLAB/toolbox/gift/GroupICAT/icatb/icatb_templates/Neuromark_fMRI_1.0.nii",
    "derivatives/MNI152_T1_3mm.nii",
).to_filename("derivatives/Neuromark_fMRI_1.0_resampled.nii")

# for ukb fitting
image.resample_to_img(
    source_img="/Users/psadil/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz",
    target_img="tmp/sub-1000627/ses-2/func.feat/mask.nii.gz",
    interpolation="nearest",
).to_filename("derivatives/MNI152_T1_ukbmm_brain_mask_dil.nii.gz")


# without smoothing
img = nb.nifti1.Nifti1Image.load(
    "/Users/psadil/Documents/MATLAB/toolbox/gift/GroupICAT/icatb/icatb_templates/Neuromark_fMRI_1.0.nii"
)

image.concat_imgs(
    [
        processing.resample_to_output(
            img.slicer[:, :, :, i],
            voxel_sizes=2.4,
        )
        for i in range(img.shape[-1])
    ]
).to_filename("derivatives/Neuromark_fMRI_1.0_resampled24.nii")

mask = image.resample_to_img(
    source_img="/Users/psadil/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz",
    target_img="derivatives/Neuromark_fMRI_1.0_resampled24.nii",
    interpolation="nearest",
)

# erode so that the edge artifacts from downsampling are excluded
mask_eroded = ndimage.binary_erosion(
    np.asarray(mask.get_fdata(), dtype=np.bool), iterations=2
)

image.new_img_like(mask, mask_eroded, affine=mask.affine).to_filename(
    "derivatives/MNI152_T1_ukbmm_brain_mask_ero.nii"
)

mask.to_filename("derivatives/MNI152_T1_ukbmm_brain_mask_dil.nii")
# NOTE: the above look like the have truncation in superior and frontal cortex
#       not actually true -- the mask is just rather large
