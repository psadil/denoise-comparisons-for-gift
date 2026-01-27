# denoise-comparisons-for-gift

## Compute Envrionment

Environments (R and python) are managed with [`pixi`](https://pixi.prefix.dev/latest/).

## To Run

### Prep work

This produces masks and templates used during ICA.

```{shell}
pixi run python write_gift_masks.py
```

Generates

- derivatives/MNI152_T1_3mm_brain_mask_ero.nii
- derivatives/MNI152_T1_ukbmm_brain_mask_ero.nii
- derivatives/Neuromark_fMRI_1.0_resampled.nii
- derivatives/Neuromark_fMRI_1.0_resampled24.nii

### Generate ICA maps

```{shell}
sbatch --array 0-999 ukb
```

Uses

- derivatives/MNI152_T1_3mm_brain_mask_ero.nii
- derivatives/Neuromark_fMRI_1.0_resampled.nii

Generates

- derivatives/ukb (intermediate files, including preprocessed niftis and rerun of FEAT)
- derivatives/ukb_confounds (confounds used during ABCD processing)
- derivatives/ic (component spatial maps)
- derivatives/tc (component timeseries)

```{shell}
sbatch --array 800-999 ukb3mm
```

Uses

- derivatives/ukb
- derivatives/MNI152_T1_ukbmm_brain_mask_ero.nii
- derivatives/Neuromark_fMRI_1.0_resampled24.nii

Generates

- derivatives/ic (component spatial maps, higher res results)
- derivatives/tc (component timeseries, higher res templates)

### Generate Connectivities

```{shell}
# probably better to run interactively, like in positron
pixi run -e r main.R
```

Uses

- derivatives/tc

Generates

- derivatives/connectivity (atanh-transformed connectivities between timeseries in derivatives/tc)
- some additional plots

### Fit Predictive Models

```{shell}
# run everything
# sbatch --array 1-438 entrypoint

# do just one fit (with fluid intelligence)
sbatch --array 360 entrypoint
```

Generates

- ${MYSCRATCH}/predictions/results (scores and predictions for training and test)
- ${MYSCRATCH}/predictions/test-predictions (predictions for test, in case extra scores wanted)
- ${MYSCRATCH}/predictions/features (training model features)
