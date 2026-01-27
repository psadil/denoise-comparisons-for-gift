from pathlib import Path
import logging

import numpy as np
import polars as pl
from scipy import signal, stats
import nibabel as nb
from nilearn import maskers, masking, image
from gigica import estimator

FD_THRESH_FIRST = 0.3
FD_THRESH_SECOND = 0.2
LOW_PASS = 0.08
HIGH_PASS = 0.009
BAND_LOWER = 0.31
BAND_UPPER = 0.43
FD_RADIUS = 50.0
STANDARDIZE = False
# all VC's papers seem to use this
FWHM = 6


def get_masker_abcd(
    mask: Path, tr: float, smoothing_fwhm: float | None = FWHM
) -> maskers.NiftiMasker:
    # not detrending to preserve mean for later, global PSC normalization
    return maskers.NiftiMasker(
        mask_img=mask,
        standardize=STANDARDIZE,
        low_pass=LOW_PASS,
        high_pass=HIGH_PASS,
        t_r=tr,
        reports=False,
        smoothing_fwhm=smoothing_fwhm,
    ).fit()


def do_additional_abcd_mask(
    arr: np.ndarray,
    masker: maskers.NiftiMasker,
    confounds: pl.DataFrame,
    smoothing_fwhm: float | None = FWHM,
) -> nb.nifti1.Nifti1Image:
    current_confounds = confounds.filter(pl.col("sample_mask_first"))

    not_all_zero_nii: nb.nifti1.Nifti1Image = masker.inverse_transform(
        np.logical_not(np.isclose(arr, 0).sum(axis=0) == arr.shape[0])
    )  # type:ignore

    # filter with tighter threshold
    nii0: nb.nifti1.Nifti1Image = masker.inverse_transform(arr)  # type:ignore

    # old mask was not always stored as int/bool
    old_mask = image.binarize_img(masker.mask_img)

    new_mask = image.math_img("i0 & i1", i0=old_mask, i1=not_all_zero_nii)

    masker2 = maskers.NiftiMasker(
        mask_img=new_mask,
        standardize=False,
        reports=False,
        smoothing_fwhm=None,
    ).fit()

    arr2 = masker2.transform(
        nii0,
        sample_mask=current_confounds.select("sample_mask_second")
        .to_series()
        .to_numpy(),
    )

    # now, filter based on any remaining weird variance
    # not perfectly ABCD, but should be close
    # also, with this dvars approach, it is standard to
    # ignore cases where dvars is suspiciously low, but
    # we've already removed some variance, so a dvars dip
    # may be super informative
    dvars = get_dvars(arr2).with_columns(
        sample_mask=(pl.col("DPD").abs() < 5) | (pl.col("ZD").abs() < 5)
    )
    prop_to_keep = dvars.select("sample_mask").to_series().to_numpy().mean()
    if prop_to_keep < 0.95:
        logging.warning(
            f"removing more than 5% of remaining data with dvars: {prop_to_keep}"
        )

    # it's only here that we smooth
    masker3 = maskers.NiftiMasker(
        mask_img=new_mask,
        standardize=False,
        reports=False,
        smoothing_fwhm=smoothing_fwhm,
    ).fit()

    nii1: nb.nifti1.Nifti1Image = masker3.inverse_transform(arr2)  # type:ignore

    arr3 = masker3.transform(
        nii1, sample_mask=dvars.select("sample_mask").to_series().to_numpy()
    )

    return masker3.inverse_transform(arr3)  # type: ignore


def get_masker_hcp(
    mask: Path, smoothing_fwhm: float | None = FWHM
) -> maskers.NiftiMasker:
    return maskers.NiftiMasker(
        mask_img=mask,
        standardize=STANDARDIZE,
        reports=False,
        smoothing_fwhm=smoothing_fwhm,
    ).fit()


def filter_short_runs(arr: np.ndarray, min_length=5):
    # Ensure input is a numpy array
    arr = np.asanyarray(arr)

    # Pad with False to handle runs at the very beginning or end
    padded = np.concatenate([[False], arr, [False]])

    # Find indices where the value changes (False -> True or True -> False)
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # Calculate lengths of each run
    durations = ends - starts

    # Identify runs that are too short
    short_runs = starts[durations < min_length]
    short_durations = durations[durations < min_length]

    # Create a copy of the original array to modify
    result = arr.copy()

    # "Erase" the short runs
    for start, length in zip(short_runs, short_durations):
        result[start : start + length] = False

    return result


def filter_band_stop(series: pl.Series, tr: float, bw: float, w0: float) -> pl.Series:
    b, a = signal.iirnotch(w0=w0, Q=w0 / bw, fs=1 / tr)
    return pl.Series(signal.filtfilt(b, a, series.to_numpy()))


def add_fd(d: pl.DataFrame, radius: float = FD_RADIUS) -> pl.DataFrame:
    return (
        d.sort("t")
        .with_columns(
            rot_x_mm=pl.col("rot_x") * radius,
            rot_y_mm=pl.col("rot_y") * radius,
            rot_z_mm=pl.col("rot_z") * radius,
            trans_x_mm=pl.col("trans_x"),
            trans_y_mm=pl.col("trans_y"),
            trans_z_mm=pl.col("trans_z"),
        )
        .with_columns(pl.selectors.ends_with("_mm").diff().abs())
        .with_columns(
            FD=pl.col("rot_x_mm")
            + pl.col("rot_y_mm")
            + pl.col("rot_z_mm")
            + pl.col("trans_x_mm")
            + pl.col("trans_y_mm")
            + pl.col("trans_z_mm")
        )
        .drop(pl.selectors.ends_with("_mm"))
    )


def do_ica(src: Path, mask: Path, references: Path, dst_tc: Path, dst_ic: Path):
    # and now estimate components!
    gigica = estimator.GIGICA()

    x = masking.apply_mask(src, mask_img=mask)
    r = masking.apply_mask(references, mask_img=mask)
    tc = gigica.fit_transform(X=x, references=r)

    if not dst_tc.parent.exists():
        dst_tc.parent.mkdir(parents=True)

    pl.from_numpy(tc).with_row_index("t").unpivot(
        index="t", variable_name="component"
    ).with_columns(
        pl.col("component").str.strip_prefix("column_").cast(pl.UInt16),
        pl.col("t").cast(pl.UInt16),
    ).write_parquet(dst_tc)

    if not dst_ic.parent.exists():
        dst_ic.parent.mkdir(parents=True)

    masking.unmask(gigica.components_, mask_img=mask).to_filename(dst_ic)  # type:ignore


def expand_motion(d: pl.DataFrame, radius: float = FD_RADIUS) -> pl.DataFrame:
    return (
        add_fd(d.with_row_index("t"), radius=radius)
        .with_columns(
            trans_x_power2=pl.col("trans_x").pow(2),
            trans_y_power2=pl.col("trans_y").pow(2),
            trans_z_power2=pl.col("trans_z").pow(2),
            rot_x_power2=pl.col("rot_x").pow(2),
            rot_y_power2=pl.col("rot_y").pow(2),
            rot_z_power2=pl.col("rot_z").pow(2),
        )
        .with_columns(
            trans_x_derivative1=pl.col("trans_x").diff(),
            trans_y_derivative1=pl.col("trans_y").diff(),
            trans_z_derivative1=pl.col("trans_z").diff(),
            rot_x_derivative1=pl.col("rot_x").diff(),
            rot_y_derivative1=pl.col("rot_y").diff(),
            rot_z_derivative1=pl.col("rot_z").diff(),
            trans_x_power2_derivative1=pl.col("trans_x_power2").diff(),
            trans_y_power2_derivative1=pl.col("trans_y_power2").diff(),
            trans_z_power2_derivative1=pl.col("trans_z_power2").diff(),
            rot_x_power2_derivative1=pl.col("rot_x_power2").diff(),
            rot_y_power2_derivative1=pl.col("rot_y_power2").diff(),
            rot_z_power2_derivative1=pl.col("rot_z_power2").diff(),
        )
    )


def get_spatial_confounds(
    src: Path, wholebrain: Path, wm: Path, csf: Path
) -> pl.DataFrame:
    # make maskers
    # these will be standardized later
    masker_wholebrain = maskers.NiftiMasker(mask_img=wholebrain, reports=False).fit()
    masker_white = maskers.NiftiMasker(mask_img=wm, reports=False).fit()
    masker_csf = maskers.NiftiMasker(mask_img=csf, reports=False).fit()

    src_nii = nb.nifti1.Nifti1Image.load(src)

    # get average signals
    wholebrain_avg = masker_wholebrain.transform(src_nii).mean(axis=1)
    white_avg = masker_white.transform(src_nii).mean(axis=1)
    csf_avg = masker_csf.transform(src_nii).mean(axis=1)

    return (
        pl.DataFrame({"white": white_avg, "brain": wholebrain_avg, "csf": csf_avg})
        .with_columns(
            white_derivative1=pl.col("white").diff(),
            brain_derivative1=pl.col("brain").diff(),
            csf_derivative1=pl.col("csf").diff(),
        )
        .with_row_index("t")
    )


def finalize_confounds(
    motion: pl.DataFrame, spatial_confounds: pl.DataFrame, dst: Path
) -> pl.DataFrame:
    motion = motion.with_columns(
        sample_mask_first=pl.col("FD") < FD_THRESH_FIRST,
        sample_mask_second=pl.col("FD") < FD_THRESH_SECOND,
    )

    sample_mask_first = (
        motion.select("sample_mask_first").to_series().fill_null(True).to_numpy()
    )
    sample_mask_second = filter_short_runs(
        motion.select("sample_mask_second").to_series().fill_null(True).to_numpy()
    )
    out = (
        motion.join(spatial_confounds, on="t")
        .sort("t")
        .drop(pl.selectors.starts_with("sample_mask"))
        .fill_null(0)
        .with_columns(
            sample_mask_first=sample_mask_first, sample_mask_second=sample_mask_second
        )
    )
    out.lazy().sink_parquet(dst, mkdir=True)
    return out


def sd_hIQR(x, d=1):
    w = x ** (1 / d)  # Power trans.: w~N(mu_w, sigma_w^2)
    sd = (np.quantile(w, 0.5) - np.quantile(w, 0.25)) / (1.349 / 2)  # hIQR
    out = (d * np.median(w) ** (d - 1)) * sd  # Delta method
    # In the paper, the above formula incorrectly has d^2 instead of d.
    # The code on github correctly uses d.
    return out


def get_zd(D: pl.Series) -> np.ndarray:
    DV2 = D.drop_nans().drop_nulls().to_numpy() * 4
    mu_0 = np.median(DV2)  # pg 305
    sigma_0 = sd_hIQR(DV2, d=3)  # pg 305: cube root power trans
    v = 2 * mu_0**2 / sigma_0**2
    X = v / mu_0 * DV2  # pg 298: ~X^2(v=2*mu_0^2/sigma_0^2)
    P = stats.chi2.cdf(X, v)
    return np.concatenate(
        [
            [np.nan],
            np.where(
                np.abs(P - 0.5) < 0.49999,
                stats.norm.ppf(1 - stats.chi2.cdf(X, v)),
                (DV2 - mu_0) / sigma_0,
            ),
        ]
    )


def get_dvars(x: np.ndarray) -> pl.DataFrame:
    # normalize
    X = x.copy()
    X = X / np.median(np.mean(X, axis=0)) * 100
    avgs = np.mean(X, 0)
    X -= avgs[:, None].T

    out = (
        pl.from_numpy(X)
        .rename(lambda col: col.removeprefix("column_"))
        .with_row_index(name="t", offset=0)
        .unpivot(index="t")
        .with_columns(
            A=pl.col("value") ** 2,
            D=pl.col("value").diff().over("variable", order_by="t"),
            S=pl.col("value")
            .rolling_mean(window_size=2)
            .over("variable", order_by="t"),
        )
        .with_columns((pl.selectors.by_name("D", "S") ** 2) / 4)
        .group_by("t")
        .agg((~pl.selectors.by_name("variable")).mean())
        .sort("t")
        .with_columns(
            DPD=(pl.col("D") - pl.col("D").median()) / pl.col("A").mean() * 100,
            DVARS=pl.col("D").sqrt() * 2,
        )
    )
    zd = get_zd(out.select("D").to_series())

    return out.with_columns(ZD=zd).fill_nan(None).fill_null(0)
