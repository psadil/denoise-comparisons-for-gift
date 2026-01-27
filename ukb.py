import argparse
from pathlib import Path
import subprocess
import typing
import re
import logging
import os
import tempfile
import shutil
import sys
from enum import Enum

import polars as pl
import numpy as np
import pandas as pd
import nibabel as nb
from nilearn import image
from scipy import stats
from scipy import ndimage
import pydantic
from fsl import wrappers
import nitransforms as nt

import utils

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s |  %(message)s",
    level=logging.INFO,
    force=True,
)


FSLDIR = os.environ["FSLDIR"]
SRC_ROOT = Path("/dcs07/smart/data/ukb/rawdata")
DST_ROOT = Path("derivatives/ukb").resolve()
REF = Path("derivatives")
# FD_THRESH = utils.FD_THRESH


class Ref(str, Enum):
    mm3 = "Neuromark_fMRI_1.0_resampled.nii"
    mm24 = "Neuromark_fMRI_1.0_resampled24.nii"


class SrcDst(pydantic.BaseModel):
    src: Path
    dst: Path


class Descrip(pydantic.BaseModel):
    TE: int
    dwell: float = 0.64

    @classmethod
    def from_header(cls, header: nb.nifti1.Nifti1Header) -> typing.Self:
        descrip: str = header.get("descrip").tobytes().decode()  # type:ignore
        return cls(**dict(re.findall(r"(\w+)=([\d\.]+)", descrip)))


class UKBSub(pydantic.BaseModel):
    sub: str
    ses: str = "2"
    task: str = "rest"
    root: Path = SRC_ROOT
    dst_root: Path = DST_ROOT

    @property
    def ses_dir(self) -> Path:
        return self.root / f"sub-{self.sub}" / f"ses-{self.ses}"

    @property
    def func_dir(self) -> Path:
        return self.ses_dir / "func"

    @property
    def dst_func_dir(self) -> Path:
        return self.dst_root / f"sub-{self.sub}" / f"ses-{self.ses}" / "func"

    @property
    def ica_dir(self) -> Path:
        return self.ses_dir / "non-bids" / "20227" / "fMRI" / "rfMRI.ica"

    @property
    def t1_dir(self) -> Path:
        return self.ses_dir / "non-bids" / "20252" / "T1"

    @property
    def transforms_dir(self) -> Path:
        return self.t1_dir / "transforms"

    @property
    def t1_fast_dir(self) -> Path:
        return self.t1_dir / "T1_fast"

    @property
    def reg_dir(self) -> Path:
        return self.ica_dir / "reg"

    @property
    def unwarp_dir(self) -> Path:
        return self.reg_dir / "unwarp"

    @property
    def rfmri(self) -> SrcDst:
        return SrcDst(
            src=(
                self.func_dir
                / f"sub-{self.sub}_ses-{self.ses}_task-{self.task}_bold.nii.gz"
            ),
            dst=self.dst_func_dir / "rfMRI.nii.gz",
        )

    @property
    def sbref(self) -> SrcDst:
        return SrcDst(
            src=self.func_dir
            / f"sub-{self.sub}_ses-{self.ses}_task-{self.task}_sbref.nii.gz",
            dst=self.dst_func_dir / "rfMRI_SBREF.nii.gz",
        )

    @property
    def fieldmap(self) -> SrcDst:
        return SrcDst(
            src=self.unwarp_dir / "fieldmap_fout_to_T1_brain_rad.nii.gz",
            dst=self.dst_func_dir / "fieldmap_fout_to_T1_brain_rad.nii.gz",
        )

    @property
    def t1(self) -> Path:
        return self.dst_func_dir / "T1.nii.gz"

    @property
    def t1_orig(self) -> Path:
        return self.t1_dir / "T1_orig_defaced.nii.gz"

    @property
    def t1_brain(self) -> SrcDst:
        return SrcDst(
            src=self.t1_dir / "T1_brain.nii.gz",
            dst=self.dst_func_dir / "T1_brain.nii.gz",
        )

    @property
    def t1_2_mni_linear(self) -> SrcDst:
        return SrcDst(
            src=self.transforms_dir / "T1_to_MNI_linear.mat",
            dst=self.dst_func_dir / "T1_brain2MNI152_T1_2mm_brain.mat",
        )

    @property
    def t1_2_mni_coef(self) -> Path:
        return self.transforms_dir / "T1_to_MNI_warp_coef.nii.gz"

    @property
    def t1_2_standard_warp(self) -> Path:
        return self.dst_func_dir / "T1_brain2MNI152_T1_2mm_brain_warp.nii.gz"

    @property
    def t1_pve_wm(self) -> Path:
        return self.t1_fast_dir / "T1_brain_pve_2.nii.gz"

    @property
    def t1_pve_csf(self) -> Path:
        return self.t1_fast_dir / "T1_brain_pve_0.nii.gz"

    @property
    def t1_wmseg(self) -> SrcDst:
        return SrcDst(
            src=self.t1_pve_wm,
            dst=self.dst_func_dir / "T1_brain_wmseg.nii.gz",
        )

    @property
    def example_func_wmseg(self) -> Path:
        return self.dst_func_dir / "example_func_wmseg.nii.gz"

    @property
    def example_func_csfseg(self) -> Path:
        return self.dst_func_dir / "example_func_csfseg.nii.gz"

    @property
    def nii(self) -> nb.nifti1.Nifti1Image:
        return nb.nifti1.Nifti1Image.load(self.rfmri.dst)

    @property
    def tr(self) -> float:
        return self.nii.header.get("pixdim")[4]  # type:ignore

    @property
    def n_tr(self) -> float:
        return self.nii.header.get("dim")[4]  # type:ignore

    @property
    def desc(self) -> Descrip:
        return Descrip.from_header(self.nii.header)

    @property
    def motion(self) -> Path:
        return self.ica_dir / "mc" / "prefiltered_func_data_mcf.par"

    @property
    def feat_dir(self) -> Path:
        return self.dst_func_dir.parent / "func.feat"

    @property
    def filtered_func(self) -> Path:
        return self.feat_dir / "filtered_func_data.nii.gz"

    @property
    def filtered_func_clean(self) -> Path:
        return self.ica_dir / "filtered_func_data_clean.nii.gz"

    @property
    def example_func2standard_warp_redone(self) -> Path:
        return self.feat_dir / "reg" / "example_func2standard_warp.nii.gz"

    @property
    def example_func2standard_warp(self) -> Path:
        return self.ica_dir / "reg" / "example_func2standard_warp.nii.gz"

    @property
    def example_func2highres_redone(self) -> Path:
        return self.feat_dir / "reg" / "example_func2highres.mat"

    @property
    def mask(self) -> Path:
        return self.ica_dir / "mask.nii.gz"

    @property
    def mask_redone(self) -> Path:
        return self.feat_dir / "mask.nii.gz"

    def update_fsf(self, src: Path, dst: Path):
        template = src.read_text()
        design = (
            template.replace("@FSLDIR@", FSLDIR)
            .replace("@OUTPUTDIR@", str(self.dst_func_dir))
            .replace("@FEAT_FILES@", str(self.rfmri.dst))
            .replace("@ALT_EX_FUNC@", str(self.sbref.dst))
            .replace("@UNWARP_FILES@", str(self.fieldmap.dst))
            .replace("@UNWARP_FILES_MAG@", str(self.t1_brain.dst))
            .replace("@HIGHRES_FILES@", str(self.t1_brain.dst))
            .replace("@TR@", str(self.tr))
            .replace("@NPTS@", str(self.n_tr))
            .replace("@DWELL@", str(self.desc.dwell))
            .replace("@TE@", str(self.desc.TE))
        )
        dst.write_text(design)


def stdize_unmask_downsample_write(
    nii: nb.nifti1.Nifti1Image, dst: Path, header_ref: Path, target: Ref, warp: Path
):
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)

    nii_with_header: nb.nifti1.Nifti1Image = image.new_img_like(
        ref_niimg=header_ref, data=nii.get_fdata()
    )  # type:ignore

    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf_mni:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tf:
            nii_with_header.to_filename(tf.name)

            # applying the warp also downsamples
            wrappers.applywarp(
                src=tf.name, ref=REF / target.value, out=tf_mni.name, warp=warp
            )

        # final zscore so that voxelwise timeseries can be knit together
        resampled = nb.nifti1.Nifti1Image.load(tf_mni.name)
        image.new_img_like(
            resampled,
            np.nan_to_num(stats.zscore(resampled.get_fdata(), axis=-1, ddof=1)),  # type: ignore
        ).to_filename(dst)  # type: ignore


def get_motion(motion: Path, tr: float, filter: bool = True) -> pl.DataFrame:
    d = pl.DataFrame(
        pd.read_csv(
            motion,
            sep=r"\s+",
            header=None,
            names=["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"],
        )
    )
    if filter:
        filtered = {}
        for col in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]:
            filtered[col] = utils.filter_band_stop(
                d.select(col).to_series(),
                tr=tr,
                bw=(utils.BAND_UPPER - utils.BAND_LOWER),
                w0=(utils.BAND_UPPER + utils.BAND_LOWER) / 2,
            )
        d = pl.DataFrame(filtered)

    return utils.expand_motion(d)


def get_spatial_confounds(ukb_sub: UKBSub) -> pl.DataFrame:
    return utils.get_spatial_confounds(
        src=ukb_sub.filtered_func,
        wholebrain=ukb_sub.mask_redone,
        wm=ukb_sub.example_func_wmseg,
        csf=ukb_sub.example_func_csfseg,
    )


def get_confounds(ukb_sub: UKBSub) -> pl.DataFrame:
    motion = get_motion(ukb_sub.motion, tr=ukb_sub.tr)
    spatial_confounds = get_spatial_confounds(ukb_sub=ukb_sub)

    return utils.finalize_confounds(
        motion=motion,
        spatial_confounds=spatial_confounds,
        dst=ukb_sub.dst_root.parent
        / "ukb_confounds"
        / f"sub={ukb_sub.sub}"
        / f"ses={ukb_sub.ses}"
        / f"task={ukb_sub.task}"
        / "data.parquet",
    )


def transform_t1_seg_to_example_func(src: Path, mat: nt.linear.Affine, dst: Path):
    seg: nb.nifti1.Nifti1Image = image.binarize_img(
        nt.resampling.apply(transform=mat, spatialimage=src), threshold=0.5
    )  # type:ignore

    seg_eroded = ndimage.binary_erosion(
        np.asarray(seg.get_fdata(), dtype=np.bool), iterations=2
    )
    image.new_img_like(seg, seg_eroded, affine=seg.affine).to_filename(  # type:ignore
        dst
    )


def prep(sub: str, dst_root: Path, design_src: Path = Path("design.fsf")) -> UKBSub:
    """Redo preprocessing to generate un-fix'd fmri

    Args:
        sub (str): _description_
    """
    ukb_sub = UKBSub(sub=sub, dst_root=dst_root)

    if ukb_sub.filtered_func.exists():
        logging.info("feat outputs exist, skipping")
        return ukb_sub
    elif not ukb_sub.t1_brain.src.exists():
        logging.info("T1 brain not found, unable to proceed")
        sys.exit(0)
    elif ukb_sub.dst_func_dir.exists():
        logging.info("some outputs exist, but not filtered_func_data. cleaning")
        shutil.rmtree(ukb_sub.dst_func_dir.parent)

    if not (parent := ukb_sub.dst_func_dir).exists():
        parent.mkdir(parents=True)

    ukb_sub.rfmri.dst.symlink_to(ukb_sub.rfmri.src)
    ukb_sub.sbref.dst.symlink_to(ukb_sub.sbref.src)
    ukb_sub.fieldmap.dst.symlink_to(ukb_sub.fieldmap.src)
    ukb_sub.t1_brain.dst.symlink_to(ukb_sub.t1_brain.src)
    ukb_sub.t1_2_mni_linear.dst.symlink_to(ukb_sub.t1_2_mni_linear.src)
    ukb_sub.t1_wmseg.dst.symlink_to(ukb_sub.t1_wmseg.src)

    # need to get a T1_head that has the same FOV as T1_brain.nii.gz
    image.resample_to_img(
        ukb_sub.t1_orig, target_img=ukb_sub.t1_brain.src, interpolation="nearest"
    ).to_filename(ukb_sub.t1)  # type:ignore

    # need to make warp from coefs
    subprocess.run(
        [
            "fnirtfileutils",
            "-i",
            str(ukb_sub.t1_2_mni_coef),
            "-r",
            f"{FSLDIR}/data/standard/MNI152_T1_2mm_brain",
            "-o",
            str(ukb_sub.t1_2_standard_warp),
            "--withaff",
        ]
    )

    design_out = ukb_sub.dst_func_dir / "design.fsf"
    ukb_sub.update_fsf(design_src, design_out)

    wrappers.feat(str(design_out.resolve()))

    # and now resample spatial masks for use with abcd preprocessing
    # note we're using the newly created transformations
    example_func2highres = nt.linear.load(
        ukb_sub.example_func2highres_redone,
        fmt="fsl",
        reference=ukb_sub.t1_brain.src,
        moving=ukb_sub.filtered_func,
    )
    highres2example_func0 = example_func2highres.__invert__()
    highres2example_func = nt.Affine(
        highres2example_func0.matrix, reference=ukb_sub.filtered_func
    )
    transform_t1_seg_to_example_func(
        src=ukb_sub.t1_pve_wm,
        mat=highres2example_func,
        dst=ukb_sub.example_func_wmseg,
    )
    transform_t1_seg_to_example_func(
        src=ukb_sub.t1_pve_csf,
        mat=highres2example_func,
        dst=ukb_sub.example_func_csfseg,
    )
    return ukb_sub


def main(sub: str, dst_root: Path = DST_ROOT, ref: Ref = Ref.mm3):
    logging.info(f"checking feat for {sub=}")
    ukb_sub = prep(sub=sub, dst_root=dst_root)

    if ref == Ref.mm3:
        smooth = utils.FWHM
        res = ""
        mask2 = REF / "MNI152_T1_3mm_brain_mask_ero.nii"
    else:
        smooth = None
        res = "_res-ukbmm"
        mask2 = REF / "MNI152_T1_ukbmm_brain_mask_ero.nii"

    # smoothing will be done later (after final round of censoring)
    masker_abcd = utils.get_masker_abcd(
        mask=ukb_sub.mask_redone, tr=ukb_sub.tr, smoothing_fwhm=None
    )
    masker_hcp = utils.get_masker_hcp(mask=ukb_sub.mask, smoothing_fwhm=smooth)

    dst_abcd = (
        ukb_sub.dst_func_dir
        / f"sub-{ukb_sub.sub}_ses-{ukb_sub.ses}_task-{ukb_sub.task}_space-MNI152NLin6Asym{res}_desc-abcd_bold.nii.gz"
    )
    dst_hcp = (
        ukb_sub.dst_func_dir
        / f"sub-{ukb_sub.sub}_ses-{ukb_sub.ses}_task-{ukb_sub.task}_space-MNI152NLin6Asym{res}_desc-hcp_bold.nii.gz"
    )

    if not (dst_abcd.exists() and dst_hcp.exists()):
        logging.info(f"making counfounds for {ukb_sub}")
        confounds = get_confounds(ukb_sub)
        n_passing_second_filter = (
            confounds.select("sample_mask_second").sum().to_series().to_list()[0]
        )
        if n_passing_second_filter < 50:
            logging.warning(f"Only {n_passing_second_filter} TRs. Skipping")
            return

        logging.info("ABCD cleaning")
        arr = masker_abcd.transform(
            ukb_sub.filtered_func,
            confounds=confounds.select(
                pl.selectors.contains("trans"),
                pl.selectors.contains("rot"),
                pl.selectors.contains("white"),
                pl.selectors.contains("brain"),
                pl.selectors.contains("csf"),
            ).to_numpy(),
            sample_mask=confounds.select("sample_mask_first").to_series().to_numpy(),
        )
        logging.info("ABCD cleaning2")
        nii = utils.do_additional_abcd_mask(
            arr, masker=masker_abcd, confounds=confounds, smoothing_fwhm=smooth
        )

        # NOTE: for transform to MNI, need to be careful about which warp is used
        # the _clean (fix'd) data did motion correction with a func->t1 that included
        # GDC, but the abcd (redone) one did not
        logging.info("ABCD standardizing and downsampling")
        stdize_unmask_downsample_write(
            nii=nii,
            dst=dst_abcd,
            header_ref=ukb_sub.filtered_func,
            target=ref,
            warp=ukb_sub.example_func2standard_warp_redone,
        )

        # do the same with the fix'd data
        logging.info("HCP cleaning")
        arr_hcp = masker_hcp.transform(ukb_sub.filtered_func_clean)

        logging.info("HCP standardizing and downsampling")
        stdize_unmask_downsample_write(
            nii=masker_hcp.inverse_transform(arr_hcp),  # type:ignore
            dst=dst_hcp,
            header_ref=ukb_sub.filtered_func_clean,
            target=ref,
            warp=ukb_sub.example_func2standard_warp,
        )

    else:
        logging.info(f"{dst_abcd} and {dst_hcp} exists, skipping")

    # and now estimate components!
    for method in ["hcp", "abcd"]:
        dst_tc = (
            dst_root.parent
            / "tc"
            / "data=ukb"
            / f"method={method}"
            / f"reference={Path(ref.value).stem}"
            / f"sub={sub}"
            / f"ses={ukb_sub.ses}"
            / "data.parquet"
        )
        dst_ic = (
            dst_root.parent
            / "ic"
            / "data=ukb"
            / f"sub={sub}"
            / f"ses={ukb_sub.ses}"
            / f"reference-{Path(ref.value).stem}_method-{method}.nii.gz"
        )
        if dst_tc.exists() and dst_ic.exists():
            logging.info(f"found ICA for {method}, skipping")
            return

        logging.info(f"running ICA for {method}")
        utils.do_ica(
            src=dst_abcd if method == "abcd" else dst_hcp,
            mask=mask2,
            references=REF / ref.value,
            dst_tc=dst_tc,
            dst_ic=dst_ic,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int)
    parser.add_argument("--root", type=Path, default=SRC_ROOT, required=False)
    parser.add_argument(
        "--references",
        default="Neuromark_fMRI_1.0_resampled.nii",
        choices=[x.value for x in list(Ref)],
        required=False,
    )

    args = parser.parse_args()
    root: Path = args.root
    all_subs = sorted([x.name for x in root.glob("*") if x.is_dir()])

    main(sub=all_subs[args.idx].removeprefix("sub-"), ref=Ref(args.references))
