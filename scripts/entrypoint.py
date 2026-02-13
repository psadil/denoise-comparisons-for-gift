import argparse
import logging
import typing
import os
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats._resampling import PermutationTestResult
from sklearn import (
    cross_decomposition,
    decomposition,
    feature_selection,
    linear_model,
    metrics,
    preprocessing,
    model_selection,
)
from sklearn import pipeline
import xgboost as xgb

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
DST_ROOT = Path(os.environ["MYSCRATCH"]) / "predictions"
CONNECTIVITY = Path("derivatives") / "connectivity"
UKB_Y = Path("derivatives") / "cognitive.parquet"


MODEL = typing.Literal["RIDGE_CV", "PCR_RIDGE", "LASSO", "PCR_LASSO", "PLSR", "XGB"]
REFERENCE = typing.Literal["smoothed", "res-native"]
METHOD = typing.Literal["abcd", "hcp"]


def cor_rank(x, y) -> float:
    if (x[0] == x).all() or (y[0] == y).all():
        # If an input is constant, the correlation coefficient
        # is not defined.
        return np.nan

    return stats.spearmanr(x, y).statistic  # type: ignore


def test_cor(y, y_hat, seed: int | None = None) -> PermutationTestResult:
    def statistic(x):  # permute only `x`
        return stats.spearmanr(x, y).statistic  # type: ignore

    return stats.permutation_test(
        (y_hat,), statistic, permutation_type="pairings", rng=seed
    )


def test_r2(y, y_hat, seed: int | None = None) -> PermutationTestResult:
    def statistic(x):  # permute only `x`
        return metrics.r2_score(y, x)

    return stats.permutation_test(
        (y_hat,), statistic, permutation_type="pairings", rng=seed
    )


def get_pipeline(m: MODEL) -> pipeline.Pipeline:
    match m:
        case "RIDGE_CV":
            model_pipeline = pipeline.make_pipeline(
                feature_selection.VarianceThreshold(0.01),  # not all regions in fov
                preprocessing.RobustScaler(),
                linear_model.RidgeCV(alphas=np.logspace(-1.0, 4.0, 20)),
            )
        case "LASSO":
            model_pipeline = pipeline.make_pipeline(
                feature_selection.VarianceThreshold(0.01),  # not all regions in fov
                preprocessing.RobustScaler(),
                linear_model.LassoCV(),
            )
        case "PCR_RIDGE":
            model_pipeline = pipeline.make_pipeline(
                feature_selection.VarianceThreshold(0.01),  # not all regions in fov
                preprocessing.RobustScaler(),
                decomposition.PCA(n_components=0.9),
                linear_model.RidgeCV(alphas=np.logspace(-1.0, 4.0, 20)),
            )
        case "PCR_LASSO":
            model_pipeline = pipeline.make_pipeline(
                feature_selection.VarianceThreshold(0.01),  # not all regions in fov
                preprocessing.RobustScaler(),
                decomposition.PCA(n_components=0.9),
                linear_model.LassoCV(),
            )
        case "PLSR":
            model_pipeline = pipeline.make_pipeline(
                feature_selection.VarianceThreshold(0.01),  # not all regions in fov
                cross_decomposition.PLSRegression(n_components=20),
            )
        case "XGB":
            model_pipeline = pipeline.make_pipeline(
                feature_selection.VarianceThreshold(0.01),  # not all regions in fov
                xgb.XGBRegressor(),
            )
        case _:
            raise AssertionError("Unknown Model")
    return model_pipeline


def test_sample(
    d_test: pl.DataFrame, d_trainval: pl.DataFrame, seed: int, m: MODEL = "RIDGE_CV"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    model_pipeline = get_pipeline(m=m)
    model = model_pipeline.fit(d_trainval.drop("sub", "y"), d_trainval["y"])
    trainval_y_hat = pl.Series("y_hat", model.predict(d_trainval.drop("sub", "y")))

    test_y_hat = model.predict(d_test.drop("sub", "y"))

    result_r2 = test_r2(d_test["y"], test_y_hat, seed=seed)
    result_cor = test_cor(d_test["y"], test_y_hat, seed=seed)
    train_rank_cor = cor_rank(d_trainval["y"], trainval_y_hat)
    train_r2 = metrics.r2_score(d_trainval["y"], trainval_y_hat)
    results = (
        d_trainval.select("y", "sub")
        .with_columns(trainval_y_hat)
        .with_columns(
            test_rank_cor=result_cor.statistic,
            test_rank_cor_p=result_cor.pvalue,
            test_r2_p=result_r2.pvalue,
            test_r2=result_r2.statistic,
            test_mae=metrics.median_absolute_error(d_test["y"], test_y_hat),  # type: ignore
            train_rank_cor=train_rank_cor,
            train_r2=train_r2,
            train_mae=metrics.mean_absolute_error(d_trainval["y"], trainval_y_hat),
        )
    )

    logging.info(f"train cor: {train_rank_cor}")
    logging.info(f"train r2: {train_r2}")

    logging.info(f"test cor: {result_cor.statistic}")
    logging.info(f"test cor p: {result_cor.pvalue}")
    logging.info(f"test r2: {result_r2.statistic}")
    logging.info(f"test r2 p: {result_r2.pvalue}")

    test_predictions = d_test.select("sub", "y").with_columns(
        pl.Series(name="y_hat", values=test_y_hat, dtype=pl.Float32),
        pl.col("y").cast(pl.Float32),
    )
    match m:
        case "RIDGE_CV" | "LASSO":
            coef: np.ndarray = model[2].coef_
        case "PCR_RIDGE" | "PCR_LASSO":
            coef: np.ndarray = model[3].coef_
        case "PLSR":
            coef: np.ndarray = model[1].coef_
        case "XGB":
            logging.warning("saving dummy coefficients for model XGB")
            coef = np.zeros((1, 0))
        case _:
            raise AssertionError("Unknown Model")

    if not coef.dtype == np.float64:
        coef = np.asarray(coef, dtype=np.float64)
    if len(coef.shape) > 1:
        coef = coef.squeeze()

    return (results, test_predictions, pl.DataFrame({"coef": coef}).with_row_index())


def get_1_measure(m: int, src: Path = UKB_Y) -> str:
    ukb = pl.scan_parquet(src).head(1).collect()
    return ukb.columns[m]


def main(
    x_in: Path,
    out_dir: Path,
    m: int,
    references: typing.Sequence[REFERENCE] = typing.get_args(REFERENCE),
    methods: typing.Sequence[METHOD] = typing.get_args(METHOD),
    model: MODEL = "RIDGE_CV",
    n_outer_folds: int = 50,
    test_size: float = 0.2,
    y_in: Path = UKB_Y,
) -> None:
    """Coordinate cross-validation of model fit

    Args:
        x_in (Path): parquet file with features used for fit.
        out_dir (Path): directory in which outputs will be saved (folders: results, test-predictions, features)
        m (int): integer column id of feature in x_in to fit
        references (REFERENCE, optional): sequence of references to work with
        methods (METHOD, optional): sequence of methods to work with
        model (MODEL, optional): pipeline that will be constructed for model fits. see get_pipeline. Defaults to "RIDGE_CV".
        n_outer_folds (int, optional): Number of folds in outer cross-validation loop (pipelines expected to have inner cv). Defaults to 10.
        test_size (float, optional): Proportion of dataset left out on each shuffle fold. Defaults to 10.
        y_in (Path, optional): parquet file path with measures to predict. Defaults to UKB_Y.
    """
    measure = get_1_measure(m, src=y_in)

    y0 = (
        pl.scan_parquet(y_in).select("sub", measure).rename({measure: "y"}).drop_nulls()
    )
    logging.info(f"{measure=}")
    for reference in references:
        logging.info(f"{reference=}")
        for method in methods:
            logging.info(f"{method=}")
            final_parent = (
                out_dir
                / "features"
                / f"{model=}"
                / f"{measure=}"
                / f"{reference=}"
                / f"{method=}"
            )
            if final_parent.exists():
                logging.info("skipping. outputs exist")
                return

            d = (
                pl.scan_parquet(x_in, hive_partitioning=True)
                .filter(pl.col("method") == method, pl.col("reference") == reference)
                .drop("method", "reference")
                .join(y0, on="sub", how="inner")
                .collect()
            )

            # drop columns that have all nulls (VOLS atlas may have different
            # numbers of regions in parcellation)
            d = d[[s.name for s in d if not (s.null_count() == d.height)]]
            d = d.drop_nans()

            outer_cv = model_selection.ShuffleSplit(
                n_splits=n_outer_folds, test_size=test_size, random_state=0
            )
            for fold, (train_index, test_index) in enumerate(outer_cv.split(d)):  # type:ignore
                logging.info(f"{fold=}")
                d_test = d[test_index, :]
                d_trainval = d[train_index, :]

                results, test_predictions, features = test_sample(
                    d_test=d_test, d_trainval=d_trainval, seed=fold, m=model
                )

                results.lazy().sink_parquet(
                    out_dir
                    / "results"
                    / f"{model=}"
                    / f"{measure=}"
                    / f"{reference=}"
                    / f"{method=}"
                    / f"{fold=}"
                    / "part-0.parquet",
                    mkdir=True,
                )

                test_predictions.lazy().sink_parquet(
                    out_dir
                    / "test-predictions"
                    / f"{model=}"
                    / f"{measure=}"
                    / f"{reference=}"
                    / f"{method=}"
                    / f"{fold=}"
                    / "part-0.parquet",
                    mkdir=True,
                )

                features.lazy().sink_parquet(
                    final_parent / f"{fold=}" / "part-0.parquet", mkdir=True
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "measure",
        type=int,
        help="Integer (>=1) for which measure to predict. A good default is 360, which is f.20016.2.0 (fluid intellligence)",
    )
    parser.add_argument("--out", type=Path, default=DST_ROOT)
    parser.add_argument("--x", type=Path, default=CONNECTIVITY)
    parser.add_argument("--model", default="RIDGE_CV", choices=typing.get_args(MODEL))
    parser.add_argument("--n-outer-folds", default=50, type=int)

    args = parser.parse_args()

    main(
        x_in=args.x,
        out_dir=args.out,
        m=args.measure,
        model=args.model,
        n_outer_folds=args.n_outer_folds,
    )
