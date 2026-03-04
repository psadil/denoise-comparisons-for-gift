from pathlib import Path

import polars as pl

DST_ROOT0 = Path("derivatives") / "connectivity0"
DST_ROOT = Path("derivatives") / "connectivity"

n_tr = (
    pl.scan_parquet("derivatives/tc")
    .filter(
        pl.col("data") == "ukb",
        pl.col("component") == 1,
        pl.col("method") == "abcd",
        pl.col("ses") == 2,
    )
    .group_by(["sub", "ses", "reference"])
    .agg(n=pl.col("t").count())
    .filter(pl.col("n") > 106)
)

components = (
    pl.scan_parquet("derivatives/tc")
    .select("component")
    .unique()
    .collect()
    .sort("component")
    .to_series()
)
print(f"{components=}")
methods = ["abcd", "hcp"]
subs = (
    pl.scan_parquet("derivatives/tc")
    .select("sub")
    .unique()
    .collect()
    .to_series()
    .to_list()
)
print(f"{subs=}")
references = ["res-native", "smooth"]


ld = (
    pl.scan_parquet("derivatives/tc")
    .join(n_tr, how="semi", on=["sub", "ses", "reference"])
    .select("t", "component", "value", "method", "sub", "reference", "ses")
    .with_columns(
        pl.when(pl.col("reference") == "Neuromark_fMRI_1.0_resampled24")
        .then(pl.lit("res-native"))
        .otherwise(pl.lit("smooth"))
        .alias("reference")
    )
    .pivot(
        on="component",
        on_columns=components,
        index=["sub", "ses", "reference", "method", "t"],
    )
    .drop("t")
)

for sub in subs:
    print(f"{sub=}")
    for method in methods:
        for reference in references:
            dst = (
                DST_ROOT0
                / f"method={method}"
                / f"reference={reference}"
                / f"sub={sub}"
                / "ses=2"
                / "0.parquet"
            )
            if dst.exists():
                continue
            (
                ld.filter(
                    pl.col("sub") == sub,
                    pl.col("method") == method,
                    pl.col("reference") == reference,
                )
                .drop("sub", "ses", "reference", "method")
                .collect()
                .corr()
                .lazy()
                .with_row_index("src")
                .sink_parquet(dst, mkdir=True)
            )

on_columns = []
for s in components:
    for d in components:
        if d > s:
            on_columns.append(f"{s}_{d}")


pl.scan_parquet(DST_ROOT0).unpivot(
    index=["method", "reference", "sub", "ses", "src"],
    variable_name="dst",
    value_name="connectivity",
).with_columns(pl.col("dst").cast(pl.UInt32)).filter(
    pl.col("src") < pl.col("dst")
).with_columns(
    pl.col("connectivity").arctanh(),
    feature=pl.concat_str([pl.col("src"), pl.col("dst")], separator="_"),
).drop("src", "dst", "ses").pivot(
    on="feature", on_columns=on_columns, index=["method", "reference", "sub"]
).sink_parquet(pl.PartitionByKey(DST_ROOT, by=["method", "reference"]))

