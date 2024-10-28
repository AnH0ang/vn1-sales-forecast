from typing import Any

import polars as pl
import polars.selectors as cs

from vn1_sales_forecast.cv import split_cv
from vn1_sales_forecast.settings import CLASS_PREFIX

from .expr import cls_all_zero, cls_seasonal, cls_small_sales, cls_sparse, cls_trailing_zero


def _classify_sales(sales: pl.LazyFrame, tsfeatures: pl.LazyFrame) -> pl.LazyFrame:
    # Classify sales
    class_splits = (
        sales.join(tsfeatures, on="id")
        .group_by("id")
        .agg(
            cls_all_zero().alias(f"{CLASS_PREFIX}all_zero"),
            cls_trailing_zero().alias(f"{CLASS_PREFIX}trailing_zero"),
            cls_sparse().alias(f"{CLASS_PREFIX}sparse"),
            cls_small_sales().alias(f"{CLASS_PREFIX}small_sales"),
            cls_seasonal().alias(f"{CLASS_PREFIX}seasonal"),
        )
    )

    # Add class column
    class_cols = class_splits.select(cs.starts_with(CLASS_PREFIX)).collect_schema().names()
    r = pl.lit(f"{CLASS_PREFIX}regular")
    for c in reversed(class_cols):
        r = pl.when(pl.col(c)).then(pl.lit(c)).otherwise(r)
    class_splits = class_splits.with_columns(r.alias("class").str.strip_prefix(CLASS_PREFIX))
    return class_splits


def calculate_cv_classification(
    sales: pl.LazyFrame, cv_tsfeatures: pl.LazyFrame, cv: dict[str, Any]
) -> pl.LazyFrame:
    class_dfs: list[pl.LazyFrame] = []
    for cv_train, _ in list(split_cv(sales, **cv)):
        cutoff_expr = pl.col("date").max().dt.offset_by("1w").alias("cutoff_date")
        cutoffs = cv_train.group_by("id").agg(cutoff_expr)

        tsfeatures = cv_tsfeatures.join(cutoffs, on=["id", "cutoff_date"]).drop("cutoff_date")

        c = _classify_sales(cv_train, tsfeatures)
        c = c.join(cutoffs, on="id")
        class_dfs.append(c)

    return pl.concat(class_dfs)


def calculate_live_classification(
    sales: pl.LazyFrame, live_tsfeatures: pl.LazyFrame
) -> pl.LazyFrame:
    return _classify_sales(sales, live_tsfeatures)
