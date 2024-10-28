from typing import Any

import numpy as np
import polars as pl
import ruptures as rpt
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv


def _pl_partition(e: pl.Expr, pen: float = 11) -> pl.Expr:
    def _inner(signal: pl.Series) -> np.ndarray:
        if len(signal) < 2:
            return np.repeat(0, len(signal))

        algo = rpt.Pelt(model="rbf").fit(signal.to_numpy())
        result = algo.predict(pen)
        return np.concat([np.repeat(i, s) for i, s in enumerate(np.diff([0, *result]))])

    return e.map_batches(_inner).cast(pl.UInt16)


def _partition_sales(sales: pl.LazyFrame) -> pl.LazyFrame:
    partition_expr = pl.col("sales").pipe(_pl_partition).over("id")
    return sales.select("id", "date", partition_expr.alias("partition"))


def calculate_cv_partitions(sales: pl.LazyFrame, cv: dict[str, Any]) -> pl.DataFrame:
    partitions: list[pl.DataFrame] = []
    for cv_sales, _ in tqdm(list(split_cv(sales, **cv))):
        cutoff_expr = pl.col("date").max().over("id").dt.offset_by("1w")
        p = _partition_sales(cv_sales).with_columns(cutoff_expr.alias("cutoff_date"))
        partitions.append(p.collect())
    return pl.concat(partitions)


def calculate_live_partition(sales: pl.LazyFrame) -> pl.LazyFrame:
    return _partition_sales(sales)
