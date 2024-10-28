from typing import Any

import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv
from vn1_sales_forecast.settings import PRED_PREFIX

from .timesfm import TimesFM


def create_model() -> TimesFM:
    return TimesFM(
        h=13,
        freq="W",
        backend="gpu",
    )


def cross_validate(
    models: TimesFM,
    sales: pl.LazyFrame,
    cv: dict[str, Any],
) -> pl.DataFrame:
    preds: list[pl.DataFrame] = []
    for train, _ in tqdm(list(split_cv(sales, **cv))):
        p = models.forecast(
            train,
            id_col="id",
            time_col="date",
            target_col="sales",
        )
        p = p.select(
            pl.col("id", "date"),
            cs.exclude("id", "date").fill_nan(0).clip(0).name.prefix(PRED_PREFIX),
        )
        p = p.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(p)
    return pl.concat(preds)


def live_forecast(models: TimesFM, sales: pl.LazyFrame) -> pl.DataFrame:
    p = models.forecast(
        sales,
        id_col="id",
        time_col="date",
        target_col="sales",
    )
    p = p.select(
        pl.col("id", "date"),
        cs.exclude("id", "date").fill_nan(0).clip(0).name.prefix(PRED_PREFIX),
    )
    return p
