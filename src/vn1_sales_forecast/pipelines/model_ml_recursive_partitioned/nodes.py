import typing
from typing import Any

import lightgbm as lgb
import polars as pl
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMax,
    RollingMean,
    RollingMin,
    RollingQuantile,
    RollingStd,
)
from mlforecast.target_transforms import LocalMinMaxScaler
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv
from vn1_sales_forecast.pipelines.model_ml_recursive.date_features import fourier_term
from vn1_sales_forecast.pipelines.model_ml_recursive.nodes import _fit_predict, _make_data

if typing.TYPE_CHECKING:
    from mlforecast import MLForecast


def create_model() -> "MLForecast":
    return MLForecast(
        models={
            "LGBMRegressorRecursivePartitioned": lgb.LGBMRegressor(
                verbose=0,
                n_estimators=120,
                reg_alpha=0,
                reg_lambda=0.005,
                num_leaves=100,
                colsample_bytree=0.54,
                objective="l2",
                seed=42,
            ),
        },  # type: ignore
        freq="1w",
        lags=[1, 2, 3, 6, 13, 26, 52],
        date_features=[
            "month",
            *[fourier_term(b, k) for b in [True, False] for k in [1, 2, 4]],
        ],
        target_transforms=[LocalMinMaxScaler()],
        lag_transforms={
            1: [
                *[RollingMean(w, w // 4) for w in [13, 52]],
                *[RollingMin(w, w // 4) for w in [52]],
                *[RollingMax(w, w // 4) for w in [52]],
                *[RollingStd(w, w // 4) for w in [52]],
                *[RollingQuantile(q, w, w // 4) for w in [52] for q in [0.25, 0.5, 0.75]],
            ],
            52: [RollingMean(min_samples=1, window_size=52)],
        },  # type: ignore
    )


def cross_validate(
    models: "MLForecast",
    sales: pl.LazyFrame,
    cv_partitions: pl.LazyFrame,
    cv: dict[str, Any],
) -> pl.DataFrame:
    preds: list[pl.DataFrame] = []
    for train, _ in tqdm(list(split_cv(sales, **cv))):
        cutoff_expr = pl.col("date").max().over("id").dt.offset_by("1w")
        _train = (
            train.with_columns(cutoff_expr.alias("cutoff_date"))
            .join(cv_partitions, on=["id", "date", "cutoff_date"])
            .filter(pl.col("partition") == pl.col("partition").max().over("id"))
            .sort("id", "date", "cutoff_date")
            .drop(["cutoff_date", "partition"])
        )

        df, df_future = _make_data(_train)
        p = _fit_predict(models, df, df_future)
        p = p.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(p)
    return pl.concat(preds)


def live_forecast(
    models: "MLForecast",
    sales: pl.LazyFrame,
    live_partitions: pl.LazyFrame,
) -> pl.DataFrame:
    _sales = (
        sales.join(live_partitions, on=["id", "date"])
        .filter(pl.col("partition") == pl.col("partition").max().over("id"))
        .sort("id", "date")
        .drop("partition")
    )

    df, df_future = _make_data(_sales)
    p = _fit_predict(models, df, df_future)
    return p
