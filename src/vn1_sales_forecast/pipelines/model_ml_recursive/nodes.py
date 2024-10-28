import typing
import warnings
from typing import Any

import lightgbm as lgb
import polars as pl
import polars.selectors as cs
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
from vn1_sales_forecast.settings import PRED_PREFIX

from .date_features import fourier_term

if typing.TYPE_CHECKING:
    from mlforecast import MLForecast


def _make_data(sales: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
    df = sales.sort("id", "date").select(
        "id",
        "date",
        "sales",
        pl.col("price").fill_null(strategy="forward").mean().over("id").alias("price_group"),
        pl.col("Client").cast(str).cast(pl.Categorical).alias("client_group"),
        pl.col("Warehouse").cast(str).cast(pl.Categorical).alias("warehouse_group"),
        pl.col("Product").cast(str).cast(pl.Categorical).alias("product_group"),
    )
    return df, None


def create_model() -> "MLForecast":
    return MLForecast(
        models={
            "LGBMRegressorRecursive": lgb.LGBMRegressor(
                verbose=0,
                n_estimators=120,
                reg_alpha=0,
                reg_lambda=0.005,
                num_leaves=100,
                colsample_bytree=0.54,
                objective="l2",
                seed=42,
            ),
            "LGBMRegressorMedianRecursive": lgb.LGBMRegressor(
                verbose=0,
                n_estimators=120,
                reg_alpha=0,
                reg_lambda=0.005,
                num_leaves=100,
                colsample_bytree=0.54,
                objective="l1",
                seed=42,
            ),
            "LGBMRegressorTweedieRecursive": lgb.LGBMRegressor(
                verbose=0,
                n_estimators=120,
                reg_alpha=0,
                reg_lambda=0.005,
                num_leaves=100,
                colsample_bytree=0.54,
                objective="tweedie",
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


def _fit_predict(
    model: "MLForecast",
    df: pl.LazyFrame,
    df_future: pl.LazyFrame | None,
    direct: bool = False,
) -> pl.DataFrame:
    # 1. Fit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        fit_kwargs = {
            "df": df.collect(),  # type: ignore
            "id_col": "id",
            "time_col": "date",
            "target_col": "sales",
            "dropna": True,
        }

        if direct:
            fit_kwargs["max_horizon"] = 13

        model.fit(**fit_kwargs)

    # 2. Predict
    p: pl.DataFrame = model.predict(
        h=13,
        new_df=df.collect(),  # type: ignore
        X_df=None if df_future is None else df_future.collect(),  # type: ignore
    )

    # 3. Add id and date columns
    p = p.select(
        pl.col("id", "date"),
        cs.exclude("id", "date").fill_nan(0).clip(0).name.prefix(PRED_PREFIX),
    )
    return p


def cross_validate(
    models: "MLForecast",
    sales: pl.LazyFrame,
    cv: dict[str, Any],
) -> pl.DataFrame:
    preds: list[pl.DataFrame] = []
    for train, _ in tqdm(list(split_cv(sales, **cv))):
        df, df_future = _make_data(train)
        p = _fit_predict(models, df, df_future)
        p = p.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(p)
    return pl.concat(preds)


def live_forecast(models: "MLForecast", sales: pl.LazyFrame, direct: bool = False) -> pl.DataFrame:
    df, df_future = _make_data(sales)
    p = _fit_predict(models, df, df_future)
    return p
