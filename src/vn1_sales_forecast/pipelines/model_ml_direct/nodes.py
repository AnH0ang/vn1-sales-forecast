import typing
from typing import Any

import lightgbm as lgb
import polars as pl
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv
from vn1_sales_forecast.pipelines.model_ml_recursive.nodes import _fit_predict

if typing.TYPE_CHECKING:
    from mlforecast import MLForecast


def _make_data(sales: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
    df = sales.sort("id", "date").select(
        "id",
        "date",
        "sales",
        pl.col("price").fill_null(strategy="forward").mean().over("id").alias("price_group"),
    )
    return df, None


def create_model() -> "MLForecast":
    return MLForecast(
        models={
            "LGBMRegressorDirect": lgb.LGBMRegressor(
                verbose=0,
                n_estimators=194,
                reg_alpha=0.017,
                reg_lambda=0.003,
                num_leaves=230,
                colsample_bytree=0.669,
                objective="l2",
                seed=42,
            ),
        },  # type: ignore
        freq="1w",
        lags=range(4, 53, 4),
        date_features=["month"],
        lag_transforms={
            1: [ExponentiallyWeightedMean(alpha=0.9)],
            52: [RollingMean(min_samples=1, window_size=52)],
        },  # type: ignore
    )


def cross_validate(
    models: "MLForecast",
    sales: pl.LazyFrame,
    cv: dict[str, Any],
) -> pl.DataFrame:
    preds: list[pl.DataFrame] = []
    for train, _ in tqdm(list(split_cv(sales, **cv))):
        df, df_future = _make_data(train)
        p = _fit_predict(models, df, df_future, direct=True)
        p = p.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(p)
    return pl.concat(preds)


def live_forecast(models: "MLForecast", sales: pl.LazyFrame) -> pl.DataFrame:
    df, df_future = _make_data(sales)
    p = _fit_predict(models, df, df_future, direct=True)
    return p
