import typing
import warnings

import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv
from vn1_sales_forecast.settings import PRED_PREFIX

if typing.TYPE_CHECKING:
    from statsforecast import StatsForecast


def create_model() -> "StatsForecast":
    from statsforecast import StatsForecast
    from statsforecast.models import (
        IMAPA,
        AutoETS,
        AutoMFLES,
        CrostonOptimized,
        DynamicOptimizedTheta,
        HistoricAverage,
        OptimizedTheta,
        SeasonalNaive,
        SimpleExponentialSmoothingOptimized,
        WindowAverage,
        ZeroModel,
    )

    models = [
        # naive
        ZeroModel(),
        WindowAverage(window_size=13),
        SeasonalNaive(season_length=52),
        # intermitten
        CrostonOptimized(),
        IMAPA(),
        # exponential smoothing
        AutoETS(season_length=52),
        SimpleExponentialSmoothingOptimized(),
        # theta
        OptimizedTheta(season_length=52),
        DynamicOptimizedTheta(season_length=52),
        # tree
        AutoMFLES(test_size=13, season_length=52, metric="mae"),
    ]

    return StatsForecast(
        models=models,
        fallback_model=HistoricAverage(),
        freq="1w",
        n_jobs=-1,
        verbose=True,
    )


def _fit_predict(model: "StatsForecast", df: pl.LazyFrame) -> pl.DataFrame:
    # Fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model.fit(
            df.select("id", "date", "sales").collect(),  # type: ignore
            id_col="id",
            time_col="date",
            target_col="sales",
        )

    # Predict
    p: pl.DataFrame = model.predict(h=13)  # type: ignore

    # Add id and date columns
    p = p.select(
        pl.col("id", "date"),
        cs.exclude("id", "date").fill_nan(0).clip(0).name.prefix(PRED_PREFIX),
    )
    return p


def cross_validate(
    model: "StatsForecast", train: pl.LazyFrame, cv: dict[str, typing.Any]
) -> pl.DataFrame:
    preds: list[pl.DataFrame] = []
    for cv_train, _ in tqdm(list(split_cv(train, **cv))):
        p = _fit_predict(model, cv_train)
        p = p.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(p)

    return pl.concat(preds)


def live_forecast(model: "StatsForecast", train: pl.LazyFrame) -> pl.DataFrame:
    return _fit_predict(model, train)
