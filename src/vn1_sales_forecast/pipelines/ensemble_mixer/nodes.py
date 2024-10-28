import polars as pl

from vn1_sales_forecast.cv import split_cv_loo
from vn1_sales_forecast.settings import PRED_PREFIX


def _calc_ensemble(forecast: pl.LazyFrame) -> pl.LazyFrame:
    mixer_models = [
        ["LGBMRegressorRecursive", "LGBMRegressorDirect"],
    ]

    return forecast.select(
        "id",
        "date",
        *(
            pl.mean_horizontal(pl.col([f"{PRED_PREFIX}{m}" for m in models])).alias(
                f"{PRED_PREFIX}Mixer{''.join((m[:2]+m[-2:]).title() for m in models)}Ensemble"
            )
            for models in mixer_models
        ),
    )


def cross_validation(cv_forecast: pl.LazyFrame) -> pl.LazyFrame:
    preds = []
    for _, forecast in split_cv_loo(cv_forecast):
        e = _calc_ensemble(forecast)
        e = e.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(e)
    return pl.concat(preds)


def live_forecast(live_forecast: pl.LazyFrame) -> pl.LazyFrame:
    return _calc_ensemble(live_forecast)
