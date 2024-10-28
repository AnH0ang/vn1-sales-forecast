import polars as pl

from vn1_sales_forecast.cv import split_cv_loo
from vn1_sales_forecast.settings import PRED_PREFIX

CLASS_MODEL_MATCHING = {
    "all_zero": "ZeroModel",
    "trailing_zero": "ZeroModel",
    "regular": "LGBMRegressorRecursive",
    "small_sales": "LGBMRegressorRecursive",
    "sparse": "DynamicOptimizedTheta",
    "seasonal": "SeasonalNaive",
}


def _class_pred_expr(cls_model_matching: dict[str, str]) -> pl.Expr:
    pred_expr = pl.lit(None)
    for cls, model in cls_model_matching.items():
        pred_expr = (
            pl.when(pl.col("class") == cls).then(pl.col(PRED_PREFIX + model)).otherwise(pred_expr)
        )
    return pred_expr


def _calc_classification_pred(forecast: pl.LazyFrame, classification: pl.LazyFrame) -> pl.LazyFrame:
    return forecast.join(classification, on=["id"]).select(
        "id",
        "date",
        _class_pred_expr(CLASS_MODEL_MATCHING).alias(f"{PRED_PREFIX}ClassificationEnsemble"),
    )


def cross_validation(cv_forecast: pl.LazyFrame, cv_classification: pl.LazyFrame) -> pl.LazyFrame:
    preds = []
    for _, forecast, _, classification in split_cv_loo(cv_forecast, cv_classification):
        e = _calc_classification_pred(forecast, classification)
        e = e.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(e)
    return pl.concat(preds)


def live_forecast(live_forecast: pl.LazyFrame, live_classification: pl.LazyFrame) -> pl.LazyFrame:
    return _calc_classification_pred(live_forecast, live_classification)
