import polars as pl

from vn1_sales_forecast.cv import split_cv_loo
from vn1_sales_forecast.settings import PRED_PREFIX
from vn1_sales_forecast.utils import multi_join


def _calc_ensemble(classification: pl.LazyFrame, *forecast: pl.LazyFrame) -> pl.LazyFrame:
    ID_MODEL_MAP = {
        "0-3-1154": "SeasonalNaive",
        "0-3-12086": "SeasonalNaive",
        "16-140-6568": "SeasonalNaive",
        "18-177-4168": "SeasonalNaive",
        "22-265-6181": "SeasonalNaive",
        "16-99-1103": "SeasonalNaive",
        "21-136-7572": "ZeroModel",
        "0-234-9927": "ZeroModel",
        "0-242-10424": "ZeroModel",
        "0-3-2838": "ZeroModel",
        "0-3-9055": "ZeroModel",
        "18-177-3443": "ZeroModel",
        "21-135-10902": "ZeroModel",
        "21-136-12541": "ZeroModel",
        "21-136-9866": "ZeroModel",
        "23-333-2827": "ZeroModel",
        "23-333-573": "ZeroModel",
        "24-226-1884": "ZeroModel",
        "40-18-4780": "ZeroModel",
        "41-212-7502": "ZeroModel",
        "42-229-2181": "ZeroModel",
        "21-113-9462": "ZeroModel",
        "22-265-3780": "ZeroModel",
        "23-333-5009": "ZeroModel",
    }

    CLASS_MODEL_MAP = {
        "all_zero": "ZeroModel",
        "trailing_zero": "ZeroModel",
        "seasonal": "SeasonalNaive",
    }

    CANDIDATE_MODELS = [
        "OptimizedWeightsEnsemble",
    ]

    def _make_divine(e: pl.Expr) -> pl.Expr:
        # class model map
        for cls, model in CLASS_MODEL_MAP.items():
            e = pl.when(pl.col("class") == cls).then(pl.col(PRED_PREFIX + model)).otherwise(e)

        # id model map
        for id, model in ID_MODEL_MAP.items():
            e = pl.when(pl.col("id") == id).then(PRED_PREFIX + model).otherwise(e)

        return e

    total = multi_join(*forecast, on=["id", "date"]).join(classification, on=["id"])
    return total.select(
        "id",
        "date",
        *(
            pl.col(PRED_PREFIX + m).pipe(_make_divine).alias(PRED_PREFIX + "Divine" + m)
            for m in CANDIDATE_MODELS
        ),
    )


def cross_validate(
    model_cv_forecast: pl.LazyFrame,
    ensemble_cv_forecast: pl.LazyFrame,
    cv_classification: pl.LazyFrame,
) -> pl.LazyFrame:
    preds = []

    splits = split_cv_loo(model_cv_forecast, ensemble_cv_forecast, cv_classification)
    for _, model_forecast, _, ensemble_forecast, _, classfication in splits:
        e = _calc_ensemble(classfication, model_forecast, ensemble_forecast)
        e = e.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(e)
    return pl.concat(preds)


def live_forecast(
    model_live_forecast: pl.LazyFrame,
    ensemble_live_forecast: pl.LazyFrame,
    live_classification: pl.LazyFrame,
) -> pl.LazyFrame:
    return _calc_ensemble(live_classification, model_live_forecast, ensemble_live_forecast)
