from pprint import pprint
from typing import Any

import polars as pl
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import Pipeline
from sklego.feature_selection import MaximumRelevanceMinimumRedundancy
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv_loo
from vn1_sales_forecast.settings import PRED_PREFIX

KS = [None]


def create_models() -> dict[str, Any]:
    return {
        "LassoLarsICPositive": LassoLarsIC(fit_intercept=False, positive=True),
        "LassoLarsICMrmr": Pipeline(
            [
                ("mrmr", MaximumRelevanceMinimumRedundancy(k=10)),
                ("lassolarsic", LassoLarsIC(fit_intercept=False)),
            ]
        ),
        "LassoLarsICPositiveMrmr": Pipeline(
            [
                ("mrmr", MaximumRelevanceMinimumRedundancy(k=10)),
                ("lassolarsic", LassoLarsIC(fit_intercept=False, positive=True)),
            ]
        ),
    }


def _fit_predict(
    models: dict[str, BaseEstimator],
    train_kernel: pl.LazyFrame,
    val_kernel: pl.LazyFrame,
    total_scores: pl.LazyFrame,
    cv: bool = True,
) -> pl.LazyFrame:
    # Sort models by score
    model_list = total_scores.sort("score").collect()["model"].to_list()

    # Fit models and predict with stacking models
    ps: list[pl.Series] = []
    for k in KS:
        used_models = model_list if k is None else model_list[:k]
        pred_cols = pl.col([f"{PRED_PREFIX}{m}" for m in used_models])

        X_train = train_kernel.select(pred_cols).collect()
        y_train = train_kernel.select("sales").collect().to_series()
        X_val = val_kernel.select(pred_cols).collect()

        for name, model in models.items():
            m = clone(model)
            m.fit(X_train, y_train)  # type: ignore
            p = m.predict(X_val)  # type: ignore

            if hasattr(m, "coef_"):
                pprint(dict(zip(X_train.columns, map(float, m.coef_))))  # type: ignore

            p_name = (
                f"{PRED_PREFIX}" f"Stacking{name}" f"{'' if k is None else f'Top{k}'}" "Ensemble"
            )
            ps.append(pl.Series(p_name, p).clip(0))

    # Add cutoff date to the output if cv is True
    return_cols = ["id", "date", *ps] + (["cutoff_date"] if cv else [])
    return val_kernel.select(*return_cols)


def cross_validate(
    models: dict[str, BaseEstimator],
    cv_pred: pl.LazyFrame,
    sales: pl.LazyFrame,
    total_scores: pl.LazyFrame,
) -> pl.LazyFrame:
    kernel = cv_pred.join(sales.select("id", "date", "sales"), on=["id", "date"], how="inner")
    kernel = kernel.sort("id", "date", "cutoff_date")

    preds: list[pl.LazyFrame] = []
    for train_kernel, val_kernel in tqdm(split_cv_loo(kernel)):
        p = _fit_predict(models, train_kernel, val_kernel, total_scores, cv=True)
        preds.append(p)
    ensemble_cv_pred = pl.concat(preds)

    return ensemble_cv_pred


def live_forecast(
    models: dict[str, BaseEstimator],
    cv_pred: pl.LazyFrame,
    sales: pl.LazyFrame,
    live_pred: pl.LazyFrame,
    total_scores: pl.LazyFrame,
) -> pl.LazyFrame:
    kernel = cv_pred.join(sales.select("id", "date", "sales"), on=["id", "date"], how="inner")
    kernel = kernel.sort("id", "date", "cutoff_date")

    live_preds = _fit_predict(models, kernel, live_pred, total_scores, cv=False)
    return live_preds
