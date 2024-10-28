import numpy as np
import polars as pl
import polars.selectors as cs
import xgboost as xgb
from scipy.special import softmax
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv_loo
from vn1_sales_forecast.expr import competition_score
from vn1_sales_forecast.settings import PRED_PREFIX, TSFEATURES_PREFIX
from vn1_sales_forecast.utils import multi_join

KS = [None]


def calculate_cv_scores(cv_forecast: pl.LazyFrame, sales: pl.LazyFrame) -> pl.LazyFrame:
    df = cv_forecast.join(sales, on=["id", "date"], how="left")
    score_expr = competition_score(cs.starts_with(PRED_PREFIX), pl.col("sales"))
    scores = df.group_by("id", "cutoff_date").agg(score_expr)

    # NOTE: Clip nan scores to prevent infinite competition score
    scores = scores.with_columns(cs.starts_with(PRED_PREFIX).clip(0, 1))
    return scores


def _error_softmax_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
    fc_errors = dtrain.get_label().reshape(predt.shape)
    weights = softmax(predt, axis=1)
    grad = weights * (fc_errors - fc_errors.mean(axis=1, keepdims=True))
    hessian = fc_errors * weights * (1 - weights) - grad * weights
    return grad, hessian


def _fit_predict_single_fforma(
    train_tsfeat: pl.LazyFrame,
    train_scores: pl.LazyFrame,
    val_tsfeat: pl.LazyFrame,
    val_fcst: pl.LazyFrame,
    name: str = "FFORMA",
) -> pl.LazyFrame:
    # Join Kernel
    train_kernel = train_tsfeat.join(train_scores, on=["id", "cutoff_date"])
    train_kernel = train_kernel.with_columns(cs.starts_with(PRED_PREFIX).fill_nan(1))
    train_kernel = train_kernel.sort("cutoff_date", "id")

    # Make DMatrix
    X_train = train_kernel.select(cs.starts_with(TSFEATURES_PREFIX)).collect()
    y_train = train_kernel.select(cs.starts_with(PRED_PREFIX)).collect()
    d_train = xgb.DMatrix(X_train, y_train)

    X_val = val_tsfeat.select(cs.starts_with(TSFEATURES_PREFIX)).collect()
    d_val = xgb.DMatrix(X_val)

    # Fit and Predict FFORMA Model
    model = xgb.train(
        params={"seed": 42},
        dtrain=d_train,
        obj=_error_softmax_obj,
        num_boost_round=100,
    )
    prediction_val = model.predict(d_val)

    # Calculate Weights
    w = softmax(prediction_val, axis=1)
    weights_val = val_tsfeat.select("id").with_columns(pl.DataFrame(w, schema=y_train.schema))

    # Multiply Predictions with Weights
    forecast_long = val_fcst.unpivot(index=["id", "date"], variable_name="model", value_name="pred")
    weights_long = weights_val.unpivot(index="id", variable_name="model", value_name="weight")
    return (
        weights_long.join(forecast_long, on=["id", "model"])
        .group_by(["id", "date"])
        .agg((pl.col("pred") * pl.col("weight")).sum().alias(f"{PRED_PREFIX}{name}"))
    )


def _fit_predict(
    train_tsfeatures: pl.LazyFrame,
    train_scores: pl.LazyFrame,
    val_tsfeatures: pl.LazyFrame,
    val_forecast: pl.LazyFrame,
    total_scores: pl.LazyFrame,
) -> pl.LazyFrame:
    # Sort models by score
    model_list = total_scores.sort("score").collect()["model"].to_list()

    preds: list[pl.LazyFrame] = []
    for k in KS:
        used_models = model_list if k is None else model_list[:k]
        pred_cols = pl.col([f"{PRED_PREFIX}{m}" for m in used_models])

        _train_scores = train_scores.select(pl.col("id", "cutoff_date"), pred_cols)
        _val_forecast = val_forecast.select(pl.col("id", "date"), pred_cols)

        name = "FFORMA" + ("" if k is None else f"Top{k}") + "Ensemble"
        p = _fit_predict_single_fforma(
            train_tsfeatures,
            _train_scores,
            val_tsfeatures,
            _val_forecast,
            name,
        )
        preds.append(p)

    return multi_join(*preds, on=["id", "date"], how="inner")


def cross_validate(
    cv_forecast: pl.LazyFrame,
    cv_tsfeatures: pl.LazyFrame,
    cv_scores: pl.LazyFrame,
    total_scores: pl.LazyFrame,
) -> pl.LazyFrame:
    preds: list[pl.LazyFrame] = []

    splits = tqdm(list(split_cv_loo(cv_tsfeatures, cv_forecast, cv_scores)))
    for train_tsfeatures, val_tsfeatures, _, val_forecast, train_scores, _ in splits:
        p = _fit_predict(train_tsfeatures, train_scores, val_tsfeatures, val_forecast, total_scores)
        p = p.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(p)

    return pl.concat(preds)


def live_forecast(
    cv_tsfeatures: pl.LazyFrame,
    cv_scores: pl.LazyFrame,
    live_forecast: pl.LazyFrame,
    live_tsfeatures: pl.LazyFrame,
    total_scores: pl.LazyFrame,
) -> pl.LazyFrame:
    return _fit_predict(cv_tsfeatures, cv_scores, live_tsfeatures, live_forecast, total_scores)
