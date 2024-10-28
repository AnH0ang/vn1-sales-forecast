import typing
from typing import Any

import polars as pl
import polars.selectors as cs
from neuralforecast import NeuralForecast
from neuralforecast.models import KAN, NHITS
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv
from vn1_sales_forecast.pipelines.model_nn.losses import CustomLoss
from vn1_sales_forecast.settings import PRED_PREFIX

if typing.TYPE_CHECKING:
    from neuralforecast import NeuralForecast


def create_model() -> "NeuralForecast":
    models = [
        KAN(
            h=13,
            input_size=3 * 13,
            start_padding_enabled=True,
            random_seed=42,
        ),
        NHITS(
            h=13,
            input_size=3 * 13,
            max_steps=1_000,
            start_padding_enabled=True,
            random_seed=42,
        ),
        NHITS(
            h=13,
            input_size=3 * 13,
            loss=CustomLoss(),  # type: ignore
            max_steps=1_000,
            start_padding_enabled=True,
            random_seed=42,
            alias="NHITSCustom",
        ),
    ]
    return NeuralForecast(models, freq="1w")


def _fit_predict(
    model: "NeuralForecast", df: pl.LazyFrame, fit: bool = True
) -> tuple["NeuralForecast", pl.DataFrame]:
    train = df.select("id", "date", "sales").collect()
    if fit:
        model.fit(
            train,  # type: ignore
            id_col="id",
            time_col="date",
            target_col="sales",
        )

    p: pl.DataFrame = model.predict(df=train)  # type: ignore

    p = p.select(
        pl.col("id", "date"),
        cs.exclude("id", "date").fill_nan(0).clip(0).name.prefix(PRED_PREFIX),
    )

    return model, p


def cross_validate(
    model: "NeuralForecast", train: pl.LazyFrame, cv: dict[str, Any]
) -> pl.DataFrame:
    preds: list[pl.DataFrame] = []
    for i, (cv_train, _) in tqdm(list(enumerate(split_cv(train, **cv)))):
        model, p = _fit_predict(model, cv_train, fit=(i == 0))
        p = p.with_columns(pl.col("date").min().over("id").alias("cutoff_date"))
        preds.append(p)
    return pl.concat(preds)


def live_forecast(model: "NeuralForecast", train: pl.LazyFrame) -> pl.DataFrame:
    model, p = _fit_predict(model, train, fit=True)
    return p
