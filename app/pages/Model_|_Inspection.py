import altair as alt
import polars as pl
import polars.selectors as cs
import streamlit as st
from kedro.config import OmegaConfigLoader
from kedro.io.data_catalog import DataCatalog

from vn1_sales_forecast.settings import PRED_PREFIX

st.set_page_config(layout="wide")


def create_catalog() -> DataCatalog:
    loader = OmegaConfigLoader(conf_source="conf")
    catalog = DataCatalog.from_config(loader.get("catalog"))
    return catalog


catalog = create_catalog()

cv_forecast: pl.LazyFrame = catalog.load("cv_forecast")
live_forecast: pl.LazyFrame = catalog.load("live_forecast")
sales: pl.LazyFrame = catalog.load("primary_sales")
cv_errors: pl.LazyFrame = catalog.load("cv_errors")
total_errors: pl.LazyFrame = catalog.load("total_errors")
horizon_errors: pl.LazyFrame = catalog.load("horizon_errors")

st.title("Model Inspection")

model_list = cv_errors.select(pl.col("model").unique().sort()).collect().to_series().to_list()
model = st.selectbox("Model", model_list, index=model_list.index("LGBMRegressorRecursive"))

st.header("Errors")

bar = (
    alt.Chart(
        horizon_errors.filter(pl.col("model") == model)
        .unpivot(on=["mae", "bias"], index=["horizon", "model"])
        .collect(),
    )
    .mark_bar()
    .encode(
        x="horizon:O",
        y="value",
        color="variable",
        xOffset="variable",
    )
)

line = (
    alt.Chart(horizon_errors.filter(pl.col("model") == model).collect())
    .mark_line(point=True)
    .encode(
        x="horizon:O",
        y="score",
    )
)

c = (bar + line).properties(title="Horizon Errors")
st.altair_chart(c, use_container_width=True)  # type: ignore

d = (
    total_errors.filter(pl.col("model") == model)
    .melt(id_vars=["model", "cutoff_date"], variable_name="metric", value_name="score")
    .collect()
)
c = (
    alt.Chart(d)
    .mark_bar()
    .encode(x="metric", y="score", xOffset="cutoff_date:O", color="metric")
    .properties(title="Scores by CV")
)
st.altair_chart(c, use_container_width=True)  # type: ignore

st.header("Worst Predictions")

st.subheader("Filter")

n = st.slider("Top N", 10, 500, 50)

other_models = list(set(model_list) - {model})
benchmark_models = st.multiselect("Benchmark Models", other_models, default=[])


show_points = st.checkbox("Show Points")

cutoffs = (
    cv_forecast.select(pl.col("cutoff_date").cast(pl.String).unique().sort()).collect().to_series()
)
selected_cutoffs = st.multiselect("Cutoff Dates", cutoffs, default=cutoffs, key="1")

st.subheader("Plots")
worst_ids = (
    cv_errors.filter(pl.col("model") == model)
    .filter(pl.col("cutoff_date").cast(pl.String).is_in(selected_cutoffs))
    .group_by("id")
    .agg(pl.col("error").abs().sum())
    .sort("error", descending=True)
    .head(n)
    .collect()
)

for i, (idx, error) in enumerate(worst_ids.select("id", "error").iter_rows()):
    actual_line = (
        alt.Chart(sales.filter(pl.col("id") == idx).collect())
        .mark_line(color="black", opacity=0.3, point=show_points)
        .encode(x="date:T", y="sales", tooltip=["date", "sales"])
    )
    pred_line = (
        alt.Chart(
            cv_forecast.select(
                "id",
                "date",
                "cutoff_date",
                f"{PRED_PREFIX}{model}",
                *[f"{PRED_PREFIX}{m}" for m in benchmark_models],
            )
            .filter(pl.col("id") == idx)
            .unpivot(
                index=["date", "cutoff_date"],
                on=cs.starts_with(PRED_PREFIX),
                value_name="prediction",
                variable_name="model",
            )
            .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
            .collect()
        )
        .mark_line(opacity=0.8)
        .encode(
            x="date:T",
            y="prediction",
            color="model",
            strokeDash="cutoff_date",
            tooltip=["date", "prediction", "model"],
        )
    )
    live_pred_line = (
        alt.Chart(
            live_forecast.select(
                "id",
                "date",
                f"{PRED_PREFIX}{model}",
                *[f"{PRED_PREFIX}{m}" for m in benchmark_models],
            )
            .filter(pl.col("id") == idx)
            .unpivot(
                index=["date"],
                on=cs.starts_with(PRED_PREFIX),
                value_name="prediction",
                variable_name="model",
            )
            .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
            .collect()
        )
        .mark_line(opacity=0.8, strokeDash=[6, 3])
        .encode(
            x="date:T",
            y="prediction",
            color="model",
            tooltip=["date", "prediction", "model"],
        )
    )

    st.markdown(f"##### {i}. Prediction (ID: {idx}, Error: {error:.2f})")
    chart = alt.layer(actual_line, pred_line, live_pred_line)
    st.altair_chart(chart, use_container_width=True)  # type: ignore
