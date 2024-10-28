import random

import altair as alt
import polars as pl
import polars.selectors as cs
import streamlit as st
from kedro.config import OmegaConfigLoader
from kedro.io.data_catalog import DataCatalog

from vn1_sales_forecast.expr import competition_score
from vn1_sales_forecast.settings import PRED_PREFIX

st.set_page_config(layout="wide")

st.title("Evaluate Forecasts")


def create_catalog() -> DataCatalog:
    loader = OmegaConfigLoader(conf_source="conf")
    catalog = DataCatalog.from_config(loader.get("catalog"))
    return catalog


# 1. Load data
catalog = create_catalog()
sales: pl.LazyFrame = catalog.load("primary_sales")
cv_forecast: pl.LazyFrame = catalog.load("cv_forecast")
live_forecast: pl.LazyFrame = catalog.load("live_forecast")
vip_series: pl.LazyFrame = catalog.load("vip_series_analysis")

# 2. Filter
st.subheader("Filter")


# VIP
ids = sales.select(pl.col("id").unique().sort())
if st.checkbox("VIP Series"):
    ids = ids.join(vip_series, on="id")
ids = ids.collect().to_series().to_list()


# Index
col1, col2 = st.columns([0.8, 0.2])
with col2:
    if st.button("Random Id"):
        id_index = random.choice(range(len(ids)))
    else:
        id_index = 0
with col1:
    idx = st.selectbox("Index", ids, index=id_index, label_visibility="collapsed")

# Model
models: list[str] = cv_forecast.select(cs.starts_with(PRED_PREFIX)).collect_schema().names()
models = [m.removeprefix(PRED_PREFIX) for m in models]

model_group = st.selectbox("Models Group", ["All", "Ensemble", "Single"])
if model_group == "Single":
    models = [m for m in models if "Ensemble" not in m]
elif model_group == "Ensemble":
    models = [m for m in models if "Ensemble" in m]


selected_models = st.multiselect("Models", models, default=models)
selected_models = [f"{PRED_PREFIX}{m}" for m in selected_models]

cv_forecast = cv_forecast.select(cs.exclude(cs.starts_with(PRED_PREFIX)), *selected_models)
live_forecast = live_forecast.select(cs.exclude(cs.starts_with(PRED_PREFIX)), *selected_models)

# 3. Scores
st.subheader("Scores")
with st.expander("Competition Score"):
    score_df = (
        cv_forecast.join(sales, on=["id", "date"], how="left")
        .group_by("id", "cutoff_date")
        .agg(competition_score(cs.starts_with(PRED_PREFIX), pl.col("sales")))
        .filter(pl.col("id") == idx)
        .unpivot(
            index=["id", "cutoff_date"],
            on=cs.starts_with(PRED_PREFIX),
            value_name="prediction",
            variable_name="model",
        )
        .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
        .collect()
        .sort("cutoff_date")
        .pivot(index="model", on="cutoff_date", values="prediction")
        .to_pandas()
    )

    def highlight_min(s):
        is_min = s == s.min()
        return ["background-color: lightyellow" if v else "" for v in is_min]

    st.dataframe(
        score_df.style.apply(highlight_min, subset=score_df.columns[score_df.columns != "model"]),
        hide_index=True,
    )

# 4. Plotting
st.subheader("CV Forecast")
actual_line = (
    alt.Chart(sales.filter(pl.col("id") == idx).collect())
    .mark_line(color="black", opacity=0.3)
    .encode(x="date:T", y="sales")
)
pred_line = (
    alt.Chart(
        cv_forecast.filter(pl.col("id") == idx)
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
    .encode(x="date:T", y="prediction", color="model", strokeDash="cutoff_date")
)

chart = alt.layer(actual_line, pred_line).properties(title=f"Sales Prediction for ID {idx}")
st.altair_chart(chart, use_container_width=True)  # type: ignore


# 5. Live Forecast
st.subheader("Live Forecast")

actual_line = (
    alt.Chart(sales.filter(pl.col("id") == idx).collect())
    .mark_line(color="black", opacity=0.3)
    .encode(x="date:T", y="sales")
)
live_pred_line = (
    alt.Chart(
        live_forecast.filter(pl.col("id") == idx)
        .unpivot(
            index=["date"],
            on=cs.starts_with(PRED_PREFIX),
            value_name="prediction",
            variable_name="model",
        )
        .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
        .collect()
    )
    .mark_line(opacity=0.8)
    .encode(x="date:T", y="prediction", color="model")
)

chart = alt.layer(actual_line, live_pred_line).properties(title=f"Sales Prediction for ID {idx}")
st.altair_chart(chart, use_container_width=True)  # type: ignore
