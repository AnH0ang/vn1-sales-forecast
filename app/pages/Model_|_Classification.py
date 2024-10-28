import altair as alt
import polars as pl
import streamlit as st
from kedro.config import OmegaConfigLoader
from kedro.io.data_catalog import DataCatalog

from vn1_sales_forecast.plot import plot_subsample

st.set_page_config(layout="wide")

st.title("Model Classification")


def create_catalog() -> DataCatalog:
    loader = OmegaConfigLoader(conf_source="conf")
    catalog = DataCatalog.from_config(loader.get("catalog"))
    return catalog


catalog = create_catalog()

sales: pl.LazyFrame = catalog.load("primary_sales")
cv_classification: pl.LazyFrame = catalog.load("cv_classification")
live_classification: pl.LazyFrame = catalog.load("live_classification")
class_errors: pl.LazyFrame = catalog.load("class_errors")

models: list[str] = class_errors.collect()["model"].unique().sort().to_list()
model_group = st.selectbox("Models Group", ["Single", "Ensemble", "All"])
if model_group == "Single":
    models = [m for m in models if "Ensemble" not in m]
elif model_group == "Ensemble":
    models = [m for m in models if "Ensemble" in m]

class_errors = class_errors.filter(pl.col("model").is_in(models))

model_sort = (
    class_errors.group_by("model")
    .agg(pl.col("score").mean())
    .sort("score")
    .collect()["model"]
    .to_list()
)

st.subheader("Share")

d = live_classification.group_by("class").agg(pl.len())
c = (
    alt.Chart(d.collect())
    .encode(
        theta=alt.Theta("len", stack=True),
        color=alt.Color("class:N"),
        tooltip=["class", "len"],
    )
    .mark_arc()
)
st.altair_chart(c, use_container_width=True)  # type: ignore

st.subheader("Errors")
with st.expander("Data Frame", expanded=True):
    cutoffs = (
        cv_classification.select(pl.col("cutoff_date").cast(pl.String).unique().sort())
        .collect()
        .to_series()
    )
    selected_cutoffs = st.multiselect("Cutoff Dates", cutoffs, default=cutoffs, key="1")

    df = (
        class_errors.sort("class")
        .filter(pl.col("cutoff_date").cast(pl.String).is_in(selected_cutoffs))
        .collect()
        .pivot(
            on="class",
            index="model",
            values="score",
            aggregate_function="mean",
        )
        .to_pandas()
    )

    def highlight_min(s):
        is_min = s == s.min()
        return ["background-color: lightyellow" if v else "" for v in is_min]

    st.dataframe(
        df.style.apply(highlight_min, subset=df.columns[df.columns != "model"]),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Error Barplot"):
    d = class_errors.group_by("class", "model").agg(pl.col("score").mean())
    classes = d.select(pl.col("class").unique().sort()).collect().to_series()

    for cls in classes:
        class_model_sort = (
            d.filter(pl.col("class") == cls).sort("score").collect()["model"].to_list()
        )
        c = (
            alt.Chart(d.filter(pl.col("class") == cls).collect())
            .mark_bar()
            .encode(
                x=alt.X("model", sort=class_model_sort),
                y="score",
                color=alt.Color("model", sort=model_sort),
                tooltip=["model", "score", "class"],
            )
            .properties(title=f"Class {cls.title()}")
        )
        st.altair_chart(c, use_container_width=True)  # type: ignore

with st.expander("Error CV Barplot"):
    cutoffs = (
        cv_classification.select(pl.col("cutoff_date").cast(pl.String).unique().sort())
        .collect()
        .to_series()
    )
    selected_cutoffs = st.multiselect("Cutoff Dates", cutoffs, default=cutoffs)

    for cls in classes:
        c = (
            alt.Chart(
                class_errors.filter(pl.col("class") == cls)
                .filter(pl.col("cutoff_date").cast(pl.String).is_in(selected_cutoffs))
                .collect()
            )
            .mark_bar()
            .encode(
                x=alt.X("model", sort=model_sort),
                y="score",
                color=alt.Color("model", sort=model_sort),
                tooltip=["model", "score", "class"],
                xOffset="cutoff_date:O",
                opacity="cutoff_date:O",
            )
            .properties(title=f"Class {cls.title()}")
        )
        st.altair_chart(c, use_container_width=True)  # type: ignore


st.subheader("Time Series Plots")
sales_classes = live_classification.select(pl.col("class").unique().sort()).collect().to_series()
selected_class = st.selectbox("Class", sales_classes)
selected_n = st.slider("Number of samples", 5, 20, 10)

class_sales = sales.join(live_classification, on="id").filter(pl.col("class") == selected_class)
c = plot_subsample(class_sales, selected_n)
st.altair_chart(c, use_container_width=True)  # type: ignore
