import altair as alt
import polars as pl
import streamlit as st
from kedro.config import OmegaConfigLoader
from kedro.io.data_catalog import DataCatalog

st.set_page_config(layout="wide")


def create_catalog() -> DataCatalog:
    loader = OmegaConfigLoader(conf_source="conf")
    catalog = DataCatalog.from_config(loader.get("catalog"))
    return catalog


catalog = create_catalog()


sales: pl.LazyFrame = catalog.load("primary_sales")
partitions: pl.LazyFrame = catalog.load("live_partitions")

st.title("Sales Partition")

st.subheader("Filter")
if st.checkbox("Show Only Multiple Partition"):
    sample_ids = (
        sales.join(partitions, on=["id", "date"])
        .filter(pl.col("partition").n_unique().over("id") > 1)
        .select(pl.col("id").unique().sort())
    )
else:
    sample_ids = sales.select(pl.col("id").unique().sort())

st.subheader("Select")
selected_id = st.selectbox("ID", sample_ids.collect()["id"])
d = sales.filter(pl.col("id") == selected_id).join(partitions, on=["id", "date"]).collect()
st.markdown(f"**ID: {selected_id}**")
c = (
    alt.Chart(d)
    .mark_line(point=True, strokeCap="round")
    .encode(x="date:T", y="sales:Q", color="partition:N")
)
st.altair_chart(c, use_container_width=True)

st.subheader("Random")
n = st.slider("Top N", 10, 100, 20)
sample_ids = sample_ids.select(pl.col("id").unique().sort().sample(n))
for id in sample_ids.collect()["id"]:
    d = sales.filter(pl.col("id") == id).join(partitions, on=["id", "date"]).collect()
    st.markdown(f"**ID: {id}**")
    c = (
        alt.Chart(d)
        .mark_line(point=True, strokeCap="round")
        .encode(x="date:T", y="sales:Q", color="partition:N")
    )
    st.altair_chart(c, use_container_width=True)
