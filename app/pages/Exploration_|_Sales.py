import altair as alt
import polars as pl
import streamlit as st
from kedro.config import OmegaConfigLoader
from kedro.io.data_catalog import DataCatalog

st.set_page_config(layout="wide")

st.title("Explore Sales")


def create_catalog() -> DataCatalog:
    loader = OmegaConfigLoader(conf_source="conf")
    catalog = DataCatalog.from_config(loader.get("catalog"))
    return catalog


# 1. Load data
catalog = create_catalog()
sales: pl.LazyFrame = catalog.load("primary_sales")

id_cols = ["Client", "Warehouse", "Product"]


st.markdown("#### Aggregate")
agg_cols = st.multiselect("Aggregate", id_cols)
id_cols = list(set(id_cols) - set(agg_cols))

id_col = pl.concat_str(id_cols, separator="-") if id_cols else pl.lit("All")
sales = sales.with_columns(pl.col("price").fill_null(strategy="forward").over("id"))
sales = (
    sales.group_by("date", *id_cols)
    .agg(
        pl.col("sales").sum(),
        # ((pl.col("price") * pl.col("sales")).sum() / (pl.col("sales") + 1).sum()).alias("price"),
        pl.col("price").mean(),
    )
    .with_columns(id_col.alias("id"))
)

st.markdown("#### Filter")
for col in id_cols:
    ids = sales.select(pl.col(col).unique().sort()).collect().to_series()
    selected_id = st.selectbox(f"Filter {col}", ids, index=None)

    if selected_id is not None:
        sales = sales.filter(pl.col(col) == selected_id)


st.markdown("#### Plot")
n = st.slider("Sample Size", 1, 100, 10)
sort_by_sales = st.checkbox("Sort by Sales", value=True)
show_price = st.checkbox("Show Price", value=True)
show_points = st.checkbox("Show Points")
seed = st.number_input("Seed", value=42)

if not sort_by_sales:
    max_n = sales.select(pl.col("id").n_unique()).collect().item()
    n = min(n, max_n)
    sample_ids = sales.select(pl.col("id").unique().sort().sample(n, seed=seed))
else:
    sample_ids = (
        sales.group_by("id")
        .agg(pl.col("sales").mean().alias("scale"))
        .sort("scale", descending=True)
        .head(n)
    )

sample_sales = sample_ids.join(sales, on="id", how="left").collect()
for sid, d in sample_sales.group_by("id", maintain_order=True):
    st.markdown(f"**{sid[0]}**")
    sales_c = alt.Chart(d).mark_line(point=show_points).encode(x="date", y="sales")
    price_c = alt.Chart(d).mark_line(color="red").encode(x="date", y="price")
    if show_price:
        c = (sales_c + price_c).resolve_scale(y="independent")
    else:
        c = sales_c
    st.altair_chart(c, use_container_width=True)  # type: ignore
