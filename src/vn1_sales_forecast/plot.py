import altair as alt
import polars as pl

from vn1_sales_forecast.utils import subsample_data


def plot_subsample(df: pl.LazyFrame | pl.DataFrame, n: int | None = 5, id_col: str = "id"):
    df = df.lazy()

    if n is not None:
        df = subsample_data(df, n=n, id_col=id_col)

    return (
        alt.Chart(df.collect())
        .mark_line(point=True, strokeCap="round")
        .encode(
            x="date",
            y="sales",
            color=alt.Color(id_col, type="nominal"),
            tooltip=[id_col, "date", "sales"],
        )
        .properties(width=700, height=150)
        .facet(row=id_col)
        .resolve_scale(y="independent")
    )
