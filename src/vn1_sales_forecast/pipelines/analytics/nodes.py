import polars as pl


def calc_scale_categories(sales: pl.LazyFrame) -> pl.LazyFrame:
    scale_categories = (
        sales.group_by("id")
        .agg(pl.col("sales").abs().mean().alias("sales_scale"))
        .with_columns(pl.col("sales_scale").qcut(20, allow_duplicates=True).alias("bin"))
        .with_columns((pl.col("sales_scale") / pl.col("sales_scale").sum()).alias("weight"))
    )
    return scale_categories


def calc_vip_series(scale_categories: pl.LazyFrame) -> pl.LazyFrame:
    vip_series = scale_categories.filter(pl.col("sales_scale") > 100)
    return vip_series
