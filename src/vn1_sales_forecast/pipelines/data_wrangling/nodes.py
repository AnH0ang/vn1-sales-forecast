import polars as pl


def join_into_cube(sales_wide: pl.LazyFrame, price_wide: pl.LazyFrame) -> pl.LazyFrame:
    idx_cols = ["Client", "Warehouse", "Product"]
    sales = sales_wide.melt(id_vars=idx_cols, variable_name="date", value_name="sales")
    price = price_wide.melt(id_vars=idx_cols, variable_name="date", value_name="price")

    return (
        sales.join(price, on=[*idx_cols, "date"], how="left")
        .with_columns(
            pl.col(["sales", "price"]).cast(pl.Float32),
            pl.col("date").str.to_date(r"%Y-%m-%d"),
            pl.concat_str(idx_cols, separator="-").alias("id"),
        )
        .sort([*idx_cols, "date"])
    )


def remove_leading_zeros(cube: pl.LazyFrame) -> pl.LazyFrame:
    idx_cols = ["Client", "Warehouse", "Product"]
    non_zero_sales = pl.col("sales").is_not_null() & (pl.col("sales") > 0)
    leading_zero_mask = (non_zero_sales.cum_sum() >= 1).over(idx_cols)
    return cube.sort([*idx_cols, "date"]).filter(leading_zero_mask)
