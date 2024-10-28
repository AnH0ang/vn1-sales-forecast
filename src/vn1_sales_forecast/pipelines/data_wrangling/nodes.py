import polars as pl


def join_dfs(*dfs: pl.LazyFrame, name: str) -> pl.LazyFrame:
    def to_long(df: pl.LazyFrame) -> pl.LazyFrame:
        idx_cols = ["Client", "Warehouse", "Product"]
        df = df.melt(id_vars=idx_cols, variable_name="date", value_name=name)
        df = df.with_columns(pl.col(name).cast(pl.Float32))
        return df

    # Convert all dataframes to long format
    dfs_long = [to_long(df) for df in dfs]

    # Add phase number to the date column
    dfs_long = [df.with_columns(pl.lit(i).alias("phase")) for i, df in enumerate(dfs_long)]

    # Concatenate dataframes
    return pl.concat(dfs_long)


def join_sales_dfs(*dfs: pl.LazyFrame) -> pl.LazyFrame:
    return join_dfs(*dfs, name="sales")


def join_price_dfs(*dfs: pl.LazyFrame) -> pl.LazyFrame:
    return join_dfs(*dfs, name="price")


def join_into_cube(sales: pl.LazyFrame, price: pl.LazyFrame) -> pl.LazyFrame:
    idx_cols = ["Client", "Warehouse", "Product"]

    return (
        sales.join(price, on=[*idx_cols, "date", "phase"], how="left")
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
