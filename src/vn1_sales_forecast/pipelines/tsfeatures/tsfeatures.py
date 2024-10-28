import pandas as pd
import polars as pl


def tsfeatures(
    df: pl.DataFrame | pl.LazyFrame,
    id_col: str = "id",
    time_col: str = "date",
    target_col: str = "sales",
    **kwargs,
) -> pl.DataFrame:
    from tsfeatures import tsfeatures as _tsfeatures

    df = df.lazy()

    panel: pd.DataFrame = (
        df.select(
            pl.col(id_col).alias("unique_id"),
            pl.col(time_col).alias("ds"),
            pl.col(target_col).alias("y"),
        )
        .collect()
        .to_pandas()
    )

    features = _tsfeatures(panel, **kwargs)

    return pl.DataFrame(features).rename({"unique_id": id_col})
