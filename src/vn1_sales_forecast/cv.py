from collections.abc import Generator

import polars as pl


def split_cv(
    df: pl.LazyFrame,
    h: int = 13,
    freq: str = "w",
    n_windows: int = 3,
    step: int | None = None,
    min_train_size: int | None = None,
    id_col: str = "id",
    time_col: str = "date",
) -> Generator[tuple[pl.LazyFrame, pl.LazyFrame], None, None]:
    step = step or h

    for i in reversed(range(n_windows)):
        offset = i * step
        max_date = pl.col(time_col).max().over(id_col)
        lw_d = max_date.dt.offset_by(f"-{offset + h}{freq}")
        up_d = max_date.dt.offset_by(f"-{offset}{freq}")

        train = df.filter(pl.col(time_col) <= lw_d)
        if min_train_size:
            train = train.filter(pl.len().over(id_col) >= min_train_size)

        train_ids = train.select(pl.col(id_col).unique())
        test = df.join(train_ids, on="id").filter(
            pl.col("date").is_between(lw_d, up_d, closed="right")
        )
        yield train, test


def split_cv_loo(
    *dfs: pl.LazyFrame, group_col: str = "cutoff_date"
) -> Generator[tuple[pl.LazyFrame, ...], None, None]:
    groups = dfs[0].select(pl.col(group_col).unique().sort()).collect().to_series().to_list()
    for g in groups:
        r = []
        for df in dfs:
            r.append(df.filter(pl.col(group_col) != g))  # train
            r.append(df.filter(pl.col(group_col) == g))  # val

        yield tuple(r)
