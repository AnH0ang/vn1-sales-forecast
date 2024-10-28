from collections.abc import Sequence
from typing import TypeVar

import polars as pl
from polars._typing import JoinStrategy

T_DF = TypeVar("T_DF", pl.LazyFrame, pl.DataFrame)


def subsample_data(data: T_DF, n: int = 100, seed: int | None = None, id_col: str = "id") -> T_DF:
    ids = data.select(pl.col(id_col).unique().sort().sample(n, seed=seed))
    return data.join(ids, on=id_col)  # type: ignore


def multi_join(
    *dfs: T_DF,
    on: str | pl.Expr | Sequence[str | pl.Expr],
    how: JoinStrategy = "inner",
) -> T_DF:
    res = dfs[0]
    for df in dfs[1:]:
        res = res.join(df, on=on, how=how)
    return res
