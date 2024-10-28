from collections.abc import Sequence

import polars as pl

from vn1_sales_forecast.utils import multi_join


def _assert_checks(dfs: Sequence[pl.LazyFrame]) -> None:
    lengths = {df.select(pl.len()).collect().item() for df in dfs}
    assert len(lengths) == 1, f"Lengths of DataFrames are not equal: {lengths}"


def _join_preds(*dfs: pl.LazyFrame, idx_cols: list[str]) -> pl.LazyFrame:
    return multi_join(*dfs, on=idx_cols)


def join_cv_preds(*dfs: pl.LazyFrame) -> pl.LazyFrame:
    _assert_checks(dfs)
    return _join_preds(*dfs, idx_cols=["id", "date", "cutoff_date"])


def join_live_preds(*dfs: pl.LazyFrame) -> pl.LazyFrame:
    _assert_checks(dfs)
    return _join_preds(*dfs, idx_cols=["id", "date"])
