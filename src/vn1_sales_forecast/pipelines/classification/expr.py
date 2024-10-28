import numpy as np
import polars as pl


def _longest_streak_of_ones(e: pl.Expr) -> pl.Expr:
    def _inner(arr: np.ndarray) -> int:
        arr = 1 - arr
        r = np.max(np.bincount(np.cumsum(arr))) - 1
        return int(r)

    return e.cast(pl.Int32).map_batches(_inner, returns_scalar=True, return_dtype=pl.Int64)  # type: ignore


def cls_all_zero() -> pl.Expr:
    return (pl.col("sales") == 0).tail(52).all() | (pl.col("sales") == 0).all()


def cls_trailing_zero() -> pl.Expr:
    return (pl.col("sales") == 0).tail(13).all()


def cls_sparse() -> pl.Expr:
    non_continuous_expr = (pl.col("sales") != 0).pipe(_longest_streak_of_ones) <= 4
    sparse_expr = (pl.col("sales") == 0).mean() > 0.4
    return non_continuous_expr & sparse_expr


def cls_small_sales() -> pl.Expr:
    return pl.col("sales").max() < 4


def cls_gaps() -> pl.Expr:
    nonzero_streak_expr = (pl.col("sales") == 0).pipe(_longest_streak_of_ones) >= 13
    return nonzero_streak_expr & cls_trailing_zero.not_() & cls_sparse.not_()  # type: ignore


def cls_seasonal() -> pl.Expr:
    return (pl.col("tsfeatures_seas_acf1") > 0.55).all() & (pl.len() > 52)
