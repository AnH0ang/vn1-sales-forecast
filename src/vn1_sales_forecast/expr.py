import polars as pl
from polars._typing import IntoExpr


def parse_into_expr(expr: IntoExpr) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, str | list):
        return pl.col(expr)
    else:
        return pl.lit(expr)


def mae(expr: pl.Expr, target: pl.Expr) -> pl.Expr:
    return (expr - target).abs().sum() / target.abs().sum()


def bias(expr: pl.Expr, target: pl.Expr) -> pl.Expr:
    return (expr - target).sum().abs() / target.abs().sum()


def competition_score(expr: pl.Expr, target: pl.Expr) -> pl.Expr:
    return bias(expr, target) + mae(expr, target)
