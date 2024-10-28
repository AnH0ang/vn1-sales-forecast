import polars as pl
import polars.selectors as cs

from vn1_sales_forecast.expr import competition_score
from vn1_sales_forecast.settings import PRED_PREFIX


def calc_cv_scores(
    cv_forecast: pl.LazyFrame, sales: pl.LazyFrame
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    df = cv_forecast.join(sales, on=["id", "date"], how="left")

    score_expr = competition_score(cs.starts_with(PRED_PREFIX), pl.col("sales"))

    scores = df.group_by("id", "cutoff_date").agg(score_expr)
    cv_scores = df.group_by("cutoff_date").agg(score_expr)

    total_scores = (
        df.select(score_expr)
        .melt(variable_name="model", value_name="score")
        .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
    )

    return (scores, cv_scores, total_scores)


def calc_errors_reports(
    cv_forecast: pl.LazyFrame, sales: pl.LazyFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cv_errors = (
        cv_forecast.unpivot(
            index=["id", "date", "cutoff_date"],
            value_name="prediction",
            variable_name="model",
        )
        .join(sales.select("id", "date", "sales"), on=["id", "date"], how="left")
        .with_columns((pl.col("prediction") - pl.col("sales")).alias("error"))
        .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
        .collect()
    )

    total_errors = (
        cv_errors.group_by("model", "cutoff_date")
        .agg(pl.col("error").mean().alias("bias"), pl.col("error").abs().mean().alias("mae"))
        .with_columns((pl.col("mae") + pl.col("bias").abs()).alias("score"))
        .sort("model", "cutoff_date")
    )

    horizon_expr = ((pl.col("date") - pl.col("cutoff_date")).dt.total_days() // 7) + 1
    horizon_errors = (
        cv_errors.with_columns(horizon_expr.alias("horizon"))
        .group_by("horizon", "model")
        .agg(
            pl.col("error").abs().mean().alias("mae"),
            pl.col("error").mean().alias("bias"),
        )
        .with_columns(
            (pl.col("mae") + pl.col("bias").abs()).alias("score"),
        )
        .sort("horizon")
    )
    return (cv_errors, total_errors, horizon_errors)


def calc_class_error_report(
    cv_errors: pl.LazyFrame, cv_classification: pl.LazyFrame
) -> pl.LazyFrame:
    class_errors = (
        cv_errors.join(
            cv_classification,
            on=["id", "cutoff_date"],
            how="left",
        )
        .group_by("cutoff_date", "class", "model")
        .agg(
            pl.col("error").abs().mean().alias("mae"),
            pl.col("error").mean().alias("bias"),
        )
        .with_columns(
            (pl.col("mae") + pl.col("bias").abs()).alias("score"),
        )
    )
    return class_errors


def calc_winner_model(scores: pl.LazyFrame) -> pl.LazyFrame:
    return (
        scores.group_by("id")
        .agg(cs.starts_with(PRED_PREFIX).mean())
        .unpivot(index="id", variable_name="model", value_name="score")
        .with_columns(pl.col("model").str.strip_prefix(PRED_PREFIX))
        .filter(pl.col("score") == pl.col("score").min().over("id"))
    )
