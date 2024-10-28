import polars as pl
import polars.selectors as cs

from vn1_sales_forecast.settings import PRED_PREFIX


def _format_submission(preds: pl.LazyFrame, raw_sales: pl.LazyFrame) -> pl.LazyFrame:
    assert {"id", "date", "pred"} <= set(preds.collect_schema().names())

    id_cols = ["Client", "Warehouse", "Product"]
    submission = (
        preds.select(
            pl.col("id")
            .str.split_exact("-", 2)
            .struct.rename_fields(id_cols)
            .struct.field("*")
            .cast(pl.Int64),
            pl.col("date").dt.strftime("%Y-%m-%d"),
            pl.col("pred"),
        )
        .collect()
        .sort("date")
        .pivot(index=id_cols, on="date")
        .lazy()
    )

    assert (
        raw_sales.select(pl.len()).collect().item() == submission.select(pl.len()).collect().item()
    )

    raw_sales, pred_df = pl.align_frames(raw_sales, submission, on=id_cols)

    assert pred_df.collect_schema().names()[3:] == sorted(pred_df.collect_schema().names()[3:])

    return pred_df


def make_submission(df: pl.LazyFrame, raw_sales: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    submissions = {}
    for p in df.select(cs.starts_with(PRED_PREFIX)).collect_schema().names():
        s = _format_submission(df.select("id", "date", pl.col(p).alias("pred")), raw_sales)
        submissions[p.removeprefix(PRED_PREFIX).lower()] = s

    return submissions
