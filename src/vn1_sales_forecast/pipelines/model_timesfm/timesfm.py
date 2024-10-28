import typing
from typing import Literal

import polars as pl

if typing.TYPE_CHECKING:
    from timesfm import TimesFm


class TimesFM:
    def __init__(
        self,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        h: int = 13,
        freq: str = "W",
        batch_size: int = 32,
        alias: str = "TimesFM",
        backend: Literal["cpu", "gpu", "tpu"] = "gpu",
    ) -> None:
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias
        self.backend = backend
        self.h = h
        self.freq = freq

    def get_predictor(
        self,
    ) -> "TimesFm":
        import timesfm

        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=self.backend,  # type: ignore
                per_core_batch_size=self.batch_size,
                horizon_len=self.h,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo_id),
        )
        return tfm

    def forecast(
        self,
        df: pl.LazyFrame,
        id_col="id",
        time_col="date",
        target_col="sales",
    ) -> pl.DataFrame:
        predictor = self.get_predictor()

        df_input = (
            df.select(
                pl.col(id_col).alias("unique_id"),
                pl.col(time_col).alias("ds"),
                target_col,
            )
            .collect()
            .to_pandas()
        )

        p = predictor.forecast_on_df(
            df_input,
            freq=self.freq,
            value_name=target_col,
            model_name=self.alias,
        )

        date_col = pl.col("ds").alias(time_col).cast(pl.Date)
        if self.freq == "W":
            date_col = date_col.dt.offset_by("1d")

        p = (
            pl.DataFrame(p)
            .select(
                pl.col("unique_id").alias(id_col),
                date_col,
                pl.col(self.alias),
            )
            .sort(id_col, time_col)
        )
        return p
