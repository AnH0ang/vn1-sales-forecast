from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from vn1_sales_forecast.cv import split_cv
from vn1_sales_forecast.settings import TSFEATURES_PREFIX

from .tsfeatures import tsfeatures


def _postprocess_features_df(df: pl.DataFrame, id_cols: list[str]) -> pl.DataFrame:
    # Rename
    df = df.select(*id_cols, cs.exclude(*id_cols).name.prefix(TSFEATURES_PREFIX))

    # Replace inf
    feat_cols = cs.starts_with(TSFEATURES_PREFIX)
    non_inf_feat_cols = pl.when(~feat_cols.is_infinite()).then(feat_cols).otherwise(np.nan)
    df = df.with_columns(non_inf_feat_cols)

    return df


def calculate_cv_tsfeatures(train: pl.LazyFrame, cv: dict[str, Any]) -> pl.DataFrame:
    features_dfs: list[pl.DataFrame] = []
    for cv_train, _ in tqdm(list(split_cv(train, **cv))):
        f = tsfeatures(cv_train, freq=52)
        max_date = cv_train.select(pl.col("date").max().dt.offset_by("1w")).collect().item()
        f = f.with_columns(pl.lit(max_date).alias("cutoff_date"))
        features_dfs.append(f)
    cv_feature_df: pl.DataFrame = pl.concat(features_dfs)

    return _postprocess_features_df(cv_feature_df, ["id", "cutoff_date"])


def calculate_live_tsfeatures(train: pl.LazyFrame) -> pl.DataFrame:
    feature_df = tsfeatures(train, freq=52)

    return _postprocess_features_df(feature_df, ["id"])
