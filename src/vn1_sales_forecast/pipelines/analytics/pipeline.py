from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calc_scale_categories, calc_vip_series


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                calc_scale_categories,
                inputs="primary_sales",
                outputs="scale_category_analysis",
                name="calc_scale_categories",
            ),
            node(
                calc_vip_series,
                inputs="scale_category_analysis",
                outputs="vip_series_analysis",
                name="calc_vip_series",
            ),
        ]
    )
