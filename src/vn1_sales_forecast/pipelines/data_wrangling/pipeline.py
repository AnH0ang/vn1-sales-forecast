from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import join_into_cube, remove_leading_zeros


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                join_into_cube,
                inputs=["raw_Sales", "raw_Price"],
                outputs="cube_data",
                name="join_into_cube",
            ),
            node(
                remove_leading_zeros,
                inputs="cube_data",
                outputs="primary_sales",
                name="remove_leading_zeros",
            ),
        ],
        namespace="data_wrangling",
        inputs={"raw_Sales", "raw_Price"},
        outputs={"primary_sales"},
    )
