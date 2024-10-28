from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import join_into_cube, join_price_dfs, join_sales_dfs, remove_leading_zeros


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                join_sales_dfs,
                inputs=["raw_Sales_phase_0", "raw_Sales_phase_1"],
                outputs="raw_sales_data",
                name="join_sales_dfs",
            ),
            node(
                join_price_dfs,
                inputs=["raw_Price_phase_0", "raw_Price_phase_1"],
                outputs="raw_price_data",
                name="join_price_dfs",
            ),
            node(
                join_into_cube,
                inputs=["raw_sales_data", "raw_price_data"],
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
        inputs={"raw_Sales_phase_0", "raw_Sales_phase_1", "raw_Price_phase_0", "raw_Price_phase_1"},
        outputs={"primary_sales"},
    )
