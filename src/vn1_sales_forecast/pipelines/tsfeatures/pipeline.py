from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calculate_cv_tsfeatures, calculate_live_tsfeatures


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                calculate_cv_tsfeatures,
                inputs=["primary_sales", "params:cv"],
                outputs="cv_tsfeatures",
                name="calculate_cv_tsfeatures",
            ),
            node(
                calculate_live_tsfeatures,
                inputs="primary_sales",
                outputs="live_tsfeatures",
                name="calculate_live_tsfeatures",
            ),
        ],
        namespace="tsfeatures",
        inputs={"primary_sales"},
        parameters={"cv"},
        outputs={"cv_tsfeatures", "live_tsfeatures"},
    )
