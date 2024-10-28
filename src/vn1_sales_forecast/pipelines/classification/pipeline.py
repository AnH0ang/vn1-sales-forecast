from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calculate_cv_classification, calculate_live_classification


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                calculate_cv_classification,
                inputs=[
                    "primary_sales",
                    "cv_tsfeatures",
                    "params:cv",
                ],
                outputs="cv_classification",
            ),
            node(
                calculate_live_classification,
                inputs=[
                    "primary_sales",
                    "live_tsfeatures",
                ],
                outputs="live_classification",
            ),
        ],
        namespace="classification",
        inputs={
            "primary_sales",
            "cv_tsfeatures",
            "live_tsfeatures",
        },
        outputs={"cv_classification", "live_classification"},
        parameters={"cv"},
    )
