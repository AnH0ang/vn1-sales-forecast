from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calculate_cv_scores, cross_validate, live_forecast


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                calculate_cv_scores,
                inputs=["model_cv_forecast", "primary_sales"],
                outputs="full_cv_scores",
            ),
            node(
                cross_validate,
                inputs=[
                    "model_cv_forecast",
                    "cv_tsfeatures",
                    "full_cv_scores",
                    "model_total_scores",
                ],
                outputs="cv_forecast",
                name="cross_validate",
            ),
            node(
                live_forecast,
                inputs=[
                    "cv_tsfeatures",
                    "full_cv_scores",
                    "model_live_forecast",
                    "live_tsfeatures",
                    "model_total_scores",
                ],
                outputs="live_forecast",
                name="live_forecast",
            ),
        ],
        namespace="ensemble_fforma",
        inputs={
            "primary_sales",
            "model_cv_forecast",
            "model_live_forecast",
            "model_total_scores",
            "cv_tsfeatures",
            "live_tsfeatures",
        },
    )
