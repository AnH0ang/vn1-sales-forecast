from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import cross_validate, live_forecast


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                cross_validate,
                inputs=["model_cv_forecast", "primary_sales", "model_total_scores"],
                outputs="cv_forecast",
                name="cross_validate",
            ),
            node(
                live_forecast,
                inputs=[
                    "model_cv_forecast",
                    "primary_sales",
                    "model_live_forecast",
                    "model_total_scores",
                ],
                outputs="live_forecast",
                name="live_forecast",
            ),
        ],
        namespace="ensemble_optimal_weights",
        inputs={"model_cv_forecast", "primary_sales", "model_live_forecast", "model_total_scores"},
    )
