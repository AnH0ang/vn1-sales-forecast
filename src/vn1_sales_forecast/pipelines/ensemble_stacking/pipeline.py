from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_models, cross_validate, live_forecast


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(create_models, inputs=None, outputs="models", name="create_models"),
            node(
                cross_validate,
                inputs=[
                    "models",
                    "model_cv_forecast",
                    "primary_sales",
                    "model_total_scores",
                ],
                outputs="cv_forecast",
                name="cross_validate",
            ),
            node(
                live_forecast,
                inputs=[
                    "models",
                    "model_cv_forecast",
                    "primary_sales",
                    "model_live_forecast",
                    "model_total_scores",
                ],
                outputs="live_forecast",
                name="live_forecast",
            ),
        ],
        namespace="ensemble_stacking",
        inputs={
            "model_cv_forecast",
            "primary_sales",
            "model_live_forecast",
            "model_total_scores",
        },
    )
