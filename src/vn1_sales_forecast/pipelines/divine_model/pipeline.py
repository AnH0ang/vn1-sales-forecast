from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import cross_validate, live_forecast


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                cross_validate,
                inputs=[
                    "model_cv_forecast",
                    "ensemble_cv_forecast",
                    "cv_classification",
                ],
                outputs="cv_forecast",
                name="cross_validate",
            ),
            node(
                live_forecast,
                inputs=[
                    "model_live_forecast",
                    "ensemble_live_forecast",
                    "live_classification",
                ],
                outputs="live_forecast",
                name="live_forecast",
            ),
        ],
        namespace="divine_model",
        inputs={
            "model_cv_forecast",
            "model_live_forecast",
            "ensemble_cv_forecast",
            "ensemble_live_forecast",
            "cv_classification",
            "live_classification",
        },
    )
