from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import cross_validation, live_forecast


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=cross_validation,
                inputs=["model_cv_forecast"],
                outputs="cv_forecast",
            ),
            node(
                func=live_forecast,
                inputs=["model_live_forecast"],
                outputs="live_forecast",
            ),
        ],
        namespace="ensemble_mixer",
        inputs={
            "model_cv_forecast",
            "model_live_forecast",
        },
    )
