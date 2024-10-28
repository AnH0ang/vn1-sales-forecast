from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import create_model, cross_validate, live_forecast


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_model,
                inputs=None,
                outputs="model",
                name="create_model",
            ),
            node(
                cross_validate,
                inputs=["model", "primary_sales", "params:cv"],
                outputs="cv_forecast",
                name="cross_validate",
            ),
            node(
                live_forecast,
                inputs=["model", "primary_sales"],
                outputs="live_forecast",
                name="live_forecast",
            ),
        ],
        namespace="model_ml_direct",
        inputs={"primary_sales"},
        parameters={"cv"},
    )
