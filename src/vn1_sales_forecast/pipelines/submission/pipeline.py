from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import make_submission


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                make_submission,
                inputs=["live_forecast", "raw_Sales_phase_1"],
                outputs="submissions",
                name="make_submission",
            )
        ]
    )
