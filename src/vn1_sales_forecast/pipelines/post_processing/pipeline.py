from collections.abc import Sequence

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import join_cv_preds, join_live_preds


def create_pipeline(
    model_ns: Sequence[str], ensemble_ns: Sequence[str], divine_ns: Sequence[str]
) -> dict[str, Pipeline]:
    pipelines: dict[str, Pipeline] = {}

    nss_prefix_list: list[tuple[str, Sequence[str]]] = [
        ("model", model_ns),
        ("ensemble", ensemble_ns),
    ]
    for prefix, nss in nss_prefix_list:
        pipelines[f"{prefix}_post_processing"] = pipeline(
            [
                node(
                    join_live_preds,
                    inputs=[f"{ns}.live_forecast" for ns in nss],
                    outputs=f"{prefix}_live_forecast",
                    name=f"join_{prefix}_live_forecasts",
                ),
                node(
                    join_cv_preds,
                    inputs=[f"{ns}.cv_forecast" for ns in nss],
                    outputs=f"{prefix}_cv_forecast",
                    name=f"join_{prefix}_cv_forecasts",
                ),
            ]
        )

    pipelines["post_processing"] = pipeline(
        [
            node(
                join_live_preds,
                inputs=[
                    "model_live_forecast",
                    "ensemble_live_forecast",
                    *(f"{ns}.live_forecast" for ns in divine_ns),
                ],
                outputs="live_forecast",
                name="join_live_forecasts",
            ),
            node(
                join_cv_preds,
                inputs=[
                    "model_cv_forecast",
                    "ensemble_cv_forecast",
                    *(f"{ns}.cv_forecast" for ns in divine_ns),
                ],
                outputs="cv_forecast",
                name="join_cv_forecasts",
            ),
        ]
    )

    return pipelines
