from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calc_class_error_report, calc_cv_scores, calc_errors_reports, calc_winner_model


def create_pipeline() -> dict[str, Pipeline]:
    pipelines: dict[str, Pipeline] = {}
    for prefix in ["model"]:
        pipelines[f"{prefix}_evaluation"] = pipeline(
            [
                node(
                    calc_cv_scores,
                    inputs=[f"{prefix}_cv_forecast", "primary_sales"],
                    outputs=[
                        f"{prefix}_individual_scores",
                        f"{prefix}_cv_scores",
                        f"{prefix}_total_scores",
                    ],
                    name=f"{prefix}_calc_cv_scores",
                ),
            ]
        )

    pipelines["evaluation"] = pipeline(
        [
            node(
                calc_cv_scores,
                inputs=["cv_forecast", "primary_sales"],
                outputs=["individual_scores", "cv_scores", "total_scores"],
                name="calc_cv_scores",
            ),
            node(
                calc_errors_reports,
                inputs=["cv_forecast", "primary_sales"],
                outputs=["cv_errors", "total_errors", "horizon_errors"],
                name="calc_errors_reports",
            ),
            node(
                calc_winner_model,
                inputs="individual_scores",
                outputs="winner_model_data",
                name="calc_winner_model",
            ),
            node(
                calc_class_error_report,
                inputs=["cv_errors", "cv_classification"],
                outputs="class_errors",
                name="calc_class_error_report",
            ),
        ]
    )
    return pipelines
