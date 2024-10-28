from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calculate_cv_partitions, calculate_live_partition


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                calculate_cv_partitions,
                name="calculate_cv_partitions",
                inputs=["primary_sales", "params:cv"],
                outputs="cv_partitions",
            ),
            node(
                calculate_live_partition,
                name="calculate_live_partition",
                inputs="primary_sales",
                outputs="live_partitions",
            ),
        ],
        namespace="partition",
        inputs={"primary_sales"},
        outputs={"cv_partitions", "live_partitions"},
        parameters={"cv"},
    )
