from kedro.pipeline import Pipeline

from vn1_sales_forecast.pipelines import (
    analytics,
    classification,
    data_wrangling,
    divine_model,
    ensemble_classification,
    ensemble_fforma,
    ensemble_mixer,
    ensemble_optimal_weights,
    ensemble_stacking,
    evaluation,
    model_ml_direct,
    model_ml_recursive,
    model_ml_recursive_partitioned,
    model_nn,
    model_stat,
    model_timesfm,
    partition,
    post_processing,
    submission,
    tsfeatures,
)


def register_pipelines() -> dict[str, Pipeline]:
    model_pipelines: dict[str, Pipeline] = {
        "model_stat": model_stat.create_pipeline(),
        "model_ml_recursive": model_ml_recursive.create_pipeline(),
        "model_ml_direct": model_ml_direct.create_pipeline(),
        "model_ml_recursive_partitioned": model_ml_recursive_partitioned.create_pipeline(),
        "model_nn": model_nn.create_pipeline(),
        "model_timesfm": model_timesfm.create_pipeline(),
    }

    ensemble_pipelines: dict[str, Pipeline] = {
        "ensemble_stacking": ensemble_stacking.create_pipeline(),
        "ensemble_optimal_weights": ensemble_optimal_weights.create_pipeline(),
        "ensemble_fforma": ensemble_fforma.create_pipeline(),
        "ensemble_classification": ensemble_classification.create_pipeline(),
        "ensemble_mixer": ensemble_mixer.create_pipeline(),
    }

    divine_pipeline = {
        "divine_model": divine_model.create_pipeline(),
    }

    post_processing_pipelines = post_processing.create_pipeline(
        model_ns=list(model_pipelines),
        ensemble_ns=list(ensemble_pipelines),
        divine_ns=list(divine_pipeline),
    )

    evaluation_pipelines = evaluation.create_pipeline()

    pipelines = (
        {
            "tsfeatures": tsfeatures.create_pipeline(),
            "classification": classification.create_pipeline(),
            "analytics": analytics.create_pipeline(),
            "data_wrangling": data_wrangling.create_pipeline(),
            "submission": submission.create_pipeline(),
            "partition": partition.create_pipeline(),
        }
        | model_pipelines
        | ensemble_pipelines
        | post_processing_pipelines
        | evaluation_pipelines
        | divine_pipeline
    )

    pipelines["model"] = sum(model_pipelines.values(), Pipeline([]))
    pipelines["ensemble"] = sum(ensemble_pipelines.values(), Pipeline([]))

    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))

    pipelines["full_eval"] = (
        pipelines["model_post_processing"]
        + pipelines["model_evaluation"]
        + pipelines["ensemble_post_processing"]
        + pipelines["post_processing"]
        + pipelines["evaluation"]
        + pipelines["submission"]
    )
    return pipelines
