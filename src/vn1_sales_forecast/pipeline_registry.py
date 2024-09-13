from kedro.pipeline import Pipeline

from vn1_sales_forecast.pipelines import data_wrangling


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {"data_wrangling": data_wrangling.create_pipeline()}

    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))
    return pipelines
