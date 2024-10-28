from kedro_datasets._typing import TablePreview
from kedro_datasets.polars.lazy_polars_dataset import LazyPolarsDataset as _LazyPolarsDataset


class LazyPolarsDataset(_LazyPolarsDataset):
    def preview(self) -> TablePreview:
        d = self._load().head(10).collect().to_pandas().to_dict(orient="split")
        return TablePreview(d)
