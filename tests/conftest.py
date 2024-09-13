from pathlib import Path

import pytest
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.project import settings


@pytest.fixture()
def config_loader() -> OmegaConfigLoader:
    return OmegaConfigLoader(
        conf_source=str(Path.cwd() / settings.CONF_SOURCE), **settings.CONFIG_LOADER_ARGS
    )


@pytest.fixture()
def project_context(config_loader: OmegaConfigLoader) -> KedroContext:
    return KedroContext(
        project_path=Path.cwd(),
        config_loader=config_loader,
        env=None,
        hook_manager=_create_hook_manager(),  # type: ignore
        package_name="numforecast",  # type: ignore
    )
