import pytest

from smartshop.ingest.sample import write_sample
from smartshop.warehouse import run_dbt


@pytest.fixture(scope="session")
def sample_warehouse(tmp_path_factory):
    """Synthetic raw data, built and tested by the real dbt project (runs in CI)."""
    d = tmp_path_factory.mktemp("sample")
    path = write_sample(d / "sample.duckdb")
    run_dbt("build", warehouse=path, target_path=d / "target")
    return path
