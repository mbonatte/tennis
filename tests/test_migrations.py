from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sqlalchemy.dialects.postgresql import ENUM


def test_render_outputs_migration_reuses_existing_job_status_enum() -> None:
    migration_path = Path(__file__).parents[1] / "alembic" / "versions" / "20260719_0004_render_outputs.py"
    spec = spec_from_file_location("render_outputs_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)

    status_enum = migration._existing_job_status_enum()

    assert isinstance(status_enum, ENUM)
    assert status_enum.name == "jobstatus"
    assert status_enum.create_type is False
