import pytest
from pydantic import ValidationError

from app.core.config import Settings
from tennis_analyzer.schemas import AnalysisOptions, OptionValidationError, PipelineOptions, VisualizationOptions


def test_settings_parse_hosts_and_positive_limits(tmp_path):
    settings = Settings(data_root=tmp_path, allowed_hosts="one.test,two.test", worker_concurrency=1)
    assert settings.allowed_hosts == ["one.test", "two.test"]
    with pytest.raises(ValidationError):
        Settings(worker_concurrency=0)


def test_settings_build_database_url_from_single_password():
    settings = Settings(_env_file=None, database_url="", postgres_password="p@ss: /word")
    from sqlalchemy.engine import make_url

    url = make_url(settings.database_url)
    assert url.host == "tennis-postgres"
    assert url.username == "tennis"
    assert url.password == "p@ss: /word"


def test_settings_reject_mismatched_duplicate_database_password():
    with pytest.raises(ValidationError, match="must match POSTGRES_PASSWORD"):
        Settings(
            _env_file=None,
            database_url="postgresql+psycopg://tennis:first@postgres:5432/tennis",
            postgres_password="second",
        )


def test_analysis_dependencies_are_enabled():
    options = AnalysisOptions(statistics=True, pose_tracking=True).validated()
    assert options.ball_tracking and options.court_detection and options.player_tracking


def test_visualization_dependency_validation():
    with pytest.raises(OptionValidationError):
        PipelineOptions(visualization=VisualizationOptions(ball_trail=True)).validated()
