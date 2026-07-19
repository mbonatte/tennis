from pathlib import Path

import tomllib


def test_wheel_configuration_includes_runtime_model_adapters():
    configuration = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    packages = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]

    for runtime_module in ("ball.py", "court.py", "player.py", "BallTrack"):
        assert runtime_module in packages
