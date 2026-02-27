from pathlib import Path
import tomllib

from pydantic import ValidationError

from qe.models.genome import Blueprint


def load_blueprint(path: Path) -> Blueprint:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return Blueprint.model_validate(data)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ValueError(f"Failed to load blueprint from {path}: {exc}") from exc
