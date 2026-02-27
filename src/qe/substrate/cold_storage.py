import json
from pathlib import Path
from typing import Optional

from qe.models.envelope import Envelope


class ColdStorage:
    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)

    def append(self, envelope: Envelope) -> Path:
        """
        Write envelope to cold/YYYY/MM/{envelope_id}.json.
        Creates directories if missing. Returns path written.
        """
        year = envelope.timestamp.strftime("%Y")
        month = envelope.timestamp.strftime("%m")

        dir_path = self.base_path / year / month
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"{envelope.envelope_id}.json"

        with open(file_path, "w") as f:
            f.write(envelope.model_dump_json(indent=2))

        return file_path

    def read(self, envelope_id: str, year: int, month: int) -> Optional[Envelope]:
        """
        Read and deserialize envelope from cold/YYYY/MM/{envelope_id}.json.
        Returns None if not found.
        """
        file_path = self.base_path / str(year) / f"{month:02d}" / f"{envelope_id}.json"

        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        return Envelope(**data)
