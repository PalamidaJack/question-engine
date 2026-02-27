from pathlib import Path

from qe.substrate.belief_ledger import BeliefLedger
from qe.substrate.cold_storage import ColdStorage


class Substrate:
    def __init__(self, db_path: str = "data/belief_ledger.db", cold_path: str = "cold") -> None:
        self.ledger = BeliefLedger(db_path, Path(__file__).parent / "migrations")
        self.cold_storage = ColdStorage(cold_path)

    async def initialize(self) -> None:
        await self.ledger.initialize()

    async def commit_claim(self, claim):
        return await self.ledger.commit_claim(claim)

    async def get_claims(self, **kwargs):
        return await self.ledger.get_claims(**kwargs)
