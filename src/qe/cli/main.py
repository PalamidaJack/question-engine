from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.table import Table

from qe.bus import get_bus
from qe.kernel.supervisor import Supervisor
from qe.models.envelope import Envelope
from qe.substrate import Substrate

load_dotenv()

app = typer.Typer(help="Question Engine CLI")
claims_app = typer.Typer(help="Claim inspection commands")
hil_app = typer.Typer(help="Human-in-the-loop commands")
app.add_typer(claims_app, name="claims")
app.add_typer(hil_app, name="hil")

console = Console()
INBOX_DIR = Path("data/runtime_inbox")


def _genome_paths() -> list[Path]:
    genomes_dir = Path("genomes")
    if not genomes_dir.exists():
        return []
    return sorted(genomes_dir.glob("*.toml"))


async def _inbox_relay_loop() -> None:
    """Relay cross-process submissions into the in-memory bus singleton."""
    bus = get_bus()
    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        for item in sorted(INBOX_DIR.glob("*.json")):
            try:
                payload = json.loads(item.read_text(encoding="utf-8"))
                env = Envelope.model_validate(payload)
                bus.publish(env)
            finally:
                item.unlink(missing_ok=True)
        await asyncio.sleep(0.5)


@app.command()
def start() -> None:
    try:
        bus = get_bus()
        substrate = Substrate()

        async def _run() -> None:
            await substrate.initialize()
            supervisor = Supervisor(bus=bus, substrate=substrate, config_path=Path("config.toml"))
            relay_task = asyncio.create_task(_inbox_relay_loop())
            try:
                await supervisor.start(_genome_paths())
            finally:
                relay_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await relay_task

        import contextlib

        asyncio.run(_run())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise typer.Exit(code=1)


@app.command()
def submit(observation_text: str) -> None:
    envelope = Envelope(
        topic="observations.structured",
        source_service_id="cli",
        payload={"text": observation_text},
    )

    # Same-process best effort
    try:
        get_bus().publish(envelope)
    except Exception:
        pass

    # Cross-process path for qe start running in another process
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    inbox_file = INBOX_DIR / f"{envelope.envelope_id}.json"
    inbox_file.write_text(envelope.model_dump_json(), encoding="utf-8")

    console.print(f"Submitted observation envelope {envelope.envelope_id}")


@claims_app.command("list")
def claims_list(subject: str | None = typer.Option(None, "--subject")) -> None:
    async def _run() -> None:
        substrate = Substrate()
        await substrate.initialize()
        claims = await substrate.get_claims(subject_entity_id=subject)
        if not claims:
            console.print("No claims found.")
            return

        table = Table(title="Claims")
        table.add_column("claim_id")
        table.add_column("subject")
        table.add_column("predicate")
        table.add_column("object")
        table.add_column("confidence")
        table.add_column("created_at")
        for c in claims:
            table.add_row(
                c.claim_id,
                c.subject_entity_id,
                c.predicate,
                c.object_value,
                f"{c.confidence:.2f}",
                c.created_at.isoformat(),
            )
        console.print(table)

    asyncio.run(_run())


@claims_app.command("get")
def claims_get(claim_id: str) -> None:
    async def _run() -> None:
        substrate = Substrate()
        await substrate.initialize()
        claims = await substrate.get_claims(include_superseded=True)
        for claim in claims:
            if claim.claim_id == claim_id:
                console.print_json(data=claim.model_dump(mode="json"))
                return
        print(f"Claim not found: {claim_id}", file=sys.stderr)
        raise typer.Exit(code=1)

    asyncio.run(_run())


@hil_app.command("list")
def hil_list() -> None:
    pending_dir = Path("data/hil_queue/pending")
    pending_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pending_dir.glob("*.json"))
    if not files:
        console.print("No pending HIL proposals.")
        return

    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        console.print(
            f"{payload.get('envelope_id')} | reason={payload.get('reason')} | expires_at={payload.get('expires_at')}"
        )


@hil_app.command("approve")
def hil_approve(envelope_id: str) -> None:
    completed_dir = Path("data/hil_queue/completed")
    completed_dir.mkdir(parents=True, exist_ok=True)
    decision_file = completed_dir / f"{envelope_id}.json"
    decision_file.write_text(
        json.dumps({"decision": "approved", "decided_at": datetime.utcnow().isoformat()}, indent=2),
        encoding="utf-8",
    )
    console.print(f"Approved {envelope_id}")


@hil_app.command("reject")
def hil_reject(envelope_id: str, reason: str = typer.Option(..., "--reason")) -> None:
    completed_dir = Path("data/hil_queue/completed")
    completed_dir.mkdir(parents=True, exist_ok=True)
    decision_file = completed_dir / f"{envelope_id}.json"
    decision_file.write_text(
        json.dumps(
            {
                "decision": "rejected",
                "reason": reason,
                "decided_at": datetime.utcnow().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"Rejected {envelope_id}")


if __name__ == "__main__":
    app()
