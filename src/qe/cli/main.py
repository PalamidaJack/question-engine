from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from qe.bus import get_bus
from qe.kernel.supervisor import Supervisor
from qe.models.envelope import Envelope
from qe.substrate import Substrate

load_dotenv()

log = logging.getLogger(__name__)

app = typer.Typer(help="Question Engine CLI")
claims_app = typer.Typer(help="Claim inspection commands")
hil_app = typer.Typer(help="Human-in-the-loop commands")
ingest_app = typer.Typer(help="Ingestion commands")
goal_app = typer.Typer(help="Goal management commands")
app.add_typer(claims_app, name="claims")
app.add_typer(hil_app, name="hil")
app.add_typer(ingest_app, name="ingest")
app.add_typer(goal_app, name="goal")

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
    INBOX_DIR.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    while True:
        for item in sorted(INBOX_DIR.glob("*.json")):  # noqa: ASYNC240
            try:
                payload = json.loads(item.read_text(encoding="utf-8"))
                env = Envelope.model_validate(payload)
                bus.publish(env)
            finally:
                item.unlink(missing_ok=True)
        await asyncio.sleep(0.5)


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def start(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Start the Question Engine daemon."""
    _configure_logging(verbose)
    try:
        bus = get_bus()
        substrate = Substrate()

        async def _run() -> None:
            await substrate.initialize()
            supervisor = Supervisor(
                bus=bus, substrate=substrate, config_path=Path("config.toml")
            )
            relay_task = asyncio.create_task(_inbox_relay_loop())
            try:
                await supervisor.start(_genome_paths())
            finally:
                relay_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await relay_task

        asyncio.run(_run())
    except Exception as exc:
        log.error("Engine failed: %s", exc)
        raise typer.Exit(code=1) from exc


@app.command()
def submit(observation_text: str) -> None:
    """Submit an observation to the engine for claim extraction."""
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


@app.command()
def ask(question: str) -> None:
    """Ask a question against the belief ledger."""
    from qe.services.query import answer_question

    async def _run() -> None:
        substrate = Substrate()
        await substrate.initialize()
        result = await answer_question(question, substrate)

        console.print(f"\n[bold]Answer:[/bold] {result['answer']}")
        console.print(f"[bold]Confidence:[/bold] {result['confidence']:.0%}")
        console.print(f"[bold]Reasoning:[/bold] {result['reasoning']}")
        if result["supporting_claims"]:
            console.print(f"\n[bold]Supporting claims:[/bold] ({len(result['supporting_claims'])})")
            for c in result["supporting_claims"]:
                console.print(
                    f"  - [{c['confidence']:.0%}] {c['subject_entity_id']} "
                    f"{c['predicate']} {c['object_value']}"
                )

    asyncio.run(_run())


@claims_app.command("list")
def claims_list(
    subject: str | None = typer.Option(None, "--subject"),
) -> None:
    """List all claims in the belief ledger."""
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
    """Get a specific claim by ID."""
    async def _run() -> None:
        substrate = Substrate()
        await substrate.initialize()
        claims = await substrate.get_claims(include_superseded=True)
        for claim in claims:
            if claim.claim_id == claim_id:
                console.print_json(data=claim.model_dump(mode="json"))
                return
        log.error("Claim not found: %s", claim_id)
        raise typer.Exit(code=1)

    asyncio.run(_run())


@hil_app.command("list")
def hil_list() -> None:
    """List pending HIL approval requests."""
    pending_dir = Path("data/hil_queue/pending")
    pending_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pending_dir.glob("*.json"))
    if not files:
        console.print("No pending HIL proposals.")
        return

    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        eid = payload.get("envelope_id")
        reason = payload.get("reason")
        expires = payload.get("expires_at")
        console.print(f"{eid} | reason={reason} | expires_at={expires}")


@hil_app.command("approve")
def hil_approve(envelope_id: str) -> None:
    """Approve a pending HIL request."""
    completed_dir = Path("data/hil_queue/completed")
    completed_dir.mkdir(parents=True, exist_ok=True)
    decision_file = completed_dir / f"{envelope_id}.json"
    decision_file.write_text(
        json.dumps(
            {
                "decision": "approved",
                "decided_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"Approved {envelope_id}")


@hil_app.command("reject")
def hil_reject(
    envelope_id: str,
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Reject a pending HIL request."""
    completed_dir = Path("data/hil_queue/completed")
    completed_dir.mkdir(parents=True, exist_ok=True)
    decision_file = completed_dir / f"{envelope_id}.json"
    decision_file.write_text(
        json.dumps(
            {
                "decision": "rejected",
                "reason": reason,
                "decided_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"Rejected {envelope_id}")


@ingest_app.command("text")
def ingest_text(text: str) -> None:
    """Ingest a single text observation."""
    envelope = Envelope(
        topic="observations.structured",
        source_service_id="cli-ingest",
        payload={"text": text},
    )

    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    inbox_file = INBOX_DIR / f"{envelope.envelope_id}.json"
    inbox_file.write_text(envelope.model_dump_json(), encoding="utf-8")

    console.print(f"Ingested: {envelope.envelope_id}")


@ingest_app.command("file")
def ingest_file(file_path: Path) -> None:
    """Ingest observations from a text file (one per line)."""
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(code=1)

    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        envelope = Envelope(
            topic="observations.structured",
            source_service_id="cli-ingest",
            payload={"text": line},
        )
        inbox_file = INBOX_DIR / f"{envelope.envelope_id}.json"
        inbox_file.write_text(envelope.model_dump_json(), encoding="utf-8")
        count += 1

    console.print(f"Ingested {count} observations from {file_path}")


@goal_app.callback(invoke_without_command=True)
def goal_submit(
    ctx: typer.Context,
    description: str = typer.Argument(None, help="Goal description"),
) -> None:
    """Submit a goal or manage goals. Use subcommands for list/status/pause/resume/cancel."""
    if ctx.invoked_subcommand is not None:
        return
    if not description:
        console.print("[red]Usage: qe goal 'description' or qe goal list[/red]")
        raise typer.Exit(code=1)

    import httpx

    try:
        resp = httpx.post(
            "http://localhost:8000/api/goals",
            json={"description": description},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            console.print(f"[green]Goal submitted:[/green] {data['goal_id']}")
            console.print(f"  Status: {data['status']}")
            console.print(f"  Subtasks: {data['subtask_count']}")
            if data.get("strategy"):
                console.print(f"  Strategy: {data['strategy']}")
        else:
            console.print(f"[red]Error:[/red] {resp.text}")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server. Is it running?[/red]")
        raise typer.Exit(code=1) from None


@goal_app.command("list")
def goal_list() -> None:
    """List all goals."""
    import httpx

    try:
        resp = httpx.get("http://localhost:8000/api/goals", timeout=10)
        data = resp.json()
        goals = data.get("goals", [])
        if not goals:
            console.print("No goals found.")
            return

        table = Table(title="Goals")
        table.add_column("goal_id")
        table.add_column("description")
        table.add_column("status")
        table.add_column("subtasks")
        table.add_column("created_at")
        for g in goals:
            table.add_row(
                g["goal_id"],
                g["description"][:60],
                g["status"],
                str(g["subtask_count"]),
                g["created_at"],
            )
        console.print(table)
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@goal_app.command("status")
def goal_status(goal_id: str) -> None:
    """Show detailed status of a goal."""
    import httpx

    try:
        resp = httpx.get(
            f"http://localhost:8000/api/goals/{goal_id}", timeout=10
        )
        if resp.status_code == 404:
            console.print(f"[red]Goal not found: {goal_id}[/red]")
            raise typer.Exit(code=1)
        console.print_json(data=resp.json())
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@goal_app.command("pause")
def goal_pause(goal_id: str) -> None:
    """Pause a running goal."""
    import httpx

    try:
        resp = httpx.post(
            f"http://localhost:8000/api/goals/{goal_id}/pause", timeout=10
        )
        console.print(resp.json())
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@goal_app.command("resume")
def goal_resume(goal_id: str) -> None:
    """Resume a paused goal."""
    import httpx

    try:
        resp = httpx.post(
            f"http://localhost:8000/api/goals/{goal_id}/resume", timeout=10
        )
        console.print(resp.json())
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@goal_app.command("cancel")
def goal_cancel(goal_id: str) -> None:
    """Cancel a running goal."""
    import httpx

    try:
        resp = httpx.post(
            f"http://localhost:8000/api/goals/{goal_id}/cancel", timeout=10
        )
        console.print(resp.json())
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the QE API server (FastAPI + WebSocket)."""
    _configure_logging(verbose)
    import uvicorn

    console.print(f"Starting QE API server on {host}:{port}")
    uvicorn.run("qe.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
