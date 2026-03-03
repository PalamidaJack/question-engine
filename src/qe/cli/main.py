from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
dlq_app = typer.Typer(help="Dead-letter queue commands")
export_app = typer.Typer(help="Data export commands")
db_app = typer.Typer(help="Database management commands")
app.add_typer(claims_app, name="claims")
app.add_typer(hil_app, name="hil")
app.add_typer(ingest_app, name="ingest")
app.add_typer(goal_app, name="goal")
app.add_typer(dlq_app, name="dlq")
app.add_typer(export_app, name="export")
app.add_typer(db_app, name="db")
a2a_app = typer.Typer(help="Agent-to-Agent (A2A) commands")
app.add_typer(a2a_app, name="a2a")
otel_app = typer.Typer(help="OpenTelemetry commands")
app.add_typer(otel_app, name="otel")
guardrails_app = typer.Typer(help="Guardrails commands")
app.add_typer(guardrails_app, name="guardrails")
memory_app = typer.Typer(help="Memory subsystem commands")
app.add_typer(memory_app, name="memory")
arena_app = typer.Typer(help="Competitive arena commands")
app.add_typer(arena_app, name="arena")
models_app = typer.Typer(help="Model configuration commands")
app.add_typer(models_app, name="models")
doctor_app = typer.Typer(help="System diagnostics")
app.add_typer(doctor_app, name="doctor")

console = Console()
INBOX_DIR = Path("data/runtime_inbox")


async def _reindex_claim_embeddings(
    substrate: Substrate,
    *,
    include_superseded: bool = False,
    dry_run: bool = False,
    batch_size: int = 500,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Rebuild claim embeddings from the belief ledger.

    Returns stats: {"deleted": int, "indexed": int, "total": int}.
    """
    import aiosqlite

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    claims = await substrate.get_claims(include_superseded=include_superseded)
    total = len(claims)

    db_path = substrate.belief_ledger._db_path
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE id LIKE 'claim:%'"
        )
        row = await cursor.fetchone()
        existing_claim_vectors = int(row[0]) if row else 0

    if dry_run:
        return {
            "deleted": existing_claim_vectors,
            "indexed": total,
            "total": total,
            "dry_run": 1,
        }

    # Remove existing claim vectors, keep non-claim embeddings (e.g. goal patterns).
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("DELETE FROM embeddings WHERE id LIKE 'claim:%'")
        await db.commit()
        deleted = max(cursor.rowcount, 0)

    # Invalidate HNSW index so it rebuilds on next query.
    substrate.embeddings._hnsw_dirty = True

    indexed = 0
    for offset in range(0, total, batch_size):
        batch = claims[offset : offset + batch_size]
        for claim in batch:
            await substrate.embeddings.store(
                id=f"claim:{claim.claim_id}",
                text=(
                    f"{claim.subject_entity_id} "
                    f"{claim.predicate} "
                    f"{claim.object_value}"
                ),
                metadata={
                    "kind": "claim",
                    "claim_id": claim.claim_id,
                    "subject_entity_id": claim.subject_entity_id,
                    "predicate": claim.predicate,
                },
            )
            indexed += 1
            if progress_callback is not None:
                progress_callback(indexed, total)

    return {"deleted": deleted, "indexed": indexed, "total": total, "dry_run": 0}


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
    from qe.api.setup import get_settings
    from qe.runtime.logging_config import configure_from_config

    configure_from_config(get_settings(), verbose=verbose)


@a2a_app.command("discover")
def a2a_discover(url: str) -> None:
    """Fetch and display an external agent's AgentCard from <url>/.well-known/agent.json"""
    import httpx

    try:
        r = httpx.get(f"{url.rstrip('/')}/.well-known/agent.json", timeout=10)
        r.raise_for_status()
        console.print_json(data=r.json())
    except Exception as e:
        console.print(f"Failed to discover agent: {e}")


@a2a_app.command("send")
def a2a_send(url: str, task: str) -> None:
    """Send a task description to an external A2A agent at <url>."""
    import httpx

    try:
        r = httpx.post(f"{url.rstrip('/')}/api/a2a/tasks", json={"description": task}, timeout=15)
        r.raise_for_status()
        console.print_json(data=r.json())
    except Exception as e:
        console.print(f"Failed to send task: {e}")


@otel_app.command("status")
def otel_status() -> None:
    """Show current OpenTelemetry config from config.toml."""
    from qe.config import load_config

    cfg = load_config(Path("config.toml"))
    ot = cfg.otel
    console.print_json(data={
        "enabled": ot.enabled,
        "exporter": ot.exporter,
        "otlp_endpoint": ot.otlp_endpoint or "",
        "service_name": ot.service_name,
    })


@guardrails_app.command("status")
def guardrails_status() -> None:
    """Show current guardrails config from config.toml."""
    from qe.config import load_config

    cfg = load_config(Path("config.toml"))
    g = cfg.guardrails
    console.print_json(data=g.model_dump(mode="json"))


@guardrails_app.command("set")
def guardrails_set(
    content_filter_enabled: bool | None = typer.Option(
        None, "--content-filter/--no-content-filter",
    ),
    pii_detection_enabled: bool | None = typer.Option(None, "--pii/--no-pii"),
    cost_guard_enabled: bool | None = typer.Option(None, "--cost-guard/--no-cost-guard"),
    cost_guard_threshold_usd: float | None = typer.Option(None, "--cost-threshold"),
) -> None:
    """Update guardrails settings in `config.toml` (writes file).

    Only provided options are updated; others remain unchanged.
    """
    from qe.config import load_config

    cfg = load_config(Path("config.toml"))
    changed = False
    if content_filter_enabled is not None:
        cfg.guardrails.content_filter_enabled = content_filter_enabled
        changed = True
    if pii_detection_enabled is not None:
        cfg.guardrails.pii_detection_enabled = pii_detection_enabled
        changed = True
    if cost_guard_enabled is not None:
        cfg.guardrails.cost_guard_enabled = cost_guard_enabled
        changed = True
    if cost_guard_threshold_usd is not None:
        cfg.guardrails.cost_guard_threshold_usd = cost_guard_threshold_usd
        changed = True

    if not changed:
        console.print("No changes provided.")
        raise typer.Exit()

    # Write back to config.toml preserving other fields via simple toml dump
    try:
        import tomllib

        import tomli_w

        raw = {}
        if Path("config.toml").exists():
            with Path("config.toml").open("rb") as f:
                raw = tomllib.load(f)

        raw.setdefault("guardrails", {})
        raw["guardrails"]["content_filter_enabled"] = cfg.guardrails.content_filter_enabled
        raw["guardrails"]["pii_detection_enabled"] = cfg.guardrails.pii_detection_enabled
        raw["guardrails"]["cost_guard_enabled"] = cfg.guardrails.cost_guard_enabled
        raw["guardrails"]["cost_guard_threshold_usd"] = cfg.guardrails.cost_guard_threshold_usd

        with Path("config.toml").open("w", encoding="utf-8") as f:
            f.write(tomli_w.dumps(raw))
        console.print("Guardrails updated in config.toml")
    except Exception as e:
        console.print(f"Failed to update config.toml: {e}")


@a2a_app.command("list-peers")
def a2a_list_peers(
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """List registered A2A peers."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/a2a/peers", timeout=10)
        data = resp.json()
        peers = data.get("peers", [])
        if not peers:
            console.print("No peers registered.")
            return

        table = Table(title=f"A2A Peers ({data.get('total_peers', 0)})")
        table.add_column("peer_id")
        table.add_column("name")
        table.add_column("url")
        table.add_column("healthy")
        table.add_column("capabilities")
        for p in peers:
            table.add_row(
                p["peer_id"],
                p.get("name", ""),
                p["url"],
                "[green]yes[/green]" if p.get("healthy") else "[red]no[/red]",
                ", ".join(p.get("capabilities", []))[:40],
            )
        console.print(table)
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@a2a_app.command("register-peer")
def a2a_register_peer(
    url: str,
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Register a peer agent by URL (discovers capabilities automatically)."""
    import httpx

    try:
        resp = httpx.post(
            f"{base_url}/api/a2a/peers",
            json={"url": url},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            console.print(
                f"[green]Registered peer:[/green] {data.get('peer_id', '?')}"
                f" ({data.get('name', 'unknown')})"
            )
        else:
            console.print(f"[red]Error:[/red] {resp.text}")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@a2a_app.command("remove-peer")
def a2a_remove_peer(
    peer_id: str,
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Remove a registered peer."""
    import httpx

    try:
        resp = httpx.delete(
            f"{base_url}/api/a2a/peers/{peer_id}", timeout=10,
        )
        if resp.status_code == 200:
            console.print(f"[green]Removed peer: {peer_id}[/green]")
        else:
            console.print(f"[red]Peer not found: {peer_id}[/red]")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@a2a_app.command("check-peer")
def a2a_check_peer(
    peer_id: str,
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Check connectivity of a registered peer."""
    import httpx

    try:
        resp = httpx.get(
            f"{base_url}/api/a2a/peers/{peer_id}/health", timeout=10,
        )
        data = resp.json()
        if data.get("healthy"):
            console.print(f"[green]Peer {peer_id} is healthy[/green]")
        else:
            console.print(f"[red]Peer {peer_id} is unreachable[/red]")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


# ── Memory CLI ──────────────────────────────────────────────────────


@memory_app.command("search")
def memory_search(
    query: str,
    tier: str = typer.Option("all", "--tier", help="episodic|belief|procedural|all"),
    top_k: int = typer.Option(10, "--top-k"),
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Search across memory tiers."""
    import httpx

    try:
        resp = httpx.post(
            f"{base_url}/api/memory/search",
            json={"query": query, "tier": tier, "top_k": top_k},
            timeout=30,
        )
        console.print_json(data=resp.json())
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@memory_app.command("tiers")
def memory_tiers(
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Show memory tier status and sizes."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/memory/tiers", timeout=10)
        data = resp.json()
        for tier_name, info in data.items():
            if isinstance(info, dict):
                console.print(f"\n[bold]{tier_name}[/bold]")
                for k, v in info.items():
                    console.print(f"  {k}: {v}")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@memory_app.command("export")
def memory_export(
    output: Path | None = typer.Option(None, "--output"),  # noqa: B008
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Export all memory tiers to JSON."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/memory/export", timeout=30)
        data = resp.json()
        if output:
            output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console.print(f"[green]Memory exported to {output}[/green]")
        else:
            console.print_json(data=data)
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


# ── Arena CLI ───────────────────────────────────────────────────────


@arena_app.command("status")
def arena_status_cmd(
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Show competitive arena status and Elo rankings."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/arena/status", timeout=10)
        data = resp.json()
        rankings = data.get("rankings", [])
        if not rankings:
            console.print("No arena data yet.")
            console.print_json(data=data)
            return

        table = Table(title="Agent Elo Rankings")
        table.add_column("Agent")
        table.add_column("Elo")
        table.add_column("W/L/D")
        for r in rankings:
            table.add_row(
                r.get("agent_id", "?"),
                str(r.get("elo", 1200)),
                f"{r.get('wins', 0)}/{r.get('losses', 0)}/{r.get('draws', 0)}",
            )
        console.print(table)
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


# ── Models CLI ──────────────────────────────────────────────────────


@models_app.command("list")
def models_list() -> None:
    """Show configured model tiers from config.toml."""
    from qe.config import load_config

    cfg = load_config(Path("config.toml"))
    table = Table(title="Configured Model Tiers")
    table.add_column("Tier")
    table.add_column("Model ID")
    table.add_row("fast", cfg.models.fast)
    table.add_row("balanced", cfg.models.balanced)
    table.add_row("powerful", cfg.models.powerful)
    table.add_row("local", cfg.models.local or "(not set)")
    console.print(table)


@models_app.command("check")
def models_check(
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Check model availability via the running API server."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/status", timeout=10)
        data = resp.json()
        budget = data.get("budget", {})
        console.print(f"Budget used: ${budget.get('used_usd', 0):.2f}")
        console.print(f"Budget limit: ${budget.get('limit_usd', 0):.2f}")
        cb = data.get("circuit_breakers", {})
        if cb:
            console.print("\nCircuit Breakers:")
            for name, state in cb.items():
                c = "green" if state == "closed" else "red"
                console.print(f"  [{c}]{name}: {state}[/{c}]")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


# ── Doctor CLI ──────────────────────────────────────────────────────


@doctor_app.command("check")
def doctor_check() -> None:
    """Run system health diagnostics."""
    from qe.config import load_config

    console.print("[bold]QE Doctor — System Diagnostics[/bold]\n")

    # 1. Config validation
    try:
        cfg = load_config(Path("config.toml"))
        console.print("[green]Config:[/green] config.toml is valid")
    except Exception as exc:
        console.print(f"[red]Config:[/red] config.toml invalid — {exc}")
        cfg = None

    # 2. Data directory
    data_dir = Path("data")
    if data_dir.exists():
        db_files = list(data_dir.glob("*.db"))
        console.print(
            f"[green]Data:[/green] {len(db_files)} database(s) in data/"
        )
    else:
        console.print("[yellow]Data:[/yellow] data/ directory not found")

    # 3. Environment
    env_file = Path(".env")
    if env_file.exists():
        lines = env_file.read_text().splitlines()
        key_count = sum(
            1 for line in lines
            if line.strip() and not line.strip().startswith("#")
        )
        console.print(f"[green]Env:[/green] .env has {key_count} keys")
    else:
        console.print("[yellow]Env:[/yellow] .env file not found")

    # 4. Python version
    import sys
    console.print(f"[green]Python:[/green] {sys.version.split()[0]}")

    # 5. Key dependencies
    deps_ok = True
    for pkg in ["fastapi", "pydantic", "litellm", "instructor"]:
        try:
            __import__(pkg)
            console.print(f"[green]Dep:[/green] {pkg} installed")
        except ImportError:
            console.print(f"[red]Dep:[/red] {pkg} NOT installed")
            deps_ok = False

    # 6. Config summary
    if cfg:
        console.print("\n[bold]Config Summary:[/bold]")
        console.print(f"  Log level: {cfg.runtime.log_level}")
        console.print(
            f"  Budget: ${cfg.budget.monthly_limit_usd:.2f}/month"
        )
        console.print(
            f"  Models: {cfg.models.fast} / {cfg.models.balanced}"
            f" / {cfg.models.powerful}"
        )
        console.print(f"  Guardrails: {'enabled' if cfg.guardrails.enabled else 'disabled'}")
        console.print(f"  OTEL: {'enabled' if cfg.otel.enabled else 'disabled'}")
        console.print(f"  A2A: {'enabled' if cfg.a2a.enabled else 'disabled'}")
        console.print(f"  Scout: {'enabled' if cfg.scout.enabled else 'disabled'}")

    if deps_ok:
        console.print("\n[green]All checks passed.[/green]")
    else:
        console.print("\n[yellow]Some issues found. See above.[/yellow]")


@doctor_app.command("connectivity")
def doctor_connectivity(
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Test connectivity to the QE API server and its components."""
    import httpx

    console.print("[bold]Connectivity Check[/bold]\n")

    # API server
    try:
        resp = httpx.get(f"{base_url}/api/health", timeout=5)
        data = resp.json()
        console.print(
            f"[green]API Server:[/green] {data.get('status', 'ok')}"
        )
    except httpx.ConnectError:
        console.print("[red]API Server:[/red] unreachable")
        raise typer.Exit(code=1) from None

    # Readiness
    try:
        resp = httpx.get(f"{base_url}/api/health/ready", timeout=5)
        data = resp.json()
        ready = data.get("ready", False)
        c = "green" if ready else "yellow"
        console.print(f"[{c}]Readiness:[/{c}] {'ready' if ready else 'not ready'}")
        phases = data.get("phases", {})
        for phase, done in phases.items():
            pc = "green" if done else "yellow"
            console.print(f"  [{pc}]{phase}: {'done' if done else 'pending'}[/{pc}]")
    except Exception:
        console.print("[yellow]Readiness:[/yellow] check failed")


@app.command("init")
def init_wizard() -> None:
    """Interactive first-time setup wizard."""
    console.print("[bold]Question Engine — First-Time Setup[/bold]\n")

    config_path = Path("config.toml")
    env_path = Path(".env")

    if config_path.exists():
        console.print("[yellow]config.toml already exists.[/yellow]")
        overwrite = typer.confirm("Overwrite?", default=False)
        if not overwrite:
            console.print("Keeping existing config.")
            return

    # Build basic config
    console.print("\n[bold]Model Configuration[/bold]")
    fast = typer.prompt("Fast model", default="gpt-4o-mini")
    balanced = typer.prompt("Balanced model", default="gpt-4o")
    powerful = typer.prompt("Powerful model", default="o1-preview")

    console.print("\n[bold]Budget[/bold]")
    budget = typer.prompt(
        "Monthly budget limit (USD)", default="50.0", type=float,
    )

    console.print("\n[bold]Runtime[/bold]")
    log_level = typer.prompt(
        "Log level (DEBUG/INFO/WARNING/ERROR)", default="INFO",
    )

    # Write config.toml
    config_content = (
        f'[budget]\nmonthly_limit_usd = {budget}\n\n'
        f'[runtime]\nlog_level = "{log_level}"\n\n'
        f'[models]\nfast = "{fast}"\n'
        f'balanced = "{balanced}"\n'
        f'powerful = "{powerful}"\n'
    )
    config_path.write_text(config_content, encoding="utf-8")
    console.print(f"\n[green]Wrote {config_path}[/green]")

    # API key setup
    if not env_path.exists():
        console.print("\n[bold]API Key Setup[/bold]")
        api_key = typer.prompt(
            "OpenAI/LLM API key (or press Enter to skip)", default="",
        )
        if api_key:
            env_path.write_text(
                f"OPENAI_API_KEY={api_key}\n", encoding="utf-8",
            )
            console.print(f"[green]Wrote {env_path}[/green]")

    # Create data directory
    Path("data").mkdir(exist_ok=True)
    console.print("[green]Created data/ directory[/green]")
    console.print("\n[green]Setup complete! Run `qe serve` to start.[/green]")


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
def health(
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Check engine health by hitting the live health endpoint."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/health/live", timeout=10)
        data = resp.json()
        overall = data.get("overall_status", "unknown")
        color = "green" if overall == "healthy" else "red"
        console.print(f"[{color}]Health: {overall}[/{color}]")
        for check in data.get("checks", []):
            status = check.get("status", "?")
            c = "green" if status == "pass" else ("yellow" if status == "warn" else "red")
            console.print(f"  [{c}]{check.get('name', '?')}: {status}[/{c}]")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@app.command()
def metrics(
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Display engine metrics."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/metrics", timeout=10)
        console.print_json(data=resp.json())
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@app.command("config-validate")
def config_validate(
    config_path: Path = typer.Option(Path("config.toml"), "--config"),  # noqa: B008
) -> None:
    """Validate config.toml without starting the engine."""
    from pydantic import ValidationError

    from qe.config import load_config

    try:
        cfg = load_config(config_path)
        console.print("[green]Config valid.[/green]")
        console.print(f"  Budget limit: ${cfg.budget.monthly_limit_usd:.2f}")
        console.print(f"  Log level: {cfg.runtime.log_level}")
        models = f"{cfg.models.fast} / {cfg.models.balanced} / {cfg.models.powerful}"
        console.print(f"  Models: {models}")
    except ValidationError as exc:
        console.print("[red]Config validation failed:[/red]")
        for err in exc.errors():
            loc = " → ".join(str(x) for x in err["loc"])
            console.print(f"  {loc}: {err['msg']}")
        raise typer.Exit(code=1) from None
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(code=1) from None


@dlq_app.command("list")
def dlq_list(
    limit: int = typer.Option(100, "--limit"),
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """List dead-letter queue entries."""
    import httpx

    try:
        resp = httpx.get(f"{base_url}/api/dlq", params={"limit": limit}, timeout=10)
        data = resp.json()
        entries = data.get("entries", [])
        if not entries:
            console.print("DLQ is empty.")
            return

        table = Table(title=f"Dead Letter Queue ({data.get('count', 0)} entries)")
        table.add_column("envelope_id")
        table.add_column("topic")
        table.add_column("error")
        table.add_column("attempts")
        for e in entries:
            table.add_row(
                e["envelope_id"][:12] + "...",
                e["topic"],
                e["error"][:60],
                str(e["attempts"]),
            )
        console.print(table)
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@dlq_app.command("replay")
def dlq_replay(
    envelope_id: str,
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Replay a dead-lettered envelope."""
    import httpx

    try:
        resp = httpx.post(f"{base_url}/api/dlq/{envelope_id}/replay", timeout=10)
        if resp.status_code == 404:
            console.print(f"[red]Envelope not found in DLQ: {envelope_id}[/red]")
            raise typer.Exit(code=1)
        console.print(f"[green]Replayed: {envelope_id}[/green]")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@dlq_app.command("purge")
def dlq_purge(base_url: str = typer.Option("http://localhost:8000", "--url")) -> None:
    """Purge all DLQ entries."""
    import httpx

    try:
        resp = httpx.delete(f"{base_url}/api/dlq", timeout=10)
        data = resp.json()
        console.print(f"[green]Purged {data.get('count', 0)} entries.[/green]")
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@export_app.command("claims")
def export_claims(
    subject: str | None = typer.Option(None, "--subject"),
    fmt: str = typer.Option("json", "--format"),
    output: Path | None = typer.Option(None, "--output"),  # noqa: B008
) -> None:
    """Export claims from the belief ledger to JSON or CSV."""
    async def _run() -> None:
        substrate = Substrate()
        await substrate.initialize()
        claims = await substrate.get_claims(
            subject_entity_id=subject, include_superseded=True
        )
        rows = [c.model_dump(mode="json") for c in claims]

        if fmt == "csv":
            import csv
            import io

            buf = io.StringIO()
            if rows:
                writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            text = buf.getvalue()
        else:
            text = json.dumps(rows, indent=2, default=str)

        if output:
            output.write_text(text, encoding="utf-8")  # noqa: ASYNC240
            console.print(f"Exported {len(rows)} claims to {output}")
        else:
            console.print(text)

    asyncio.run(_run())


@export_app.command("events")
def export_events(
    topic: str | None = typer.Option(None, "--topic"),
    limit: int = typer.Option(1000, "--limit"),
    fmt: str = typer.Option("json", "--format"),
    output: Path | None = typer.Option(None, "--output"),  # noqa: B008
) -> None:
    """Export events from the event log to JSON or CSV."""
    from qe.bus.event_log import EventLog

    async def _run() -> None:
        el = EventLog()
        await el.initialize()
        events = await el.replay(topic=topic, limit=limit)

        if fmt == "csv":
            import csv
            import io

            buf = io.StringIO()
            if events:
                writer = csv.DictWriter(
                    buf, fieldnames=events[0].keys()
                )
                writer.writeheader()
                for e in events:
                    row = {
                        k: json.dumps(v) if isinstance(v, dict) else v
                        for k, v in e.items()
                    }
                    writer.writerow(row)
            text = buf.getvalue()
        else:
            text = json.dumps(events, indent=2, default=str)

        if output:
            output.write_text(text, encoding="utf-8")  # noqa: ASYNC240
            console.print(f"Exported {len(events)} events to {output}")
        else:
            console.print(text)

    asyncio.run(_run())


@app.command("events-replay")
def events_replay(
    topic: str | None = typer.Option(None, "--topic"),
    since: str | None = typer.Option(None, "--since"),
    limit: int = typer.Option(100, "--limit"),
    base_url: str = typer.Option("http://localhost:8000", "--url"),
) -> None:
    """Replay historical events from the event log into the bus."""
    import httpx

    try:
        body: dict[str, Any] = {"limit": limit}
        if topic:
            body["topic"] = topic
        if since:
            body["since"] = since
        resp = httpx.post(
            f"{base_url}/api/events/replay",
            json=body,
            timeout=60,
        )
        data = resp.json()
        console.print(
            f"[green]Replayed {data.get('count', 0)} events.[/green]"
        )
    except httpx.ConnectError:
        console.print("[red]Cannot connect to QE API server.[/red]")
        raise typer.Exit(code=1) from None


@db_app.command("backup")
def db_backup(
    dest: str = typer.Option("data/backups", "--dest"),
) -> None:
    """Back up all QE databases."""
    from qe.runtime.db_backup import backup_all

    result = backup_all(dest_dir=dest)
    console.print(
        f"[green]Backup complete:[/green] {result['backup_dir']}"
    )
    for db in result["databases"]:
        status = db["status"]
        c = "green" if status == "completed" else "yellow"
        size = db.get("size_bytes", 0)
        console.print(f"  [{c}]{db['source']}: {status} ({size} bytes)[/{c}]")


@db_app.command("restore")
def db_restore(
    backup_dir: str = typer.Argument(..., help="Backup directory path"),
) -> None:
    """Restore databases from a backup directory."""
    from qe.runtime.db_backup import restore_database

    backup_path = Path(backup_dir)
    if not backup_path.exists():
        console.print(f"[red]Backup dir not found: {backup_dir}[/red]")
        raise typer.Exit(code=1)

    db_files = list(backup_path.glob("*.db"))
    if not db_files:
        console.print("[red]No .db files found in backup directory.[/red]")
        raise typer.Exit(code=1)

    for db_file in db_files:
        dest = f"data/{db_file.name}"
        result = restore_database(str(db_file), dest)
        status = result["status"]
        c = "green" if status == "restored" else "red"
        console.print(f"  [{c}]{db_file.name}: {status}[/{c}]")


@db_app.command("list-backups")
def db_list_backups(
    backup_dir: str = typer.Option("data/backups", "--dir"),
) -> None:
    """List available database backups."""
    from qe.runtime.db_backup import list_backups

    backups = list_backups(backup_dir)
    if not backups:
        console.print("No backups found.")
        return

    table = Table(title="Database Backups")
    table.add_column("Name")
    table.add_column("Databases")
    table.add_column("Size")
    for b in backups:
        size_kb = b["total_size_bytes"] / 1024
        table.add_row(
            b["name"],
            ", ".join(b["databases"]),
            f"{size_kb:.1f} KB",
        )
    console.print(table)


@db_app.command("reindex-embeddings")
def db_reindex_embeddings(
    include_superseded: bool = typer.Option(
        False,
        "--include-superseded",
        help="Reindex superseded claims too.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be reindexed without modifying embeddings.",
    ),
    batch_size: int = typer.Option(
        500,
        "--batch-size",
        min=1,
        help="Number of claims to process per batch.",
    ),
) -> None:
    """Rebuild claim embeddings from the current belief ledger."""

    async def _run() -> None:
        substrate = Substrate()
        await substrate.initialize()

        def _progress(indexed: int, total: int) -> None:
            if total == 0:
                return
            if indexed % 100 == 0 or indexed == total:
                console.print(f"Indexed {indexed}/{total} claims...")

        stats = await _reindex_claim_embeddings(
            substrate,
            include_superseded=include_superseded,
            dry_run=dry_run,
            batch_size=batch_size,
            progress_callback=_progress,
        )
        if dry_run:
            console.print(
                "[yellow]Dry run complete.[/yellow] "
                f"would_delete={stats['deleted']} "
                f"would_index={stats['indexed']} total={stats['total']}"
            )
        else:
            console.print(
                "[green]Embedding reindex complete.[/green] "
                f"deleted={stats['deleted']} indexed={stats['indexed']} total={stats['total']}"
            )

    try:
        asyncio.run(_run())
    except Exception as exc:
        log.error("Embedding reindex failed: %s", exc)
        raise typer.Exit(code=1) from exc


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
