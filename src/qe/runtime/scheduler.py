"""Task Scheduler — cron-based scheduling for periodic tasks.

Executes tasks via the workflow executor based on cron syntax.
Gated behind ``task_scheduler`` feature flag.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled task with cron-like timing."""

    task_id: str
    name: str
    cron_expr: str  # Simplified: "*/5 * * * *" or interval seconds
    action: str  # Action identifier
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: float = 0.0
    run_count: int = 0
    max_runs: int = 0  # 0 = unlimited

    @property
    def interval_seconds(self) -> int | None:
        """Parse simple interval from cron_expr like '*/5' (minutes)."""
        m = re.match(r"\*/(\d+)", self.cron_expr)
        if m:
            return int(m.group(1)) * 60
        # Support raw seconds: "every:300"
        m = re.match(r"every:(\d+)", self.cron_expr)
        if m:
            return int(m.group(1))
        return None

    def is_due(self, now: float | None = None) -> bool:
        """Check if this task is due to run."""
        if not self.enabled:
            return False
        if self.max_runs > 0 and self.run_count >= self.max_runs:
            return False
        interval = self.interval_seconds
        if interval is None:
            return False
        current = now or time.time()
        return (current - self.last_run) >= interval

    def mark_run(self) -> None:
        self.last_run = time.time()
        self.run_count += 1


class TaskScheduler:
    """Simple cron-like scheduler for periodic tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._next_id = 0

    def register_handler(
        self, action: str, handler: Callable[..., Any],
    ) -> None:
        """Register a handler function for an action type."""
        self._handlers[action] = handler

    def schedule(
        self,
        name: str,
        cron_expr: str,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        max_runs: int = 0,
    ) -> str:
        """Schedule a new task. Returns task_id."""
        self._next_id += 1
        task_id = f"sched_{self._next_id:04d}"
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            cron_expr=cron_expr,
            action=action,
            params=params or {},
            max_runs=max_runs,
        )
        self._tasks[task_id] = task
        log.info(
            "scheduler.task_added id=%s name=%s cron=%s",
            task_id, name, cron_expr,
        )
        return task_id

    def unschedule(self, task_id: str) -> bool:
        """Remove a scheduled task."""
        return self._tasks.pop(task_id, None) is not None

    def enable(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.enabled = True
            return True
        return False

    def disable(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.enabled = False
            return True
        return False

    def get_due_tasks(
        self, now: float | None = None,
    ) -> list[ScheduledTask]:
        """Return all tasks that are due to run."""
        return [
            t for t in self._tasks.values()
            if t.is_due(now)
        ]

    async def tick(self) -> list[dict[str, Any]]:
        """Execute all due tasks. Called periodically."""
        results: list[dict[str, Any]] = []
        due = self.get_due_tasks()
        for task in due:
            handler = self._handlers.get(task.action)
            if handler is None:
                log.warning(
                    "scheduler.no_handler action=%s",
                    task.action,
                )
                continue
            try:
                result = handler(**task.params)
                # Handle coroutines
                import asyncio
                if asyncio.iscoroutine(result):
                    result = await result
                task.mark_run()
                results.append({
                    "task_id": task.task_id,
                    "name": task.name,
                    "status": "success",
                    "result": result,
                })
            except Exception as e:
                task.mark_run()
                results.append({
                    "task_id": task.task_id,
                    "name": task.name,
                    "status": "error",
                    "error": str(e),
                })
                log.exception(
                    "scheduler.task_failed id=%s",
                    task.task_id,
                )
        return results

    def list_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "cron_expr": t.cron_expr,
                "action": t.action,
                "enabled": t.enabled,
                "run_count": t.run_count,
                "last_run": t.last_run,
            }
            for t in self._tasks.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "total_tasks": len(self._tasks),
            "enabled": sum(
                1 for t in self._tasks.values() if t.enabled
            ),
            "handlers": list(self._handlers.keys()),
        }
