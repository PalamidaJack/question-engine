"""Structured concurrency helpers using asyncio.TaskGroup.

Provides managed task scopes for services and the supervisor,
ensuring proper cleanup and exception propagation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

log = logging.getLogger(__name__)


class TaskScope:
    """A managed scope for async tasks with structured lifecycle.

    Uses asyncio.TaskGroup for coordinated startup, execution,
    and shutdown. All tasks in a scope share the same lifecycle:
    if any task fails with an unhandled exception, the scope
    cancels all remaining tasks and re-raises.

    Usage:
        async with TaskScope("supervisor") as scope:
            scope.spawn(heartbeat_loop())
            scope.spawn(stall_detector())
            scope.spawn(budget_flush())
            # All tasks run until scope exits or any fails

    For daemon tasks that should be tolerant of individual failures:
        async with TaskScope("supervisor", fail_fast=False) as scope:
            scope.spawn(task1())  # if this fails, others continue
            scope.spawn(task2())
    """

    def __init__(self, name: str = "", *, fail_fast: bool = True) -> None:
        self.name = name
        self._fail_fast = fail_fast
        self._task_group: asyncio.TaskGroup | None = None
        self._tasks: list[asyncio.Task] = []
        self._errors: list[Exception] = []
        self._running = False

    async def __aenter__(self) -> TaskScope:
        self._running = True
        if self._fail_fast:
            self._task_group = asyncio.TaskGroup()
            await self._task_group.__aenter__()
        log.debug("task_scope.enter name=%s fail_fast=%s", self.name, self._fail_fast)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self._running = False

        if self._fail_fast and self._task_group is not None:
            try:
                await self._task_group.__aexit__(exc_type, exc_val, exc_tb)
            except* Exception as eg:
                self._errors.extend(eg.exceptions)
                log.error(
                    "task_scope.errors name=%s count=%d",
                    self.name,
                    len(eg.exceptions),
                )
                raise
        else:
            # Graceful mode: cancel all tasks, collect errors
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            for task in self._tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self._errors.append(e)
                    log.error(
                        "task_scope.task_error name=%s error=%s",
                        self.name,
                        e,
                    )
            self._tasks.clear()

        log.debug(
            "task_scope.exit name=%s errors=%d",
            self.name,
            len(self._errors),
        )
        return False  # don't suppress exceptions

    def spawn(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
    ) -> asyncio.Task:
        """Spawn a task within this scope.

        In fail_fast mode, uses TaskGroup for coordinated cancellation.
        In graceful mode, uses create_task with manual tracking.
        """
        if self._fail_fast and self._task_group is not None:
            task = self._task_group.create_task(coro, name=name)
        else:
            task = asyncio.create_task(coro, name=name)
            self._tasks.append(task)
        return task

    @property
    def errors(self) -> list[Exception]:
        """Errors collected from failed tasks."""
        return list(self._errors)

    @property
    def running(self) -> bool:
        return self._running


class DaemonScope:
    """A scope for long-running daemon tasks that should be resilient.

    Unlike TaskScope with fail_fast=True, DaemonScope restarts
    failed tasks automatically up to a max retry count.
    """

    def __init__(
        self,
        name: str = "",
        max_restarts: int = 3,
    ) -> None:
        self.name = name
        self._max_restarts = max_restarts
        self._task_factories: list[tuple[str, Callable[[], Coroutine]]] = []
        self._tasks: dict[str, asyncio.Task] = {}
        self._restart_counts: dict[str, int] = {}
        self._running = False

    def register(
        self,
        task_name: str,
        factory: Callable[[], Coroutine],
    ) -> None:
        """Register a daemon task factory for supervised execution."""
        self._task_factories.append((task_name, factory))

    async def run(self) -> None:
        """Start all registered daemons and supervise them."""
        self._running = True

        # Start all daemons
        for task_name, factory in self._task_factories:
            self._start_daemon(task_name, factory)

        # Supervise: check for failed tasks and restart
        try:
            while self._running:
                await asyncio.sleep(1.0)
                for task_name, factory in self._task_factories:
                    task = self._tasks.get(task_name)
                    if task and task.done() and self._running:
                        exc = task.exception() if not task.cancelled() else None
                        if exc:
                            restarts = self._restart_counts.get(task_name, 0)
                            if restarts < self._max_restarts:
                                log.warning(
                                    "daemon.restarting name=%s task=%s "
                                    "restarts=%d error=%s",
                                    self.name,
                                    task_name,
                                    restarts,
                                    exc,
                                )
                                self._restart_counts[task_name] = restarts + 1
                                self._start_daemon(task_name, factory)
                            else:
                                log.error(
                                    "daemon.max_restarts name=%s task=%s",
                                    self.name,
                                    task_name,
                                )
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    def _start_daemon(
        self,
        task_name: str,
        factory: Callable[[], Coroutine],
    ) -> None:
        task = asyncio.create_task(factory(), name=f"{self.name}.{task_name}")
        self._tasks[task_name] = task

    async def stop(self) -> None:
        """Stop all daemon tasks."""
        self._running = False
        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        for task in self._tasks.values():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    def status(self) -> dict[str, Any]:
        """Return status of all daemon tasks."""
        return {
            "name": self.name,
            "running": self._running,
            "daemons": {
                name: {
                    "running": name in self._tasks and not self._tasks[name].done(),
                    "restarts": self._restart_counts.get(name, 0),
                }
                for name, _ in self._task_factories
            },
        }
