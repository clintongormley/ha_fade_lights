"""Per-entity fade state for the Fado integration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from homeassistant.core import State

from .const import FADE_CANCEL_TIMEOUT_S
from .expected_state import ExpectedState

_LOGGER = logging.getLogger(__name__)


@dataclass
class EntityFadeState:
    """Transient state for a single entity's fade operations.

    Attributes:
        active_task: The running asyncio.Task executing the current fade.
            Cancelled when a new fade starts or the entity is cleaned up.
        cancel_event: Signals the active fade loop to stop early, e.g. when
            a new fade is requested or the integration is unloaded.
        complete_condition: Allows callers to wait until the current fade
            finishes (used by blocking service calls).
        expected_state: Tracks brightness/state values the fade intends to
            set, so that state-change listeners can distinguish our own
            updates from external changes by the user.
        intended_queue: Queued state snapshots representing the brightness
            the light should reach once external interference is resolved.
        restore_task: A delayed task that restores the light to its original
            brightness after a temporary notification fade completes.
    """

    active_task: asyncio.Task | None = None
    cancel_event: asyncio.Event | None = None
    complete_condition: asyncio.Condition | None = None
    expected_state: ExpectedState | None = None
    intended_queue: list[State] = field(default_factory=list)
    restore_task: asyncio.Task | None = None

    @property
    def is_fading(self) -> bool:
        """True when a fade is actively running."""
        return self.active_task is not None and self.expected_state is not None

    @property
    def is_restoring(self) -> bool:
        """True when a restore task is running."""
        return self.restore_task is not None

    def start_fade(self, task: asyncio.Task | None) -> None:
        """Set up state for a new fade operation."""
        self.cancel_event = asyncio.Event()
        self.complete_condition = asyncio.Condition()
        if task is not None:
            self.active_task = task

    async def finish_fade(self) -> None:
        """Clean up after a fade completes (success, cancel, or error).

        Clears the active task and cancel event, then notifies any waiters
        via the completion condition.
        """
        self.active_task = None
        self.cancel_event = None

        condition = self.complete_condition
        self.complete_condition = None
        if condition:
            async with condition:
                condition.notify_all()

    def signal_cancel(self) -> None:
        """Signal the active fade to stop early."""
        if self.cancel_event is not None:
            self.cancel_event.set()

    def cancel_all_tasks(self) -> list[asyncio.Task]:
        """Cancel active and restore tasks. Returns cancelled tasks for awaiting."""
        tasks: list[asyncio.Task] = []
        if self.active_task is not None:
            self.active_task.cancel()
            tasks.append(self.active_task)
        if self.restore_task is not None:
            self.restore_task.cancel()
            tasks.append(self.restore_task)
        return tasks

    async def cancel_and_wait(self) -> None:
        """Cancel the active fade, wait for cleanup, and flush stale events."""
        if self.active_task is not None:
            task = self.active_task
            condition = self.complete_condition

            self.signal_cancel()

            if not task.done():
                task.cancel()

            if condition:
                async with condition:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(
                            condition.wait_for(lambda: self.active_task is None),
                            timeout=FADE_CANCEL_TIMEOUT_S,
                        )

        await self.wait_for_expected_state_flush()

    async def wait_for_expected_state_flush(self, timeout: float = 5.0) -> None:
        """Wait until all expected state values have been confirmed via state changes."""
        expected_state = self.expected_state
        if not expected_state or expected_state.is_empty:
            return

        condition = expected_state.get_condition()
        with contextlib.suppress(TimeoutError):
            async with condition:
                await asyncio.wait_for(
                    condition.wait_for(lambda: expected_state.is_empty),
                    timeout=timeout,
                )

    async def flush_and_clear_expected_state(self) -> None:
        """Wait for remaining expected state events then clear."""
        if self.expected_state:
            _LOGGER.debug(
                "%s: Fade finished. Waiting for expected state events to be flushed"
                " (remaining: %d)",
                self.expected_state.entity_id,
                len(self.expected_state.values),
            )
            await self.expected_state.wait_and_clear()

    async def cleanup(self) -> None:
        """Full teardown: cancel tasks, signal events, clear all state."""
        if self.active_task is not None:
            task = self.active_task
            self.active_task = None
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if self.cancel_event is not None:
            self.cancel_event.set()
            self.cancel_event = None

        if self.expected_state is not None:
            self.expected_state.values.clear()
            self.expected_state = None

        self.complete_condition = None
        self.intended_queue = []

        if self.restore_task is not None:
            task = self.restore_task
            self.restore_task = None
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
