"""FadeCoordinator for the Fado integration.

Centralises all fade-related state and logic that was previously spread across
module-level global dictionaries and functions in __init__.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_SUPPORTED_COLOR_MODES,
)
from homeassistant.components.light import (
    ATTR_COLOR_TEMP_KELVIN as HA_ATTR_COLOR_TEMP_KELVIN,
)
from homeassistant.components.light import (
    ATTR_HS_COLOR as HA_ATTR_HS_COLOR,
)
from homeassistant.components.light.const import DOMAIN as LIGHT_DOMAIN
from homeassistant.components.light.const import ColorMode
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    ServiceCall,
    State,
    callback,
)
from homeassistant.helpers.service import remove_entity_service_fields
from homeassistant.helpers.storage import Store
from homeassistant.helpers.target import (
    TargetSelection,
    async_extract_referenced_entity_ids,
)

from .const import (
    DOMAIN,
    NATIVE_TRANSITION_MS,
)
from .entity_fade_state import EntityFadeState
from .expected_state import ExpectedState, ExpectedValues
from .fade_change import FadeChange, FadeStep
from .fade_params import FadeParams

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# FadeCoordinator
# =============================================================================


class FadeCoordinator:
    """Coordinate all fade operations for the Fado integration.

    Stored as ``hass.data[DOMAIN]``. Owns per-entity fade state, persistent
    storage, and all fade/restore/notification logic.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        store: Store[dict[str, int]],
        min_step_delay_ms: int,
    ) -> None:
        self.hass = hass
        self.store = store
        self.data: dict[str, Any] = {}
        self.min_step_delay_ms = min_step_delay_ms
        self._entities: dict[str, EntityFadeState] = {}

    async def async_load(self) -> None:
        """Load persistent data from store."""
        self.data = await self.store.async_load() or {}

    # --------------------------------------------------------------------- #
    # Entity helpers
    # --------------------------------------------------------------------- #

    def get_entity(self, entity_id: str) -> EntityFadeState | None:
        """Return the EntityFadeState for *entity_id*, or ``None``."""
        return self._entities.get(entity_id)

    def get_or_create_entity(self, entity_id: str) -> EntityFadeState:
        """Return (or create) the EntityFadeState for *entity_id*."""
        if entity_id not in self._entities:
            self._entities[entity_id] = EntityFadeState()
        return self._entities[entity_id]

    # --------------------------------------------------------------------- #
    # Service handler: fade_lights
    # --------------------------------------------------------------------- #

    async def handle_fade_lights(self, call: ServiceCall) -> None:
        """Handle the fade_lights service call."""
        # Remove target fields (entity_id, device_id, area_id, etc.) from service data
        # before parsing fade parameters - these are handled separately via TargetSelection
        service_data = remove_entity_service_fields(call)
        fade_params = FadeParams.from_service_data(service_data)

        if not fade_params.has_target() and not fade_params.has_from_target():
            _LOGGER.debug("No fade parameters specified, nothing to do")
            return

        # Resolve targets to entity IDs
        target_selection = TargetSelection(call.data)
        selected = async_extract_referenced_entity_ids(self.hass, target_selection)
        all_entity_ids = selected.referenced | selected.indirectly_referenced

        # Expand groups, filter to light domain, and remove excluded lights
        expanded_entities = self._expand_light_groups(list(all_entity_ids))

        if not expanded_entities:
            _LOGGER.debug("No light entities found in target")
            return

        tasks = []
        for entity_id in expanded_entities:
            state = self.hass.states.get(entity_id)
            if not state or state.state == "unavailable":
                _LOGGER.debug("%s: Skipping - entity unavailable", entity_id)
                continue
            if not _can_apply_fade_params(state, fade_params):
                _LOGGER.info(
                    "%s: Skipping - light cannot apply any requested fade parameters",
                    entity_id,
                )
                continue
            tasks.append(
                asyncio.create_task(
                    self._fade_light(
                        entity_id,
                        fade_params,
                        self.min_step_delay_ms,
                    )
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # --------------------------------------------------------------------- #
    # State change handler
    # --------------------------------------------------------------------- #

    def _should_process_state_change(self, new_state: State | None) -> bool:
        """Check if this state change should be processed."""
        if not new_state:
            return False
        if new_state.domain != LIGHT_DOMAIN:
            return False
        # Ignore group helpers (lights that contain other lights)
        if new_state.attributes.get(ATTR_ENTITY_ID) is not None:
            return False
        # Skip excluded lights
        return not self.get_light_config(new_state.entity_id).get("exclude", False)

    @callback
    def handle_state_change(self, event: Event[EventStateChangedData]) -> None:
        """Handle light state changes - detects manual intervention and tracks brightness."""
        new_state: State | None = event.data.get("new_state")
        old_state: State | None = event.data.get("old_state")

        if not self._should_process_state_change(new_state):
            return

        # Type narrowing: new_state is guaranteed non-None after _should_process_state_change
        assert new_state is not None

        entity_id = new_state.entity_id

        # Check if this is an expected state change (from our service calls)
        if self._match_and_remove_expected(entity_id, new_state):
            return

        # During fade or restore: if we get here, state didn't match expected - manual intervention
        entity = self.get_entity(entity_id)
        is_during_fade = entity is not None and entity.is_fading
        is_during_restore = entity is not None and entity.is_restoring
        if is_during_fade or is_during_restore:
            # Manual intervention detected - add to intended state queue
            old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS) if old_state else None
            _LOGGER.info(
                "%s: Manual intervention detected (state=%s, brightness=%s->%s)",
                entity_id,
                new_state.state,
                old_brightness,
                new_state.attributes.get(ATTR_BRIGHTNESS),
            )

            ent = self.get_or_create_entity(entity_id)

            # Initialize queue with old_state if this is the first manual event
            if not ent.intended_queue:
                ent.intended_queue = [old_state] if old_state else []

            # Append the new intended state
            ent.intended_queue.append(new_state)

            # Only spawn restore task if one isn't already running
            if not ent.is_restoring:
                task = self.hass.async_create_task(self._restore_intended_state(entity_id))
                ent.restore_task = task
            else:
                _LOGGER.debug("%s: Restore task already running, queued intended state", entity_id)
            return

        # Normal state handling (no active fade)
        if _is_off_to_on_transition(old_state, new_state):
            self._handle_off_to_on(entity_id, new_state)
            return

        if _is_brightness_change(old_state, new_state):
            new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
            if new_brightness is not None:
                _LOGGER.debug(
                    "%s: Storing new brightness as original: %s", entity_id, new_brightness
                )
                self.store_orig_brightness(entity_id, new_brightness)

    # --------------------------------------------------------------------- #
    # Fade execution
    # --------------------------------------------------------------------- #

    async def _fade_light(
        self,
        entity_id: str,
        fade_params: FadeParams,
        min_step_delay_ms: int,
    ) -> None:
        """Fade a single light to the specified brightness and/or color.

        This is the entry point for fading a single light. It handles:
        - Cancelling any existing fade for the same entity
        - Setting up tracking state
        - Delegating to _execute_fade for the actual work
        - Cleaning up tracking state when done (success, cancel, or error)
        """
        # Get per-light delay config
        light_config = self.get_light_config(entity_id)
        delay_ms = light_config.get("min_delay_ms") or min_step_delay_ms

        # Cancel any existing fade for this entity - only one fade per light at a time
        entity = self.get_or_create_entity(entity_id)
        await entity.cancel_and_wait()

        entity.start_fade(asyncio.current_task())
        assert entity.cancel_event is not None  # set by start_fade
        cancel_event = entity.cancel_event

        try:
            await self._execute_fade(entity_id, fade_params, delay_ms, cancel_event)
        except asyncio.CancelledError:
            pass  # Normal cancellation, not an error
        finally:
            # Clean up tracking state regardless of how the fade ended.
            # Note: expected_state is NOT cleared here - values persist
            # for event matching and are pruned when next fade starts.
            await entity.finish_fade()

    async def _execute_fade(
        self,
        entity_id: str,
        fade_params: FadeParams,
        min_step_delay_ms: int,
        cancel_event: asyncio.Event,
    ) -> None:
        """Execute the fade operation using FadeChange iterator pattern.

        Uses _resolve_fade to create a single FadeChange that handles all fade types
        including hybrid transitions internally. The iterator generates steps
        seamlessly across mode switches.
        """
        state = self.hass.states.get(entity_id)
        if not state:
            _LOGGER.warning("%s: Entity not found", entity_id)
            return

        # Store original brightness for restoration after OFF->ON
        # Update if: (1) nothing stored yet, or (2) user changed brightness since last fade
        current_brightness = state.attributes.get(ATTR_BRIGHTNESS)
        start_brightness = int(current_brightness) if current_brightness is not None else 0
        existing_orig = self.get_orig_brightness(entity_id)
        if start_brightness > 0 and start_brightness != existing_orig:
            self.store_orig_brightness(entity_id, start_brightness)

        # Get stored brightness for auto-turn-on when fading color from off
        stored_brightness = start_brightness if start_brightness > 0 else existing_orig

        # Get per-light minimum brightness from config (detected by autoconfigure)
        light_config = self.get_light_config(entity_id)
        min_brightness = light_config.get("min_brightness") or 1

        # Resolve fade parameters into a configured FadeChange
        fade = FadeChange.resolve(
            fade_params, state.attributes, min_step_delay_ms, stored_brightness, min_brightness
        )

        if fade is None:
            _LOGGER.debug("%s: Nothing to fade", entity_id)
            return

        total_steps = fade.step_count()
        delay_ms = fade.delay_ms()

        # Check if light supports native transitions and if "from" was specified
        native_transitions = light_config.get("native_transitions") is True
        has_from = fade_params.has_from_target()

        _LOGGER.info(
            "%s: Fading in %s steps, (brightness=%s->%s, hs=%s->%s, mireds=%s->%s, "
            "easing=%s, hybrid=%s, crossover_step=%s, delay_ms=%s, native_transitions=%s)",
            entity_id,
            total_steps,
            fade.start_brightness,
            fade.end_brightness,
            fade.start_hs,
            fade.end_hs,
            fade.start_mireds,
            fade.end_mireds,
            fade.easing_name,
            fade.hybrid_direction,
            fade.crossover_step,
            delay_ms,
            native_transitions,
        )

        # Execute fade steps
        step_num = 0
        prev_step: FadeStep | None = None

        while fade.has_next():
            step_start = time.monotonic()

            if cancel_event.is_set():
                return

            step = fade.next_step()
            step_num += 1

            # Determine if using transition for THIS step
            use_transition = native_transitions and not (step_num == 1 and has_from)

            # Build expected values - track ranges when using transitions
            if use_transition and prev_step is not None:
                # Range-based: track transition from prev_step -> step
                expected = ExpectedValues(
                    brightness=step.brightness,
                    from_brightness=prev_step.brightness,
                    hs_color=step.hs_color,
                    from_hs_color=prev_step.hs_color,
                    color_temp_kelvin=step.color_temp_kelvin,
                    from_color_temp_kelvin=prev_step.color_temp_kelvin,
                )
            else:
                # Point-based: no from values
                expected = ExpectedValues(
                    brightness=step.brightness,
                    hs_color=step.hs_color,
                    color_temp_kelvin=step.color_temp_kelvin,
                )
            self._add_expected_values(entity_id, expected)

            await self._apply_step(entity_id, step, use_transition=use_transition)

            # Save for next iteration
            prev_step = step

            if cancel_event.is_set():
                return

            # Sleep remaining time (skip after last step)
            if fade.has_next():
                await _sleep_remaining_step_time(step_start, delay_ms)

        # Wait for any late events and clear expected state
        entity = self.get_entity(entity_id)
        if entity:
            await entity.flush_and_clear_expected_state()

        # Store final brightness after successful fade completion
        if not cancel_event.is_set():
            final_brightness = fade.end_brightness

            if final_brightness is not None and final_brightness > 0:
                self.store_orig_brightness(entity_id, final_brightness)
                await self.save_storage()
                _LOGGER.info("%s: Fade complete at brightness %s", entity_id, final_brightness)
            elif final_brightness == 0:
                _LOGGER.info("%s: Fade complete (turned off)", entity_id)
            else:
                _LOGGER.info("%s: Fade complete", entity_id)

    async def _apply_step(
        self,
        entity_id: str,
        step: FadeStep,
        *,
        use_transition: bool = False,
    ) -> None:
        """Apply a fade step to a light.

        Handles brightness, hs_color, and color_temp_kelvin in a single service call.
        If brightness is 0, turns off the light. If step is empty, does nothing.

        Args:
            entity_id: Light entity ID
            step: The fade step to apply
            use_transition: If True, add transition: 0.1 to smooth the step
        """
        # Build service data based on what's in the step
        service_data: dict = {ATTR_ENTITY_ID: entity_id}

        if step.brightness is not None:
            if step.brightness == 0:
                # Turn off - no other attributes needed
                await self.hass.services.async_call(
                    LIGHT_DOMAIN,
                    SERVICE_TURN_OFF,
                    {ATTR_ENTITY_ID: entity_id},
                    blocking=True,
                )
                return
            service_data[ATTR_BRIGHTNESS] = step.brightness

        if step.hs_color is not None:
            service_data[HA_ATTR_HS_COLOR] = step.hs_color

        if step.color_temp_kelvin is not None:
            service_data[HA_ATTR_COLOR_TEMP_KELVIN] = step.color_temp_kelvin

        # Add short transition for smoother steps on lights that support native transitions
        if use_transition:
            service_data["transition"] = NATIVE_TRANSITION_MS / 1000

        _LOGGER.debug("%s", service_data)

        # Only call service if there's something to set (beyond entity_id)
        if len(service_data) > 1:
            await self.hass.services.async_call(
                LIGHT_DOMAIN,
                SERVICE_TURN_ON,
                service_data,
                blocking=True,
            )

    # --------------------------------------------------------------------- #
    # Expected state tracking & matching
    # --------------------------------------------------------------------- #

    def _match_and_remove_expected(self, entity_id: str, new_state: State) -> bool:
        """Check if state matches expected, remove if found, notify if empty.

        Returns True if this was an expected state change (caller should ignore it).
        """
        entity = self.get_entity(entity_id)
        expected_state = entity.expected_state if entity else None
        if not expected_state or expected_state.is_empty:
            return False

        # Build ExpectedValues from the new state
        if new_state.state == STATE_OFF:
            actual = ExpectedValues(brightness=0)
        else:
            brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
            if brightness is None:
                return False

            # Extract color attributes
            hs_raw = new_state.attributes.get(HA_ATTR_HS_COLOR)
            hs_color = (float(hs_raw[0]), float(hs_raw[1])) if hs_raw else None

            # Read kelvin directly from state attributes
            kelvin_raw = new_state.attributes.get(HA_ATTR_COLOR_TEMP_KELVIN)
            color_temp_kelvin = int(kelvin_raw) if kelvin_raw else None

            actual = ExpectedValues(
                brightness=brightness,
                hs_color=hs_color,
                color_temp_kelvin=color_temp_kelvin,
            )

        matched = expected_state.match_and_remove(actual)
        return matched is not None

    def _add_expected_values(self, entity_id: str, values: ExpectedValues) -> None:
        """Register expected values before making a service call."""
        entity = self.get_or_create_entity(entity_id)
        if entity.expected_state is None:
            entity.expected_state = ExpectedState(entity_id=entity_id)
        entity.expected_state.add(values)

    def _add_expected_brightness(self, entity_id: str, brightness: int) -> None:
        """Register an expected brightness value (convenience wrapper)."""
        self._add_expected_values(entity_id, ExpectedValues(brightness=brightness))

    # --------------------------------------------------------------------- #
    # OFF -> ON handling
    # --------------------------------------------------------------------- #

    def _handle_off_to_on(self, entity_id: str, new_state: State) -> None:
        """Handle OFF -> ON transition by restoring original brightness."""
        _LOGGER.info("%s: Light turned on", entity_id)

        # Non-dimmable lights can't have brightness restored
        if ColorMode.BRIGHTNESS not in new_state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []):
            return

        orig_brightness = self.get_orig_brightness(entity_id)
        current_brightness = new_state.attributes.get(ATTR_BRIGHTNESS, 0)

        _LOGGER.debug(
            "%s: orig_brightness=%s, current_brightness=%s",
            entity_id,
            orig_brightness,
            current_brightness,
        )

        if orig_brightness > 0 and current_brightness != orig_brightness:
            _LOGGER.info("%s: Restoring to brightness %s", entity_id, orig_brightness)
            self.hass.async_create_task(
                self._restore_original_brightness(entity_id, orig_brightness)
            )

    async def _restore_original_brightness(
        self,
        entity_id: str,
        brightness: int,
    ) -> None:
        """Restore original brightness and wait for confirmation."""
        self._add_expected_brightness(entity_id, brightness)
        await self.hass.services.async_call(
            LIGHT_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: entity_id, ATTR_BRIGHTNESS: brightness},
            blocking=True,
        )
        entity = self.get_or_create_entity(entity_id)
        await entity.wait_for_expected_state_flush()

    # --------------------------------------------------------------------- #
    # Manual intervention / restore intended state
    # --------------------------------------------------------------------- #

    async def _restore_intended_state(
        self,
        entity_id: str,
    ) -> None:
        """Restore intended state after manual intervention during fade.

        When manual intervention is detected during a fade, late fade events may
        overwrite the user's intended state. This function:
        1. Cancels the fade and waits for cleanup
        2. Loops: reads most recent intended state from queue and restores if needed
        3. Continues until no more intended states are queued

        The queue structure is: [old_state, intended_1, intended_2, ...]
        - First entry is the state before the first manual intervention
        - Subsequent entries are intended states from manual interventions

        When processing, we compare adjacent states to determine transitions
        (e.g., OFF->ON vs ON->ON) and only store original brightness when
        the previous state had non-zero brightness.
        """
        try:
            _LOGGER.debug(
                "%s: Waiting for state events to flush before restoring intended state",
                entity_id,
            )
            entity = self.get_or_create_entity(entity_id)
            await entity.cancel_and_wait()

            # Loop until no more intended states are queued
            # (handles case where another manual event arrives during restore)
            while True:
                # Get the queue for this entity
                queue = entity.intended_queue

                # Need at least 2 entries: previous state + intended state
                if len(queue) < 2:
                    _LOGGER.debug("%s: No more intended states in queue, done", entity_id)
                    entity.intended_queue = []
                    break

                # Get the most recent intended state (last in queue)
                intended_state = queue[-1]
                # Get the previous state (second to last) for comparison
                previous_state = queue[-2]

                # Remove intended_state and all previous states from queue
                # Keep only the intended_state as the new "previous" for any future events
                entity.intended_queue = [intended_state]

                intended_brightness = self._get_intended_brightness(
                    entity_id, previous_state, intended_state
                )
                _LOGGER.debug("%s: Got intended brightness (%s)", entity_id, intended_brightness)
                if intended_brightness is None:
                    break

                # Store as new original brightness only if:
                # - Previous state had non-zero brightness (was ON, not coming from OFF)
                # - Intended brightness is > 0 (not turning off)
                # - Brightness is actually changing
                # This ensures we track the user's intended brightness for OFF->ON restoration
                previous_brightness = (
                    previous_state.attributes.get(ATTR_BRIGHTNESS, 0)
                    if previous_state and previous_state.state != STATE_OFF
                    else 0
                )
                if (
                    previous_brightness > 0
                    and intended_brightness > 0
                    and intended_brightness != previous_brightness
                ):
                    _LOGGER.debug(
                        "%s: Storing original brightness (%s) from transition %s->%s",
                        entity_id,
                        intended_brightness,
                        previous_brightness,
                        intended_brightness,
                    )
                    self.store_orig_brightness(entity_id, intended_brightness)

                # Get current state after fade cleanup
                current_state = self.hass.states.get(entity_id)
                if not current_state:
                    _LOGGER.debug("%s: No current state found, exiting", entity_id)
                    break

                current_brightness = current_state.attributes.get(ATTR_BRIGHTNESS) or 0
                if current_state.state == STATE_OFF:
                    current_brightness = 0

                # Handle OFF case
                if intended_brightness == 0:
                    if current_brightness != 0:
                        _LOGGER.info("%s: Restoring to off as intended", entity_id)
                        self._add_expected_brightness(entity_id, 0)
                        await self.hass.services.async_call(
                            LIGHT_DOMAIN,
                            SERVICE_TURN_OFF,
                            {ATTR_ENTITY_ID: entity_id},
                            blocking=True,
                        )
                        await entity.wait_for_expected_state_flush()
                    else:
                        _LOGGER.debug("%s: already off, nothing to restore", entity_id)
                    continue  # Check for more intended states

                # Handle ON case - check brightness and colors
                # Build service data for restoration
                service_data: dict = {ATTR_ENTITY_ID: entity_id}
                need_restore = False

                # Check brightness
                if current_brightness != intended_brightness:
                    service_data[ATTR_BRIGHTNESS] = intended_brightness
                    need_restore = True

                # Get intended colors from manual intervention (intended_state)
                intended_hs = intended_state.attributes.get(HA_ATTR_HS_COLOR)
                intended_kelvin = intended_state.attributes.get(HA_ATTR_COLOR_TEMP_KELVIN)

                # Get current colors
                current_hs = current_state.attributes.get(HA_ATTR_HS_COLOR)
                current_kelvin = current_state.attributes.get(HA_ATTR_COLOR_TEMP_KELVIN)

                # Check HS color
                if intended_hs and intended_hs != current_hs:
                    service_data[HA_ATTR_HS_COLOR] = intended_hs
                    need_restore = True

                # Check color temp (mutually exclusive with HS)
                if (
                    intended_kelvin
                    and intended_kelvin != current_kelvin
                    and HA_ATTR_HS_COLOR not in service_data
                ):
                    service_data[HA_ATTR_COLOR_TEMP_KELVIN] = intended_kelvin
                    need_restore = True

                if need_restore:
                    _LOGGER.info("%s: Restoring intended state: %s", entity_id, service_data)

                    # Track expected values (ExpectedValues uses kelvin)
                    self._add_expected_values(
                        entity_id,
                        ExpectedValues(
                            brightness=service_data.get(ATTR_BRIGHTNESS),
                            hs_color=service_data.get(HA_ATTR_HS_COLOR),
                            color_temp_kelvin=service_data.get(HA_ATTR_COLOR_TEMP_KELVIN),
                        ),
                    )

                    await self.hass.services.async_call(
                        LIGHT_DOMAIN,
                        SERVICE_TURN_ON,
                        service_data,
                        blocking=True,
                    )
                    await entity.wait_for_expected_state_flush()
                else:
                    _LOGGER.debug("%s: already in intended state, nothing to restore", entity_id)
        finally:
            # Clean up restore task tracking
            entity = self.get_entity(entity_id)
            if entity is not None:
                entity.restore_task = None

    def _get_intended_brightness(
        self,
        entity_id: str,
        old_state: State | None,
        new_state: State,
    ) -> int | None:
        """Determine the intended brightness from a manual intervention.

        Returns:
            0: Light should be OFF
            >0: Light should be ON at this brightness
            None: Could not determine (integration unloaded)
        """
        if DOMAIN not in self.hass.data:
            return None

        if new_state.state == STATE_OFF:
            return 0

        new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)

        if old_state and old_state.state == STATE_OFF:
            # OFF -> ON: restore to original brightness
            orig = self.get_orig_brightness(entity_id)
            return orig if orig > 0 else new_brightness

        # ON -> ON: use the brightness from the event
        return new_brightness

    # --------------------------------------------------------------------- #
    # Light group expansion
    # --------------------------------------------------------------------- #

    def _expand_light_groups(self, entity_ids: list[str]) -> list[str]:
        """Expand light groups to individual light entities.

        Light groups have an entity_id attribute containing member lights.
        Expands iteratively (not recursively) and deduplicates results.
        Lights with exclude=True in their config are filtered out.

        Example:
            Input: ["light.living_room_group", "light.bedroom"]
            If light.living_room_group contains [light.lamp, light.ceiling]
            Output: ["light.lamp", "light.ceiling", "light.bedroom"]
        """
        pending = list(entity_ids)
        result: set[str] = set()
        light_prefix = f"{LIGHT_DOMAIN}."

        while pending:
            entity_id = pending.pop()
            state = self.hass.states.get(entity_id)

            if state is None:
                _LOGGER.warning("%s: Entity not found, skipping", entity_id)
                continue

            # Check if this is a group (has entity_id attribute with member lights)
            if ATTR_ENTITY_ID in state.attributes:
                group_members = state.attributes[ATTR_ENTITY_ID]
                if isinstance(group_members, str):
                    group_members = [group_members]
                # Filter to lights only (groups can technically contain non-lights)
                pending.extend(m for m in group_members if m.startswith(light_prefix))
            elif entity_id.startswith(light_prefix):
                result.add(entity_id)

        # Filter out excluded lights
        final_result = []
        for eid in result:
            if self.get_light_config(eid).get("exclude", False):
                _LOGGER.debug("%s: Excluded from fade", eid)
            else:
                final_result.append(eid)
        return final_result

    # --------------------------------------------------------------------- #
    # Storage helpers
    # --------------------------------------------------------------------- #

    def get_light_config(self, entity_id: str) -> dict[str, Any]:
        """Get per-light configuration.

        Returns the config dict for the light, or empty dict if not configured.
        """
        return self.data.get(entity_id, {})

    def get_or_create_light_config(self, entity_id: str) -> dict[str, Any]:
        """Get per-light configuration, creating it if it doesn't exist."""
        if entity_id not in self.data:
            self.data[entity_id] = {}
        return self.data[entity_id]

    def get_orig_brightness(self, entity_id: str) -> int:
        """Get stored original brightness for an entity."""
        return self.get_light_config(entity_id).get("orig_brightness", 0)

    def store_orig_brightness(self, entity_id: str, level: int) -> None:
        """Store original brightness for an entity."""
        if entity_id not in self.data:
            self.data[entity_id] = {}
        self.data[entity_id]["orig_brightness"] = level

    async def save_storage(self) -> None:
        """Save storage data to disk."""
        await self.store.async_save(self.data)

    # --------------------------------------------------------------------- #
    # Cleanup
    # --------------------------------------------------------------------- #

    async def cleanup_entity(self, entity_id: str) -> None:
        """Clean up all data associated with a deleted entity.

        This is called when an entity is removed from the entity registry.
        It cleans up:
        - Active fade tasks and cancellation events
        - Expected state tracking
        - Completion conditions
        - Intended state queues for brightness restoration
        - Restore tasks
        - Testing lights set (autoconfigure)
        - Persistent storage data
        """
        _LOGGER.debug("%s: Cleaning up data for deleted entity", entity_id)

        entity = self._entities.get(entity_id)
        if entity is not None:
            await entity.cleanup()

        # Remove entity from tracking
        self._entities.pop(entity_id, None)

        # Remove from persistent storage
        if entity_id in self.data:
            del self.data[entity_id]
            # Save updated storage
            await self.store.async_save(self.data)
            _LOGGER.info("%s: Removed persistent data for deleted entity", entity_id)

    async def shutdown(self) -> None:
        """Shut down all active fades and clean up state."""
        for entity in self._entities.values():
            entity.signal_cancel()
        tasks = []
        for entity in self._entities.values():
            tasks.extend(entity.cancel_all_tasks())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._entities.clear()


# =============================================================================
# Module-level utility functions (stateless)
# =============================================================================


def _is_off_to_on_transition(old_state: State | None, new_state: State) -> bool:
    """Check if this is an OFF -> ON transition."""
    return old_state is not None and old_state.state == STATE_OFF and new_state.state == STATE_ON


def _is_brightness_change(old_state: State | None, new_state: State) -> bool:
    """Check if this is an ON -> ON brightness change."""
    if not old_state or old_state.state != STATE_ON or new_state.state != STATE_ON:
        return False
    old_brightness = old_state.attributes.get(ATTR_BRIGHTNESS)
    new_brightness = new_state.attributes.get(ATTR_BRIGHTNESS)
    return new_brightness is not None and new_brightness != old_brightness


async def _sleep_remaining_step_time(step_start: float, delay_ms: float) -> None:
    """Sleep for the remaining time in a fade step.

    Subtracts elapsed time from target delay to maintain consistent fade duration
    regardless of how long the service call took.
    """
    elapsed_ms = (time.monotonic() - step_start) * 1000
    sleep_ms = max(0, delay_ms - elapsed_ms)
    if sleep_ms > 0:
        await asyncio.sleep(sleep_ms / 1000)


def _can_apply_fade_params(state: State, params: FadeParams) -> bool:
    """Check if a light can perform at least one of the requested fade operations.

    Returns True if the light can do ANY of:
    - Brightness fade (any light, including on/off only)
    - HS color fade (if requested and light supports HS/RGB/RGBW/RGBWW/XY)
    - Color temp fade (if requested and light supports COLOR_TEMP)
    """
    modes = set(state.attributes.get(ATTR_SUPPORTED_COLOR_MODES, []))

    # Check if brightness is requested - any light can handle it
    # (on/off lights get turned on/off, dimmable lights get faded)
    brightness_requested = (
        params.brightness_pct is not None or params.from_brightness_pct is not None
    )
    if brightness_requested:
        return True

    # Check if HS color is requested and light supports it
    hs_requested = params.hs_color is not None or params.from_hs_color is not None
    hs_capable = modes & {
        ColorMode.HS,
        ColorMode.RGB,
        ColorMode.RGBW,
        ColorMode.RGBWW,
        ColorMode.XY,
    }
    if hs_requested and hs_capable:
        return True

    # Check if color temp is requested and light supports it
    color_temp_requested = (
        params.color_temp_kelvin is not None or params.from_color_temp_kelvin is not None
    )
    return color_temp_requested and ColorMode.COLOR_TEMP in modes
