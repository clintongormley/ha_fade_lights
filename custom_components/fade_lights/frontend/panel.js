import {
  LitElement,
  html,
  css,
} from "https://unpkg.com/lit-element@2.4.0/lit-element.js?module";

class FadeLightsPanel extends LitElement {
  static get properties() {
    return {
      hass: { type: Object },
      narrow: { type: Boolean },
      panel: { type: Object },
      _data: { type: Object },
      _loading: { type: Boolean },
      _refreshing: { type: Boolean },
      _collapsed: { type: Object },
      _configureChecked: { type: Object },
      _testing: { type: Object },
      _testErrors: { type: Object },
      _globalMinDelayMs: { type: Number },
      _logLevel: { type: String },
    };
  }

  static get styles() {
    return css`
      :host {
        display: block;
        padding: 16px;
        max-width: 1200px;
        margin: 0 auto;
        font-family: var(--paper-font-body1_-_font-family, Roboto, sans-serif);
        font-size: var(--paper-font-body1_-_font-size, 14px);
        color: var(--primary-text-color);
      }

      h1 {
        margin: 0;
        font-size: var(--ha-card-header-font-size, 24px);
        font-weight: 400;
        color: var(--primary-text-color);
      }

      .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
      }

      .header h1 {
        margin: 0;
      }

      ha-button {
        --mdc-theme-primary: var(--primary-color);
      }

      ha-button[disabled] {
        --mdc-theme-primary: var(--disabled-text-color);
      }

      .settings-row {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
        padding: 12px 16px;
        background: var(--card-background-color, #fff);
        border-radius: 8px;
        border: 1px solid var(--divider-color);
      }

      .settings-row label {
        font-weight: 500;
        color: var(--primary-text-color);
      }

      .settings-row ha-textfield {
        width: 140px;
        --mdc-text-field-fill-color: transparent;
      }

      .settings-row .hint {
        font-size: var(--paper-font-caption_-_font-size, 12px);
        color: var(--secondary-text-color);
      }

      .settings-row select {
        padding: 8px 12px;
        border: 1px solid var(--divider-color);
        border-radius: 4px;
        background: var(--card-background-color);
        color: var(--primary-text-color);
        font-size: var(--paper-font-body1_-_font-size, 14px);
        cursor: pointer;
      }

      .settings-row select:focus {
        outline: none;
        border-color: var(--primary-color);
      }

      .header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
      }

      .log-level-selector {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .log-level-selector label {
        font-size: var(--paper-font-body1_-_font-size, 14px);
        color: var(--secondary-text-color);
      }

      .controls-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
      }

      .refresh-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.3);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        pointer-events: all;
      }

      .refresh-overlay .spinner {
        width: 48px;
        height: 48px;
        border-width: 4px;
      }

      .chevron {
        margin-right: 8px;
        transition: transform 0.2s;
        display: inline-block;
      }

      .chevron.collapsed {
        transform: rotate(-90deg);
      }

      .header-icon {
        margin-right: 8px;
        --mdc-icon-size: 20px;
      }

      .lights-table {
        width: 100%;
        border-collapse: collapse;
        margin: 8px 0;
        table-layout: fixed;
      }

      .lights-table th,
      .lights-table td {
        padding: 4px 8px;
        border-bottom: 1px solid var(--divider-color);
      }

      .lights-table th {
        text-align: left;
        font-size: var(--paper-font-caption_-_font-size, 12px);
        font-weight: 500;
        color: var(--secondary-text-color);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .lights-table th.col-configure {
        text-align: center;
      }

      /* Fixed column widths for consistency */
      .col-light {
        width: auto;
      }

      .col-delay {
        width: 120px;
        text-align: center;
      }

      .col-exclude {
        width: 80px;
        text-align: center;
      }

      .col-configure {
        width: 80px;
        text-align: center;
      }

      .lights-table td.col-delay,
      .lights-table td.col-exclude,
      .lights-table td.col-configure {
        text-align: center;
      }

      ha-textfield {
        width: 100px;
        --mdc-text-field-fill-color: transparent;
      }

      ha-checkbox {
        --mdc-checkbox-unchecked-color: var(--secondary-text-color);
      }

      .floor-row td {
        background: var(--primary-background-color);
        font-size: var(--paper-font-subhead_-_font-size, 16px);
        font-weight: 500;
        color: var(--primary-text-color);
        cursor: pointer;
        user-select: none;
      }

      .area-row td {
        background: var(--secondary-background-color, rgba(0,0,0,0.05));
        font-size: var(--paper-font-body1_-_font-size, 14px);
        font-weight: 500;
        color: var(--primary-text-color);
        cursor: pointer;
        user-select: none;
      }

      .area-row.with-floor .group-cell {
        padding-left: 24px;
      }

      .group-cell {
        display: flex;
        align-items: center;
      }

      .light-row td {
        background: transparent;
      }

      .light-cell {
        display: flex;
        align-items: center;
        cursor: pointer;
      }

      .light-cell:hover {
        opacity: 0.8;
      }

      .light-icon {
        margin-right: 12px;
        --mdc-icon-size: 24px;
        color: var(--secondary-text-color);
      }

      .light-icon.on {
        color: var(--amber-color, #ffc107);
      }

      .light-info {
        overflow: hidden;
      }

      .light-name {
        font-size: var(--paper-font-body1_-_font-size, 14px);
        font-weight: 500;
        color: var(--primary-text-color);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .entity-id {
        font-size: var(--paper-font-caption_-_font-size, 12px);
        color: var(--secondary-text-color);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      input[type="number"] {
        width: 80px;
        padding: 4px 8px;
        border: 1px solid var(--divider-color);
        border-radius: 4px;
        background: var(--card-background-color);
        color: var(--primary-text-color);
      }

      input[type="checkbox"] {
        width: 18px;
        height: 18px;
        cursor: pointer;
      }

      .hidden {
        display: none;
      }

      .no-lights {
        color: var(--secondary-text-color);
      }

      .testing-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
      }

      .spinner {
        width: 20px;
        height: 20px;
        border: 2px solid var(--divider-color);
        border-top-color: var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }

      .loading-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 24px 16px;
        color: var(--secondary-text-color);
        font-size: var(--paper-font-body1_-_font-size, 14px);
      }

      @keyframes spin {
        to { transform: rotate(360deg); }
      }

      .button-spinner {
        width: 16px;
        height: 16px;
        border: 2px solid rgba(255,255,255,0.3);
        border-top-color: white;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        display: inline-block;
        vertical-align: middle;
        margin-right: 8px;
      }

      .test-error {
        color: var(--error-color, #db4437);
        font-size: var(--paper-font-caption_-_font-size, 12px);
        margin-top: 4px;
      }

      .light-row.excluded td {
        opacity: 0.5;
      }

      .light-row.excluded td.col-exclude {
        opacity: 1;
      }

      .col-native-transitions {
        width: 130px;
        text-align: center;
      }

      .native-transitions-select {
        padding: 4px 8px;
        border: 1px solid var(--divider-color);
        border-radius: 4px;
        background: var(--card-background-color);
        color: var(--primary-text-color);
        font-size: 12px;
        cursor: pointer;
        min-width: 80px;
      }

      .native-transitions-select:focus {
        outline: none;
        border-color: var(--primary-color);
      }

      .native-transitions-select:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
    `;
  }

  constructor() {
    super();
    this._data = null;
    this._loading = true;
    this._refreshing = false;
    this._collapsed = this._loadCollapsedState();
    this._configureChecked = new Set();
    this._testing = new Set();
    this._testErrors = new Map();
    this._globalMinDelayMs = 100;
    this._logLevel = "warning";
  }

  connectedCallback() {
    super.connectedCallback();
    // Only fetch if hass is already available
    if (this.hass) {
      this._fetchAll();
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this._fetchTimeout) {
      clearTimeout(this._fetchTimeout);
    }
    if (this._autoconfigureUnsub) {
      this._autoconfigureUnsub();
      this._autoconfigureUnsub = null;
    }
  }

  updated(changedProperties) {
    super.updated(changedProperties);
    if (changedProperties.has("hass") && this.hass) {
      // Fetch immediately when hass first becomes available and we haven't loaded yet
      if (!this._data && this._loading) {
        this._fetchAll();
      } else if (!this._loading) {
        // Debounce subsequent updates (only lights, not settings)
        this._debouncedFetch();
      }
    }
  }

  _debouncedFetch() {
    // Debounce re-fetches to avoid excessive API calls
    if (this._fetchTimeout) {
      clearTimeout(this._fetchTimeout);
    }
    this._fetchTimeout = setTimeout(async () => {
      this._refreshing = true;
      await this._fetchLights();
      // Enforce global minimum on any existing per-light values
      await this._enforceGlobalMinimum(this._globalMinDelayMs);
      this._refreshing = false;
    }, 1000);
  }

  _loadCollapsedState() {
    // Version 2: Default to collapsed (v1 defaulted to expanded)
    const STORAGE_VERSION = 2;
    try {
      const stored = JSON.parse(localStorage.getItem("fade_lights_collapsed") || "{}");
      // If version mismatch or missing, clear and start fresh
      if (stored._version !== STORAGE_VERSION) {
        localStorage.removeItem("fade_lights_collapsed");
        return {};
      }
      return stored;
    } catch {
      return {};
    }
  }

  _saveCollapsedState() {
    const toSave = { ...this._collapsed, _version: 2 };
    localStorage.setItem("fade_lights_collapsed", JSON.stringify(toSave));
  }

  async _fetchAll() {
    // Fetch both settings and lights in parallel
    await Promise.all([this._fetchSettings(), this._fetchLights()]);
    // After loading, enforce global minimum on any existing per-light values
    await this._enforceGlobalMinimum(this._globalMinDelayMs);
  }

  async _fetchSettings() {
    try {
      const result = await this.hass.callWS({ type: "fade_lights/get_settings" });
      this._globalMinDelayMs = result.default_min_delay_ms;
      this._logLevel = result.log_level || "warning";
    } catch (err) {
      console.error("Failed to fetch settings:", err);
    }
  }

  async _saveLogLevel(value) {
    try {
      await this.hass.callWS({
        type: "fade_lights/save_settings",
        log_level: value,
      });
      this._logLevel = value;
    } catch (err) {
      console.error("Failed to save log level:", err);
    }
  }

  _handleLogLevelChange(e) {
    const value = e.target.value;
    this._saveLogLevel(value);
  }

  async _saveGlobalMinDelay(value) {
    try {
      await this.hass.callWS({
        type: "fade_lights/save_settings",
        default_min_delay_ms: value,
      });
      this._globalMinDelayMs = value;

      // Update any per-light delays that are now below the global minimum
      await this._enforceGlobalMinimum(value);
    } catch (err) {
      console.error("Failed to save global min delay:", err);
    }
  }

  async _enforceGlobalMinimum(globalMin) {
    // Find all lights with per-light delays below the new global minimum and update them
    if (!this._data?.floors) return;

    for (const floor of this._data.floors) {
      for (const area of floor.areas) {
        for (const light of area.lights) {
          if (light.min_delay_ms && light.min_delay_ms < globalMin) {
            light.min_delay_ms = globalMin;
            await this._saveConfig(light.entity_id, "min_delay_ms", globalMin);
          }
        }
      }
    }
    this.requestUpdate();
  }

  _handleGlobalDelayChange(e) {
    const value = e.target.value ? parseInt(e.target.value, 10) : null;
    if (value && value >= 50 && value <= 1000) {
      this._saveGlobalMinDelay(value);
    }
  }

  async _fetchLights() {
    this._loading = true;
    try {
      const result = await this.hass.callWS({ type: "fade_lights/get_lights" });
      this._data = result;

      // Auto-check lights that don't have a custom delay and are not excluded
      const toCheck = new Set();
      // Set default collapsed state for all floors and areas (collapsed by default)
      const newCollapsed = { ...this._collapsed };
      if (result && result.floors) {
        for (const floor of result.floors) {
          const floorKey = `floor_${floor.floor_id || "none"}`;
          if (!(floorKey in newCollapsed)) {
            newCollapsed[floorKey] = true; // Collapsed by default
          }
          for (const area of floor.areas) {
            const areaKey = `area_${floor.floor_id || "none"}_${area.area_id || "none"}`;
            if (!(areaKey in newCollapsed)) {
              newCollapsed[areaKey] = true; // Collapsed by default
            }
            for (const light of area.lights) {
              if (!light.min_delay_ms && !light.exclude) {
                toCheck.add(light.entity_id);
              }
            }
          }
        }
      }
      this._collapsed = newCollapsed;
      this._configureChecked = toCheck;
    } catch (err) {
      console.error("Failed to fetch lights:", err);
    }
    this._loading = false;
  }

  _toggleCollapse(key) {
    this._collapsed = {
      ...this._collapsed,
      [key]: !this._collapsed[key],
    };
    this._saveCollapsedState();
  }

  async _saveConfig(entityId, field, value) {
    try {
      await this.hass.callWS({
        type: "fade_lights/save_light_config",
        entity_id: entityId,
        [field]: value,
      });
    } catch (err) {
      console.error("Failed to save config:", err);
    }
  }

  _handleDelayChange(entityId, e) {
    let value = e.target.value ? parseInt(e.target.value, 10) : null;

    // Enforce minimum: per-light delay cannot be less than global minimum
    if (value !== null && value < this._globalMinDelayMs) {
      value = this._globalMinDelayMs;
      e.target.value = value;
    }

    // Update local data
    const light = this._findLight(entityId);
    if (light) {
      light.min_delay_ms = value;
    }

    this._saveConfig(entityId, "min_delay_ms", value);
  }

  _handleCheckboxChange(entityId, field, e) {
    const checked = e.target.checked;
    this._saveConfig(entityId, field, checked);

    // Handle exclude field changes dynamically
    if (field === "exclude") {
      // Update local data so UI reflects the change
      const light = this._findLight(entityId);
      if (light) {
        light.exclude = checked;
      }

      const newSet = new Set(this._configureChecked);
      if (checked) {
        // Excluding: uncheck from autoconfigure
        newSet.delete(entityId);
      } else {
        // Un-excluding: check if light has no custom delay
        if (light && !light.min_delay_ms) {
          newSet.add(entityId);
        }
      }
      this._configureChecked = newSet;
      this.requestUpdate();
    }
  }

  _findLight(entityId) {
    // Find a light object by entity_id in _data
    if (!this._data || !this._data.floors) {
      return null;
    }
    for (const floor of this._data.floors) {
      for (const area of floor.areas) {
        for (const light of area.lights) {
          if (light.entity_id === entityId) {
            return light;
          }
        }
      }
    }
    return null;
  }

  _handleConfigureChange(entityId, e) {
    const newSet = new Set(this._configureChecked);
    if (e.target.checked) {
      newSet.add(entityId);
    } else {
      newSet.delete(entityId);
    }
    this._configureChecked = newSet;
  }

  _openLightDialog(entityId) {
    // Fire event to open the more-info dialog for this entity
    const event = new CustomEvent("hass-more-info", {
      bubbles: true,
      composed: true,
      detail: { entityId },
    });
    this.dispatchEvent(event);
  }

  _getFloorLightIds(floor) {
    // Returns all entity_ids in the floor
    const entityIds = [];
    for (const area of floor.areas) {
      for (const light of area.lights) {
        entityIds.push(light.entity_id);
      }
    }
    return entityIds;
  }

  _getAreaLightIds(area) {
    // Returns all entity_ids in the area
    return area.lights.map((light) => light.entity_id);
  }

  _getAllLightIds() {
    // Returns all entity_ids across all floors and areas
    if (!this._data || !this._data.floors) {
      return [];
    }
    const entityIds = [];
    for (const floor of this._data.floors) {
      for (const area of floor.areas) {
        for (const light of area.lights) {
          entityIds.push(light.entity_id);
        }
      }
    }
    return entityIds;
  }

  _getCheckboxState(entityIds) {
    // Returns "all" | "none" | "some" based on how many are checked
    if (entityIds.length === 0) {
      return "none";
    }
    const checkedCount = entityIds.filter((id) => this._configureChecked.has(id)).length;
    if (checkedCount === 0) {
      return "none";
    }
    if (checkedCount === entityIds.length) {
      return "all";
    }
    return "some";
  }

  _handleFloorCheckboxChange(floor, e) {
    e.stopPropagation(); // Prevent collapse toggle
    const entityIds = this._getFloorLightIds(floor);
    const currentState = this._getCheckboxState(entityIds);
    const newSet = new Set(this._configureChecked);

    if (currentState === "all") {
      // Uncheck all
      for (const id of entityIds) {
        newSet.delete(id);
      }
    } else {
      // Check all (for "none" or "some")
      for (const id of entityIds) {
        newSet.add(id);
      }
    }
    this._configureChecked = newSet;
  }

  _handleAreaCheckboxChange(area, e) {
    e.stopPropagation(); // Prevent collapse toggle
    const entityIds = this._getAreaLightIds(area);
    const currentState = this._getCheckboxState(entityIds);
    const newSet = new Set(this._configureChecked);

    if (currentState === "all") {
      // Uncheck all
      for (const id of entityIds) {
        newSet.delete(id);
      }
    } else {
      // Check all (for "none" or "some")
      for (const id of entityIds) {
        newSet.add(id);
      }
    }
    this._configureChecked = newSet;
  }

  _handleAllLightsCheckboxChange() {
    const entityIds = this._getAllLightIds();
    const currentState = this._getCheckboxState(entityIds);
    const newSet = new Set(this._configureChecked);

    if (currentState === "all") {
      // Uncheck all
      for (const id of entityIds) {
        newSet.delete(id);
      }
    } else {
      // Check all (for "none" or "some")
      for (const id of entityIds) {
        newSet.add(id);
      }
    }
    this._configureChecked = newSet;
  }

  _getFloorLights(floor) {
    // Returns all light objects in the floor
    const lights = [];
    for (const area of floor.areas) {
      for (const light of area.lights) {
        lights.push(light);
      }
    }
    return lights;
  }

  _getExcludeState(lights) {
    // Returns "all" | "none" | "some" based on how many are excluded
    if (lights.length === 0) {
      return "none";
    }
    const excludedCount = lights.filter((light) => light.exclude).length;
    if (excludedCount === 0) {
      return "none";
    }
    if (excludedCount === lights.length) {
      return "all";
    }
    return "some";
  }

  async _handleFloorExcludeChange(floor, e) {
    e.stopPropagation(); // Prevent collapse toggle
    const lights = this._getFloorLights(floor);
    const currentState = this._getExcludeState(lights);
    const newExclude = currentState !== "all";

    // Save exclude state for all lights in floor
    const newConfigureSet = new Set(this._configureChecked);
    for (const light of lights) {
      light.exclude = newExclude;
      await this._saveConfig(light.entity_id, "exclude", newExclude);
      // Uncheck from configure when excluding
      if (newExclude) {
        newConfigureSet.delete(light.entity_id);
      }
    }
    this._configureChecked = newConfigureSet;
    this.requestUpdate();
  }

  async _handleAreaExcludeChange(area, e) {
    e.stopPropagation(); // Prevent collapse toggle
    const currentState = this._getExcludeState(area.lights);
    const newExclude = currentState !== "all";

    // Save exclude state for all lights in area
    const newConfigureSet = new Set(this._configureChecked);
    for (const light of area.lights) {
      light.exclude = newExclude;
      await this._saveConfig(light.entity_id, "exclude", newExclude);
      // Uncheck from configure when excluding
      if (newExclude) {
        newConfigureSet.delete(light.entity_id);
      }
    }
    this._configureChecked = newConfigureSet;
    this.requestUpdate();
  }

  _hasRealFloors() {
    // Check if there are any floors with a real floor_id (not null/none)
    if (!this._data || !this._data.floors) return false;
    return this._data.floors.some((floor) => floor.floor_id !== null);
  }

  _getButtonText() {
    const checkedCount = this._configureChecked.size;

    if (checkedCount > 0) {
      return `Autoconfigure (${checkedCount})`;
    }
    return "Autoconfigure";
  }

  _isTesting() {
    return this._testing.size > 0;
  }

  _getTestingText() {
    const completed = this._totalToTest - this._testing.size;
    return `Configuring ${this._testing.size} light${this._testing.size === 1 ? "" : "s"}... (${completed}/${this._totalToTest})`;
  }

  _isButtonDisabled() {
    return this._configureChecked.size === 0 || this._testing.size > 0;
  }

  async _runAutoconfigure() {
    const entityIds = Array.from(this._configureChecked);
    if (entityIds.length === 0) {
      return;
    }

    this._testErrors = new Map();
    this._totalToTest = entityIds.length;

    try {
      const unsub = await this.hass.connection.subscribeMessage(
        (event) => this._handleAutoconfigureEvent(event),
        { type: "fade_lights/autoconfigure", entity_ids: entityIds }
      );
      // Store unsub function if we need to cancel later
      this._autoconfigureUnsub = unsub;
    } catch (err) {
      console.error("Failed to start autoconfigure:", err);
      this._testing = new Set();
    }
  }

  _handleAutoconfigureEvent(event) {
    if (event.type === "started") {
      // Add to testing set
      const newTesting = new Set(this._testing);
      newTesting.add(event.entity_id);
      this._testing = newTesting;
    } else if (event.type === "result") {
      // Remove from testing
      const newTesting = new Set(this._testing);
      newTesting.delete(event.entity_id);
      this._testing = newTesting;

      // Update local data with new values
      this._updateLightConfig(event.entity_id, event.min_delay_ms, event.native_transitions);

      // Uncheck from configure
      const newChecked = new Set(this._configureChecked);
      newChecked.delete(event.entity_id);
      this._configureChecked = newChecked;
    } else if (event.type === "error") {
      // Remove from testing
      const newTesting = new Set(this._testing);
      newTesting.delete(event.entity_id);
      this._testing = newTesting;

      // Add to errors
      const newErrors = new Map(this._testErrors);
      newErrors.set(event.entity_id, event.message);
      this._testErrors = newErrors;

      // Uncheck from configure (even on error)
      const newChecked = new Set(this._configureChecked);
      newChecked.delete(event.entity_id);
      this._configureChecked = newChecked;
    }

    // Check if testing is complete (all done)
    if (this._testing.size === 0 && this._configureChecked.size === 0) {
      // All done, clean up
      if (this._autoconfigureUnsub) {
        this._autoconfigureUnsub();
        this._autoconfigureUnsub = null;
      }
    }
  }

  _updateLightConfig(entityId, minDelayMs, nativeTransitions) {
    // Find and update the light in _data
    if (!this._data?.floors) return;

    for (const floor of this._data.floors) {
      for (const area of floor.areas) {
        const light = area.lights.find((l) => l.entity_id === entityId);
        if (light) {
          light.min_delay_ms = minDelayMs;
          if (nativeTransitions !== undefined) {
            light.native_transitions = nativeTransitions;
          }
          this.requestUpdate(); // Trigger re-render
          return;
        }
      }
    }
  }

  async _handleNativeTransitionsChange(entityId, e) {
    const value = e.target.value;
    let nativeTransitions = null;
    if (value === "true") {
      nativeTransitions = true;
    } else if (value === "false") {
      nativeTransitions = false;
    }

    // Update local data
    const light = this._findLight(entityId);
    if (light) {
      light.native_transitions = nativeTransitions;
    }

    // Save to backend
    await this.hass.callWS({
      type: "fade_lights/save_light_config",
      entity_id: entityId,
      native_transitions: nativeTransitions,
    });
  }

  _renderHeader() {
    const isTesting = this._isTesting();
    return html`
      <div class="header-row">
        <h1>Fade Lights</h1>
      </div>
      <div class="controls-row">
        <div class="log-level-selector">
          <label>Log level:</label>
          <select .value=${this._logLevel} @change=${(e) => this._handleLogLevelChange(e)}>
            <option value="warning" ?selected=${this._logLevel === "warning"}>Warning</option>
            <option value="info" ?selected=${this._logLevel === "info"}>Info</option>
            <option value="debug" ?selected=${this._logLevel === "debug"}>Debug</option>
          </select>
        </div>
        ${isTesting
          ? html`<ha-button unelevated disabled>
              <span class="button-spinner"></span>${this._getTestingText()}
            </ha-button>`
          : html`<ha-button
              unelevated
              ?disabled=${this._isButtonDisabled()}
              @click=${this._runAutoconfigure}
            >${this._getButtonText()}</ha-button>`
        }
      </div>
    `;
  }

  render() {
    // Show loading if hass isn't available yet or we're still loading data
    if (!this.hass || this._loading) {
      return html`
        <div class="header-row">
          <h1>Fade Lights</h1>
        </div>
        <div class="loading-container">
          <div class="spinner"></div>
          <span>Loading...</span>
        </div>
      `;
    }

    if (!this._data || !this._data.floors) {
      return html`${this._renderHeader()}<p>No lights found.</p>`;
    }

    const hasRealFloors = this._hasRealFloors();

    return html`
      ${this._refreshing ? html`<div class="refresh-overlay"><div class="spinner"></div></div>` : ""}
      ${this._renderHeader()}
      <table class="lights-table">
        <thead>
          <tr>
            <th class="col-light"></th>
            <th class="col-delay">Min Delay (ms)</th>
            <th class="col-native-transitions">Native Transitions</th>
            <th class="col-exclude">Exclude</th>
            <th class="col-configure">
              <ha-checkbox
                .checked=${this._getCheckboxState(this._getAllLightIds()) === "all"}
                .indeterminate=${this._getCheckboxState(this._getAllLightIds()) === "some"}
                @change=${() => this._handleAllLightsCheckboxChange()}
              ></ha-checkbox>
            </th>
          </tr>
        </thead>
        <tbody>
          ${hasRealFloors
            ? this._data.floors.map((floor) => this._renderFloor(floor))
            : this._renderAreasOnly()}
        </tbody>
      </table>
      <div class="settings-row">
        <label>Global min delay:</label>
        <ha-textfield
          type="number"
          min="50"
          max="1000"
          step="10"
          suffix="ms"
          .value=${this._globalMinDelayMs || ""}
          @change=${(e) => this._handleGlobalDelayChange(e)}
        ></ha-textfield>
        <span class="hint">The absolute minimum delay for all lights</span>
      </div>
    `;
  }

  _renderAreasOnly() {
    // When no real floors exist, render areas directly without floor grouping
    const allAreas = this._data.floors.flatMap((floor) => floor.areas);
    return allAreas.map((area) => this._renderAreaRows(area, null, false));
  }

  _renderFloor(floor) {
    const floorKey = `floor_${floor.floor_id || "none"}`;
    const isCollapsed = this._collapsed[floorKey];
    const floorIcon = floor.icon || "mdi:floor-plan";
    const floorLights = this._getFloorLights(floor);
    const floorLightIds = this._getFloorLightIds(floor);
    const configureState = this._getCheckboxState(floorLightIds);
    const excludeState = this._getExcludeState(floorLights);

    return html`
      <tr class="floor-row" @click=${() => this._toggleCollapse(floorKey)}>
        <td colspan="2">
          <div class="group-cell">
            <span class="chevron ${isCollapsed ? "collapsed" : ""}">▼</span>
            <ha-icon class="header-icon" icon="${floorIcon}"></ha-icon>
            ${floor.name}
          </div>
        </td>
        <td class="col-native-transitions"></td>
        <td class="col-exclude">
          <ha-checkbox
            .checked=${excludeState === "all"}
            .indeterminate=${excludeState === "some"}
            @click=${(e) => e.stopPropagation()}
            @change=${(e) => this._handleFloorExcludeChange(floor, e)}
          ></ha-checkbox>
        </td>
        <td class="col-configure">
          <ha-checkbox
            .checked=${configureState === "all"}
            .indeterminate=${configureState === "some"}
            @click=${(e) => e.stopPropagation()}
            @change=${(e) => this._handleFloorCheckboxChange(floor, e)}
          ></ha-checkbox>
        </td>
      </tr>
      ${isCollapsed ? "" : floor.areas.map((area) => this._renderAreaRows(area, floor.floor_id, true))}
    `;
  }

  _renderAreaRows(area, floorId, withFloor) {
    const areaKey = `area_${floorId || "none"}_${area.area_id || "none"}`;
    const isCollapsed = this._collapsed[areaKey];
    const areaIcon = area.icon || "mdi:texture-box";
    const areaLightIds = this._getAreaLightIds(area);
    const configureState = this._getCheckboxState(areaLightIds);
    const excludeState = this._getExcludeState(area.lights);

    return html`
      <tr class="area-row ${withFloor ? "with-floor" : ""}" @click=${() => this._toggleCollapse(areaKey)}>
        <td colspan="2">
          <div class="group-cell">
            <span class="chevron ${isCollapsed ? "collapsed" : ""}">▼</span>
            <ha-icon class="header-icon" icon="${areaIcon}"></ha-icon>
            ${area.name}
          </div>
        </td>
        <td class="col-native-transitions"></td>
        <td class="col-exclude">
          <ha-checkbox
            .checked=${excludeState === "all"}
            .indeterminate=${excludeState === "some"}
            @click=${(e) => e.stopPropagation()}
            @change=${(e) => this._handleAreaExcludeChange(area, e)}
          ></ha-checkbox>
        </td>
        <td class="col-configure">
          <ha-checkbox
            .checked=${configureState === "all"}
            .indeterminate=${configureState === "some"}
            @click=${(e) => e.stopPropagation()}
            @change=${(e) => this._handleAreaCheckboxChange(area, e)}
          ></ha-checkbox>
        </td>
      </tr>
      ${isCollapsed
        ? ""
        : area.lights.length > 0
          ? area.lights.map((light) => this._renderLightRow(light, withFloor))
          : html`<tr><td colspan="5" class="no-lights">No lights in this area</td></tr>`}
    `;
  }

  _renderLightRow(light, withFloor) {
    const lightIcon = light.icon || "mdi:lightbulb";
    const state = this.hass.states[light.entity_id];
    const isOn = state && state.state === "on";
    const isTesting = this._testing.has(light.entity_id);
    const errorMessage = this._testErrors.get(light.entity_id);
    const indent = withFloor ? "padding-left: 48px;" : "padding-left: 24px;";
    const isExcluded = light.exclude;

    return html`
      <tr class="light-row ${isExcluded ? "excluded" : ""}">
        <td class="col-light" style="${indent}">
          <div class="light-cell" @click=${() => this._openLightDialog(light.entity_id)}>
            <ha-icon class="light-icon ${isOn ? "on" : ""}" icon="${lightIcon}"></ha-icon>
            <div class="light-info">
              <div class="light-name">${light.name}</div>
              <div class="entity-id">${light.entity_id}</div>
            </div>
          </div>
        </td>
        <td class="col-delay">
          ${isTesting
            ? html`<div class="testing-spinner"><div class="spinner"></div></div>`
            : html`
                <ha-textfield
                  type="number"
                  min="${this._globalMinDelayMs}"
                  max="1000"
                  step="10"
                  placeholder="global"
                  ?disabled=${isExcluded}
                  .value=${light.min_delay_ms || ""}
                  @change=${(e) => this._handleDelayChange(light.entity_id, e)}
                ></ha-textfield>
                ${errorMessage ? html`<div class="test-error">${errorMessage}</div>` : ""}
              `
          }
        </td>
        <td class="col-native-transitions">
          <select
            class="native-transitions-select"
            ?disabled=${isExcluded}
            @change=${(e) => this._handleNativeTransitionsChange(light.entity_id, e)}
          >
            <option value="" ?selected=${light.native_transitions === null || light.native_transitions === undefined}></option>
            <option value="true" ?selected=${light.native_transitions === true}>Yes</option>
            <option value="false" ?selected=${light.native_transitions === false}>No</option>
          </select>
        </td>
        <td class="col-exclude">
          <ha-checkbox
            .checked=${light.exclude}
            @change=${(e) => this._handleCheckboxChange(light.entity_id, "exclude", e)}
          ></ha-checkbox>
        </td>
        <td class="col-configure">
          <ha-checkbox
            ?disabled=${isTesting || isExcluded}
            .checked=${this._configureChecked.has(light.entity_id)}
            @change=${(e) => this._handleConfigureChange(light.entity_id, e)}
          ></ha-checkbox>
        </td>
      </tr>
    `;
  }
}

customElements.define("fade-lights-panel", FadeLightsPanel);
