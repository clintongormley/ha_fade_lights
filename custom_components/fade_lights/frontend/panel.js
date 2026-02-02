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
      _collapsed: { type: Object },
      _configureChecked: { type: Object },
    };
  }

  static get styles() {
    return css`
      :host {
        display: block;
        padding: 16px;
        max-width: 1200px;
        margin: 0 auto;
      }

      h1 {
        margin: 0 0 16px 0;
        font-size: 24px;
        font-weight: 400;
      }

      .floor-section {
        margin-bottom: 16px;
      }

      .floor-header, .area-header {
        display: flex;
        align-items: center;
        cursor: pointer;
        padding: 8px;
        background: var(--primary-background-color);
        border-radius: 4px;
        user-select: none;
      }

      .floor-header {
        font-size: 18px;
        font-weight: 500;
      }

      .area-header {
        font-size: 16px;
        margin-top: 8px;
      }

      .area-section {
        /* No left margin when no floors shown */
      }

      .area-section.with-floor {
        margin-left: 16px;
      }

      .chevron {
        margin-right: 8px;
        transition: transform 0.2s;
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

      .with-floor .lights-table {
        margin-left: 16px;
        width: calc(100% - 16px);
      }

      .lights-table th,
      .lights-table td {
        padding: 8px;
        border-bottom: 1px solid var(--divider-color);
      }

      .lights-table th {
        text-align: left;
        font-weight: 500;
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
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .entity-id {
        font-size: 12px;
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
        padding: 8px 0;
      }

      .with-floor .no-lights {
        margin-left: 16px;
      }
    `;
  }

  constructor() {
    super();
    this._data = null;
    this._loading = true;
    this._collapsed = this._loadCollapsedState();
    this._configureChecked = new Set();
  }

  connectedCallback() {
    super.connectedCallback();
    this._fetchLights();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this._fetchTimeout) {
      clearTimeout(this._fetchTimeout);
    }
  }

  updated(changedProperties) {
    super.updated(changedProperties);
    // Re-fetch when hass changes to pick up icon/state updates
    if (changedProperties.has("hass") && this.hass && !this._loading) {
      this._debouncedFetch();
    }
  }

  _debouncedFetch() {
    // Debounce re-fetches to avoid excessive API calls
    if (this._fetchTimeout) {
      clearTimeout(this._fetchTimeout);
    }
    this._fetchTimeout = setTimeout(() => {
      this._fetchLights();
    }, 1000);
  }

  _loadCollapsedState() {
    try {
      return JSON.parse(localStorage.getItem("fade_lights_collapsed") || "{}");
    } catch {
      return {};
    }
  }

  _saveCollapsedState() {
    localStorage.setItem("fade_lights_collapsed", JSON.stringify(this._collapsed));
  }

  async _fetchLights() {
    this._loading = true;
    const isInitialLoad = this._data === null;
    try {
      const result = await this.hass.callWS({ type: "fade_lights/get_lights" });
      this._data = result;
      // Only populate _configureChecked on initial load to preserve user changes
      if (isInitialLoad) {
        const checkedSet = new Set();
        if (result && result.floors) {
          for (const floor of result.floors) {
            for (const area of floor.areas) {
              for (const light of area.lights) {
                if (light.min_delay_ms == null) {
                  checkedSet.add(light.entity_id);
                }
              }
            }
          }
        }
        this._configureChecked = checkedSet;
      }
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
    const value = e.target.value ? parseInt(e.target.value, 10) : null;
    this._saveConfig(entityId, "min_delay_ms", value);
  }

  _handleCheckboxChange(entityId, field, e) {
    this._saveConfig(entityId, field, e.target.checked);
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

    if (currentState === "none") {
      // Check all
      for (const id of entityIds) {
        newSet.add(id);
      }
    } else {
      // Uncheck all (for "all" or "some")
      for (const id of entityIds) {
        newSet.delete(id);
      }
    }
    this._configureChecked = newSet;
  }

  _handleAreaCheckboxChange(area, e) {
    e.stopPropagation(); // Prevent collapse toggle
    const entityIds = this._getAreaLightIds(area);
    const currentState = this._getCheckboxState(entityIds);
    const newSet = new Set(this._configureChecked);

    if (currentState === "none") {
      // Check all
      for (const id of entityIds) {
        newSet.add(id);
      }
    } else {
      // Uncheck all (for "all" or "some")
      for (const id of entityIds) {
        newSet.delete(id);
      }
    }
    this._configureChecked = newSet;
  }

  _hasRealFloors() {
    // Check if there are any floors with a real floor_id (not null/none)
    if (!this._data || !this._data.floors) return false;
    return this._data.floors.some((floor) => floor.floor_id !== null);
  }

  render() {
    if (this._loading) {
      return html`<h1>Fade Lights</h1><p>Loading...</p>`;
    }

    if (!this._data || !this._data.floors) {
      return html`<h1>Fade Lights</h1><p>No lights found.</p>`;
    }

    const hasRealFloors = this._hasRealFloors();

    return html`
      <h1>Fade Lights</h1>
      ${hasRealFloors
        ? this._data.floors.map((floor) => this._renderFloor(floor))
        : this._renderAreasOnly()}
    `;
  }

  _renderAreasOnly() {
    // When no real floors exist, render areas directly without floor grouping
    const allAreas = this._data.floors.flatMap((floor) => floor.areas);
    return allAreas.map((area) => this._renderArea(area, null, false));
  }

  _renderFloor(floor) {
    const floorKey = `floor_${floor.floor_id || "none"}`;
    const isCollapsed = this._collapsed[floorKey];
    const floorIcon = floor.icon || "mdi:floor-plan";
    const floorLightIds = this._getFloorLightIds(floor);
    const checkboxState = this._getCheckboxState(floorLightIds);

    return html`
      <div class="floor-section">
        <div class="floor-header" @click=${() => this._toggleCollapse(floorKey)}>
          <input
            type="checkbox"
            .checked=${checkboxState === "all"}
            .indeterminate=${checkboxState === "some"}
            @click=${(e) => e.stopPropagation()}
            @change=${(e) => this._handleFloorCheckboxChange(floor, e)}
            style="margin-right: 8px;"
          />
          <span class="chevron ${isCollapsed ? "collapsed" : ""}">▼</span>
          <ha-icon class="header-icon" icon="${floorIcon}"></ha-icon>
          ${floor.name}
        </div>
        <div class="${isCollapsed ? "hidden" : ""}">
          ${floor.areas.map((area) => this._renderArea(area, floor.floor_id, true))}
        </div>
      </div>
    `;
  }

  _renderArea(area, floorId, withFloor) {
    const areaKey = `area_${floorId || "none"}_${area.area_id || "none"}`;
    const isCollapsed = this._collapsed[areaKey];
    const areaIcon = area.icon || "mdi:texture-box";
    const areaLightIds = this._getAreaLightIds(area);
    const checkboxState = this._getCheckboxState(areaLightIds);

    return html`
      <div class="area-section ${withFloor ? "with-floor" : ""}">
        <div class="area-header" @click=${() => this._toggleCollapse(areaKey)}>
          <input
            type="checkbox"
            .checked=${checkboxState === "all"}
            .indeterminate=${checkboxState === "some"}
            @click=${(e) => e.stopPropagation()}
            @change=${(e) => this._handleAreaCheckboxChange(area, e)}
            style="margin-right: 8px;"
          />
          <span class="chevron ${isCollapsed ? "collapsed" : ""}">▼</span>
          <ha-icon class="header-icon" icon="${areaIcon}"></ha-icon>
          ${area.name}
        </div>
        <div class="${isCollapsed ? "hidden" : ""}">
          ${area.lights.length > 0
            ? html`
                <table class="lights-table">
                  <thead>
                    <tr>
                      <th class="col-light">Light</th>
                      <th class="col-delay">Min Delay (ms)</th>
                      <th class="col-exclude">Exclude</th>
                      <th class="col-configure">Configure</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${area.lights.map((light) => this._renderLight(light))}
                  </tbody>
                </table>
              `
            : html`<p class="no-lights">No lights</p>`}
        </div>
      </div>
    `;
  }

  _renderLight(light) {
    const lightIcon = light.icon || "mdi:lightbulb";
    const state = this.hass.states[light.entity_id];
    const isOn = state && state.state === "on";

    return html`
      <tr>
        <td class="col-light">
          <div class="light-cell" @click=${() => this._openLightDialog(light.entity_id)}>
            <ha-icon class="light-icon ${isOn ? "on" : ""}" icon="${lightIcon}"></ha-icon>
            <div class="light-info">
              <div class="light-name">${light.name}</div>
              <div class="entity-id">${light.entity_id}</div>
            </div>
          </div>
        </td>
        <td class="col-delay">
          <input
            type="number"
            min="50"
            max="1000"
            step="10"
            placeholder="default"
            .value=${light.min_delay_ms || ""}
            @change=${(e) => this._handleDelayChange(light.entity_id, e)}
          />
        </td>
        <td class="col-exclude">
          <input
            type="checkbox"
            .checked=${light.exclude}
            @change=${(e) => this._handleCheckboxChange(light.entity_id, "exclude", e)}
          />
        </td>
        <td class="col-configure">
          <input
            type="checkbox"
            .checked=${this._configureChecked.has(light.entity_id)}
            @change=${(e) => this._handleConfigureChange(light.entity_id, e)}
          />
        </td>
      </tr>
    `;
  }
}

customElements.define("fade-lights-panel", FadeLightsPanel);
