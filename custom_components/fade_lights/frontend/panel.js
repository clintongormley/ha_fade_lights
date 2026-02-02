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
        margin-left: 16px;
        margin-top: 8px;
      }

      .chevron {
        margin-right: 8px;
        transition: transform 0.2s;
      }

      .chevron.collapsed {
        transform: rotate(-90deg);
      }

      .area-section {
        margin-left: 16px;
      }

      .lights-table {
        width: 100%;
        border-collapse: collapse;
        margin: 8px 0 8px 32px;
      }

      .lights-table th {
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid var(--divider-color);
        font-weight: 500;
      }

      .lights-table td {
        padding: 8px;
        border-bottom: 1px solid var(--divider-color);
      }

      .light-name {
        font-weight: 500;
      }

      .entity-id {
        font-size: 12px;
        color: var(--secondary-text-color);
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
    `;
  }

  constructor() {
    super();
    this._data = null;
    this._loading = true;
    this._collapsed = this._loadCollapsedState();
  }

  connectedCallback() {
    super.connectedCallback();
    this._fetchLights();
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
    try {
      const result = await this.hass.callWS({ type: "fade_lights/get_lights" });
      this._data = result;
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

  render() {
    if (this._loading) {
      return html`<h1>Fade Lights</h1><p>Loading...</p>`;
    }

    if (!this._data || !this._data.floors) {
      return html`<h1>Fade Lights</h1><p>No lights found.</p>`;
    }

    return html`
      <h1>Fade Lights</h1>
      ${this._data.floors.map((floor) => this._renderFloor(floor))}
    `;
  }

  _renderFloor(floor) {
    const floorKey = `floor_${floor.floor_id || "none"}`;
    const isCollapsed = this._collapsed[floorKey];

    return html`
      <div class="floor-section">
        <div class="floor-header" @click=${() => this._toggleCollapse(floorKey)}>
          <span class="chevron ${isCollapsed ? "collapsed" : ""}">▼</span>
          ${floor.name}
        </div>
        <div class="${isCollapsed ? "hidden" : ""}">
          ${floor.areas.map((area) => this._renderArea(area, floor.floor_id))}
        </div>
      </div>
    `;
  }

  _renderArea(area, floorId) {
    const areaKey = `area_${floorId || "none"}_${area.area_id || "none"}`;
    const isCollapsed = this._collapsed[areaKey];

    return html`
      <div class="area-section">
        <div class="area-header" @click=${() => this._toggleCollapse(areaKey)}>
          <span class="chevron ${isCollapsed ? "collapsed" : ""}">▼</span>
          ${area.name}
        </div>
        <div class="${isCollapsed ? "hidden" : ""}">
          ${area.lights.length > 0
            ? html`
                <table class="lights-table">
                  <thead>
                    <tr>
                      <th>Light</th>
                      <th>Min Delay (ms)</th>
                      <th>Exclude</th>
                      <th>Native Transition</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${area.lights.map((light) => this._renderLight(light))}
                  </tbody>
                </table>
              `
            : html`<p style="margin-left: 32px; color: var(--secondary-text-color);">No lights</p>`}
        </div>
      </div>
    `;
  }

  _renderLight(light) {
    return html`
      <tr>
        <td>
          <div class="light-name">${light.name}</div>
          <div class="entity-id">${light.entity_id}</div>
        </td>
        <td>
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
        <td>
          <input
            type="checkbox"
            .checked=${light.exclude}
            @change=${(e) => this._handleCheckboxChange(light.entity_id, "exclude", e)}
          />
        </td>
        <td>
          <input
            type="checkbox"
            .checked=${light.use_native_transition}
            @change=${(e) => this._handleCheckboxChange(light.entity_id, "use_native_transition", e)}
          />
        </td>
      </tr>
    `;
  }
}

customElements.define("fade-lights-panel", FadeLightsPanel);
