(function bootstrapSettingsHelpers(globalObj) {
  const DEFAULT_SETTINGS = {
    budget: { monthly_limit_usd: 50, alert_at_pct: 0.8 },
    runtime: { log_level: "INFO", hil_timeout_seconds: 3600 },
    agent_access: { mode: "balanced" },
  };

  const DEFAULT_DISPLAY_PREFS = {
    eventFilters: [
      "observations",
      "claims",
      "goals",
      "tasks",
      "hil",
      "system",
      "chat",
      "channels",
      "inference",
      "entities",
      "queries",
      "ingestion",
      "analysis",
      "synthesis",
      "memory",
      "monitor",
      "notification",
      "voice",
      "document",
      "predictions",
    ],
    minConfidence: 0,
    showSuperseded: false,
    timeRange: "all",
  };

  function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function getDefaultSettings() {
    return cloneJson(DEFAULT_SETTINGS);
  }

  function getDefaultDisplayPrefs() {
    return cloneJson(DEFAULT_DISPLAY_PREFS);
  }

  function mergeSettings(currentSettings, nextSettings) {
    const current = currentSettings || getDefaultSettings();
    const incoming = nextSettings || {};
    return {
      ...current,
      ...incoming,
      budget: { ...current.budget, ...(incoming.budget || {}) },
      runtime: { ...current.runtime, ...(incoming.runtime || {}) },
      agent_access: { ...current.agent_access, ...(incoming.agent_access || {}) },
    };
  }

  function resetSection(settings, section) {
    if (!DEFAULT_SETTINGS[section]) return settings;
    return { ...settings, [section]: cloneJson(DEFAULT_SETTINGS[section]) };
  }

  function buildExportPayload(settings, displayPrefs) {
    return {
      exported_at: new Date().toISOString(),
      settings,
      displayPrefs,
    };
  }

  function parseImportText(text) {
    const data = JSON.parse(text);
    return {
      settings: data.settings && typeof data.settings === "object" ? data.settings : null,
      displayPrefs:
        data.displayPrefs && typeof data.displayPrefs === "object"
          ? data.displayPrefs
          : null,
    };
  }

  globalObj.QESettingsHelpers = {
    getDefaultSettings,
    getDefaultDisplayPrefs,
    mergeSettings,
    resetSection,
    buildExportPayload,
    parseImportText,
  };
})(window);
