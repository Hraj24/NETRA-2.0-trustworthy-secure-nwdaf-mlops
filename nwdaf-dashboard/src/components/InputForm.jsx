// NETRA 2.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";

// ─── IMT-2030 Scenario KPI Bounds (ITU-R M.2160) ───
const SCENARIO_BOUNDS = {
  IC: {
    throughput_mbps: [300, 500],
    latency_ms: [1, 5],
    reliability_target: [99.999, 99.999],
    connection_density_km2: [1e4, 1e6],
    ai_load_score: [0.2, 0.5],
    resilience_score: [0.6, 0.9],
  },
  HRLLC: {
    throughput_mbps: [100, 300],
    latency_ms: [0.1, 1.0],
    reliability_target: [99.99999, 99.99999],
    connection_density_km2: [1e3, 1e5],
    ai_load_score: [0.1, 0.3],
    resilience_score: [0.9, 1.0],
  },
  MC: {
    throughput_mbps: [1, 10],
    latency_ms: [10, 100],
    reliability_target: [99.9, 99.9],
    connection_density_km2: [1e6, 1e8],
    ai_load_score: [0.1, 0.2],
    resilience_score: [0.5, 0.8],
  },
  UC: {
    throughput_mbps: [50, 300],
    latency_ms: [5, 50],
    reliability_target: [99.99, 99.99],
    connection_density_km2: [1e4, 1e7],
    ai_load_score: [0.2, 0.4],
    resilience_score: [0.8, 1.0],
  },
  AIAC: {
    throughput_mbps: [100, 500],
    latency_ms: [1, 10],
    reliability_target: [99.999, 99.999],
    connection_density_km2: [1e4, 1e6],
    ai_load_score: [0.7, 1.0],
    resilience_score: [0.7, 0.9],
  },
  ISAC: {
    throughput_mbps: [100, 300],
    latency_ms: [1, 5],
    reliability_target: [99.999, 99.999],
    connection_density_km2: [1e4, 1e6],
    ai_load_score: [0.5, 0.8],
    resilience_score: [0.7, 0.9],
  },
};

// ─── Scenario Metadata ───
const SCENARIOS = {
  IC: {
    label: "Immersive Communication",
    short: "IC",
    icon: "🌐",
    color: "var(--scenario-ic)",
    colorHex: "#00d4ff",
    desc: "Ultra-high throughput immersive experiences including XR, holographic telepresence, and 3D media.",
  },
  HRLLC: {
    label: "Hyper Reliable & Low Latency",
    short: "HRLLC",
    icon: "⚡",
    color: "var(--scenario-hrllc)",
    colorHex: "#ef4444",
    desc: "Mission-critical ultra-low latency applications for industrial automation and remote surgery.",
  },
  MC: {
    label: "Massive Communication",
    short: "MC",
    icon: "📡",
    color: "var(--scenario-mc)",
    colorHex: "#10b981",
    desc: "Massive-scale IoT with billions of low-power devices in smart cities and agriculture.",
  },
  UC: {
    label: "Ubiquitous Connectivity",
    short: "UC",
    icon: "🌍",
    color: "var(--scenario-uc)",
    colorHex: "#f59e0b",
    desc: "Seamless coverage everywhere including rural, aerial, and maritime communication.",
  },
  AIAC: {
    label: "AI and Communication",
    short: "AIAC",
    icon: "🤖",
    color: "var(--scenario-aiac)",
    colorHex: "#7c3aed",
    desc: "AI-native networks with integrated intelligence for autonomous optimization and inference.",
  },
  ISAC: {
    label: "Integrated Sensing & Communication",
    short: "ISAC",
    icon: "📊",
    color: "var(--scenario-isac)",
    colorHex: "#06b6d4",
    desc: "Joint sensing and communication for environmental awareness and precise positioning.",
  },
};

const FIELD_DEFS = [
  { key: "time_of_day", label: "Time of Day (0-23)", step: 1, min: 0, max: 23, type: "number" },
  { key: "usage_scenario", label: "Usage Scenario", type: "select" },
  { key: "throughput_mbps", label: "Throughput (Mbps)", step: 0.1, min: 0, boundsKey: "throughput_mbps" },
  { key: "latency_ms", label: "Latency (ms)", step: 0.01, min: 0, boundsKey: "latency_ms" },
  { key: "jitter_ms", label: "Jitter (ms)", step: 0.01, min: 0 },
  { key: "packet_loss_rate", label: "Packet Loss Rate", step: 0.00001, min: 0, max: 1 },
  { key: "reliability_target", label: "Reliability Target (%)", step: 0.001, min: 99, max: 100, boundsKey: "reliability_target" },
  { key: "connection_density_km2", label: "Connection Density (/km²)", step: 1000, min: 0, boundsKey: "connection_density_km2" },
  { key: "mobility_kmph", label: "Mobility (km/h)", step: 1, min: 0 },
  { key: "area_traffic_capacity_score", label: "Area Traffic Capacity (0-1)", step: 0.01, min: 0, max: 1 },
  { key: "ai_load_score", label: "AI Load Score (0-1)", step: 0.01, min: 0, max: 1, boundsKey: "ai_load_score" },
  { key: "resilience_score", label: "Resilience Score (0-1)", step: 0.01, min: 0, max: 1, boundsKey: "resilience_score" },
];

const DEFAULTS = {
  time_of_day: 12,
  usage_scenario: "IC",
  throughput_mbps: 400,
  latency_ms: 3,
  jitter_ms: 1.5,
  packet_loss_rate: 0.0001,
  reliability_target: 99.999,
  connection_density_km2: 500000,
  mobility_kmph: 100,
  area_traffic_capacity_score: 0.35,
  ai_load_score: 0.35,
  resilience_score: 0.75,
};

function formatBound(val) {
  if (val >= 1e6) return `${(val / 1e6).toFixed(0)}M`;
  if (val >= 1e3) return `${(val / 1e3).toFixed(0)}K`;
  return String(val);
}

export default function InputForm({ onSubmit, onFormChange, loading }) {
  const [form, setForm] = React.useState(DEFAULTS);

  const handleChange = (key, value) => {
    const parsed = key === "usage_scenario" ? value : parseFloat(value) || value;
    const next = { ...form, [key]: parsed };
    setForm(next);
    onFormChange?.(next);
  };

  React.useEffect(() => {
    onFormChange?.(form);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const scenario = form.usage_scenario;
  const bounds = SCENARIO_BOUNDS[scenario] || {};
  const meta = SCENARIOS[scenario];

  const isInRange = (key, value) => {
    const b = bounds[key];
    if (!b) return null;
    const v = parseFloat(value);
    if (isNaN(v)) return null;
    return v >= b[0] && v <= b[1];
  };

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">IMT-2030 Traffic Input</h3>
        <p className="card__subtitle">6G Network Traffic Parameters</p>
      </div>

      <div className="form-grid">
        {FIELD_DEFS.map((field) => {
          if (field.type === "select") {
            return (
              <div className="form-group" key={field.key}>
                <label>{field.label}</label>
                <select
                  value={form[field.key]}
                  onChange={(e) => handleChange(field.key, e.target.value)}
                >
                  {Object.entries(SCENARIOS).map(([code, s]) => (
                    <option key={code} value={code}>
                      {s.icon} {code} — {s.label}
                    </option>
                  ))}
                </select>
              </div>
            );
          }

          const valid = field.boundsKey ? isInRange(field.boundsKey, form[field.key]) : null;
          const b = bounds[field.boundsKey];

          return (
            <div className="form-group" key={field.key}>
              <label>{field.label}</label>
              <input
                type="number"
                step={field.step}
                min={field.min}
                max={field.max}
                value={form[field.key]}
                onChange={(e) => handleChange(field.key, e.target.value)}
                className={
                  valid === true
                    ? "input--valid"
                    : valid === false
                    ? "input--invalid"
                    : ""
                }
              />
              {b && (
                <span
                  className={`form-hint ${
                    valid === true
                      ? "form-hint--valid"
                      : valid === false
                      ? "form-hint--invalid"
                      : ""
                  }`}
                >
                  {scenario} range: {formatBound(b[0])} – {formatBound(b[1])}
                  {valid === true ? " ✓ ITU-R M.2160" : valid === false ? " ✗ Out of range" : ""}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Scenario Context Card */}
      {meta && (
        <div className="scenario-ctx">
          <div
            className="scenario-ctx__accent"
            style={{ background: meta.color }}
          />
          <span className="scenario-ctx__icon">{meta.icon}</span>
          <div className="scenario-ctx__body">
            <div className="scenario-ctx__name">{meta.label}</div>
            <div className="scenario-ctx__desc">{meta.desc}</div>
            <span className="scenario-ctx__cite">ITU-R M.2160 §4</span>
          </div>
        </div>
      )}

      {/* Run Analysis Button */}
      <button
        className="btn-analyze"
        onClick={() => onSubmit(form)}
        disabled={loading}
        style={{
          background: loading
            ? "var(--bg-elevated)"
            : `linear-gradient(135deg, ${meta?.colorHex || "#00d4ff"}, var(--accent-secondary))`,
        }}
      >
        {loading ? (
          <>
            <span className="spinner" />
            Analyzing 6G Network Conditions...
          </>
        ) : (
          <>⚡ Run Analysis</>
        )}
      </button>
    </div>
  );
}

export { SCENARIO_BOUNDS, SCENARIOS };
