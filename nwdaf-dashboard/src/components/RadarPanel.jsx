// NETRA 2.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { SCENARIO_BOUNDS, SCENARIOS } from "./InputForm";

// Normalization ranges (absolute max/min across all scenarios for 0-1 mapping)
const NORM_RANGES = {
  throughput: { min: 0, max: 500 },
  latency: { min: 0, max: 100 },      // inverted: lower = better
  reliability: { min: 99, max: 100 },
  density: { min: 0, max: 1e8 },
  aiLoad: { min: 0, max: 1 },
  resilience: { min: 0, max: 1 },
};

function normalize(value, min, max) {
  if (max === min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function buildRadarData(formData, scenarioBounds) {
  const b = scenarioBounds || {};

  const throughputMid = b.throughput_mbps
    ? (b.throughput_mbps[0] + b.throughput_mbps[1]) / 2
    : 0;
  const latencyMid = b.latency_ms
    ? (b.latency_ms[0] + b.latency_ms[1]) / 2
    : 50;
  const reliabilityMid = b.reliability_target
    ? (b.reliability_target[0] + b.reliability_target[1]) / 2
    : 99.9;
  const densityMid = b.connection_density_km2
    ? (b.connection_density_km2[0] + b.connection_density_km2[1]) / 2
    : 1e4;
  const aiMid = b.ai_load_score
    ? (b.ai_load_score[0] + b.ai_load_score[1]) / 2
    : 0.5;
  const resMid = b.resilience_score
    ? (b.resilience_score[0] + b.resilience_score[1]) / 2
    : 0.7;

  return [
    {
      axis: "Throughput",
      standard: normalize(throughputMid, NORM_RANGES.throughput.min, NORM_RANGES.throughput.max),
      input: normalize(formData.throughput_mbps || 0, NORM_RANGES.throughput.min, NORM_RANGES.throughput.max),
    },
    {
      axis: "Latency",
      // Inverted: lower latency = higher radar value
      standard: 1 - normalize(latencyMid, NORM_RANGES.latency.min, NORM_RANGES.latency.max),
      input: 1 - normalize(formData.latency_ms || 0, NORM_RANGES.latency.min, NORM_RANGES.latency.max),
    },
    {
      axis: "Reliability",
      standard: normalize(reliabilityMid, NORM_RANGES.reliability.min, NORM_RANGES.reliability.max),
      input: normalize(formData.reliability_target || 99, NORM_RANGES.reliability.min, NORM_RANGES.reliability.max),
    },
    {
      axis: "Density",
      standard: normalize(Math.log10(densityMid || 1), 0, Math.log10(NORM_RANGES.density.max)),
      input: normalize(
        Math.log10(formData.connection_density_km2 || 1),
        0,
        Math.log10(NORM_RANGES.density.max)
      ),
    },
    {
      axis: "AI Load",
      standard: normalize(aiMid, NORM_RANGES.aiLoad.min, NORM_RANGES.aiLoad.max),
      input: normalize(formData.ai_load_score || 0, NORM_RANGES.aiLoad.min, NORM_RANGES.aiLoad.max),
    },
    {
      axis: "Resilience",
      standard: normalize(resMid, NORM_RANGES.resilience.min, NORM_RANGES.resilience.max),
      input: normalize(formData.resilience_score || 0, NORM_RANGES.resilience.min, NORM_RANGES.resilience.max),
    },
  ];
}

export default function RadarPanel({ formData, scenario }) {
  const [activeScenario, setActiveScenario] = React.useState(scenario || "IC");

  React.useEffect(() => {
    setActiveScenario(scenario || "IC");
  }, [scenario]);

  const bounds = SCENARIO_BOUNDS[activeScenario] || {};
  const meta = SCENARIOS[activeScenario];
  const data = buildRadarData(formData || {}, bounds);

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">IMT-2030 Scenario KPI Profile</h3>
        <p className="card__subtitle">
          Current input vs ITU-R M.2160 standard bounds
        </p>
      </div>

      {/* Scenario Tabs */}
      <div className="radar-tabs">
        {Object.entries(SCENARIOS).map(([code, s]) => (
          <button
            key={code}
            className={`radar-tab ${
              activeScenario === code ? "radar-tab--active" : ""
            }`}
            style={
              activeScenario === code
                ? { borderColor: s.colorHex, color: s.colorHex }
                : {}
            }
            onClick={() => setActiveScenario(code)}
          >
            <span
              className="radar-tab__dot"
              style={{ background: s.colorHex }}
            />
            {code}
          </button>
        ))}
      </div>

      {/* Radar Chart */}
      <ResponsiveContainer width="100%" height={380}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="72%">
          <PolarGrid
            stroke="var(--border)"
            strokeDasharray="3 3"
          />
          <PolarAngleAxis
            dataKey="axis"
            tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 1]}
            tick={false}
            axisLine={false}
          />
          <Radar
            name="IMT-2030 Standard"
            dataKey="standard"
            stroke={meta?.colorHex || "#00d4ff"}
            fill={meta?.colorHex || "#00d4ff"}
            fillOpacity={0.2}
            strokeWidth={2}
          />
          <Radar
            name="Your Input"
            dataKey="input"
            stroke="#f0f6ff"
            fill="#00d4ff"
            fillOpacity={0.35}
            strokeWidth={2}
            strokeDasharray="4 2"
          />
        </RadarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="radar-legend">
        <div className="radar-legend__item">
          <span
            className="radar-legend__swatch"
            style={{
              background: meta?.colorHex || "#00d4ff",
              opacity: 0.4,
            }}
          />
          IMT-2030 Standard
        </div>
        <div className="radar-legend__item">
          <span
            className="radar-legend__swatch"
            style={{
              background: "#00d4ff",
              opacity: 0.6,
              border: "1px dashed #f0f6ff",
            }}
          />
          Your Input
        </div>
      </div>
    </div>
  );
}
