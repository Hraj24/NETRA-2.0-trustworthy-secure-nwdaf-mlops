// NETRA 1.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

// ITU-R feature → KPI category mapping
const FEATURE_TAGS = {
  time_of_day: "Temporal",
  usage_scenario: "Scenario",
  throughput_mbps: "Throughput",
  latency_ms: "Latency",
  jitter_ms: "Latency",
  packet_loss_rate: "Reliability",
  reliability_target: "Reliability",
  connection_density_km2: "Density",
  mobility_kmph: "Mobility",
  area_traffic_capacity_score: "Capacity",
  ai_load_score: "AI-KPI",
  resilience_score: "Resilience",
};

function formatFeatureName(name) {
  return name
    .replace(/_/g, " ")
    .replace(/\b(mbps|ms|km2|kmph)\b/gi, (m) => m.toUpperCase());
}

export default function ExplainPanel({ shap, explainData }) {
  if (!shap) {
    return (
      <div className="card">
        <div className="card__header">
          <h3 className="card__title">SHAP Feature Attribution</h3>
          <p className="card__subtitle">
            Why did the model predict this load?
          </p>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">🔬</div>
          <div className="empty-state__text">
            Run an analysis to see feature attributions.
          </div>
        </div>
      </div>
    );
  }

  // Sort by absolute value descending
  const data = Object.entries(shap)
    .map(([feature, value]) => ({
      feature: formatFeatureName(feature),
      rawFeature: feature,
      value: parseFloat(value.toFixed(4)),
      tag: FEATURE_TAGS[feature] || "",
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const baseValue = explainData?.base_value;

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">SHAP Feature Attribution</h3>
        <p className="card__subtitle">
          Why did the model predict this load?
        </p>
      </div>

      <div className="shap-subtitle">
        Positive = increases predicted load &nbsp;|&nbsp; Negative =
        decreases predicted load
      </div>

      {/* ─── Diverging Horizontal Bar Chart ─── */}
      <div className="shap-chart-container">
        <ResponsiveContainer width="100%" height={Math.max(400, data.length * 36)}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 10, right: 60, left: 160, bottom: 10 }}
          >
            <XAxis
              type="number"
              tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
              axisLine={{ stroke: "var(--border)" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={<CustomYTick data={data} />}
              width={150}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
            />
            <ReferenceLine
              x={0}
              stroke="var(--border-light)"
              strokeWidth={1}
            />
            {baseValue != null && (
              <ReferenceLine
                x={baseValue}
                stroke="var(--accent-secondary)"
                strokeDasharray="4 4"
                label={{
                  value: `Base: ${baseValue.toFixed(2)}`,
                  fill: "var(--accent-secondary)",
                  fontSize: 10,
                }}
              />
            )}
            <Bar dataKey="value" barSize={18} radius={[4, 4, 4, 4]}>
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={
                    entry.value >= 0
                      ? "var(--accent-primary)"
                      : "var(--color-danger)"
                  }
                  style={{
                    animation: `bar-enter 0.5s ease-out ${i * 0.06}s both`,
                    transformOrigin: "center",
                  }}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ─── Legend ─── */}
      <div className="shap-legend">
        <div className="shap-legend__item">
          <span
            className="shap-legend__dot"
            style={{ background: "var(--accent-primary)" }}
          />
          Increases load
        </div>
        <div className="shap-legend__item">
          <span
            className="shap-legend__dot"
            style={{ background: "var(--color-danger)" }}
          />
          Decreases load
        </div>
      </div>
    </div>
  );
}

// Custom Y-axis tick with ITU-R tags
function CustomYTick({ x, y, payload, data }) {
  const entry = data?.find((d) => d.feature === payload?.value);
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={-8}
        y={0}
        dy={4}
        textAnchor="end"
        fill="var(--text-primary)"
        fontSize={12}
      >
        {payload?.value}
      </text>
      {entry?.tag && (
        <text
          x={-8}
          y={14}
          textAnchor="end"
          fill="var(--accent-primary)"
          fontSize={9}
          fontWeight={600}
          opacity={0.7}
        >
          [{entry.tag}]
        </text>
      )}
    </g>
  );
}

// Custom tooltip
function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div
      style={{
        background: "var(--bg-elevated)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-input)",
        padding: "8px 12px",
        fontSize: 12,
      }}
    >
      <div style={{ color: "var(--text-primary)", fontWeight: 600 }}>
        {d.feature}
      </div>
      <div
        style={{
          color: d.value >= 0 ? "var(--accent-primary)" : "var(--color-danger)",
          fontFamily: "var(--font-mono)",
          marginTop: 2,
        }}
      >
        SHAP: {d.value >= 0 ? "+" : ""}
        {d.value.toFixed(4)}
      </div>
      {d.tag && (
        <div style={{ color: "var(--text-muted)", marginTop: 2 }}>
          IMT-2030: {d.tag}
        </div>
      )}
    </div>
  );
}
