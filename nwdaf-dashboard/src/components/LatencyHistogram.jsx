// NETRA 2.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
  Cell,
} from "recharts";
import { latencyStats } from "../api";

// Scenario metadata for tabs (same palette as RadarPanel)
const SCENARIOS = [
  { key: "all", label: "All", color: "#00d4ff", icon: "🌐" },
  { key: "IC", label: "IC", color: "#00d4ff", icon: "🥽" },
  { key: "HRLLC", label: "HRLLC", color: "#ef4444", icon: "⚡" },
  { key: "MC", label: "MC", color: "#10b981", icon: "📡" },
  { key: "UC", label: "UC", color: "#f59e0b", icon: "🔗" },
  { key: "AIAC", label: "AIAC", color: "#7c3aed", icon: "🤖" },
  { key: "ISAC", label: "ISAC", color: "#06b6d4", icon: "📡" },
];

function getKpiColor(value, slaTarget) {
  if (value <= slaTarget) return "var(--color-success)";
  if (value <= slaTarget * 2) return "var(--color-warning)";
  return "var(--color-danger)";
}

function getComplianceColor(pct) {
  if (pct >= 95) return "var(--color-success)";
  if (pct >= 80) return "var(--color-warning)";
  return "var(--color-danger)";
}

export default function LatencyHistogram({ scenario: activeScenario }) {
  const [activeTab, setActiveTab] = React.useState("all");
  const [data, setData] = React.useState(null);
  const [loading, setLoading] = React.useState(false);

  // Fetch latency data when tab changes
  const fetchData = React.useCallback(async (tab) => {
    try {
      setLoading(true);
      const res = await latencyStats(tab === "all" ? null : tab);
      setData(res.data);
    } catch {
      // silently ignore
    } finally {
      setLoading(false);
    }
  }, []);

  // Re-fetch when tab or activeScenario changes
  React.useEffect(() => {
    fetchData(activeTab);
  }, [activeTab, fetchData]);

  // Auto-refresh every 10 seconds
  React.useEffect(() => {
    const interval = setInterval(() => fetchData(activeTab), 10000);
    return () => clearInterval(interval);
  }, [activeTab, fetchData]);

  // Sync with parent scenario on initial load
  React.useEffect(() => {
    if (activeScenario && activeScenario !== activeTab) {
      // Don't override user's manual tab selection
    }
  }, [activeScenario]);

  const handleTabClick = (key) => {
    setActiveTab(key);
  };

  // ── Empty State ──
  if (!data || data.total_samples === 0) {
    return (
      <div className="card">
        <div className="card__header">
          <h3 className="card__title">Prediction Latency Distribution</h3>
          <p className="card__subtitle">
            Inference Latency Histogram per IMT-2030 Scenario
          </p>
        </div>

        {/* Scenario tabs even in empty state */}
        <div className="radar-tabs">
          {SCENARIOS.map((s) => (
            <button
              key={s.key}
              className={`radar-tab ${activeTab === s.key ? "radar-tab--active" : ""}`}
              style={activeTab === s.key ? { borderColor: s.color, color: s.color } : {}}
              onClick={() => handleTabClick(s.key)}
            >
              <span className="radar-tab__dot" style={{ background: s.color }} />
              {s.label}
            </button>
          ))}
        </div>

        <div className="empty-state">
          <div className="empty-state__icon">⏱</div>
          <div className="empty-state__text">
            No latency data yet — run predictions to populate histogram.
          </div>
        </div>
      </div>
    );
  }

  const { percentiles, histogram, total_samples, mean_ms, sla_target_ms, sla_compliance_pct } = data;
  const activeColor = SCENARIOS.find((s) => s.key === activeTab)?.color || "#00d4ff";

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">Prediction Latency Distribution</h3>
        <p className="card__subtitle">
          Inference Latency Histogram per IMT-2030 Scenario
          <span
            className="badge-itu-tag"
            style={{ marginLeft: 8 }}
          >
            {total_samples} samples
          </span>
        </p>
      </div>

      {/* ─── Scenario Tabs ─── */}
      <div className="radar-tabs">
        {SCENARIOS.map((s) => (
          <button
            key={s.key}
            className={`radar-tab ${activeTab === s.key ? "radar-tab--active" : ""}`}
            style={activeTab === s.key ? { borderColor: s.color, color: s.color } : {}}
            onClick={() => handleTabClick(s.key)}
          >
            <span className="radar-tab__dot" style={{ background: s.color }} />
            {s.label}
          </button>
        ))}
      </div>

      {/* ─── Percentile KPI Strip ─── */}
      <div className="latency-kpi-strip">
        {[
          { label: "p50", value: percentiles.p50 },
          { label: "p95", value: percentiles.p95 },
          { label: "p99", value: percentiles.p99 },
        ].map(({ label, value }) => (
          <div
            key={label}
            className="latency-kpi"
            style={{ borderTopColor: getKpiColor(value, sla_target_ms) }}
          >
            <div
              className="latency-kpi__value"
              style={{ color: getKpiColor(value, sla_target_ms) }}
            >
              {value.toFixed(2)}
              <span className="latency-kpi__unit">ms</span>
            </div>
            <div className="latency-kpi__label">{label.toUpperCase()}</div>
          </div>
        ))}
      </div>

      {/* ─── Histogram Chart ─── */}
      <div className="latency-chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={histogram}
            margin={{ top: 20, right: 20, left: 0, bottom: 20 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.08)"
            />
            <XAxis
              dataKey="bucket"
              tick={{
                fill: "var(--text-muted)",
                fontSize: 11,
                fontFamily: "var(--font-mono)",
              }}
              label={{
                value: "Latency (ms)",
                position: "insideBottom",
                offset: -10,
                fill: "var(--text-muted)",
                fontSize: 12,
                fontFamily: "var(--font-mono)",
              }}
            />
            <YAxis
              tick={{
                fill: "var(--text-muted)",
                fontSize: 11,
                fontFamily: "var(--font-mono)",
              }}
              label={{
                value: "Requests",
                angle: -90,
                position: "insideLeft",
                fill: "var(--text-muted)",
                fontSize: 12,
                fontFamily: "var(--font-mono)",
              }}
            />
            <Tooltip
              contentStyle={{
                background: "var(--bg-secondary, #151d35)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                fontSize: 12,
                fontFamily: "var(--font-mono)",
              }}
              labelStyle={{
                color: "var(--text-primary)",
                marginBottom: 8,
              }}
              formatter={(value) => [`${value} requests`, "Count"]}
              labelFormatter={(label) => `${label} ms`}
            />
            {/* SLA threshold reference line */}
            <ReferenceLine
              x={(() => {
                // Find the bucket that contains the SLA target
                const slaBucket = histogram.find(
                  (h) =>
                    h.range_end !== null
                      ? h.range_start <= sla_target_ms && sla_target_ms < h.range_end
                      : h.range_start <= sla_target_ms
                );
                return slaBucket ? slaBucket.bucket : null;
              })()}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth={2}
              label={{
                value: `SLA ≤${sla_target_ms}ms`,
                fill: "#f59e0b",
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                position: "top",
              }}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={50}>
              {histogram.map((entry, index) => {
                const isOverSla =
                  entry.range_start >= sla_target_ms ||
                  (entry.range_end === null && entry.range_start >= sla_target_ms);
                return (
                  <Cell
                    key={`cell-${index}`}
                    fill={isOverSla ? "rgba(239, 68, 68, 0.7)" : activeColor}
                    fillOpacity={0.85}
                  />
                );
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ─── Legend ─── */}
      <div
        style={{
          display: "flex",
          gap: "var(--space-4)",
          justifyContent: "center",
          marginTop: "var(--space-2)",
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: 3,
              background: activeColor,
              opacity: 0.85,
            }}
          />
          <span
            style={{
              fontSize: 12,
              color: "var(--text-muted)",
              fontFamily: "var(--font-mono)",
            }}
          >
            Within SLA
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: 3,
              background: "rgba(239, 68, 68, 0.7)",
            }}
          />
          <span
            style={{
              fontSize: 12,
              color: "var(--text-muted)",
              fontFamily: "var(--font-mono)",
            }}
          >
            Exceeds SLA
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <div style={{ width: 20, height: 0, borderTop: "2px dashed #f59e0b" }} />
          <span
            style={{
              fontSize: 12,
              color: "var(--text-muted)",
              fontFamily: "var(--font-mono)",
            }}
          >
            SLA Threshold ({sla_target_ms}ms)
          </span>
        </div>
      </div>

      {/* ─── SLA Compliance + Mean ─── */}
      <div className="latency-sla-bar">
        <div className="latency-sla-bar__label">
          <span>
            SLA Compliance ({activeTab === "all" ? "Global" : activeTab})
          </span>
          <span
            style={{
              color: getComplianceColor(sla_compliance_pct),
              fontFamily: "var(--font-mono)",
              fontWeight: 600,
            }}
          >
            {sla_compliance_pct}%
          </span>
        </div>
        <div className="latency-sla-bar__track">
          <div
            className="latency-sla-bar__fill"
            style={{
              width: `${sla_compliance_pct}%`,
              background: getComplianceColor(sla_compliance_pct),
            }}
          />
        </div>
      </div>

      <div className="latency-mean">
        <span>
          Mean Latency:{" "}
          <strong style={{ color: "var(--text-primary)" }}>
            {mean_ms.toFixed(3)} ms
          </strong>
        </span>
        <span>
          SLA Target:{" "}
          <strong style={{ color: "#f59e0b" }}>
            ≤ {sla_target_ms} ms
          </strong>
        </span>
      </div>
    </div>
  );
}
