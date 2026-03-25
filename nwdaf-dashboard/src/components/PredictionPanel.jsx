// NETRA 1.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";
import CountUp from "react-countup";
import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";

// ─── Scenario Colors ───
const SCENARIO_COLORS = {
  IC: "#00d4ff",
  HRLLC: "#ef4444",
  MC: "#10b981",
  UC: "#f59e0b",
  AIAC: "#7c3aed",
  ISAC: "#06b6d4",
};

// ─── SHAP Helpers ───
function getShapVariance(shap) {
  if (!shap) return null;
  const values = Object.values(shap);
  if (values.length === 0) return null;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
}

function getConfidence(variance, load) {
  if (variance === null || variance === undefined) return "Low";
  // Confidence based on SHAP variance magnitude (lower variance = higher confidence)
  if (variance < 0.5) return "High";
  if (variance < 2.0) return "Medium";
  return "Low";
}

const CONFIDENCE_MAP = {
  High: { pct: 100, color: "var(--color-success)" },
  Medium: { pct: 66, color: "var(--color-warning)" },
  Low: { pct: 33, color: "var(--color-danger)" },
};

// ─── Gauge severity zones ───
function getGaugeSeverity(pct) {
  if (pct < 25) return { label: "Stable", color: "#10b981" };
  if (pct < 60) return { label: "Moderate", color: "#f59e0b" };
  if (pct < 85) return { label: "High", color: "#f97316" };
  return { label: "Critical", color: "#ef4444" };
}

// ─── Top 3 SHAP features ───
function getTopDrivers(shap) {
  if (!shap) return [];
  return Object.entries(shap)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3)
    .map(([feature, value]) => ({
      feature: feature.replace(/_/g, " "),
      value,
      positive: value >= 0,
    }));
}

export default function PredictionPanel({
  result,
  shap,
  loading,
  scenario,
  predictionHistory,
}) {
  const [prevLoad, setPrevLoad] = React.useState(0);

  React.useEffect(() => {
    if (result?.predicted_future_load != null) {
      setPrevLoad((prev) =>
        prev === result.predicted_future_load ? prev : prev
      );
    }
  }, [result]);

  if (!result && !loading) {
    return (
      <div className="card">
        <div className="card__header">
          <h3 className="card__title">Prediction</h3>
          <p className="card__subtitle">Run analysis to see results</p>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">📊</div>
          <div className="empty-state__text">
            No prediction yet. Submit traffic parameters to start.
          </div>
        </div>
      </div>
    );
  }

  const load = result?.predicted_future_load ?? 0;
  const scenarioColor = SCENARIO_COLORS[scenario] || "#00d4ff";
  const variance = getShapVariance(shap);
  const confidence = getConfidence(variance, load);
  const confMeta = CONFIDENCE_MAP[confidence];
  const drivers = getTopDrivers(shap);

  // Gauge: load % of max capacity (10 Gbps = 10000 Mbps per IMT-2030)
  const gaugeMax = 10000;
  const gaugePct = Math.min(100, Math.max(0, (load / gaugeMax) * 100));
  const severity = getGaugeSeverity(gaugePct);

  // Gauge data for RadialBarChart
  const gaugeData = [
    { name: "load", value: gaugePct, fill: severity.color },
  ];

  // Sparkline data
  const sparkData = (predictionHistory || []).map((p, i) => ({
    idx: i,
    val: p.value,
  }));

  // Prediction timestamp
  const now = new Date().toLocaleTimeString("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">Prediction</h3>
        <p className="card__subtitle">6G Network Load Forecast</p>
      </div>

      {loading ? (
        <div className="empty-state">
          <div className="spinner" style={{ width: 28, height: 28, borderWidth: 3 }} />
          <div className="empty-state__text" style={{ marginTop: 12 }}>
            Analyzing...
          </div>
        </div>
      ) : (
        <>
          {/* ─── Predicted Load Number ─── */}
          <div className="prediction-load" style={{ color: scenarioColor }}>
            <CountUp
              start={prevLoad}
              end={load}
              decimals={2}
              duration={0.5}
              separator=","
              onEnd={() => setPrevLoad(load)}
            />
          </div>
          <div className="prediction-label">
            Mbps — Predicted Future Load
          </div>
          <div className="prediction-timestamp">
            Predicted at {now} IST
          </div>

          {/* ─── Semicircular Gauge ─── */}
          <div className="gauge-container">
            <ResponsiveContainer width="100%" height={140}>
              <RadialBarChart
                cx="50%"
                cy="100%"
                innerRadius="70%"
                outerRadius="100%"
                startAngle={180}
                endAngle={0}
                data={gaugeData}
                barSize={14}
              >
                <RadialBar
                  dataKey="value"
                  cornerRadius={8}
                  background={{ fill: "rgba(255,255,255,0.04)" }}
                />
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="gauge-labels">
              <span>0%</span>
              <span>100%</span>
            </div>
            <div
              className="gauge-value-label"
              style={{ color: severity.color }}
            >
              {severity.label} — {gaugePct.toFixed(0)}%
            </div>
          </div>

          {/* ─── Confidence Bar ─── */}
          <div className="confidence-bar">
            <div className="confidence-bar__label">
              <span>Prediction Confidence</span>
              <span style={{ color: confMeta.color, fontWeight: 600 }}>
                {confidence}
              </span>
            </div>
            <div className="confidence-bar__track">
              <div
                className="confidence-bar__fill"
                style={{
                  width: `${confMeta.pct}%`,
                  background: confMeta.color,
                }}
              />
            </div>
          </div>

          {/* ─── Key Drivers ─── */}
          {drivers.length > 0 && (
            <div className="key-drivers">
              <div className="key-drivers__title">Key Drivers</div>
              <div className="key-drivers__list">
                {drivers.map((d) => (
                  <span
                    key={d.feature}
                    className={`badge-shap ${
                      d.positive
                        ? "badge-shap--positive"
                        : "badge-shap--negative"
                    }`}
                  >
                    {d.positive ? "↑" : "↓"} {d.feature}{" "}
                    {d.value >= 0 ? "+" : ""}
                    {d.value.toFixed(2)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* ─── Sparkline ─── */}
          {sparkData.length > 1 && (
            <div className="sparkline-container">
              <div className="sparkline-container__title">
                Last {sparkData.length} Predictions
              </div>
              <ResponsiveContainer width="100%" height={60}>
                <LineChart data={sparkData}>
                  <Line
                    type="monotone"
                    dataKey="val"
                    stroke={scenarioColor}
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ─── Model Reasoning ─── */}
          {result.explanation && (
            <div className="reasoning-card">
              <div className="reasoning-card__text">
                {result.explanation}
              </div>
            </div>
          )}

          {/* ─── Warning ─── */}
          {result.warning && (
            <div className="warning-text">{result.warning}</div>
          )}
        </>
      )}
    </div>
  );
}
