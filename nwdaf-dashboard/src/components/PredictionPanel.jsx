// NETRA 2.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";
import CountUp from "react-countup";
import {
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";

// ─── Custom SVG Semi-Circle Gauge ───
// Two arcs forming one continuous 180° semicircle:
//   • Filled arc (color): sweeps left → value
//   • Remaining arc (grey): sweeps value → right
function SemiCircleGauge({ percentage, color, label }) {
  const width = 240;
  const height = 140;
  const strokeWidth = 16;
  const cx = width / 2;
  const cy = height - 18;
  const radius = (width - strokeWidth - 16) / 2;

  const pct = Math.min(100, Math.max(0, percentage));

  // Convert angle (0°=right, 180°=left in math coords) to SVG pixel coords
  const toXY = (deg) => ({
    x: cx + radius * Math.cos((deg * Math.PI) / 180),
    y: cy - radius * Math.sin((deg * Math.PI) / 180),
  });

  // Angles: 180° = leftmost (0%), 0° = rightmost (100%)
  const valueDeg = 180 - (pct / 100) * 180; // meeting point angle
  const left = toXY(180);
  const right = toXY(0);
  const mid = toXY(valueDeg);

  // SVG arc: sweep-flag=1 means clockwise in SVG coords
  // Going from left(180°) to right(0°) clockwise = through the top = semicircle shape
  const makeArc = (from, to, angleDiff) => {
    const large = angleDiff > 180 ? 1 : 0;
    return `M ${from.x} ${from.y} A ${radius} ${radius} 0 ${large} 1 ${to.x} ${to.y}`;
  };

  // Filled arc: left → meeting point (clockwise through top)
  const filledAngleDiff = 180 - valueDeg; // how many degrees the fill covers
  const filledPath = pct > 0.2 ? makeArc(left, mid, filledAngleDiff) : null;

  // Remaining arc: meeting point → right (clockwise continues through top)
  const remainAngleDiff = valueDeg; // remaining degrees to right
  const remainPath = pct < 99.8 ? makeArc(mid, right, remainAngleDiff) : null;

  return (
    <div className="gauge-container">
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Remaining arc (grey): meeting point → right end */}
        {remainPath && (
          <path
            d={remainPath}
            fill="none"
            stroke="rgba(255,255,255,0.12)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        )}

        {/* Filled arc (color): left end → meeting point */}
        {filledPath && (
          <path
            d={filledPath}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            style={{
              filter: `drop-shadow(0 0 8px ${color}50)`,
            }}
          />
        )}

        {/* Full empty state */}
        {pct <= 0.2 && (
          <path
            d={makeArc(left, right, 180)}
            fill="none"
            stroke="rgba(255,255,255,0.12)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        )}

        {/* 0% label at left endpoint */}
        <text
          x={left.x}
          y={left.y + 20}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize="11"
          fontFamily="var(--font-mono)"
        >
          0%
        </text>
        {/* 100% label at right endpoint */}
        <text
          x={right.x}
          y={right.y + 20}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize="11"
          fontFamily="var(--font-mono)"
        >
          100%
        </text>
      </svg>
      <div className="gauge-value-label" style={{ color }}>
        {label}
      </div>
    </div>
  );
}

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
  // Typical SHAP variance ranges from 100-1000+
  if (variance < 300) return "High";
  if (variance < 800) return "Medium";
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

  // Hook 1: Track previous load for CountUp animation
  React.useEffect(() => {
    if (result?.predicted_future_load != null) {
      setPrevLoad((prev) =>
        prev === result.predicted_future_load ? prev : prev
      );
    }
  }, [result]);

  // Hook 2: Debug logging (only when result exists)
  React.useEffect(() => {
    if (result) {
      const load = result.predicted_future_load ?? 0;
      const gaugeMax = 10000;
      const gaugePct = Math.min(100, Math.max(0, (load / gaugeMax) * 100));
      const variance = getShapVariance(shap);
      const confidence = getConfidence(variance, load);
      console.log("=== PREDICTION DEBUG ===");
      console.log("  load:", load);
      console.log("  gaugePct:", gaugePct);
      console.log("  variance:", variance);
      console.log("  confidence:", confidence);
    }
  }, [result, shap]);

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
  const confMeta = CONFIDENCE_MAP[confidence] || CONFIDENCE_MAP["Low"];
  const drivers = getTopDrivers(shap);

  // Gauge: load % of max capacity (10 Gbps = 10000 Mbps per IMT-2030)
  const gaugeMax = 10000;
  const gaugePct = Math.min(100, Math.max(0, (load / gaugeMax) * 100));
  const severity = getGaugeSeverity(gaugePct);



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

          {/* ─── Semicircular Gauge (SVG) ─── */}
          <SemiCircleGauge
            percentage={gaugePct}
            color={severity.color}
            label={`${severity.label} — ${gaugePct.toFixed(1)}%`}
          />

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
