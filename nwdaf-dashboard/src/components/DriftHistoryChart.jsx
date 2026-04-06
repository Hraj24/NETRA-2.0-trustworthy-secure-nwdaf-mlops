// NETRA 2.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
} from "recharts";

export default function DriftHistoryChart({ driftHistory }) {
  // Transform drift events into time-series data
  const chartData = React.useMemo(() => {
    if (!driftHistory || driftHistory.length === 0) return [];

    // Reverse to show chronological order (oldest first)
    const reversed = [...driftHistory].reverse();

    return reversed.map((evt, index) => {
      const timestamp = evt.timestamp
        ? new Date(evt.timestamp).toLocaleTimeString("en-IN", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false,
          })
        : `T-${reversed.length - index}`;

      return {
        timestamp,
        sla_metric: evt.sla_metric,
        action: evt.action,
        detector: evt.detector,
        type: evt.type,
        isRollback: evt.action === "rollback",
        isRecovery: evt.action === "recovered",
      };
    });
  }, [driftHistory]);

  // Get SLA threshold line based on scenario (default to HRLLC as most stringent)
  const slaThreshold = 0.3; // Moving average drift threshold

  if (!driftHistory || driftHistory.length === 0) {
    return (
      <div className="card">
        <div className="card__header">
          <h3 className="card__title">Drift History</h3>
          <p className="card__subtitle">SLA Metric & Rollback Events Timeline</p>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">📈</div>
          <div className="empty-state__text">
            No drift events recorded yet. Run predictions to populate history.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">Drift History</h3>
        <p className="card__subtitle">
          SLA Metric & Rollback Events Timeline ({driftHistory.length} events)
        </p>
      </div>

      <div style={{ height: 280, width: "100%" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 40 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.1)"
            />
            <XAxis
              dataKey="timestamp"
              tick={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              angle={-45}
              textAnchor="end"
              height={60}
              interval={0}
            />
            <YAxis
              tick={{ fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              label={{
                value: "SLA Metric",
                angle: -90,
                position: "insideLeft",
                fill: "var(--text-muted)",
                fontSize: 12,
                fontFamily: "var(--font-mono)",
              }}
            />
            <Tooltip
              contentStyle={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                fontSize: 12,
                fontFamily: "var(--font-mono)",
              }}
              labelStyle={{ color: "var(--text-primary)", marginBottom: 8 }}
              formatter={(value, name) => {
                if (name === "SLA Metric") return [value.toFixed(3), name];
                return [value, name];
              }}
              labelFormatter={(label) => `Time: ${label}`}
            />
            {/* SLA Threshold Reference Line */}
            <ReferenceLine
              y={slaThreshold}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              label={{
                value: "Drift Threshold",
                fill: "#f59e0b",
                fontSize: 11,
                fontFamily: "var(--font-mono)",
              }}
            />
            {/* SLA Metric Line */}
            <Line
              type="monotone"
              dataKey="sla_metric"
              stroke="#00d4ff"
              strokeWidth={2}
              dot={(props) => {
                const { cx, cy, payload } = props;
                const fill = payload.isRollback
                  ? "#ef4444"
                  : payload.isRecovery
                  ? "#10b981"
                  : "#00d4ff";
                return (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={payload.isRollback || payload.isRecovery ? 6 : 4}
                    fill={fill}
                    stroke="var(--bg-primary)"
                    strokeWidth={2}
                  />
                );
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: "var(--space-4)", justifyContent: "center", marginTop: "var(--space-4)", flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#00d4ff" }} />
          <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>SLA Metric</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#ef4444" }} />
          <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>Rollback Event</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <div style={{ width: 12, height: 12, borderRadius: "50%", background: "#10b981" }} />
          <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>Recovery Event</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <div style={{ width: 20, height: 0, borderTop: "2px dashed #f59e0b" }} />
          <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>Drift Threshold</span>
        </div>
      </div>
    </div>
  );
}
