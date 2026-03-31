// NETRA 2.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";
import { driftLog } from "../api";

export default function StatusPanel({ status }) {
  const [driftEvents, setDriftEvents] = React.useState([]);

  // Fetch drift log on mount and when status changes
  React.useEffect(() => {
    driftLog().then((events) => {
      if (Array.isArray(events)) {
        setDriftEvents(events.slice(-5));
      }
    });
  }, [status]);

  if (!status) {
    return (
      <div className="card">
        <div className="card__header">
          <h3 className="card__title">System Health</h3>
          <p className="card__subtitle">NWDAF Model & Drift Status</p>
        </div>
        <div className="empty-state">
          <div className="empty-state__icon">🔄</div>
          <div className="empty-state__text">
            Connecting to health endpoint...
          </div>
        </div>
      </div>
    );
  }

  const {
    model_version = "Unknown",
    rollback_active = false,
    recovery_counter = 0,
    primary_detector = "MovingAverage",
  } = status;

  // Derive drift states from available data
  const maDrift = rollback_active;
  const adwinWarning = rollback_active;
  const ddmWarning = rollback_active;

  // Drift risk calculation - dynamic based on recovery counter and rollback state
  let driftRisk = 10;
  let driftRiskColor = "var(--color-success)";

  if (rollback_active) {
    driftRisk = 100;
    driftRiskColor = "var(--color-danger)";
  } else if (recovery_counter > 0 && recovery_counter < 3) {
    driftRisk = 40;
    driftRiskColor = "var(--color-warning)";
  } else if (recovery_counter >= 3 && recovery_counter < 5) {
    driftRisk = 25;
    driftRiskColor = "var(--color-warning)";
  } else if (recovery_counter >= 5) {
    driftRisk = 10;
    driftRiskColor = "var(--color-success)";
  }

  const stableCount = recovery_counter || 0;

  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">System Health</h3>
        <p className="card__subtitle">NWDAF Model & Drift Status</p>
      </div>

      {/* ─── Model Version ─── */}
      <div style={{ marginBottom: "var(--space-4)" }}>
        <span
          className={`badge-model ${
            rollback_active ? "badge-model--rollback" : ""
          }`}
        >
          {rollback_active ? `⚠ ROLLBACK ACTIVE` : model_version}
        </span>
      </div>

      {/* ─── Drift Detectors ─── */}
      <DetectorRow
        name={`● ${primary_detector}`}
        stable={!maDrift}
        type={maDrift ? "drift" : "stable"}
        label={maDrift ? "DRIFT ⚠️" : "STABLE ✅"}
      />
      <DetectorRow
        name="● ADWIN"
        stable={!adwinWarning}
        type={adwinWarning ? "warning" : "stable"}
        label={adwinWarning ? "WARNING ⚠️" : "STABLE ✅"}
      />
      <DetectorRow
        name="● DDM"
        stable={!ddmWarning}
        type={ddmWarning ? "warning" : "stable"}
        label={ddmWarning ? "WARNING ⚠️" : "STABLE ✅"}
      />

      <hr className="section-divider" />

      {/* ─── Drift Risk Meter ─── */}
      <div className="drift-risk-meter">
        <div className="drift-risk-meter__label">
          <span>Drift Risk Level</span>
          <span
            style={{
              color: driftRiskColor,
              fontFamily: "var(--font-mono)",
              fontWeight: 600,
            }}
          >
            {driftRisk}%
          </span>
        </div>
        <div className="drift-risk-meter__track">
          <div
            className="drift-risk-meter__fill"
            style={{
              width: `${driftRisk}%`,
              background: driftRiskColor,
            }}
          />
        </div>
      </div>

      {/* ─── Rollback Status ─── */}
      <div style={{ marginTop: "var(--space-4)" }}>
        {rollback_active ? (
          <div className="badge-rollback-active">
            ⚠️ AUTO-ROLLBACK ACTIVE — Running on Stable Model
          </div>
        ) : (
          <div className="badge-stable">✅ Primary Model Active</div>
        )}
      </div>

      {/* ─── Stable Counter ─── */}
      <div className="stable-counter">
        Stable windows: {stableCount}/5
      </div>

      <hr className="section-divider" />

      {/* ─── Last 5 Drift Events ─── */}
      <div>
        <div
          className="card__subtitle"
          style={{ marginBottom: "var(--space-2)" }}
        >
          Recent Drift Events
        </div>
        {driftEvents.length === 0 ? (
          <div className="drift-events-empty">
            No drift events recorded
          </div>
        ) : (
          <table className="drift-events-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Detector</th>
                <th>Type</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {driftEvents.map((evt, i) => (
                <tr key={i}>
                  <td>{evt.timestamp || "—"}</td>
                  <td>{evt.detector || "—"}</td>
                  <td>{evt.type || "—"}</td>
                  <td>{evt.action || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function DetectorRow({ name, type, label }) {
  const dotClass =
    type === "drift"
      ? "detector-dot--drift"
      : type === "warning"
      ? "detector-dot--warning"
      : "detector-dot--stable";

  const statusClass =
    type === "drift"
      ? "detector-row__status--drift"
      : type === "warning"
      ? "detector-row__status--warning"
      : "detector-row__status--stable";

  return (
    <div className="detector-row">
      <div className="detector-row__name">
        <span className={`detector-dot ${dotClass}`} />
        {name}
      </div>
      <div className={`detector-row__status ${statusClass}`}>{label}</div>
    </div>
  );
}
