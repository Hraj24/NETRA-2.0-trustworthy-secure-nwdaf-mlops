// NETRA 2.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";
import { predict, explain, health, checkAlerts, driftLog } from "./api";
import "./App.css";

import NavBar from "./components/NavBar";
import InputForm from "./components/InputForm";
import PredictionPanel from "./components/PredictionPanel";
import ExplainPanel from "./components/ExplainPanel";
import StatusPanel from "./components/StatusPanel";
import RadarPanel from "./components/RadarPanel";
import DriftHistoryChart from "./components/DriftHistoryChart";
import LatencyHistogram from "./components/LatencyHistogram";

import toast, { Toaster } from "react-hot-toast";

// Default form values (IMT-2030 IC scenario)
const DEFAULT_FORM = {
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

export default function App() {
  const [prediction, setPrediction] = React.useState(null);
  const [explainData, setExplainData] = React.useState(null);
  const [shap, setShap] = React.useState(null);
  const [status, setStatus] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [formData, setFormData] = React.useState(DEFAULT_FORM);
  const [predictionHistory, setPredictionHistory] = React.useState([]);
  const [alerts, setAlerts] = React.useState(null);
  const [driftHistory, setDriftHistory] = React.useState([]);

  // Auto-refresh health and drift log every 10 seconds
  React.useEffect(() => {
    const fetchHealth = async () => {
      try {
        const [s, driftEvents] = await Promise.all([
          health(),
          driftLog()
        ]);
        setStatus(s.data);
        setDriftHistory(Array.isArray(driftEvents) ? driftEvents : []);
      } catch {
        // silently ignore
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Run analysis pipeline
  const run = async (input) => {
    try {
      setLoading(true);

      // 1. Predict
      const p = await predict(input);
      setPrediction(p.data);

      // Append to history (keep last 10)
      setPredictionHistory((prev) => {
        const next = [
          ...prev,
          {
            value: p.data.predicted_future_load,
            ts: Date.now(),
          },
        ];
        return next.slice(-10);
      });

      // 2. Explain (auto-called)
      const e = await explain(input);
      setExplainData(e.data);
      setShap(e.data.shap_values);

      // 3. Check SLA alerts
      try {
        const a = await checkAlerts(input);
        setAlerts(a.data);
        // Toast on SLA breach
        if (a.data.alerts && a.data.alerts.length > 0) {
          const criticalAlerts = a.data.alerts.filter(alert => alert.severity === "critical");
          if (criticalAlerts.length > 0) {
            toast.error(`⚠️ SLA Breach Predicted: ${criticalAlerts[0].message}`, {
              duration: 6000,
              style: {
                background: "#151d35",
                color: "#ef4444",
                border: "1px solid #ef4444",
              },
            });
          } else if (a.data.alerts.length > 0) {
            toast.warning(`⚠️ SLA Warning: ${a.data.alerts[0].message}`, {
              duration: 5000,
              style: {
                background: "#151d35",
                color: "#f59e0b",
                border: "1px solid #f59e0b",
              },
            });
          }
        }
      } catch (alertErr) {
        console.warn("Alerts check failed:", alertErr);
      }

      // 4. Health refresh
      const s = await health();
      setStatus(s.data);

      // Toast on drift
      if (s.data.rollback_active) {
        toast.error("⚠️ Drift Detected — Model Rollback Initiated", {
          duration: 5000,
          style: {
            background: "#151d35",
            color: "#ef4444",
            border: "1px solid #ef4444",
          },
        });
      }
    } catch (err) {
      console.error("Analysis failed:", err);
      toast.error("Analysis failed. Check backend connection.", {
        style: {
          background: "#151d35",
          color: "#ef4444",
          border: "1px solid #1e2d4a",
        },
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle form data changes for live radar update
  const handleFormChange = (newFormData) => {
    setFormData(newFormData);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg-primary)",
      }}
    >
      <NavBar status={status} />

      <main className="dashboard">
        {/* ─── 3-Column Grid ─── */}
        <div className="dashboard-grid">
          {/* Column 1: Input Form */}
          <div>
            <InputForm
              onSubmit={run}
              onFormChange={handleFormChange}
              loading={loading}
              scenario={formData.usage_scenario}
            />
          </div>

          {/* Column 2: Prediction */}
          <div>
            <PredictionPanel
              result={prediction}
              shap={shap}
              loading={loading}
              scenario={formData.usage_scenario}
              predictionHistory={predictionHistory}
              alerts={alerts}
            />
          </div>

          {/* Column 3: System Health */}
          <div>
            <StatusPanel status={status} />
          </div>
        </div>

        {/* ─── SHAP Explainability (full width) ─── */}
        <div className="dashboard-full">
          <ExplainPanel shap={shap} explainData={explainData} />
        </div>

        {/* ─── IMT-2030 Radar (full width) ─── */}
        <div className="dashboard-full">
          <RadarPanel
            formData={formData}
            scenario={formData.usage_scenario}
          />
        </div>

        {/* ─── Prediction Latency Histogram (full width) ─── */}
        <div className="dashboard-full">
          <LatencyHistogram scenario={formData.usage_scenario} />
        </div>

        {/* ─── Drift History Chart (full width) ─── */}
        <div className="dashboard-full">
          <DriftHistoryChart driftHistory={driftHistory} />
        </div>
      </main>

      <Toaster position="top-right" />
    </div>
  );
}
