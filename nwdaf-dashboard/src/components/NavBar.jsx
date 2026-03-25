// NETRA 2.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import React from "react";

export default function NavBar({ status }) {
  const [time, setTime] = React.useState(new Date());

  React.useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const istTime = time.toLocaleTimeString("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  const modelVersion = status?.model_version || "—";
  const isRollback = status?.rollback_active === true;

  return (
    <nav className="navbar">
      {/* Left: Brand */}
      <div className="navbar__brand">
        <span className="navbar__icon">🛡️</span>
        <div>
          <div className="navbar__title">NETRA 2.0</div>
          <div className="navbar__subtitle">
            Trustworthy 6G NWDAF Platform
          </div>
        </div>
      </div>

      {/* Center: Badges */}
      <div className="navbar__center">
        <span className="badge-imt">
          <span>◆</span> IMT-2030 Compliant
        </span>
        <span
          className={`badge-model ${isRollback ? "badge-model--rollback" : ""
            }`}
        >
          {isRollback ? `⚠ ROLLBACK` : modelVersion}
        </span>
      </div>

      {/* Right: Clock + Live */}
      <div className="navbar__right">
        <span className="navbar__clock">{istTime} IST</span>
        <div className="navbar__live">
          <span className="navbar__live-dot" />
          Live
        </div>
      </div>
    </nav>
  );
}
