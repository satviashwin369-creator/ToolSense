
import { Component, useEffect, useState } from "react";
import {
  Chart as ChartJS,
  registerables,
} from "chart.js";

import AlgorithmComparison from "./components/AlgorithmComparison";
import DataFeatureAnalysis from "./components/DataFeatureAnalysis";
import InteractivePrediction from "./components/InteractivePrediction";
import "./App.css";

// Register all Chart.js controllers (bar, line, scatter, etc.), scales, and elements
ChartJS.register(...registerables);

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ToolSense Dashboard caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="section-error" style={{ margin: "2rem auto", maxWidth: "680px" }}>
          <h3 style={{ color: "#f43f5e", marginBottom: "0.5rem" }}>Rendering Error Detected</h3>
          <p style={{ color: "var(--text-muted)", marginBottom: "1rem" }}>
            {this.state.error?.message || "An unexpected error occurred while rendering this page."}
          </p>
          <button
            className="btn-filter active"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Retry View
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// In local dev, point directly to the Flask backend; fallback gracefully
const API_BASE = "http://127.0.0.1:5000/api";
const REPORTS_BASE = "/reports";

const NAVIGATION_ITEMS = [
  { id: "algorithms", label: "Algorithm Comparison", section: "Section B", icon: "📊" },
  { id: "data_analysis", label: "Data & Feature Analysis", section: "Section C", icon: "🔬" },
  { id: "prediction", label: "Interactive Prediction", section: "Section D", icon: "⚙️" },
];

export default function App() {
  const [activeSection, setActiveSection] = useState("algorithms");
  const [theme, setTheme] = useState(() => localStorage.getItem("toolsense-theme") || "dark");
  const [backendHealth, setBackendHealth] = useState(null);
  const [checkingBackend, setCheckingBackend] = useState(true);

  // Apply theme to HTML root
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("toolsense-theme", theme);
  }, [theme]);

  // Check backend health
  useEffect(() => {
    async function checkHealth() {
      setCheckingBackend(true);
      try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
          const data = await res.json();
          setBackendHealth(data);
        } else {
          setBackendHealth({ status: "offline" });
        }
      } catch {
        setBackendHealth({ status: "offline" });
      } finally {
        setCheckingBackend(false);
      }
    }

    checkHealth();
  }, []);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  return (
    <div className="app-layout">
      {/* Sidebar Navigation */}
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <div className="brand-logo-wrap">
            <span className="brand-logo-icon">⚙️</span>
            <div>
              <h1 className="brand-title">ToolSense</h1>
              <span className="brand-tagline">CNC Machining Intelligence</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-group-title">RESEARCH DASHBOARD</div>
          {NAVIGATION_ITEMS.map((item) => {
            const isActive = activeSection === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? "active" : ""}`}
                onClick={() => setActiveSection(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <div className="nav-text">
                  <span className="nav-label">{item.label}</span>
                  <span className="nav-section-tag">{item.section}</span>
                </div>
                {isActive && <span className="nav-active-pip"></span>}
              </button>
            );
          })}
        </nav>

        {/* System Status in Sidebar Footer */}
        <div className="sidebar-footer">
          <div className="system-status-card">
            <div className="status-indicator-row">
              <span
                className={`status-dot ${
                  backendHealth?.status === "ok" ? "dot-online" : "dot-warning"
                }`}
              ></span>
              <span className="status-text">
                {checkingBackend
                  ? "Checking Backend..."
                  : backendHealth?.status === "ok"
                  ? "Flask Backend Online"
                  : "Using Direct Data Fallback"}
              </span>
            </div>
            <div className="status-meta">
              Port 5000 &bull; 50 Features &bull; ExtraTrees
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="app-main">
        {/* Top Navbar */}
        <header className="app-topbar">
          <div className="topbar-left">
            <span className="topbar-breadcrumb">ToolSense ML Dashboard</span>
            <span className="topbar-slash">/</span>
            <span className="topbar-current">
              {NAVIGATION_ITEMS.find((n) => n.id === activeSection)?.section}:{" "}
              {NAVIGATION_ITEMS.find((n) => n.id === activeSection)?.label}
            </span>
          </div>

          <div className="topbar-right">
            <div className="dataset-tag">
              <span className="tag-pulse"></span>
              NASA/UC Berkeley Milling Dataset (167 Cuts)
            </div>

            <button
              className="theme-toggle-btn"
              onClick={toggleTheme}
              title="Toggle Light/Dark Theme"
            >
              {theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode"}
            </button>
          </div>
        </header>

        {/* Dynamic Section Content */}
        <main className="app-content">
          <ErrorBoundary>
            {activeSection === "algorithms" && (
              <AlgorithmComparison apiBase={API_BASE} reportsBase={REPORTS_BASE} />
            )}

            {activeSection === "data_analysis" && (
              <DataFeatureAnalysis apiBase={API_BASE} reportsBase={REPORTS_BASE} />
            )}

            {activeSection === "prediction" && (
              <InteractivePrediction apiBase={API_BASE} reportsBase={REPORTS_BASE} />
            )}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
