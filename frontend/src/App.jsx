

import { useEffect, useMemo, useState } from "react";
import { Bar, Line, Scatter } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  LineController,
  BarController,
  ScatterController,
  Tooltip,
  Legend,
} from "chart.js";
import "./App.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  LineController,
  BarController,
  ScatterController,
  Tooltip,
  Legend
);



const API = "https://toolsense-backend.vercel.app/api";
const REPORTS = "/reports";

const SENSOR_GROUPS = [
  { key: "smcAC", title: "Spindle motor AC current", description: "Electrical current signal" },
  { key: "smcDC", title: "Spindle motor DC current", description: "Electrical current signal" },
  { key: "vib_table", title: "Table vibration", description: "Machine-table vibration signal" },
  { key: "vib_spindle", title: "Spindle vibration", description: "Spindle vibration signal" },
  { key: "AE_table", title: "Table acoustic emission", description: "Acoustic emission signal" },
  { key: "AE_spindle", title: "Spindle acoustic emission", description: "Acoustic emission signal" },
];

const STATISTICS = [
  { key: "mean", label: "Mean" },
  { key: "std", label: "Standard deviation" },
  { key: "rms", label: "RMS" },
  { key: "minimum", label: "Minimum" },
  { key: "maximum", label: "Maximum" },
  { key: "peak_to_peak", label: "Peak-to-peak" },
];

const SHAPE_STATISTICS = [
  { key: "skewness", label: "Skewness" },
  { key: "kurtosis", label: "Fisher excess kurtosis" },
];

const FEATURES = [
  ...SENSOR_GROUPS.flatMap((group) => [
    ...STATISTICS.map((stat) => `${group.key}_${stat.key}`),
    ...SHAPE_STATISTICS.map((stat) => `${group.key}_${stat.key}`),
  ]),
  "DOC",
  "feed",
];

const NAVIGATION = [
  { label: "Overview", icon: "◉" },
  { label: "Sensor Analytics", icon: "⌁" },
  { label: "Model Laboratory", icon: "⌘" },
  { label: "Wear Prediction", icon: "◎" },
];

const COLORS = [
  "#45cde0",
  "#8b83ef",
  "#47c99a",
  "#f0b65b",
  "#5c9ded",
  "#d28be8",
  "#ed806d",
];

const SENSOR_LABELS = {
  smcAC: "Spindle motor AC current",
  smcDC: "Spindle motor DC current",
  vib_table: "Table vibration",
  vib_spindle: "Spindle vibration",
  AE_table: "Table acoustic emission",
  AE_spindle: "Spindle acoustic emission",
};

function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem("toolsense-theme") || "dark"
  );
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [apiError, setApiError] = useState("");
  const [activePage, setActivePage] = useState("Overview");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("toolsense-theme", theme);
  }, [theme]);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      try {
        const [healthResponse, metricsResponse] = await Promise.all([
          fetch(`${API}/health`),
          fetch(`${API}/metrics`),
        ]);

        if (!healthResponse.ok || !metricsResponse.ok) {
          throw new Error("The Flask API returned an unsuccessful response.");
        }

        const [healthData, metricsData] = await Promise.all([
          healthResponse.json(),
          metricsResponse.json(),
        ]);

        if (!cancelled) {
          setHealth(healthData);
          setMetrics(metricsData);
          setApiError("");
        }
      } catch (error) {
        if (!cancelled) {
          setApiError(
            `${error.message} Ensure the Flask backend is running on port 5000.`
          );
        }
      }
    }

    loadDashboard();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">T</div>
          <div>
            <div className="brand-name">ToolSense</div>
            <div className="brand-caption">MACHINING INTELLIGENCE</div>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>
        <nav className="navigation">
          {NAVIGATION.map((item) => (
            <button
              key={item.label}
              className={`nav-item ${activePage === item.label ? "active" : ""}`}
              onClick={() => setActivePage(item.label)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
              {activePage === item.label && <span className="nav-indicator" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="status-card">
            <span
              className={`status-dot ${
                health?.status === "ok" ? "" : "offline"
              }`}
            />
            <div>
              <strong>
                {health?.status === "ok" ? "Backend online" : "Backend status"}
              </strong>
              <p>
                {health?.model_loaded
                  ? `Model loaded · ${health.expected_features} features`
                  : "Waiting for connection"}
              </p>
            </div>
          </div>
          <div className="sidebar-footer">TOOLSENSE · RESEARCH BUILD</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>{activePage}</strong>
          </div>
          <div className="topbar-actions">
            <span className="environment-label">
              <span className="status-dot" />
              LOCAL ENVIRONMENT
            </span>
            <button
              className="theme-toggle"
              onClick={() =>
                setTheme((current) =>
                  current === "dark" ? "light" : "dark"
                )
              }
            >
              <span>{theme === "dark" ? "☀" : "☾"}</span>
              {theme === "dark" ? "Light mode" : "Dark mode"}
            </button>
          </div>
        </header>

        <div className="page-content">
          {apiError && (
            <div className="alert">
              <strong>Backend connection issue</strong>
              <span>{apiError}</span>
            </div>
          )}

          {activePage === "Overview" && (
            <OverviewPage
              metrics={metrics}
              health={health}
              onOpenLaboratory={() => setActivePage("Model Laboratory")}
            />
          )}

          {activePage === "Sensor Analytics" && (
            <SensorAnalytics theme={theme} />
          )}

          {activePage === "Model Laboratory" && (
            <ModelLaboratory theme={theme} metrics={metrics} />
          )}

          {activePage === "Wear Prediction" && (
            <WearPrediction health={health} />
          )}
        </div>
      </main>
    </div>
  );
}

function OverviewPage({ metrics, health, onOpenLaboratory }) {
  const overall = metrics?.overall;
  const highWearMae = overall?.high_wear_MAE_mm;

  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-line" /> CNC INTELLIGENCE PLATFORM
          </div>
          <h1>Machining overview</h1>
          <p>
            Research-driven cutting-tool wear estimation and model evaluation.
          </p>
        </div>
        <div className="model-status">
          <span className="status-dot" />
          <div>
            <strong>{metrics?.model_name || "Extra Trees"}</strong>
            <span>{metrics?.validation || "Loading validation details"}</span>
          </div>
        </div>
      </section>

      <section className="metric-grid">
        <MetricCard
          label="MEAN ABSOLUTE ERROR"
          value={formatMetric(overall?.MAE_mm)}
          unit="mm"
          note="Out-of-fold estimate"
          accent="cyan"
          icon="↘"
        />
        <MetricCard
          label="ROOT MEAN SQUARED ERROR"
          value={formatMetric(overall?.RMSE_mm)}
          unit="mm"
          note="Penalizes larger errors"
          accent="violet"
          icon="⌁"
        />
        <MetricCard
          label="R² SCORE"
          value={formatMetric(overall?.R2)}
          note="Out-of-fold predictions"
          accent="amber"
          icon="Ⅱ"
        />
        <MetricCard
          label="MODEL INPUTS"
          value={health?.expected_features ?? metrics?.feature_count ?? "—"}
          unit="features"
          note="Sensor shape + statistics + conditions"
          accent="green"
          icon="⌘"
        />
      </section>

      <section className="content-grid">
        <article className="panel performance-panel">
          <div className="panel-heading">
            <div>
              <div className="panel-kicker">MODEL EVALUATION</div>
              <h2>Current candidate</h2>
              <p>Five-fold grouped validation by machining case</p>
            </div>
            <span className="panel-badge">RESEARCH</span>
          </div>

          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>MODEL</th>
                  <th>FEATURES</th>
                  <th>MAE ↓</th>
                  <th>RMSE ↓</th>
                  <th>R² ↑</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <span className="model-name">Extra Trees Regressor</span>
                    <span className="model-subtitle">Current candidate</span>
                  </td>
                  <td>{metrics?.feature_count ?? "—"}</td>
                  <td>{formatMetric(overall?.MAE_mm)}</td>
                  <td>{formatMetric(overall?.RMSE_mm)}</td>
                  <td>{formatMetric(overall?.R2)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="panel-footnote">
            These are out-of-fold cross-validation estimates, not results from
            an independent external test set.
          </div>
        </article>

        <article className="panel insight-panel">
          <div className="panel-kicker">HIGH-WEAR PERFORMANCE</div>
          <h2>Wear above 0.50 mm</h2>
          <div className="insight-score">
            <span>{formatMetric(highWearMae)}</span>
            <span>mm MAE</span>
          </div>
          <div className="insight-track">
            <div
              className="insight-fill"
              style={{
                width: `${Math.min(100, (highWearMae ?? 0) * 100)}%`,
              }}
            />
          </div>
          <p>
            The high-wear subset has a larger prediction error than the
            overall dataset. This is an important limitation to investigate.
          </p>
          <div className="insight-tag">
            <span>!</span> Research prototype · Not safety-certified
          </div>
        </article>
      </section>

      <section className="bottom-grid">
        <article className="panel status-panel">
          <div className="panel-heading">
            <div>
              <div className="panel-kicker">SYSTEM STATUS</div>
              <h2>Backend connectivity</h2>
            </div>
            <span
              className={`status-pill ${
                health?.status === "ok" ? "online" : "offline"
              }`}
            >
              {health?.status === "ok" ? "ONLINE" : "CHECK"}
            </span>
          </div>
          <div className="system-row">
            <span>Flask REST API</span>
            <strong>
              {health?.status === "ok" ? "Connected" : "Not verified"}
            </strong>
          </div>
          <div className="system-row">
            <span>Candidate artifact</span>
            <strong>{health?.model_loaded ? "Loaded" : "Not verified"}</strong>
          </div>
          <div className="system-row">
            <span>Prediction endpoint</span>
            <strong>Available for testing</strong>
          </div>
        </article>

        <article className="panel next-panel">
          <div className="panel-kicker">EXPLORE THE EVIDENCE</div>
          <h2>Inspect model behaviour</h2>
          <p>
            Explore out-of-fold predictions, residuals, wear-range errors, and
            feature importance calculated from the current model.
          </p>
          <button className="primary-button" onClick={onOpenLaboratory}>
            Open model laboratory <span>→</span>
          </button>
        </article>
      </section>
      <Footer label="CNC WEAR ANALYTICS" />
    </>
  );
}

function SensorAnalytics({ theme }) {
  const palette = getChartPalette(theme);
  const [importanceRows, setImportanceRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadImportance() {
      try {
        const response = await fetch(
          `${REPORTS}/candidate_feature_importance.csv`
        );
        if (!response.ok) {
          throw new Error(
            "Importance CSV not found. Copy the generated report into frontend/public/reports."
          );
        }

        const csv = await response.text();
        const rows = parseCsvRows(csv).map((row) => ({
          ...row,
          rank: Number(row.rank),
          importance: Number(row.importance),
          importance_percent: Number(row.importance_percent),
        })).filter((row) =>
          row.feature &&
          Number.isFinite(row.importance) &&
          Number.isFinite(row.importance_percent)
        );

        if (!rows.length) {
          throw new Error("The importance report contains no valid feature rows.");
        }

        if (!cancelled) {
          setImportanceRows(rows.sort((a, b) => a.rank - b.rank));
          setError("");
        }
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadImportance();
    return () => {
      cancelled = true;
    };
  }, []);

  const sensorGroups = useMemo(() => {
    const totals = Object.fromEntries(
      SENSOR_GROUPS.map((sensor) => [sensor.key, 0])
    );
    let conditions = 0;

    for (const row of importanceRows) {
      const sensor = SENSOR_GROUPS.find((group) =>
        row.feature.startsWith(`${group.key}_`)
      );
      if (sensor) totals[sensor.key] += row.importance;
      else if (["DOC", "feed"].includes(row.feature)) {
        conditions += row.importance;
      }
    }

    return [
      ...SENSOR_GROUPS.map((sensor) => ({
        key: sensor.key,
        label: sensor.title,
        importance: totals[sensor.key],
        percent: totals[sensor.key] * 100,
      })),
      {
        key: "conditions",
        label: "Machining conditions",
        importance: conditions,
        percent: conditions * 100,
      },
    ].sort((a, b) => b.importance - a.importance);
  }, [importanceRows]);

  const topFeatures = importanceRows.slice(0, 15);
  const topFeature = topFeatures[0];
  const topSensor = sensorGroups[0];
  const topFeatureData = {
    labels: topFeatures.map((row) => row.feature),
    datasets: [{
      label: "Feature importance (%)",
      data: topFeatures.map((row) => row.importance_percent),
      backgroundColor: topFeatures.map((_, index) => COLORS[index % COLORS.length]),
      borderRadius: 5,
      borderSkipped: false,
      barThickness: 17,
    }],
  };

  const sensorData = {
    labels: sensorGroups.map((row) => row.label),
    datasets: [{
      label: "Summed feature importance (%)",
      data: sensorGroups.map((row) => row.percent),
      backgroundColor: sensorGroups.map((_, index) => COLORS[index % COLORS.length]),
      borderRadius: 5,
      borderSkipped: false,
      barThickness: 20,
    }],
  };

  const horizontalOptions = (title, decimals = 2) => ({
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: palette.tooltipBg,
        titleColor: palette.text,
        bodyColor: palette.text,
        borderColor: palette.border,
        borderWidth: 1,
        callbacks: {
          label: (context) =>
            ` Importance: ${Number(context.raw).toFixed(decimals)}%`,
        },
      },
    },
    scales: {
      x: {
        beginAtZero: true,
        grid: { color: palette.grid },
        ticks: { color: palette.muted },
        title: { display: true, text: title, color: palette.muted },
      },
      y: {
        grid: { display: false },
        ticks: { color: palette.text, font: { size: 10 } },
      },
    },
  });

  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-line" /> SENSOR INTELLIGENCE
          </div>
          <h1>Sensor analytics</h1>
          <p>
            Feature contributions estimated from the saved 50-feature Extra
            Trees candidate model.
          </p>
        </div>
        <div className="model-status">
          <span className={`status-dot ${error ? "offline" : ""}`} />
          <div>
            <strong>Extra Trees feature importance</strong>
            <span>50 candidate inputs · impurity-based</span>
          </div>
        </div>
      </section>

      {loading && (
        <div className="panel placeholder-panel">Loading candidate feature importance…</div>
      )}
      {error && (
        <div className="alert">
          <strong>Feature importance unavailable</strong>
          <span>{error}</span>
        </div>
      )}

      {!loading && !error && (
        <>
          <section className="metric-grid sensor-summary-grid">
            <MetricCard
              label="FEATURES ANALYSED"
              value={importanceRows.length}
              note="Saved candidate model"
              accent="cyan"
              icon="⌁"
            />
            <MetricCard
              label="TOP INDIVIDUAL FEATURE"
              value={topFeature?.feature ?? "—"}
              note={`${formatMetric(topFeature?.importance_percent)}% importance`}
              accent="violet"
              icon="↗"
            />
            <MetricCard
              label="TOP SENSOR GROUP"
              value={topSensor?.key ?? "—"}
              note={`${formatMetric(topSensor?.percent)}% summed importance`}
              accent="green"
              icon="◉"
            />
            <MetricCard
              label="CONDITIONS CONTRIBUTION"
              value={`${formatMetric(sensorGroups.find((row) => row.key === "conditions")?.percent)}%`}
              note="DOC and feed combined"
              accent="amber"
              icon="⌘"
            />
          </section>

          <section className="lab-chart-grid">
            <article className="panel sensor-chart-panel">
              <div className="panel-heading">
                <div>
                  <div className="panel-kicker">FEATURE RANKING</div>
                  <h2>Top 15 individual features</h2>
                  <p>Higher values indicate greater impurity-based model importance.</p>
                </div>
              </div>
              <div className="sensor-chart">
                <Bar
                  data={topFeatureData}
                  options={horizontalOptions("Importance (%)")}
                />
              </div>
            </article>

            <article className="panel sensor-chart-panel">
              <div className="panel-heading">
                <div>
                  <div className="panel-kicker">CHANNEL AGGREGATION</div>
                  <h2>Importance by sensor group</h2>
                  <p>Importance summed across each sensor's eight features.</p>
                </div>
              </div>
              <div className="sensor-chart">
                <Bar
                  data={sensorData}
                  options={horizontalOptions("Summed importance (%)")}
                />
              </div>
            </article>
          </section>

          <section className="panel lab-table-panel">
            <div className="panel-heading">
              <div>
                <div className="panel-kicker">MODEL INTERPRETABILITY</div>
                <h2>All feature importance scores</h2>
                <p>Ranked values from the saved candidate model.</p>
              </div>
            </div>
            <div className="comparison-table-wrap">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>RANK</th>
                    <th>FEATURE</th>
                    <th>IMPORTANCE</th>
                    <th>SHARE (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {importanceRows.map((row) => (
                    <tr key={row.feature}>
                      <td>{row.rank}</td>
                      <td>{row.feature}</td>
                      <td>{row.importance.toFixed(6)}</td>
                      <td>{row.importance_percent.toFixed(3)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="panel-footnote">
              Importance scores sum to approximately 1 (100%). These are
              impurity-based Extra Trees scores, not causal effects or
              independent evidence of sensor usefulness.
            </div>
          </section>

          <section className="panel sensor-findings">
            <div className="panel-kicker">RESEARCH INTERPRETATION</div>
            <h2>What the results suggest</h2>
            <Finding
              number="01"
              title="Spindle signals rank highly"
              text="The candidate model assigns substantial combined importance to spindle vibration and spindle acoustic-emission features. This is a hypothesis for further investigation, not proof of causation."
            />
            <Finding
              number="02"
              title="Signal shape carries information"
              text="Skewness and kurtosis describe signal distribution shape beyond mean, RMS, and range. Their ranking suggests that these features may help the model distinguish sensor patterns."
            />
            <Finding
              number="03"
              title="Validate beyond built-in importance"
              text="Impurity-based importance can favour features with particular distributions and can divide importance among correlated inputs. Permutation importance on held-out cases is a useful next robustness check."
            />
          </section>
        </>
      )}
      <Footer label="SENSOR ANALYTICS" />
    </>
  );
}

function ModelLaboratory({ theme, metrics }) {
  const palette = getChartPalette(theme);
  const [predictions, setPredictions] = useState([]);
  const [dataError, setDataError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadPredictions() {
      try {
        const response = await fetch(`${REPORTS}/candidate_model_predictions.csv`);
        if (!response.ok) throw new Error("Candidate prediction report unavailable.");
        const parsed = parsePredictionCsv(await response.text());
        if (!parsed.length) throw new Error("The prediction report has no valid rows.");

        if (!cancelled) {
          setPredictions(parsed);
          setDataError("");
        }
      } catch (error) {
        if (!cancelled) setDataError(error.message);
      }
    }

    loadPredictions();
    return () => {
      cancelled = true;
    };
  }, []);

  const evaluated = useMemo(
    () => predictions
      .filter((row) => Number.isFinite(row.VB) && Number.isFinite(row.predicted_VB))
      .map((row) => ({
        ...row,
        residual: row.predicted_VB - row.VB,
        absoluteError: Math.abs(row.predicted_VB - row.VB),
      })),
    [predictions]
  );

  const overall = metrics?.overall;

  const actualPredictedData = {
    datasets: [
      {
        label: "Out-of-fold predictions",
        data: evaluated.map((row) => ({ x: row.VB, y: row.predicted_VB })),
        backgroundColor: "#45cde0b8",
        pointRadius: 4,
        pointHoverRadius: 6,
      },
      {
        type: "line",
        label: "Ideal prediction",
        data: [{ x: 0, y: 0 }, { x: 1.6, y: 1.6 }],
        borderColor: palette.reference,
        borderDash: [6, 5],
        borderWidth: 2,
        pointRadius: 0,
        showLine: true,
      },
    ],
  };

  const scatterOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: palette.text, usePointStyle: true } },
      tooltip: {
        backgroundColor: palette.tooltipBg,
        titleColor: palette.text,
        bodyColor: palette.text,
        borderColor: palette.border,
        borderWidth: 1,
        callbacks: {
          label: (context) =>
            `${context.dataset.label}: ${context.parsed.y.toFixed(3)} mm`,
          afterLabel: (context) =>
            context.datasetIndex === 0
              ? `Actual: ${context.parsed.x.toFixed(3)} mm`
              : "",
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        min: 0,
        max: 1.6,
        title: { display: true, text: "Actual wear VB (mm)", color: palette.muted },
        ticks: { color: palette.muted },
        grid: { color: palette.grid },
      },
      y: {
        min: 0,
        max: 1.6,
        title: { display: true, text: "Predicted wear (mm)", color: palette.muted },
        ticks: { color: palette.muted },
        grid: { color: palette.grid },
      },
    },
  };

  const residualData = {
    datasets: [{
      label: "Residual",
      data: evaluated.map((row) => ({ x: row.VB, y: row.residual })),
      backgroundColor: evaluated.map((row) =>
        row.residual >= 0 ? "#e6a84fce" : "#45cde0c9"
      ),
      pointRadius: 4,
    }],
  };

  const residualOptions = {
    ...scatterOptions,
    plugins: {
      ...scatterOptions.plugins,
      tooltip: {
        ...scatterOptions.plugins.tooltip,
        callbacks: {
          label: (context) => `Residual: ${context.parsed.y.toFixed(3)} mm`,
          afterLabel: (context) => `Actual wear: ${context.parsed.x.toFixed(3)} mm`,
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Actual wear (mm)", color: palette.muted },
        ticks: { color: palette.muted },
        grid: { color: palette.grid },
      },
      y: {
        title: { display: true, text: "Residual: predicted − actual (mm)", color: palette.muted },
        ticks: { color: palette.muted },
        grid: { color: palette.grid },
      },
    },
  };

  const residualBins = useMemo(() => {
    const bins = [
      { label: "≤ −0.30", min: -Infinity, max: -0.3 },
      { label: "−0.30 to −0.15", min: -0.3, max: -0.15 },
      { label: "−0.15 to 0", min: -0.15, max: 0 },
      { label: "0 to 0.15", min: 0, max: 0.15 },
      { label: "0.15 to 0.30", min: 0.15, max: 0.3 },
      { label: "≥ 0.30", min: 0.3, max: Infinity },
    ];
    return bins.map((bin) => ({
      label: bin.label,
      count: evaluated.filter(
        (row) => row.residual >= bin.min && row.residual < bin.max
      ).length,
    }));
  }, [evaluated]);

  const residualHistogram = {
    labels: residualBins.map((bin) => bin.label),
    datasets: [{
      label: "Records",
      data: residualBins.map((bin) => bin.count),
      backgroundColor: "#8b83ef",
      borderRadius: 5,
    }],
  };

  const wearRanges = useMemo(() => {
    const ranges = [
      { label: "≤ 0.15", min: -Infinity, max: 0.15 },
      { label: "0.15–0.30", min: 0.15, max: 0.3 },
      { label: "0.30–0.50", min: 0.3, max: 0.5 },
      { label: "> 0.50", min: 0.5, max: Infinity },
    ];

    return ranges.map((range) => {
      const rows = evaluated.filter(
        (row) => row.VB > range.min && row.VB <= range.max
      );
      return {
        ...range,
        count: rows.length,
        mae: rows.length
          ? rows.reduce((sum, row) => sum + row.absoluteError, 0) / rows.length
          : null,
      };
    });
  }, [evaluated]);

  const wearRangeData = {
    labels: wearRanges.map((range) => `${range.label} (n=${range.count})`),
    datasets: [{
      label: "MAE (mm)",
      data: wearRanges.map((range) => range.mae),
      backgroundColor: ["#47c99a", "#45cde0", "#f0b65b", "#ed806d"],
      borderRadius: 5,
    }],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: palette.tooltipBg,
        titleColor: palette.text,
        bodyColor: palette.text,
      },
    },
    scales: {
      x: { ticks: { color: palette.muted }, grid: { display: false } },
      y: {
        beginAtZero: true,
        ticks: { color: palette.muted },
        grid: { color: palette.grid },
        title: { display: true, text: "Records / MAE (mm)", color: palette.muted },
      },
    },
  };

  const residualHistogramOptions = {
    ...chartOptions,
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        title: { display: true, text: "Number of records", color: palette.muted },
      },
    },
  };

  const wearRangeOptions = {
    ...chartOptions,
    plugins: {
      ...chartOptions.plugins,
      tooltip: {
        ...chartOptions.plugins.tooltip,
        callbacks: {
          label: (context) =>
            context.raw == null ? "No records" : ` MAE: ${context.raw.toFixed(4)} mm`,
        },
      },
    },
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        title: { display: true, text: "Mean absolute error (mm)", color: palette.muted },
      },
    },
  };

  const folds = metrics?.fold_results ?? [];

  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-line" /> MODEL VALIDATION WORKSPACE
          </div>
          <h1>Model laboratory</h1>
          <p>Inspect out-of-fold predictions, residual patterns, and error by wear level.</p>
        </div>
        <div className="model-status">
          <span className="status-dot" />
          <div>
            <strong>Extra Trees</strong>
            <span>{metrics?.validation || "Grouped cross-validation"}</span>
          </div>
        </div>
      </section>

      {dataError && (
        <div className="alert">
          <strong>Evaluation data unavailable</strong>
          <span>{dataError} Copy the candidate prediction CSV into frontend/public/reports.</span>
        </div>
      )}

      <section className="metric-grid lab-metrics-grid">
        <MetricCard label="MEAN ABSOLUTE ERROR" value={formatMetric(overall?.MAE_mm)} unit="mm" note="Out-of-fold estimate" accent="cyan" icon="↘" />
        <MetricCard label="ROOT MEAN SQUARED ERROR" value={formatMetric(overall?.RMSE_mm)} unit="mm" note="Penalizes larger errors" accent="violet" icon="⌁" />
        <MetricCard label="R² SCORE" value={formatMetric(overall?.R2)} note="Out-of-fold predictions" accent="amber" icon="Ⅱ" />
        <MetricCard label="EVALUATED RECORDS" value={evaluated.length || "—"} note="Valid held-out predictions" accent="green" icon="⌘" />
      </section>

      <section className="lab-chart-grid">
        <article className="panel lab-chart-panel">
          <div className="panel-heading">
            <div><div className="panel-kicker">PREDICTION QUALITY</div><h2>Actual vs. predicted wear</h2><p>Points closer to the diagonal indicate closer predictions.</p></div>
            <span className="panel-badge">OOF</span>
          </div>
          <div className="lab-chart scatter-chart">
            {evaluated.length > 0 && <Scatter data={actualPredictedData} options={scatterOptions} />}
          </div>
        </article>
        <article className="panel lab-chart-panel">
          <div className="panel-heading">
            <div><div className="panel-kicker">ERROR STRUCTURE</div><h2>Residual distribution</h2><p>Residual = predicted − actual; positive values indicate overprediction.</p></div>
          </div>
          <div className="lab-chart histogram-chart">
            {evaluated.length > 0 && <Bar data={residualHistogram} options={residualHistogramOptions} />}
          </div>
        </article>
      </section>

      <section className="lab-chart-grid">
        <article className="panel lab-chart-panel">
          <div className="panel-heading">
            <div><div className="panel-kicker">ERROR BY WEAR LEVEL</div><h2>MAE across wear ranges</h2><p>Bars show mean absolute error in millimetres.</p></div>
          </div>
          <div className="lab-chart histogram-chart">
            {evaluated.length > 0 && <Bar data={wearRangeData} options={wearRangeOptions} />}
          </div>
          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <thead><tr><th>WEAR RANGE</th><th>RECORDS</th><th>MAE (mm)</th></tr></thead>
              <tbody>
                {wearRanges.map((range) => (
                  <tr key={range.label}>
                    <td>{range.label}</td>
                    <td>{range.count}</td>
                    <td>{range.mae == null ? "—" : range.mae.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="panel lab-chart-panel">
          <div className="panel-heading">
            <div><div className="panel-kicker">FOLD STABILITY</div><h2>Grouped validation folds</h2><p>Fold-specific MAE and RMSE from the saved report.</p></div>
          </div>
          <div className="comparison-table-wrap">
            <table className="comparison-table">
              <thead><tr><th>FOLD</th><th>RECORDS</th><th>MAE (mm)</th><th>RMSE (mm)</th></tr></thead>
              <tbody>
                {folds.map((fold) => (
                  <tr key={fold.fold}>
                    <td>{fold.fold}</td>
                    <td>{fold.test_records}</td>
                    <td>{formatMetric(fold.MAE_mm)}</td>
                    <td>{formatMetric(fold.RMSE_mm)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel-footnote">
            Fold 3 has notably higher error than the other folds, indicating case-level variability.
          </div>
        </article>
      </section>

      <section className="panel lab-table-panel">
        <div className="panel-heading">
          <div><div className="panel-kicker">RECORD INSPECTION</div><h2>Largest prediction errors</h2><p>Records with the largest absolute out-of-fold errors.</p></div>
        </div>
        <div className="comparison-table-wrap">
          <table className="comparison-table">
            <thead><tr><th>CASE</th><th>RUN</th><th>ACTUAL VB (mm)</th><th>PREDICTED VB (mm)</th><th>RESIDUAL (mm)</th><th>ABS. ERROR (mm)</th></tr></thead>
            <tbody>
              {[...evaluated]
                .sort((a, b) => b.absoluteError - a.absoluteError)
                .slice(0, 10)
                .map((row) => (
                  <tr key={`${row.case}-${row.run}`}>
                    <td>{row.case}</td><td>{row.run}</td><td>{row.VB.toFixed(3)}</td>
                    <td>{row.predicted_VB.toFixed(3)}</td><td>{row.residual.toFixed(3)}</td>
                    <td>{row.absoluteError.toFixed(3)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <div className="panel-footnote">
          These are held-out cross-validation predictions, not predictions on the same records used to fit each fold model.
        </div>
      </section>

      <section className="panel research-note">
        <div className="panel-kicker">INTERPRETATION</div><h2>What to look for</h2>
        <Finding number="01" title="Scatter around the ideal line" text="A systematic shift above or below the diagonal indicates overprediction or underprediction." />
        <Finding number="02" title="Error by wear level" text="The high-wear group has larger MAE, so severe-wear estimates deserve particular caution." />
        <Finding number="03" title="Cross-validation is not deployment validation" text="Grouped cross-validation helps avoid case leakage, but this small dataset is not evidence of certified industrial performance." />
      </section>
      <Footer label="MODEL LABORATORY" />
    </>
  );
}

function WearPrediction({ health }) {
  const [form, setForm] = useState(() =>
    Object.fromEntries(FEATURES.map((feature) => [feature, ""]))
  );
  const [samples, setSamples] = useState([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [samplesError, setSamplesError] = useState("");
  const [selectedKey, setSelectedKey] = useState("");
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [loadingSample, setLoadingSample] = useState(false);
  const [sampleMessage, setSampleMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    let active = true;
    async function fetchSamplesList() {
      setLoadingSamples(true);
      setSamplesError("");
      try {
        const response = await fetch(`${API}/samples`);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Failed to load dataset samples list.");
        }
        if (active) {
          setSamples(Array.isArray(data.records) ? data.records : []);
        }
      } catch (err) {
        if (active) {
          setSamplesError(err.message || "Failed to load sample options.");
        }
      } finally {
        if (active) {
          setLoadingSamples(false);
        }
      }
    }
    fetchSamplesList();
    return () => {
      active = false;
    };
  }, []);

  function updateField(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
    setResult(null);
    setError("");
    if (selectedRecord) {
      setSelectedRecord(null);
      setSelectedKey("");
      setSampleMessage("Manual feature adjustments made.");
    }
  }

  async function loadSampleByCaseRun(caseNum, runNum) {
    setLoadingSample(true);
    setError("");
    setResult(null);
    setSampleMessage("");

    try {
      const response = await fetch(`${API}/sample?case=${caseNum}&run=${runNum}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Could not load sample for Case ${caseNum} Run ${runNum}.`);
      }
      if (!data.features || data.feature_count !== 50) {
        throw new Error("The API did not return the expected 50 features.");
      }

      setForm(
        Object.fromEntries(
          FEATURES.map((feature) => [feature, String(data.features[feature] ?? "")])
        )
      );
      const hasVB = Number.isFinite(data.measured_VB_mm);
      setSelectedRecord({
        case: data.case,
        run: data.run,
        measured_VB_mm: data.measured_VB_mm,
        has_measured_wear: hasVB,
      });
      setSelectedKey(`${data.case}-${data.run}`);
      setSampleMessage(
        `Loaded real sensor features · Case ${data.case} · Run ${data.run}` +
          (hasVB
            ? ` · Measured VB ${Number(data.measured_VB_mm).toFixed(3)} mm`
            : " · Unlabelled (no measured wear)")
      );
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoadingSample(false);
    }
  }

  async function loadDefaultSample() {
    setLoadingSample(true);
    setError("");
    setResult(null);
    setSampleMessage("");

    try {
      const response = await fetch(`${API}/sample`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Could not load default dataset sample.");
      }
      if (!data.features || data.feature_count !== 50) {
        throw new Error("The API did not return the expected 50 features.");
      }

      setForm(
        Object.fromEntries(
          FEATURES.map((feature) => [feature, String(data.features[feature] ?? "")])
        )
      );
      const hasVB = Number.isFinite(data.measured_VB_mm);
      setSelectedRecord({
        case: data.case,
        run: data.run,
        measured_VB_mm: data.measured_VB_mm,
        has_measured_wear: hasVB,
      });
      setSelectedKey(`${data.case}-${data.run}`);
      setSampleMessage(
        `Loaded real sensor features · Case ${data.case} · Run ${data.run}` +
          (hasVB
            ? ` · Measured VB ${Number(data.measured_VB_mm).toFixed(3)} mm`
            : " · Unlabelled")
      );
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoadingSample(false);
    }
  }

  function handleSelectSample(event) {
    const key = event.target.value;
    if (!key) {
      setSelectedKey("");
      setSelectedRecord(null);
      return;
    }
    const [c, r] = key.split("-").map(Number);
    loadSampleByCaseRun(c, r);
  }

  async function submitPrediction(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    const missing = FEATURES.filter(
      (feature) =>
        form[feature] === "" || !Number.isFinite(Number(form[feature]))
    );

    if (missing.length) {
      setError(
        `Enter valid numeric values for all ${FEATURES.length} features. ` +
          `Missing or invalid: ${missing.slice(0, 5).join(", ")}` +
          (missing.length > 5 ? "…" : "")
      );
      return;
    }

    const payload = Object.fromEntries(
      FEATURES.map((feature) => [feature, Number(form[feature])])
    );

    setSubmitting(true);
    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `Prediction failed (${response.status}).`);
      }

      const predictedValue = findPredictionValue(data);
      if (!Number.isFinite(predictedValue)) {
        throw new Error("The API response did not contain a numeric prediction.");
      }

      setResult({
        value: predictedValue,
        raw: data,
        record: selectedRecord ? { ...selectedRecord } : null,
      });
    } catch (requestError) {
      setError(
        `${requestError.message} Check that Flask is running and /api/predict is available.`
      );
    } finally {
      setSubmitting(false);
    }
  }

  const filledCount = FEATURES.filter(
    (feature) =>
      form[feature] !== "" && Number.isFinite(Number(form[feature]))
  ).length;

  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> INFERENCE WORKSPACE</div>
          <h1>Wear prediction</h1>
          <p>Submit the 50 model inputs to estimate flank wear.</p>
        </div>
        <div className="model-status">
          <span className={`status-dot ${health?.model_loaded ? "" : "offline"}`} />
          <div>
            <strong>{health?.model_loaded ? "Extra Trees loaded" : "Model not verified"}</strong>
            <span>{health?.expected_features ?? 50} required numeric features</span>
          </div>
        </div>
      </section>

      <div className="prediction-warning">
        <strong>Experimental model</strong>
        <span>
          This is a research estimate of flank wear in millimetres, not a live
          sensor connection or a certified tool-change recommendation.
        </span>
      </div>

      <section className="panel sample-selector-panel">
        <div className="panel-kicker">DATASET SAMPLE SELECTOR</div>
        <h2>Load real NASA milling records</h2>
        <p>
          Select from {samples.length || "167"} verified experimental records to populate all 50 features and compare model estimates against ground-truth optical wear measurements.
        </p>

        {samplesError && (
          <div className="sample-list-error">
            Unable to load sample catalog: {samplesError}. Default sample loader remains available.
          </div>
        )}

        <div className="sample-selector-row">
          <div className="sample-selector-field">
            <span>CHOOSE DATASET RECORD (CASE · RUN)</span>
            <select
              id="dataset-sample-select"
              className="sample-select"
              value={selectedKey}
              onChange={handleSelectSample}
              disabled={loadingSample || loadingSamples}
            >
              <option value="">
                {loadingSamples
                  ? "Loading sample catalog…"
                  : "Select a record (Case · Run · Measured VB)…"}
              </option>
              {samples.map((item) => (
                <option
                  key={`${item.case}-${item.run}`}
                  value={`${item.case}-${item.run}`}
                >
                  Case {item.case} · Run {item.run}
                  {item.has_measured_wear
                    ? ` (Measured VB: ${Number(item.measured_VB_mm).toFixed(3)} mm)`
                    : " (Unlabelled / No VB)"}
                </option>
              ))}
            </select>
          </div>

          <button
            className="secondary-button"
            type="button"
            onClick={loadDefaultSample}
            disabled={loadingSample}
          >
            {loadingSample ? "Loading sample…" : "↧ Load default sample (Case 1 · Run 1)"}
          </button>
        </div>

        {selectedRecord && (
          <div className="sample-meta">
            <span className="sample-meta-tag case-run">
              Case {selectedRecord.case} · Run {selectedRecord.run}
            </span>
            {selectedRecord.has_measured_wear ? (
              <span className="sample-meta-tag vb-label">
                Actual measured VB: {Number(selectedRecord.measured_VB_mm).toFixed(3)} mm
              </span>
            ) : (
              <span className="sample-meta-tag vb-missing">
                Actual wear label unavailable (unlabelled run)
              </span>
            )}
          </div>
        )}
      </section>

      {sampleMessage && <div className="success-message">{sampleMessage}</div>}
      {error && (
        <div className="alert prediction-alert">
          <strong>Unable to complete request</strong>
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={submitPrediction}>
        {SENSOR_GROUPS.map((group, index) => (
          <section className="panel sensor-input-panel" key={group.key}>
            <div className="sensor-input-heading">
              <div className="sensor-number">{String(index + 1).padStart(2, "0")}</div>
              <div><h2>{group.title}</h2><p>{group.description} · 8 features</p></div>
            </div>
            <div className="feature-input-grid">
              {STATISTICS.map((stat) => {
                const name = `${group.key}_${stat.key}`;
                return (
                  <label className="feature-field" key={name}>
                    <span>{stat.label}</span>
                    <input type="number" step="any" value={form[name]} onChange={(event) => updateField(name, event.target.value)} placeholder="Enter value" required />
                    <small>{name}</small>
                  </label>
                );
              })}
            </div>
            <div className="panel-kicker" style={{ marginTop: 18 }}>SIGNAL SHAPE STATISTICS</div>
            <div className="feature-input-grid">
              {SHAPE_STATISTICS.map((stat) => {
                const name = `${group.key}_${stat.key}`;
                return (
                  <label className="feature-field" key={name}>
                    <span>{stat.label}</span>
                    <input type="number" step="any" value={form[name]} onChange={(event) => updateField(name, event.target.value)} placeholder="Enter value" required />
                    <small>{name} · {stat.key === "skewness" ? "Sample skewness, bias=False" : "Fisher=true, bias=False"}</small>
                  </label>
                );
              })}
            </div>
          </section>
        ))}

        <section className="panel sensor-input-panel">
          <div className="sensor-input-heading">
            <div className="sensor-number">07</div>
            <div><h2>Machining conditions</h2><p>Process parameters used by the trained model</p></div>
          </div>
          <div className="feature-input-grid conditions-grid">
            <label className="feature-field">
              <span>Depth of cut (DOC)</span>
              <input type="number" step="any" value={form.DOC} onChange={(event) => updateField("DOC", event.target.value)} placeholder="Enter DOC" required />
              <small>DOC</small>
            </label>
            <label className="feature-field">
              <span>Feed rate</span>
              <input type="number" step="any" value={form.feed} onChange={(event) => updateField("feed", event.target.value)} placeholder="Enter feed" required />
              <small>feed</small>
            </label>
          </div>
        </section>

        <div className="prediction-actions">
          <span>{filledCount} of {FEATURES.length} features filled</span>
          <div>
            <button className="secondary-button" type="button" onClick={() => {
              setForm(Object.fromEntries(FEATURES.map((feature) => [feature, ""])));
              setSelectedRecord(null);
              setSelectedKey("");
              setResult(null);
              setError("");
              setSampleMessage("");
            }}>Clear form</button>
            <button className="primary-button" type="submit" disabled={submitting || !health?.model_loaded}>
              {submitting ? "Predicting…" : "Predict tool wear"} <span>→</span>
            </button>
          </div>
        </div>
      </form>

      {result && (
        <section className="panel prediction-result">
          <div className="result-icon">✓</div>
          <div className="result-copy">
            <div className="panel-kicker">MODEL OUTPUT</div>
            <h2>Estimated flank wear</h2>
            <div className="result-value">{result.value.toFixed(4)} <span>mm</span></div>
            <p>Prediction returned by the saved Extra Trees model.</p>
          </div>

          {result.record && (
            <div className={`wear-comparison ${result.record.has_measured_wear ? "has-error-card" : "two-cards"}`}>
              <div className="wear-comparison-card">
                <div className="panel-kicker">DATASET REFERENCE</div>
                <h3>Actual measured wear (VB)</h3>
                {result.record.has_measured_wear ? (
                  <>
                    <div className="wear-comparison-value">
                      {Number(result.record.measured_VB_mm).toFixed(4)} <span>mm</span>
                    </div>
                    <div className="wear-comparison-note">
                      Ground-truth optical microscopy measurement from NASA Milling dataset (Case {result.record.case}, Run {result.record.run}).
                    </div>
                  </>
                ) : (
                  <>
                    <div className="wear-comparison-value" style={{ color: "var(--muted)" }}>
                      N/A
                    </div>
                    <div className="no-vb-message">
                      Actual wear label unavailable for this record. Prediction can still be explored, but prediction error cannot be calculated.
                    </div>
                  </>
                )}
              </div>

              <div className="wear-comparison-card">
                <div className="panel-kicker">MODEL ESTIMATE</div>
                <h3>Model prediction (VB)</h3>
                <div className="wear-comparison-value">
                  {result.value.toFixed(4)} <span>mm</span>
                </div>
                <div className="wear-comparison-note">
                  ExtraTreesRegressor inference from 50 extracted sensor & condition features.
                </div>
              </div>

              {result.record.has_measured_wear && (
                <div className="wear-comparison-card error-card">
                  <div className="panel-kicker">PREDICTION ACCURACY</div>
                  <h3>Absolute error (|Δ|)</h3>
                  <div className="wear-comparison-value error-highlight">
                    {Math.abs(result.value - result.record.measured_VB_mm).toFixed(4)} <span>mm</span>
                  </div>
                  <div className="wear-comparison-note">
                    Difference between model estimate and actual measured flank wear for this record.
                  </div>
                </div>
              )}
            </div>
          )}

          <details className="result-details">
            <summary>View API response</summary>
            <pre>{JSON.stringify(result.raw, null, 2)}</pre>
          </details>
          <div className="result-disclaimer">Experimental estimate only · Not a safety threshold or certified maintenance recommendation.</div>
        </section>
      )}

      <Footer label="WEAR PREDICTION" />
    </>
  );
}

function findPredictionValue(data) {
  const keys = ["predicted_VB_mm", "predicted_wear_mm", "prediction_mm", "predicted_VB", "prediction"];
  for (const key of keys) {
    if (typeof data?.[key] === "number" && Number.isFinite(data[key])) return data[key];
    if (typeof data?.[key] === "string" && data[key].trim() && Number.isFinite(Number(data[key]))) return Number(data[key]);
  }
  return NaN;
}

function parsePredictionCsv(csv) {
  return parseCsvRows(csv).map((row) => {
    for (const key of Object.keys(row)) {
      if (row[key] !== "" && Number.isFinite(Number(row[key]))) row[key] = Number(row[key]);
    }
    return row;
  });
}

function parseCsvRows(csv) {
  const lines = csv.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((header) => header.trim());
  return lines.slice(1).filter((line) => line.trim()).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, (values[index] ?? "").trim()]));
  });
}

function formatMetric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
}

function getChartPalette(theme) {
  const light = theme === "light";
  return {
    text: light ? "#344256" : "#d7e1ed",
    muted: light ? "#687b91" : "#a9b8cc",
    grid: light ? "rgba(75, 95, 120, 0.14)" : "rgba(130, 150, 175, 0.12)",
    border: light ? "#d5deea" : "#354458",
    tooltipBg: light ? "#ffffff" : "#111923",
    reference: light ? "#65758b" : "#9aacc2",
  };
}

function MetricCard({ label, value, unit, note, accent, icon }) {
  return (
    <article className={`metric-card accent-${accent}`}>
      <div className="metric-card-top"><span>{label}</span><span className="metric-icon">{icon}</span></div>
      <div className="metric-value">{value}{unit && <span>{unit}</span>}</div>
      <div className="metric-note"><span className="metric-note-dot" />{note}</div>
    </article>
  );
}

function Finding({ number, title, text }) {
  return <div className="finding-row"><span className="finding-number">{number}</span><div><strong>{title}</strong><p>{text}</p></div></div>;
}

function Footer({ label }) {
  return <footer className="page-footer"><span>TOOLSENSE <span className="footer-separator">/</span> {label}</span><span>Experimental ML system · Use with engineering oversight</span></footer>;
}

export default App;
