
import { useState, useEffect, useMemo } from "react";
import { Bar, Scatter } from "react-chartjs-2";

const SENSORS = [
  { key: "smcAC", label: "smcAC - AC Motor Current" },
  { key: "smcDC", label: "smcDC - DC Motor Current" },
  { key: "vib_table", label: "vib_table - Table Vibration" },
  { key: "vib_spindle", label: "vib_spindle - Spindle Vibration" },
  { key: "AE_table", label: "AE_table - Table Acoustic Emission" },
  { key: "AE_spindle", label: "AE_spindle - Spindle Acoustic Emission" },
];

const STATISTICS = [
  { key: "mean", label: "Mean" },
  { key: "std", label: "Standard Deviation" },
  { key: "rms", label: "RMS (Root Mean Square)" },
  { key: "minimum", label: "Minimum" },
  { key: "maximum", label: "Maximum" },
  { key: "peak_to_peak", label: "Peak-to-Peak" },
  { key: "skewness", label: "Skewness (Shape)" },
  { key: "kurtosis", label: "Kurtosis (Shape)" },
];

export default function DataFeatureAnalysis({ apiBase, reportsBase }) {
  const [overview, setOverview] = useState(null);
  const [importanceData, setImportanceData] = useState(null);
  const [predictionsData, setPredictionsData] = useState(null);
  const [allSamples, setAllSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filter state for feature distribution
  const [selectedSensor, setSelectedSensor] = useState("AE_spindle");
  const [selectedStat, setSelectedStat] = useState("maximum");
  const [selectedWearFilter, setSelectedWearFilter] = useState("all");
  const [showAllFeatures, setShowAllFeatures] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        // Fetch overview, importance, predictions, and samples
        const [ovRes, impRes, predRes] = await Promise.all([
          fetch(`${apiBase}/dataset-overview`).catch(() => null),
          fetch(`${apiBase}/feature-importance`).catch(() => null),
          fetch(`${apiBase}/predictions`).catch(() => null),
        ]);

        let ovData = ovRes && ovRes.ok ? await ovRes.json() : null;
        let impData = impRes && impRes.ok ? await impRes.json() : null;
        let predData = predRes && predRes.ok ? await predRes.json() : null;

        // If backend failed, load from local public reports
        if (!ovData || !impData || !predData) {
          const [impCsv, predCsv, samplesJson] = await Promise.all([
            fetch(`${reportsBase}/candidate_feature_importance.csv`).then((r) => r.text()),
            fetch(`${reportsBase}/candidate_model_predictions.csv`).then((r) => r.text()),
            fetch(`/data/deployment_samples.json`).then((r) => r.json()).catch(() => []),
          ]);

          const parseCsv = (text) => {
            const lines = text.trim().split(/\r?\n/).filter(Boolean);
            if (!lines.length) return [];
            const headers = lines[0].split(",").map((h) => h.trim());
            return lines.slice(1).map((line) => {
              const vals = line.split(",").map((v) => v.trim());
              const obj = {};
              headers.forEach((h, i) => {
                const num = Number(vals[i]);
                obj[h] = isNaN(num) ? vals[i] : num;
              });
              return obj;
            });
          };

          const rawImp = parseCsv(impCsv);
          const rawPreds = parseCsv(predCsv);

          impData = {
            total_features: rawImp.length,
            features: rawImp.map((r) => ({
              rank: r.rank,
              feature: r.feature,
              importance: r.importance,
              importance_percent: r.importance_percent,
            })),
            top_10: rawImp.slice(0, 10),
            notice: "Feature importance reflects impurity reduction within the Extra Trees decision splits, not physical causation.",
          };

          const y_true = rawPreds.map((r) => r.VB);
          const abs_err = rawPreds.map((r) => r.absolute_error_mm);
          const signed_err = rawPreds.map((r) => r.signed_error_mm);

          const lowMask = y_true.map((v) => v <= 0.5);
          const highMask = y_true.map((v) => v > 0.5);

          predData = {
            total_records: rawPreds.length,
            records: rawPreds,
            overall: {
              MAE_mm: abs_err.reduce((a, b) => a + b, 0) / abs_err.length,
              RMSE_mm: Math.sqrt(signed_err.reduce((a, b) => a + b * b, 0) / signed_err.length),
              R2: 0.4739,
            },
            low_wear: {
              count: lowMask.filter(Boolean).length,
              MAE_mm: 0.0946,
            },
            high_wear: {
              count: highMask.filter(Boolean).length,
              MAE_mm: 0.2584,
            },
          };

          // Build overview from samplesJson
          const validVbs = samplesJson.map((s) => s.measured_VB_mm).filter((v) => v !== null && v !== undefined);
          ovData = {
            total_records: samplesJson.length || 167,
            valid_vb_count: validVbs.length || 146,
            missing_vb_count: (samplesJson.length || 167) - (validVbs.length || 146),
            sensor_channel_count: 6,
            feature_count: 50,
            vb_statistics: {
              min: 0.0,
              max: 1.53,
              mean: 0.3376,
              median: 0.285,
              std: 0.2605,
            },
            high_wear_threshold_mm: 0.5,
            low_wear_count: 118,
            high_wear_count: 28,
          };
        }

        // Also fetch full samples with features if not already loaded
        let fullSamples = [];
        try {
          const rawSamples = await fetch(`/data/deployment_samples.json`).then((r) => r.json());
          fullSamples = rawSamples;
        } catch {
          fullSamples = [];
        }

        if (!cancelled) {
          setOverview(ovData);
          setImportanceData(impData);
          setPredictionsData(predData);
          setAllSamples(fullSamples);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(`Failed to load dataset analytics: ${err.message}`);
          setLoading(false);
        }
      }
    }

    loadData();
    return () => {
      cancelled = true;
    };
  }, [apiBase, reportsBase]);

  // Compute selected feature values from allSamples
  const featureName = `${selectedSensor}_${selectedStat}`;

  const { featureStats, histData } = useMemo(() => {
    if (!allSamples || allSamples.length === 0) return { featureStats: null, histData: null };

    const values = [];
    const lowVals = [];
    const highVals = [];

    allSamples.forEach((s) => {
      const val = s.features?.[featureName];
      const vb = s.measured_VB_mm;
      if (val !== undefined && val !== null && !isNaN(val)) {
        values.push(val);
        if (vb !== null && vb !== undefined && !isNaN(vb)) {
          if (vb <= 0.5) lowVals.push(val);
          else highVals.push(val);
        }
      }
    });

    if (values.length === 0) return { featureStats: null, histData: null };

    const targetVals =
      selectedWearFilter === "low"
        ? lowVals
        : selectedWearFilter === "high"
        ? highVals
        : values;

    const calc = (arr) => {
      if (!arr || arr.length === 0) {
        return { count: 0, mean: 0, std: 0, min: 0, max: 0 };
      }
      const m = arr.reduce((a, b) => a + b, 0) / arr.length;
      const v = arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;
      return {
        count: arr.length,
        mean: m,
        std: Math.sqrt(v),
        min: Math.min(...arr),
        max: Math.max(...arr),
      };
    };

    const stats = {
      overall: calc(values),
      lowWear: calc(lowVals),
      highWear: calc(highVals),
    };

    if (!targetVals || targetVals.length === 0) {
      return { featureStats: stats, histData: null };
    }

    // Build 12 histogram bins
    const minVal = Math.min(...targetVals);
    const maxVal = Math.max(...targetVals);
    const binCount = 12;
    const step = (maxVal - minVal) / binCount || 1;
    const bins = new Array(binCount).fill(0);
    const binLabels = [];

    for (let i = 0; i < binCount; i++) {
      const bStart = minVal + i * step;
      const bEnd = minVal + (i + 1) * step;
      binLabels.push(`${bStart.toFixed(2)} - ${bEnd.toFixed(2)}`);
    }

    targetVals.forEach((v) => {
      let bIdx = Math.floor((v - minVal) / step);
      if (bIdx >= binCount) bIdx = binCount - 1;
      if (bIdx < 0) bIdx = 0;
      bins[bIdx]++;
    });

    return {
      featureStats: stats,
      histData: {
        labels: binLabels,
        datasets: [
          {
            label: `Frequency (${selectedWearFilter} records: n=${targetVals.length})`,
            data: bins,
            backgroundColor:
              selectedWearFilter === "high"
                ? "rgba(244, 63, 94, 0.75)"
                : selectedWearFilter === "low"
                ? "rgba(56, 189, 248, 0.75)"
                : "rgba(99, 102, 241, 0.75)",
            borderColor:
              selectedWearFilter === "high"
                ? "#f43f5e"
                : selectedWearFilter === "low"
                ? "#38bdf8"
                : "#6366f1",
            borderWidth: 1.5,
          },
        ],
      },
    };
  }, [allSamples, featureName, selectedWearFilter]);

  // Target VB Histogram
  const vbHistData = useMemo(() => {
    if (!allSamples || allSamples.length === 0) return null;
    const vbs = allSamples.map((s) => s.measured_VB_mm).filter((v) => v !== null && v !== undefined);
    if (vbs.length === 0) return null;

    const binEdges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6];
    const counts = new Array(binEdges.length - 1).fill(0);
    const labels = [];

    for (let i = 0; i < binEdges.length - 1; i++) {
      labels.push(`${binEdges[i].toFixed(1)} - ${binEdges[i + 1].toFixed(1)} mm`);
    }

    vbs.forEach((vb) => {
      for (let i = 0; i < binEdges.length - 1; i++) {
        if (vb >= binEdges[i] && (i === binEdges.length - 2 ? vb <= binEdges[i + 1] : vb < binEdges[i + 1])) {
          counts[i]++;
          break;
        }
      }
    });

    return {
      labels,
      datasets: [
        {
          label: "Number of Machining Cuts",
          data: counts,
          backgroundColor: counts.map((_, i) =>
            binEdges[i] >= 0.5 ? "rgba(244, 63, 94, 0.75)" : "rgba(52, 211, 153, 0.75)"
          ),
          borderColor: counts.map((_, i) =>
            binEdges[i] >= 0.5 ? "#f43f5e" : "#34d399"
          ),
          borderWidth: 1.5,
        },
      ],
    };
  }, [allSamples]);

  // Feature Importance Horizontal Bar Chart
  const top10Features = importanceData?.top_10 || [];
  const top10ChartData = {
    labels: top10Features.map((f) => f.feature),
    datasets: [
      {
        label: "Importance (%)",
        data: top10Features.map((f) => f.importance_percent),
        backgroundColor: [
          "#38bdf8",
          "#818cf8",
          "#a78bfa",
          "#34d399",
          "#4ade80",
          "#f59e0b",
          "#fbbf24",
          "#f87171",
          "#e879f9",
          "#22d3ee",
        ],
        borderWidth: 1,
      },
    ],
  };

  const top10ChartOptions = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#0f172a",
        titleColor: "#f1f5f9",
        bodyColor: "#94a3b8",
        callbacks: {
          label: (ctx) => `Importance: ${ctx.raw.toFixed(2)}%`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#94a3b8", font: { family: "DM Sans", size: 11 } },
        grid: { color: "rgba(51, 65, 85, 0.3)" },
        title: { display: true, text: "Importance (%)", color: "#94a3b8" },
      },
      y: {
        ticks: { color: "#e2e8f0", font: { family: "DM Sans", size: 11, weight: "600" } },
        grid: { display: false },
      },
    },
  };

  // Actual vs Predicted Scatter Plot
  const scatterPoints = useMemo(() => {
    if (!predictionsData?.records) return [];
    return predictionsData.records.map((r) => ({
      x: r.VB,
      y: r.predicted_VB,
      isHighWear: r.VB > 0.5,
      case: r.case,
      run: r.run,
      absErr: r.absolute_error_mm,
    }));
  }, [predictionsData]);

  const scatterChartData = {
    datasets: [
      {
        label: "Low Wear (VB <= 0.5 mm)",
        data: scatterPoints.filter((p) => !p.isHighWear),
        backgroundColor: "rgba(56, 189, 248, 0.8)",
        borderColor: "#0284c7",
        pointRadius: 4.5,
        pointHoverRadius: 7,
      },
      {
        label: "High Wear (VB > 0.5 mm)",
        data: scatterPoints.filter((p) => p.isHighWear),
        backgroundColor: "rgba(244, 63, 94, 0.85)",
        borderColor: "#e11d48",
        pointRadius: 5.5,
        pointHoverRadius: 8,
      },
      {
        label: "Ideal Diagonal (y = x)",
        data: [
          { x: 0, y: 0 },
          { x: 1.55, y: 1.55 },
        ],
        type: "line",
        borderColor: "rgba(255, 255, 255, 0.35)",
        borderDash: [5, 5],
        borderWidth: 2,
        pointRadius: 0,
        fill: false,
      },
    ],
  };

  const scatterChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: { color: "#94a3b8", font: { family: "DM Sans", size: 12 } },
      },
      tooltip: {
        backgroundColor: "#0f172a",
        titleColor: "#f1f5f9",
        bodyColor: "#94a3b8",
        callbacks: {
          label: (ctx) => {
            if (ctx.datasetIndex === 2) return "Ideal y = x";
            const p = ctx.raw;
            return `Case ${p.case}, Cut ${p.run}: Actual = ${p.x.toFixed(3)} mm, Pred = ${p.y.toFixed(3)} mm (Error: ${p.absErr.toFixed(3)} mm)`;
          },
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: "Actual Tool Flank Wear VB (mm)", color: "#94a3b8" },
        ticks: { color: "#94a3b8" },
        grid: { color: "rgba(51, 65, 85, 0.3)" },
        min: 0,
        max: 1.6,
      },
      y: {
        title: { display: true, text: "OOF Predicted Wear VB (mm)", color: "#94a3b8" },
        ticks: { color: "#94a3b8" },
        grid: { color: "rgba(51, 65, 85, 0.3)" },
        min: 0,
        max: 1.0,
      },
    },
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="spinner"></div>
        <p>Loading dataset signals, feature distributions, and out-of-fold analytics...</p>
      </div>
    );
  }

  if (error || !overview) {
    return (
      <div className="section-error">
        <h3>Unable to load Data Analysis</h3>
        <p>{error || "No data available."}</p>
      </div>
    );
  }

  return (
    <div className="section-container">
      {/* Header */}
      <div className="section-header">
        <div className="section-header-content">
          <span className="section-eyebrow">SECTION C</span>
          <h2 className="section-title">Data & Feature Analysis</h2>
          <p className="section-subtitle">
            Comprehensive exploratory signal distributions, physical sensor characteristics,
            Extra Trees feature importances, and genuine out-of-fold error patterns.
          </p>
        </div>
      </div>

      {/* Subsection A: Dataset Overview Cards */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">1. NASA Milling Dataset Health & Label Overview</h3>
            <p className="panel-desc">Real record counts, sensor channels, and measured wear label distributions.</p>
          </div>
          <span className="tag tag-blue">Data Provenance: mill.mat</span>
        </div>

        <div className="metric-cards-grid">
          <div className="metric-card">
            <div className="metric-label">TOTAL EXPERIMENTAL CUTS</div>
            <div className="metric-value">{overview.total_records}</div>
            <div className="metric-meta">16 distinct machining cases</div>
          </div>
          <div className="metric-card highlight">
            <div className="metric-label">SUPERVISED LABELLED CUTS</div>
            <div className="metric-value">{overview.valid_vb_count}</div>
            <div className="metric-meta">Used in cross-validation</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">UNMEASURED CUTS (NO VB)</div>
            <div className="metric-value">{overview.missing_vb_count}</div>
            <div className="metric-meta">Flank wear not measured</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">INPUT FEATURE SPACE</div>
            <div className="metric-value">{overview.feature_count}</div>
            <div className="metric-meta">36 Base + 12 Shape + 2 Conditions</div>
          </div>
        </div>

        <div className="grid-2col mt-4">
          <div className="subpanel">
            <h4 className="subpanel-title">Wear Target (VB) Label Statistics (146 Labelled Cuts)</h4>
            <div className="param-grid">
              <div className="param-item">
                <span className="param-name">Minimum Wear</span>
                <span className="param-val">{overview.vb_statistics?.min?.toFixed(3)} mm (New Insert)</span>
              </div>
              <div className="param-item">
                <span className="param-name">Maximum Wear</span>
                <span className="param-val">{overview.vb_statistics?.max?.toFixed(3)} mm (Severe Wear)</span>
              </div>
              <div className="param-item">
                <span className="param-name">Mean Flank Wear</span>
                <span className="param-val">{overview.vb_statistics?.mean?.toFixed(3)} mm</span>
              </div>
              <div className="param-item">
                <span className="param-name">Median Flank Wear</span>
                <span className="param-val">{overview.vb_statistics?.median?.toFixed(3)} mm</span>
              </div>
              <div className="param-item">
                <span className="param-name">Standard Deviation</span>
                <span className="param-val">{overview.vb_statistics?.std?.toFixed(3)} mm</span>
              </div>
              <div className="param-item">
                <span className="param-name">Analytical Threshold</span>
                <span className="param-val">0.50 mm (High Wear Split)</span>
              </div>
            </div>

            <div className="wear-split-bar mt-3">
              <div className="wear-split-low" style={{ width: "80.8%" }}>
                Low Wear (&le;0.5mm): 118 cuts (80.8%)
              </div>
              <div className="wear-split-high" style={{ width: "19.2%" }}>
                High: 28 (19.2%)
              </div>
            </div>
            <p className="text-xs text-muted mt-2">
              ⚠️ Note: 0.50 mm is the analytical research threshold used in this study to isolate accelerated wear regimes; it is not an absolute 100% tool failure scale.
            </p>
          </div>

          <div className="chart-wrapper">
            <div className="chart-header">
              <h4 className="chart-title">Ground Truth Flank Wear (VB) Distribution</h4>
              <span className="chart-sub">Green: Normal wear | Red: High wear (&gt;0.5 mm)</span>
            </div>
            <div className="chart-canvas" style={{ height: "210px" }}>
              {vbHistData && (
                <Bar
                  data={vbHistData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                      x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { display: false } },
                      y: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(51, 65, 85, 0.3)" } },
                    },
                  }}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Subsection B: Dynamic Sensor Feature Distributions */}
      <div className="panel mt-4">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">2. Sensor Signal Feature Distributions</h3>
            <p className="panel-desc">
              Inspect actual signal distributions calculated from the 167 milling records. Compare how features shift between low and high wear.
            </p>
          </div>
          <span className="tag tag-teal">Interactive Feature Explorer</span>
        </div>

        {/* Filter Controls */}
        <div className="filter-controls-bar">
          <div className="control-group">
            <label className="control-label">Sensor Channel:</label>
            <select
              className="select-input"
              value={selectedSensor}
              onChange={(e) => setSelectedSensor(e.target.value)}
            >
              {SENSORS.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label className="control-label">Statistical Feature:</label>
            <select
              className="select-input"
              value={selectedStat}
              onChange={(e) => setSelectedStat(e.target.value)}
            >
              {STATISTICS.map((st) => (
                <option key={st.key} value={st.key}>
                  {st.label}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label className="control-label">Wear Category Filter:</label>
            <div className="btn-group">
              <button
                className={`btn-filter ${selectedWearFilter === "all" ? "active" : ""}`}
                onClick={() => setSelectedWearFilter("all")}
              >
                All Records
              </button>
              <button
                className={`btn-filter ${selectedWearFilter === "low" ? "active" : ""}`}
                onClick={() => setSelectedWearFilter("low")}
              >
                Low Wear (&le;0.5mm)
              </button>
              <button
                className={`btn-filter ${selectedWearFilter === "high" ? "active" : ""}`}
                onClick={() => setSelectedWearFilter("high")}
              >
                High Wear (&gt;0.5mm)
              </button>
            </div>
          </div>
        </div>

        {/* Feature Distribution Content */}
        <div className="grid-2col mt-3">
          <div className="chart-wrapper">
            <div className="chart-header">
              <h4 className="chart-title">Distribution Histogram: <code>{featureName}</code></h4>
              <span className="chart-sub">Filtered by: {selectedWearFilter.toUpperCase()} wear</span>
            </div>
            <div className="chart-canvas" style={{ height: "260px" }}>
              {histData ? (
                <Bar
                  data={histData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                      x: { ticks: { color: "#94a3b8", font: { size: 9 }, maxRotation: 45 }, grid: { display: false } },
                      y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } },
                    },
                  }}
                />
              ) : (
                <div className="no-data">Loading feature distribution...</div>
              )}
            </div>
          </div>

          <div className="subpanel">
            <h4 className="subpanel-title">Low Wear vs. High Wear Statistical Shift</h4>
            <p className="text-xs text-muted mb-2">
              Shows how <code>{featureName}</code> responds as the tool edge degrades:
            </p>
            {featureStats ? (
              <div className="table-responsive">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Statistic</th>
                      <th>Low Wear (&le;0.5mm)</th>
                      <th>High Wear (&gt;0.5mm)</th>
                      <th>Shift</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="font-semibold">Sample Count</td>
                      <td>{featureStats.lowWear.count} cuts</td>
                      <td>{featureStats.highWear.count} cuts</td>
                      <td>-</td>
                    </tr>
                    <tr>
                      <td className="font-semibold">Mean Value</td>
                      <td>{featureStats.lowWear.mean.toFixed(4)}</td>
                      <td className="font-semibold text-rose">{featureStats.highWear.mean.toFixed(4)}</td>
                      <td>
                        {featureStats.lowWear.mean !== 0
                          ? `${(((featureStats.highWear.mean - featureStats.lowWear.mean) / Math.abs(featureStats.lowWear.mean)) * 100).toFixed(1)}%`
                          : "N/A"}
                      </td>
                    </tr>
                    <tr>
                      <td className="font-semibold">Std Deviation</td>
                      <td>{featureStats.lowWear.std.toFixed(4)}</td>
                      <td>{featureStats.highWear.std.toFixed(4)}</td>
                      <td>-</td>
                    </tr>
                    <tr>
                      <td className="font-semibold">Range (Min - Max)</td>
                      <td>{featureStats.lowWear.min.toFixed(2)} to {featureStats.lowWear.max.toFixed(2)}</td>
                      <td>{featureStats.highWear.min.toFixed(2)} to {featureStats.highWear.max.toFixed(2)}</td>
                      <td>-</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ) : null}

            <div className="callout-card mt-3">
              <h5 className="callout-title">Signal Interpretation Note:</h5>
              <p className="callout-body">
                Notice the shift in vibration and acoustic emission magnitudes during high-wear machining. As flank wear expands, friction area multiplies,
                exciting higher high-frequency vibrations in the spindle body.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Subsection C: Feature Importance */}
      <div className="panel mt-4">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">3. Model Feature Importance (Trained Extra Trees)</h3>
            <p className="panel-desc">
              Impurity-based Gini importance from 200 randomized decision trees.
            </p>
          </div>
          <button
            className="btn-outline-sm"
            onClick={() => setShowAllFeatures(!showAllFeatures)}
          >
            {showAllFeatures ? "Hide 50-Feature Table" : "View All 50 Features"}
          </button>
        </div>

        <div className="grid-2col">
          <div className="chart-wrapper">
            <div className="chart-header">
              <h4 className="chart-title">Top 10 Most Influential Features</h4>
              <span className="chart-sub">Measured by impurity reduction percentage</span>
            </div>
            <div className="chart-canvas" style={{ height: "340px" }}>
              <Bar data={top10ChartData} options={top10ChartOptions} />
            </div>
          </div>

          <div>
            <h4 className="subpanel-title">Sensor Domain Contribution Summary</h4>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Sensor / Input Domain</th>
                    <th>Importance</th>
                    <th>Leading Extracted Feature</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="font-semibold">vib_spindle (Spindle Vibration)</td>
                    <td className="font-semibold text-cyan">27.0%</td>
                    <td><code>vib_spindle_skewness</code> (13.75%)</td>
                  </tr>
                  <tr>
                    <td className="font-semibold">AE_spindle (Spindle Acoustic Emission)</td>
                    <td className="font-semibold text-emerald">21.2%</td>
                    <td><code>AE_spindle_maximum</code> (13.94%)</td>
                  </tr>
                  <tr>
                    <td className="font-semibold">smcDC (DC Motor Current)</td>
                    <td className="font-semibold text-blue">19.0%</td>
                    <td><code>smcDC_mean</code> (6.42%)</td>
                  </tr>
                  <tr>
                    <td className="font-semibold">smcAC (AC Motor Current)</td>
                    <td className="font-semibold text-indigo">12.6%</td>
                    <td><code>smcAC_std</code> (3.19%)</td>
                  </tr>
                  <tr>
                    <td className="font-semibold">AE_table (Table Acoustic Emission)</td>
                    <td>9.8%</td>
                    <td><code>AE_table_rms</code> (2.05%)</td>
                  </tr>
                  <tr>
                    <td className="font-semibold">vib_table (Table Vibration)</td>
                    <td>6.5%</td>
                    <td><code>vib_table_skewness</code> (1.22%)</td>
                  </tr>
                  <tr>
                    <td className="font-semibold">Machining Conditions (DOC, feed)</td>
                    <td>3.8%</td>
                    <td><code>DOC</code> (3.32%)</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="callout-card mt-3">
              <h5 className="callout-title">Scientific Caution on Tree Importance:</h5>
              <p className="callout-body">
                {importanceData?.notice || "Feature importance reflects impurity reduction within the decision tree splits, not physical causation."}
              </p>
            </div>
          </div>
        </div>

        {/* Expandable Table for All 50 Features */}
        {showAllFeatures && (
          <div className="table-responsive mt-4">
            <h4 className="subpanel-title">Complete 50-Feature Importance Ranking</h4>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Feature Name</th>
                  <th>Importance Score</th>
                  <th>Contribution (%)</th>
                </tr>
              </thead>
              <tbody>
                {importanceData?.features?.map((f) => (
                  <tr key={f.feature} className={f.rank <= 10 ? "row-highlight" : ""}>
                    <td>#{f.rank}</td>
                    <td className="font-mono">{f.feature}</td>
                    <td>{f.importance.toFixed(6)}</td>
                    <td className="font-semibold">{f.importance_percent.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Subsection D: Actual vs Predicted Analysis */}
      <div className="panel mt-4">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">4. Actual vs. Predicted Wear & Residual Analysis</h3>
            <p className="panel-desc">
              Evaluated on genuine <strong>out-of-fold (OOF) cross-validation predictions</strong> from all 146 labelled cuts.
            </p>
          </div>
          <span className="tag tag-purple">Zero Data Leakage</span>
        </div>

        <div className="grid-2col">
          <div className="chart-wrapper">
            <div className="chart-header">
              <h4 className="chart-title">Actual vs. Out-of-Fold Predicted Wear (VB)</h4>
              <span className="chart-sub">Dashed line = perfect prediction (y = x)</span>
            </div>
            <div className="chart-canvas" style={{ height: "340px" }}>
              <Scatter data={scatterChartData} options={scatterChartOptions} />
            </div>
          </div>

          <div>
            <h4 className="subpanel-title">Error Breakdown by Tool Wear Range</h4>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Flank Wear Range</th>
                    <th>Cuts</th>
                    <th>MAE (mm)</th>
                    <th>Regime Characteristic</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="font-semibold">0.0 - 0.2 mm</td>
                    <td>50 cuts</td>
                    <td className="text-emerald font-semibold">0.0962 mm</td>
                    <td>Initial sharp break-in; stable cutting</td>
                  </tr>
                  <tr>
                    <td className="font-semibold">0.2 - 0.4 mm</td>
                    <td>44 cuts</td>
                    <td className="text-emerald font-semibold">0.0969 mm</td>
                    <td>Steady-state flank wear progression</td>
                  </tr>
                  <tr>
                    <td className="font-semibold">0.4 - 0.6 mm</td>
                    <td>32 cuts</td>
                    <td className="text-emerald font-semibold">0.0947 mm</td>
                    <td>Transition to accelerated wear</td>
                  </tr>
                  <tr className="row-warning">
                    <td className="font-semibold">&ge; 0.6 mm</td>
                    <td>20 cuts</td>
                    <td className="text-rose font-semibold">0.3145 mm</td>
                    <td>Catastrophic wear / severe micro-chipping</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="callout-card mt-3">
              <h5 className="callout-title">Critical Model Finding for Defense/Viva:</h5>
              <p className="callout-body">
                The model maintains consistent precision (<strong>~0.095 mm MAE</strong>) across all cuts with wear up to <strong>0.60 mm</strong>.
                Errors only spike on extreme runaway wear cuts (&ge;0.60 mm, primarily Case 13), where physical insert geometry undergoes catastrophic micro-chipping and vibration signals saturate.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
