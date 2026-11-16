
import { useState, useEffect } from "react";
import { Bar } from "react-chartjs-2";

export default function AlgorithmComparison({ apiBase, reportsBase }) {
  const [algoData, setAlgoData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      setLoading(true);
      setError(null);

      // Attempt backend API first
      try {
        const res = await fetch(`${apiBase}/algorithm-comparison`);
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) {
            setAlgoData(data);
            setLoading(false);
            return;
          }
        }
      } catch {
        // Fallback to static reports files
      }

      try {
        const [algoRes, shapeRes, metricsRes] = await Promise.all([
          fetch(`${reportsBase}/algorithm_comparison.csv`).then((r) => r.text()),
          fetch(`${reportsBase}/shape_feature_comparison.csv`).then((r) => r.text()),
          fetch(`${reportsBase}/candidate_model_metrics.json`).then((r) => r.json()),
        ]);

        const parseCsv = (text) => {
          const lines = text.trim().split("\n");
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

        const rawAlgo = parseCsv(algoRes);
        const rawShape = parseCsv(shapeRes);

        const models = rawAlgo.map((r) => ({
          model: r.Model,
          feature_count: r.Feature_count,
          MAE_mm: r.MAE_mm,
          RMSE_mm: r.RMSE_mm,
          R2: r.R2,
          high_wear_count: r.High_wear_count,
          high_wear_MAE_mm: r.High_wear_MAE_mm,
        }));

        const featureComp = rawShape.map((r) => ({
          key: r.Model,
          name:
            r.Model === "Baseline_38"
              ? "Baseline (38 features)"
              : r.Model === "Baseline_plus_shape_50"
              ? "Baseline + Skewness & Kurtosis (50 features)"
              : "Shape + All Conditions (51 features)",
          feature_count: r.Feature_count,
          MAE_mm: r.MAE_mm,
          RMSE_mm: r.RMSE_mm,
          R2: r.R2,
          high_wear_count: r.High_wear_count,
          high_wear_MAE_mm: r.High_wear_MAE_mm,
          high_wear_signed_error_mm: r.High_wear_signed_error_mm,
        }));

        if (!cancelled) {
          setAlgoData({
            baseline_models: models,
            feature_comparison: featureComp,
            final_candidate: metricsRes,
            best_indicators: {
              lowest_mae: "Extra Trees",
              lowest_rmse: "Histogram Gradient Boosting",
              highest_r2: "Histogram Gradient Boosting",
            },
            validation_method: "5-fold GroupKFold by machining case (unseen case evaluation)",
            explanation: {
              problem_type: "Regression (continuous cutting-tool flank wear VB in mm)",
              metric_definitions: {
                MAE: "Mean Absolute Error in millimetres. Direct average distance between predicted wear and actual physical wear. Lower is better.",
                RMSE: "Root Mean Squared Error in millimetres. Penalizes large prediction errors more severely. Lower is better.",
                R2: "Coefficient of determination (fraction of target variance explained by model). Higher is better. This is NOT an accuracy percentage.",
              },
              why_extra_trees:
                "Extra Trees (Extremely Randomized Trees) randomizes split thresholds in addition to feature subsets. This randomization significantly dampens variance and noise present in dynamic CNC cutting signals (spindle vibration and acoustic emissions), providing the best generalization across machining cases.",
            },
          });
          setLoading(false);
        }
      } catch {
        if (!cancelled) {
          setError("Failed to load algorithm comparison data. Ensure backend or static reports are accessible.");
          setLoading(false);
        }
      }
    }

    fetchData();
    return () => {
      cancelled = true;
    };
  }, [apiBase, reportsBase]);

  if (loading) {
    return (
      <div className="section-loading">
        <div className="spinner"></div>
        <p>Loading algorithm comparison and validation benchmarks...</p>
      </div>
    );
  }

  if (error || !algoData) {
    return (
      <div className="section-error">
        <h3>Unable to load Algorithm Comparison</h3>
        <p>{error || "No data available."}</p>
      </div>
    );
  }

  const { baseline_models, feature_comparison, final_candidate, best_indicators, explanation } = algoData;

  // Chart 1: Baseline Algorithms Bar Chart
  const modelLabels = baseline_models.map((m) => m.model);
  const maeValues = baseline_models.map((m) => m.MAE_mm);
  const rmseValues = baseline_models.map((m) => m.RMSE_mm);
  const r2Values = baseline_models.map((m) => m.R2);

  const baselineChartData = {
    labels: modelLabels,
    datasets: [
      {
        label: "MAE (mm) - Lower is better",
        data: maeValues,
        backgroundColor: "rgba(56, 189, 248, 0.75)",
        borderColor: "#38bdf8",
        borderWidth: 1.5,
      },
      {
        label: "RMSE (mm) - Lower is better",
        data: rmseValues,
        backgroundColor: "rgba(168, 85, 247, 0.75)",
        borderColor: "#a855f7",
        borderWidth: 1.5,
      },
      {
        label: "R² Score - Higher is better",
        data: r2Values,
        backgroundColor: "rgba(52, 211, 153, 0.75)",
        borderColor: "#34d399",
        borderWidth: 1.5,
      },
    ],
  };

  const chartOptions = {
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
        borderColor: "#334155",
        borderWidth: 1,
        padding: 10,
        callbacks: {
          label: (ctx) => `${ctx.dataset.label.split(" ")[0]}: ${ctx.raw.toFixed(4)}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#94a3b8", font: { family: "DM Sans", size: 11 } },
        grid: { color: "rgba(51, 65, 85, 0.3)" },
      },
      y: {
        ticks: { color: "#94a3b8", font: { family: "DM Sans", size: 11 } },
        grid: { color: "rgba(51, 65, 85, 0.3)" },
        beginAtZero: true,
      },
    },
  };

  // Feature Engineering Comparison Chart
  const featLabels = feature_comparison.map((f) => f.name);
  const featChartData = {
    labels: featLabels,
    datasets: [
      {
        label: "MAE (mm) [Lower is better]",
        data: feature_comparison.map((f) => f.MAE_mm),
        backgroundColor: "rgba(56, 189, 248, 0.8)",
        borderColor: "#38bdf8",
        borderWidth: 1.5,
        yAxisID: "y",
      },
      {
        label: "R² Score [Higher is better]",
        data: feature_comparison.map((f) => f.R2),
        backgroundColor: "rgba(52, 211, 153, 0.8)",
        borderColor: "#34d399",
        borderWidth: 1.5,
        yAxisID: "y1",
      },
    ],
  };

  const featChartOptions = {
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
        borderColor: "#334155",
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ticks: { color: "#94a3b8", font: { family: "DM Sans", size: 11 } },
        grid: { color: "rgba(51, 65, 85, 0.3)" },
      },
      y: {
        type: "linear",
        position: "left",
        ticks: { color: "#38bdf8", font: { family: "DM Sans", size: 11 } },
        title: { display: true, text: "MAE (mm)", color: "#38bdf8" },
        grid: { color: "rgba(51, 65, 85, 0.3)" },
        min: 0.11,
        max: 0.15,
      },
      y1: {
        type: "linear",
        position: "right",
        ticks: { color: "#34d399", font: { family: "DM Sans", size: 11 } },
        title: { display: true, text: "R² Score", color: "#34d399" },
        grid: { drawOnChartArea: false },
        min: 0.35,
        max: 0.50,
      },
    },
  };

  return (
    <div className="section-container">
      {/* Header */}
      <div className="section-header">
        <div className="section-header-content">
          <span className="section-eyebrow">SECTION B</span>
          <h2 className="section-title">Algorithm Comparison & Validation Benchmarks</h2>
          <p className="section-subtitle">
            Systematic evaluation of 4 machine learning regressors across 5-fold GroupKFold cross-validation
            grouped by machining case to prevent data leakage.
          </p>
        </div>
        <div className="validation-badge">
          <span className="badge-icon">🛡️</span>
          <div>
            <div className="badge-title">5-Fold GroupKFold by Case</div>
            <div className="badge-sub">Strict Case-Level Generalization</div>
          </div>
        </div>
      </div>

      {/* Overview Stat Cards */}
      <div className="metric-cards-grid">
        <div className="metric-card highlight">
          <div className="metric-label">FINAL SELECTED MODEL</div>
          <div className="metric-value">Extra Trees</div>
          <div className="metric-meta">Tuned 50-Feature Regressor</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">OVERALL TEST MAE</div>
          <div className="metric-value">{final_candidate?.overall?.MAE_mm?.toFixed(4)} <span className="unit">mm</span></div>
          <div className="metric-meta">5-fold out-of-fold average</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">OVERALL TEST RMSE</div>
          <div className="metric-value">{final_candidate?.overall?.RMSE_mm?.toFixed(4)} <span className="unit">mm</span></div>
          <div className="metric-meta">Penalizes large outlier cuts</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">EXPLAINED VARIANCE (R²)</div>
          <div className="metric-value">{final_candidate?.overall?.R2?.toFixed(4)}</div>
          <div className="metric-meta">Proportion of variance explained</div>
        </div>
      </div>

      {/* Regression Problem Notice */}
      <div className="info-banner warning-style">
        <span className="banner-icon">⚠️</span>
        <div className="banner-text">
          <strong>Regression Task Notice:</strong> Tool flank wear estimation is a continuous physical measurement problem (target <code>VB</code> in mm).
          It is <em>not</em> classification. <strong>Do not interpret R² as an accuracy percentage</strong>; R² measures the fraction of target variance captured by the model relative to a mean baseline.
        </div>
      </div>

      {/* Subsection 1: Baseline 38-Feature Algorithm Comparison */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">1. Baseline Algorithm Comparison (38 Features)</h3>
            <p className="panel-desc">
              All 4 algorithms evaluated on identical 36 sensor statistics + 2 condition features (DOC, feed) using 5-fold GroupKFold.
            </p>
          </div>
          <span className="tag tag-blue">38 Features Benchmark</span>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Algorithm</th>
                <th>Feature Count</th>
                <th>MAE (mm) <span className="th-hint">↓ Best</span></th>
                <th>RMSE (mm) <span className="th-hint">↓ Best</span></th>
                <th>R² Score <span className="th-hint">↑ Best</span></th>
                <th>High-Wear MAE (&gt;0.5mm)</th>
                <th>Comparative Finding</th>
              </tr>
            </thead>
            <tbody>
              {baseline_models.map((m) => {
                const isLowestMAE = m.model === best_indicators.lowest_mae;
                const isLowestRMSE = m.model === best_indicators.lowest_rmse;
                const isHighestR2 = m.model === best_indicators.highest_r2;

                return (
                  <tr key={m.model} className={m.model === "Extra Trees" ? "row-selected" : ""}>
                    <td className="font-semibold">
                      {m.model}
                      {m.model === "Extra Trees" && <span className="badge-pill pill-green ml-2">Selected Candidate</span>}
                    </td>
                    <td>{m.feature_count}</td>
                    <td>
                      <span className={isLowestMAE ? "metric-highlight-best" : ""}>
                        {m.MAE_mm.toFixed(4)} mm
                      </span>
                      {isLowestMAE && <span className="badge-pill pill-teal ml-1">Lowest MAE</span>}
                    </td>
                    <td>
                      <span className={isLowestRMSE ? "metric-highlight-best" : ""}>
                        {m.RMSE_mm.toFixed(4)} mm
                      </span>
                      {isLowestRMSE && <span className="badge-pill pill-purple ml-1">Lowest RMSE</span>}
                    </td>
                    <td>
                      <span className={isHighestR2 ? "metric-highlight-best" : ""}>
                        {m.R2.toFixed(4)}
                      </span>
                      {isHighestR2 && <span className="badge-pill pill-green ml-1">Highest R²</span>}
                    </td>
                    <td>{m.high_wear_MAE_mm.toFixed(4)} mm</td>
                    <td className="text-muted text-sm">
                      {m.model === "Extra Trees" && "Best average linear wear accuracy; lowest individual error."}
                      {m.model === "Histogram Gradient Boosting" && "Lowest squared error; fast binned tree splits."}
                      {m.model === "Gradient Boosting" && "Moderate performance; sensitive to case 13 outliers."}
                      {m.model === "Random Forest" && "Baseline ensemble; higher variance across folds."}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Baseline Comparison Chart */}
        <div className="chart-wrapper mt-4">
          <div className="chart-header">
            <h4 className="chart-title">Baseline Model Comparison: MAE, RMSE, and R²</h4>
            <span className="chart-sub">Lower MAE/RMSE is better; higher R² is better.</span>
          </div>
          <div className="chart-canvas" style={{ height: "300px" }}>
            <Bar data={baselineChartData} options={chartOptions} />
          </div>
        </div>
      </div>

      {/* Subsection 2: Feature Engineering Comparison */}
      <div className="panel mt-4">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">2. Feature Engineering: Adding Signal Shape Statistics</h3>
            <p className="panel-desc">
              Impact of augmenting basic sensor statistics with 12 signal shape features (skewness and Fisher excess kurtosis for each of the 6 sensors).
            </p>
          </div>
          <span className="tag tag-purple">Feature Evolution</span>
        </div>

        <div className="grid-2col">
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Feature Configuration</th>
                  <th>Features</th>
                  <th>MAE (mm)</th>
                  <th>RMSE (mm)</th>
                  <th>R² Score</th>
                </tr>
              </thead>
              <tbody>
                {feature_comparison.map((f, idx) => (
                  <tr key={f.key} className={idx === 1 ? "row-selected" : ""}>
                    <td className="font-semibold">
                      {f.name}
                      {idx === 1 && <span className="badge-pill pill-green ml-2">Adopted in Model</span>}
                    </td>
                    <td>{f.feature_count}</td>
                    <td>{f.MAE_mm.toFixed(4)} mm</td>
                    <td>{f.RMSE_mm.toFixed(4)} mm</td>
                    <td className="font-semibold text-emerald">{f.R2.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="callout-card mt-3">
              <h5 className="callout-title">Engineering Justification for Shape Statistics:</h5>
              <p className="callout-body">
                Basic statistics (Mean, RMS, Std) capture energy and magnitude, but fail to detect intermittent micro-chipping transients.
                <strong> Kurtosis</strong> detects sharp, impulsive vibration spikes caused by cutting edge fracturing, while <strong>skewness</strong> detects
                asymmetric signal deflection when the insert loses its symmetrical cutting geometry. Adding these 12 features reduced MAE from <strong>0.1328 mm to 0.1261 mm</strong> and boosted R² from <strong>0.395 to 0.452</strong>.
              </p>
            </div>
          </div>

          <div className="chart-wrapper">
            <div className="chart-header">
              <h4 className="chart-title">Feature Set Evolution vs Model Accuracy</h4>
              <span className="chart-sub">Left axis: MAE (mm) | Right axis: R² Score</span>
            </div>
            <div className="chart-canvas" style={{ height: "260px" }}>
              <Bar data={featChartData} options={featChartOptions} />
            </div>
          </div>
        </div>
      </div>

      {/* Subsection 3: Final Tuned Candidate Model */}
      <div className="panel mt-4">
        <div className="panel-header">
          <div>
            <h3 className="panel-title">3. Final Candidate Model Configuration & Fold Validation</h3>
            <p className="panel-desc">
              Hyperparameter-tuned Extra Trees Regressor fitted on 50 features with 5-fold cross-validation.
            </p>
          </div>
          <span className="tag tag-green">Production Candidate</span>
        </div>

        <div className="grid-2col">
          <div>
            <h4 className="subpanel-title">Tuned Hyperparameters</h4>
            <div className="param-grid">
              <div className="param-item">
                <span className="param-name">Algorithm</span>
                <span className="param-val">ExtraTreesRegressor</span>
              </div>
              <div className="param-item">
                <span className="param-name">n_estimators</span>
                <span className="param-val">200</span>
              </div>
              <div className="param-item">
                <span className="param-name">max_depth</span>
                <span className="param-val">12</span>
              </div>
              <div className="param-item">
                <span className="param-name">min_samples_leaf</span>
                <span className="param-val">1</span>
              </div>
              <div className="param-item">
                <span className="param-name">max_features</span>
                <span className="param-val">1.0 (all 50 features)</span>
              </div>
              <div className="param-item">
                <span className="param-name">random_state</span>
                <span className="param-val">42</span>
              </div>
            </div>

            <div className="callout-card mt-3">
              <h5 className="callout-title">Why Extra Trees Was Selected Over Random Forest:</h5>
              <p className="callout-body">
                {explanation.why_extra_trees}
              </p>
            </div>
          </div>

          <div>
            <h4 className="subpanel-title">5-Fold GroupKFold Cross-Validation Breakdown</h4>
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Fold</th>
                    <th>Test Runs</th>
                    <th>MAE (mm)</th>
                    <th>RMSE (mm)</th>
                    <th>R²</th>
                    <th>High-Wear MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {final_candidate?.fold_results?.map((f) => (
                    <tr key={f.fold} className={f.fold === 3 ? "row-warning" : ""}>
                      <td className="font-semibold">Fold {f.fold}</td>
                      <td>{f.test_records}</td>
                      <td>{f.MAE_mm.toFixed(4)} mm</td>
                      <td>{f.RMSE_mm.toFixed(4)} mm</td>
                      <td>{f.R2.toFixed(4)}</td>
                      <td>{f.high_wear_MAE_mm ? `${f.high_wear_MAE_mm.toFixed(4)} mm` : "N/A"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted mt-2">
              * Note: Fold 3 exhibits elevated error (0.2149 mm MAE) because it evaluated Case 13, which contains extreme runaway tool wear (VB reaching 1.53 mm).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
