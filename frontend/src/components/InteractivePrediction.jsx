
import { useState, useEffect, useMemo } from "react";
import MachinerySchematic from "./MachinerySchematic";

export default function InteractivePrediction({ apiBase, reportsBase }) {
  const [samplesList, setSamplesList] = useState([]);
  const [selectedCase, setSelectedCase] = useState(1);
  const [selectedRun, setSelectedRun] = useState(1);
  const [selectedSample, setSelectedSample] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("record"); // 'record' or 'whatif'
  const [filterType, setFilterType] = useState("all"); // 'all', 'labelled', 'high', 'low'

  // Live what-if simulation state
  const [customFeatures, setCustomFeatures] = useState({});
  const [livePrediction, setLivePrediction] = useState(null);
  const [livePredicting, setLivePredicting] = useState(false);
  const [liveError, setLiveError] = useState(null);

  // Load samples list
  useEffect(() => {
    let cancelled = false;

    async function fetchSamples() {
      setLoading(true);

      try {
        // Attempt backend /api/samples
        const res = await fetch(`${apiBase}/samples`).catch(() => null);
        if (res && res.ok) {
          const data = await res.json();
          if (!cancelled && data.records && data.records.length > 0) {
            setSamplesList(data.records);
            setLoading(false);
            return;
          }
        }

        // Fallback to static deployment_samples.json
        const staticRes = await fetch(`/data/deployment_samples.json`);
        const staticData = await staticRes.json();

        // Also fetch candidate_model_predictions.csv to attach OOF predictions
        const predsRes = await fetch(`${reportsBase}/candidate_model_predictions.csv`).then((r) => r.text());
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
        const predsList = parseCsv(predsRes);
        const oofMap = new Map();
        predsList.forEach((p) => {
          oofMap.set(`${p.case}_${p.run}`, p);
        });

        const records = staticData.map((s) => {
          const key = `${s.case}_${s.run}`;
          const oof = oofMap.get(key);
          return {
            case: s.case,
            run: s.run,
            measured_VB_mm: s.measured_VB_mm,
            has_measured_wear: s.measured_VB_mm !== null && s.measured_VB_mm !== undefined,
            predicted_VB_mm: oof ? oof.predicted_VB : null,
            absolute_error_mm: oof ? oof.absolute_error_mm : null,
            signed_error_mm: oof ? oof.signed_error_mm : null,
            is_out_of_fold: !!oof,
            wear_category: oof ? (oof.VB > 0.5 ? "High Wear" : "Low Wear") : "Unmeasured",
            DOC: s.features?.DOC,
            feed: s.features?.feed,
          };
        });

        if (!cancelled) {
          setSamplesList(records);
          setLoading(false);
        }
      } catch {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchSamples();
    return () => {
      cancelled = true;
    };
  }, [apiBase, reportsBase]);

  // Load specific sample details when case or run changes
  useEffect(() => {
    let cancelled = false;

    async function fetchSampleDetail() {
      try {
        // Attempt backend endpoint
        const res = await fetch(`${apiBase}/sample?case=${selectedCase}&run=${selectedRun}`).catch(() => null);
        if (res && res.ok) {
          const data = await res.json();
          if (!cancelled) {
            setSelectedSample(data);
            setCustomFeatures(data.features || {});
            return;
          }
        }

        // Fallback: search local deployment_samples.json
        const staticData = await fetch(`/data/deployment_samples.json`).then((r) => r.json());
        const match = staticData.find((s) => s.case === selectedCase && s.run === selectedRun);

        if (match && !cancelled) {
          // Attach OOF pred
          const predsRes = await fetch(`${reportsBase}/candidate_model_predictions.csv`).then((r) => r.text());
          const lines = predsRes.trim().split("\n");
          let oofMatch = null;
          for (let i = 1; i < lines.length; i++) {
            const [c, r, vb, pred, signed, abs] = lines[i].split(",");
            if (Number(c) === selectedCase && Number(r) === selectedRun) {
              oofMatch = {
                predicted_VB_mm: Number(pred),
                absolute_error_mm: Number(abs),
                signed_error_mm: Number(signed),
                is_oof: true,
                wear_category: Number(vb) > 0.5 ? "High Wear" : "Low Wear",
              };
              break;
            }
          }

          const decorated = {
            ...match,
            oof_prediction: oofMatch,
            DOC: match.features?.DOC,
            feed: match.features?.feed,
          };
          setSelectedSample(decorated);
          setCustomFeatures(decorated.features || {});
        }
      } catch {
        // Handle gracefully
      }
    }

    fetchSampleDetail();
    return () => {
      cancelled = true;
    };
  }, [selectedCase, selectedRun, apiBase, reportsBase]);

  // Filtered records
  const filteredRecords = useMemo(() => {
    if (!samplesList) return [];
    if (filterType === "labelled") return samplesList.filter((r) => r.has_measured_wear);
    if (filterType === "high") return samplesList.filter((r) => r.has_measured_wear && r.measured_VB_mm > 0.5);
    if (filterType === "low") return samplesList.filter((r) => r.has_measured_wear && r.measured_VB_mm <= 0.5);
    if (filterType === "unlabelled") return samplesList.filter((r) => !r.has_measured_wear);
    return samplesList;
  }, [samplesList, filterType]);

  // Pick a random labelled record with verified OOF prediction
  const handleRandomRecord = () => {
    const labelled = samplesList.filter((r) => r.has_measured_wear && r.is_out_of_fold);
    if (labelled.length === 0) return;
    const randomPick = labelled[Math.floor(Math.random() * labelled.length)];
    setSelectedCase(randomPick.case);
    setSelectedRun(randomPick.run);
  };

  // Live prediction simulation handler
  const handleRunLivePrediction = async () => {
    setLivePredicting(true);
    setLiveError(null);

    try {
      const res = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: customFeatures }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server returned ${res.status}`);
      }

      const data = await res.json();
      setLivePrediction(data);
    } catch (err) {
      setLiveError(`Prediction failed: ${err.message}. Ensure Flask backend is running on port 5000.`);
    } finally {
      setLivePredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="section-loading">
        <div className="spinner"></div>
        <p>Loading real dataset records and out-of-fold predictions...</p>
      </div>
    );
  }

  // Selected record data values
  const actualVB = selectedSample?.measured_VB_mm;
  const hasActual = actualVB !== null && actualVB !== undefined;
  const oofPred = selectedSample?.oof_prediction;
  const predVB = oofPred?.predicted_VB_mm;
  const absError = oofPred?.absolute_error_mm;
  const signedError = oofPred?.signed_error_mm;

  // Manufacturing status decision based on predicted VB
  let mfgStatus = "UNKNOWN";
  let mfgClass = "mfg-unknown";
  let mfgAction = "Inspect tool.";

  if (predVB !== undefined && predVB !== null) {
    if (predVB < 0.20) {
      mfgStatus = "OPTIMAL / PASS";
      mfgClass = "mfg-pass";
      mfgAction = "Tool cutting edge is sharp. Part surface finish satisfies highest precision tolerance.";
    } else if (predVB < 0.50) {
      mfgStatus = "ACCELERATED WEAR / WARNING";
      mfgClass = "mfg-warning";
      mfgAction = "Friction increasing. Surface roughness increasing. Schedule replacement insert before next run.";
    } else {
      mfgStatus = "CRITICAL / REPLACEMENT REQUIRED";
      mfgClass = "mfg-critical";
      mfgAction = "Immediate tool change required. High risk of workpiece dimensional rejection or tool chatter.";
    }
  }

  return (
    <div className="section-container">
      {/* Header */}
      <div className="section-header">
        <div className="section-header-content">
          <span className="section-eyebrow">SECTION D</span>
          <h2 className="section-title">Interactive Tool Wear Prediction</h2>
          <p className="section-subtitle">
            Inspect real cutting records from the NASA dataset using genuine{" "}
            <strong>out-of-fold (OOF) cross-validation predictions</strong>, or test live parameter what-if scenarios.
          </p>
        </div>
        <div className="validation-badge badge-green">
          <span className="badge-icon">✓</span>
          <div>
            <div className="badge-title">Honest OOF Predictions</div>
            <div className="badge-sub">Record's Machining Case Excluded From Training</div>
          </div>
        </div>
      </div>

      {/* Mode Selector Tabs */}
      <div className="tab-bar mb-4">
        <button
          className={`tab-btn ${activeTab === "record" ? "active" : ""}`}
          onClick={() => setActiveTab("record")}
        >
          📂 Dataset Record Inspector (OOF Predictions)
        </button>
        <button
          className={`tab-btn ${activeTab === "whatif" ? "active" : ""}`}
          onClick={() => setActiveTab("whatif")}
        >
          ⚡ Live Model "What-If" Inference
        </button>
      </div>

      {activeTab === "record" ? (
        <>
          {/* Record Selector Panel */}
          <div className="panel mb-4">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">1. Select Machining Record</h3>
                <p className="panel-desc">
                  Choose from 167 real milling runs across 16 machining cases.
                </p>
              </div>
              <button className="btn-random" onClick={handleRandomRecord}>
                🎲 Pick Random Labelled Record
              </button>
            </div>

            {/* Filter buttons */}
            <div className="filter-chips mb-3">
              <span className="text-xs text-muted mr-2">Filter Records:</span>
              <button
                className={`chip ${filterType === "all" ? "active" : ""}`}
                onClick={() => setFilterType("all")}
              >
                All Records (167)
              </button>
              <button
                className={`chip ${filterType === "labelled" ? "active" : ""}`}
                onClick={() => setFilterType("labelled")}
              >
                Labelled Only (146)
              </button>
              <button
                className={`chip ${filterType === "low" ? "active" : ""}`}
                onClick={() => setFilterType("low")}
              >
                Low Wear (&le;0.5mm)
              </button>
              <button
                className={`chip ${filterType === "high" ? "active" : ""}`}
                onClick={() => setFilterType("high")}
              >
                High Wear (&gt;0.5mm)
              </button>
              <button
                className={`chip ${filterType === "unlabelled" ? "active" : ""}`}
                onClick={() => setFilterType("unlabelled")}
              >
                Unlabelled (21)
              </button>
            </div>

            {/* Dropdown Selector */}
            <div className="record-selector-bar">
              <label className="selector-label">Choose Record:</label>
              <select
                className="select-input select-large"
                value={`${selectedCase}_${selectedRun}`}
                onChange={(e) => {
                  const [c, r] = e.target.value.split("_");
                  setSelectedCase(Number(c));
                  setSelectedRun(Number(r));
                }}
              >
                {filteredRecords.map((r) => (
                  <option key={`${r.case}_${r.run}`} value={`${r.case}_${r.run}`}>
                    Case {r.case}, Cut {r.run} — DOC: {r.DOC ?? 1.5}mm, Feed: {r.feed ?? 0.5}mm/rev{" "}
                    {r.has_measured_wear
                      ? `[Actual VB: ${r.measured_VB_mm.toFixed(2)} mm ${r.measured_VB_mm > 0.5 ? "🔴" : "🟢"}]`
                      : "[VB Not Measured in Raw Data]"}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Record Inspection & Prediction Display */}
          <div className="grid-2col mb-4">
            {/* Left: Prediction Results & Manufacturing Quality */}
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h3 className="panel-title">Prediction Result & Tolerances</h3>
                  <p className="panel-desc">
                    Case {selectedCase}, Cut #{selectedRun}
                  </p>
                </div>
                {oofPred ? (
                  <span className="badge-pill pill-green">Verified OOF Prediction</span>
                ) : (
                  <span className="badge-pill pill-amber">Unmeasured Record</span>
                )}
              </div>

              {/* Main Metric Cards */}
              <div className="pred-cards-grid">
                <div className="pred-card highlight">
                  <div className="pred-label">OOF PREDICTED WEAR</div>
                  <div className="pred-value">
                    {predVB !== undefined && predVB !== null ? (
                      <>
                        {predVB.toFixed(4)} <span className="unit">mm</span>
                      </>
                    ) : (
                      "Not fitted"
                    )}
                  </div>
                  <div className="pred-meta">Extra Trees Model Output</div>
                </div>

                <div className="pred-card">
                  <div className="pred-label">ACTUAL MEASURED WEAR</div>
                  <div className="pred-value">
                    {hasActual ? (
                      <>
                        {actualVB.toFixed(4)} <span className="unit">mm</span>
                      </>
                    ) : (
                      <span className="text-muted text-base">Unmeasured (None)</span>
                    )}
                  </div>
                  <div className="pred-meta">Optical microscope flank wear</div>
                </div>

                <div className="pred-card">
                  <div className="pred-label">ABSOLUTE PREDICTION ERROR</div>
                  <div className="pred-value">
                    {absError !== undefined && absError !== null ? (
                      <>
                        {absError.toFixed(4)} <span className="unit">mm</span>
                      </>
                    ) : (
                      "N/A"
                    )}
                  </div>
                  <div className="pred-meta">
                    {signedError !== undefined && signedError !== null
                      ? `Signed Error: ${signedError >= 0 ? "+" : ""}${signedError.toFixed(4)} mm`
                      : "Ground truth unavailable"}
                  </div>
                </div>
              </div>

              {/* Manufacturing Requirement Check */}
              <div className={`mfg-decision-box ${mfgClass} mt-4`}>
                <div className="mfg-decision-header">
                  <span className="mfg-status-title">MANUFACTURING STATUS: {mfgStatus}</span>
                  <span className="mfg-threshold-tag">Threshold: 0.50 mm</span>
                </div>
                <p className="mfg-action-text">{mfgAction}</p>
                <div className="mfg-tolerance-meter mt-2">
                  <div className="meter-label">
                    <span>Flank Wear vs. 0.50mm Threshold</span>
                    <span className="font-mono">
                      {predVB ? `${(predVB * 1000).toFixed(0)} µm` : "0 µm"}
                    </span>
                  </div>
                  <div className="meter-track">
                    <div
                      className={`meter-bar ${mfgClass}`}
                      style={{
                        width: `${Math.min(100, Math.max(5, (predVB / 0.8) * 100))}%`,
                      }}
                    ></div>
                    <div className="meter-threshold-marker" style={{ left: `${(0.5 / 0.8) * 100}%` }}>
                      <span>0.50mm limit</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Machining Run Process Parameters */}
              <div className="param-grid mt-4">
                <div className="param-item">
                  <span className="param-name">Machining Case</span>
                  <span className="param-val">Case #{selectedCase}</span>
                </div>
                <div className="param-item">
                  <span className="param-name">Cut / Pass Number</span>
                  <span className="param-val">Run #{selectedRun}</span>
                </div>
                <div className="param-item">
                  <span className="param-name">Depth of Cut (DOC)</span>
                  <span className="param-val">{selectedSample?.DOC ?? selectedSample?.features?.DOC ?? 1.5} mm</span>
                </div>
                <div className="param-item">
                  <span className="param-name">Feed Rate</span>
                  <span className="param-val">{selectedSample?.feed ?? selectedSample?.features?.feed ?? 0.5} mm/rev</span>
                </div>
                <div className="param-item">
                  <span className="param-name">Workpiece Material</span>
                  <span className="param-val">Cast Iron / Steel (Mat 1)</span>
                </div>
                <div className="param-item">
                  <span className="param-name">Cutting Tool Insert</span>
                  <span className="param-val">KC710 Coated Carbide</span>
                </div>
              </div>
            </div>

            {/* Right: 2D Machinery Schematic */}
            <div className="panel">
              <MachinerySchematic selectedSample={selectedSample} />
            </div>
          </div>

          {/* Extracted 50 Features Breakdown for Selected Record */}
          <div className="panel mb-4">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">2. Extracted 50 Signal Features for this Cut</h3>
                <p className="panel-desc">
                  Exact engineered feature values passed to the Extra Trees model for Case {selectedCase}, Cut {selectedRun}.
                </p>
              </div>
              <span className="tag tag-blue">50 Numerical Inputs</span>
            </div>

            <div className="features-accordion-grid">
              {/* Spindle Motor Current */}
              <div className="feature-group-card">
                <h4 className="group-title text-cyan">⚡ Motor Current (smcAC & smcDC)</h4>
                <div className="group-stats">
                  <div><span>smcAC Mean:</span> <code>{selectedSample?.features?.smcAC_mean?.toFixed(4)}</code></div>
                  <div><span>smcAC RMS:</span> <code>{selectedSample?.features?.smcAC_rms?.toFixed(4)}</code></div>
                  <div><span>smcAC Std:</span> <code>{selectedSample?.features?.smcAC_std?.toFixed(4)}</code></div>
                  <div><span>smcAC Skewness:</span> <code>{selectedSample?.features?.smcAC_skewness?.toFixed(4)}</code></div>
                  <div><span>smcAC Kurtosis:</span> <code>{selectedSample?.features?.smcAC_kurtosis?.toFixed(4)}</code></div>
                  <div><span>smcDC Mean:</span> <code className="text-emerald">{selectedSample?.features?.smcDC_mean?.toFixed(4)}</code></div>
                  <div><span>smcDC RMS:</span> <code className="text-emerald">{selectedSample?.features?.smcDC_rms?.toFixed(4)}</code></div>
                  <div><span>smcDC Peak-to-Peak:</span> <code>{selectedSample?.features?.smcDC_peak_to_peak?.toFixed(4)}</code></div>
                </div>
              </div>

              {/* Spindle Vibration */}
              <div className="feature-group-card">
                <h4 className="group-title text-purple">⌁ Spindle Vibration (vib_spindle)</h4>
                <div className="group-stats">
                  <div><span>RMS:</span> <code>{selectedSample?.features?.vib_spindle_rms?.toFixed(4)}</code></div>
                  <div><span>Mean:</span> <code>{selectedSample?.features?.vib_spindle_mean?.toFixed(4)}</code></div>
                  <div><span>Std:</span> <code>{selectedSample?.features?.vib_spindle_std?.toFixed(4)}</code></div>
                  <div><span>Skewness:</span> <code className="text-cyan font-bold">{selectedSample?.features?.vib_spindle_skewness?.toFixed(4)}</code></div>
                  <div><span>Kurtosis:</span> <code className="text-cyan font-bold">{selectedSample?.features?.vib_spindle_kurtosis?.toFixed(4)}</code></div>
                  <div><span>Peak-to-Peak:</span> <code>{selectedSample?.features?.vib_spindle_peak_to_peak?.toFixed(4)}</code></div>
                  <div><span>Minimum:</span> <code>{selectedSample?.features?.vib_spindle_minimum?.toFixed(4)}</code></div>
                  <div><span>Maximum:</span> <code>{selectedSample?.features?.vib_spindle_maximum?.toFixed(4)}</code></div>
                </div>
              </div>

              {/* Spindle Acoustic Emission */}
              <div className="feature-group-card">
                <h4 className="group-title text-emerald">🔊 Acoustic Emission (AE_spindle)</h4>
                <div className="group-stats">
                  <div><span>Maximum (Top Feature):</span> <code className="text-emerald font-bold">{selectedSample?.features?.AE_spindle_maximum?.toFixed(4)}</code></div>
                  <div><span>RMS:</span> <code>{selectedSample?.features?.AE_spindle_rms?.toFixed(4)}</code></div>
                  <div><span>Mean:</span> <code>{selectedSample?.features?.AE_spindle_mean?.toFixed(4)}</code></div>
                  <div><span>Std:</span> <code>{selectedSample?.features?.AE_spindle_std?.toFixed(4)}</code></div>
                  <div><span>Skewness:</span> <code>{selectedSample?.features?.AE_spindle_skewness?.toFixed(4)}</code></div>
                  <div><span>Kurtosis:</span> <code>{selectedSample?.features?.AE_spindle_kurtosis?.toFixed(4)}</code></div>
                  <div><span>Peak-to-Peak:</span> <code>{selectedSample?.features?.AE_spindle_peak_to_peak?.toFixed(4)}</code></div>
                  <div><span>Minimum:</span> <code>{selectedSample?.features?.AE_spindle_minimum?.toFixed(4)}</code></div>
                </div>
              </div>

              {/* Table Sensors & Conditions */}
              <div className="feature-group-card">
                <h4 className="group-title text-amber">📐 Table Sensors & Conditions</h4>
                <div className="group-stats">
                  <div><span>Depth of Cut (DOC):</span> <code className="font-bold">{selectedSample?.features?.DOC} mm</code></div>
                  <div><span>Feed Rate:</span> <code className="font-bold">{selectedSample?.features?.feed} mm/rev</code></div>
                  <div><span>vib_table RMS:</span> <code>{selectedSample?.features?.vib_table_rms?.toFixed(4)}</code></div>
                  <div><span>vib_table Skewness:</span> <code>{selectedSample?.features?.vib_table_skewness?.toFixed(4)}</code></div>
                  <div><span>AE_table RMS:</span> <code>{selectedSample?.features?.AE_table_rms?.toFixed(4)}</code></div>
                  <div><span>AE_table Mean:</span> <code>{selectedSample?.features?.AE_table_mean?.toFixed(4)}</code></div>
                  <div><span>AE_table Peak-to-Peak:</span> <code>{selectedSample?.features?.AE_table_peak_to_peak?.toFixed(4)}</code></div>
                  <div><span>AE_table Maximum:</span> <code>{selectedSample?.features?.AE_table_maximum?.toFixed(4)}</code></div>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        /* What-If Live Simulator */
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">Live Model "What-If" Inference Simulator</h3>
              <p className="panel-desc">
                Adjust key machining conditions or leading sensor features to test live model inference via <code>POST /api/predict</code>.
              </p>
            </div>
            <span className="tag tag-purple">Real Model Pipeline</span>
          </div>

          <div className="grid-2col mt-4">
            <div>
              <h4 className="subpanel-title">Adjust Machining & Leading Sensor Inputs</h4>
              <div className="slider-controls">
                {/* DOC Slider */}
                <div className="slider-group">
                  <div className="slider-label">
                    <span>Depth of Cut (DOC):</span>
                    <span className="font-mono">{customFeatures.DOC ?? 1.5} mm</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="2.5"
                    step="0.25"
                    value={customFeatures.DOC ?? 1.5}
                    onChange={(e) => setCustomFeatures({ ...customFeatures, DOC: parseFloat(e.target.value) })}
                    className="range-slider"
                  />
                </div>

                {/* Feed Slider */}
                <div className="slider-group">
                  <div className="slider-label">
                    <span>Feed Rate:</span>
                    <span className="font-mono">{customFeatures.feed ?? 0.5} mm/rev</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.8"
                    step="0.05"
                    value={customFeatures.feed ?? 0.5}
                    onChange={(e) => setCustomFeatures({ ...customFeatures, feed: parseFloat(e.target.value) })}
                    className="range-slider"
                  />
                </div>

                {/* Spindle Vibration Skewness Slider */}
                <div className="slider-group">
                  <div className="slider-label">
                    <span>vib_spindle_skewness (Leading Shape Stat):</span>
                    <span className="font-mono">{customFeatures.vib_spindle_skewness?.toFixed(3) ?? "0.000"}</span>
                  </div>
                  <input
                    type="range"
                    min="-2.0"
                    max="3.0"
                    step="0.1"
                    value={customFeatures.vib_spindle_skewness ?? 0.0}
                    onChange={(e) => setCustomFeatures({ ...customFeatures, vib_spindle_skewness: parseFloat(e.target.value) })}
                    className="range-slider"
                  />
                </div>

                {/* AE Spindle Maximum Slider */}
                <div className="slider-group">
                  <div className="slider-label">
                    <span>AE_spindle_maximum (Top Model Feature):</span>
                    <span className="font-mono">{customFeatures.AE_spindle_maximum?.toFixed(3) ?? "0.450"} V</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    value={customFeatures.AE_spindle_maximum ?? 0.45}
                    onChange={(e) => setCustomFeatures({ ...customFeatures, AE_spindle_maximum: parseFloat(e.target.value) })}
                    className="range-slider"
                  />
                </div>

                {/* smcDC Mean */}
                <div className="slider-group">
                  <div className="slider-label">
                    <span>smcDC_mean (DC Motor Current):</span>
                    <span className="font-mono">{customFeatures.smcDC_mean?.toFixed(2) ?? "5.40"} A</span>
                  </div>
                  <input
                    type="range"
                    min="2.0"
                    max="10.0"
                    step="0.2"
                    value={customFeatures.smcDC_mean ?? 5.4}
                    onChange={(e) => setCustomFeatures({ ...customFeatures, smcDC_mean: parseFloat(e.target.value) })}
                    className="range-slider"
                  />
                </div>
              </div>

              <button
                className="btn-primary mt-4"
                onClick={handleRunLivePrediction}
                disabled={livePredicting}
              >
                {livePredicting ? "Computing Inference..." : "⚡ Execute Model Inference"}
              </button>

              {liveError && <p className="text-rose text-sm mt-2">{liveError}</p>}
            </div>

            {/* Live Result Output */}
            <div>
              <h4 className="subpanel-title">Model Inference Output</h4>
              {livePrediction ? (
                <div className="pred-cards-grid mt-2">
                  <div className="pred-card highlight">
                    <div className="pred-label">LIVE PREDICTED WEAR</div>
                    <div className="pred-value">
                      {livePrediction.predicted_VB_mm.toFixed(4)} <span className="unit">mm</span>
                    </div>
                    <div className="pred-meta">ExtraTreesRegressor (50 inputs)</div>
                  </div>

                  <div className="pred-card">
                    <div className="pred-label">ESTIMATED WEAR REGIME</div>
                    <div className="pred-value text-base">
                      {livePrediction.high_wear ? (
                        <span className="text-rose font-bold">🔴 High Wear (&ge; 0.50 mm)</span>
                      ) : (
                        <span className="text-emerald font-bold">🟢 Normal Wear (&lt; 0.50 mm)</span>
                      )}
                    </div>
                    <div className="pred-meta">Threshold: 0.50 mm</div>
                  </div>

                  <div className="callout-card mt-3">
                    <h5 className="callout-title">Notice:</h5>
                    <p className="callout-body">{livePrediction.notice}</p>
                  </div>
                </div>
              ) : (
                <div className="placeholder-box">
                  <p>Click "Execute Model Inference" to send the adjusted 50-feature array to the Flask prediction backend.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
