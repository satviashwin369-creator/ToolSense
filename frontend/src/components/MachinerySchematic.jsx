
/**
 * 2D Technical Schematic of the CNC Milling Machine & Sensor Locations.
 * Renders an engineering vector diagram indicating the six sensor channels
 * and the cutting tool flank wear location.
 */
export default function MachinerySchematic({ selectedSample }) {
  const smcAC_rms = selectedSample?.features?.smcAC_rms?.toFixed(2);
  const smcDC_mean = selectedSample?.features?.smcDC_mean?.toFixed(2);
  const vib_spin_rms = selectedSample?.features?.vib_spindle_rms?.toFixed(3);
  const vib_table_rms = selectedSample?.features?.vib_table_rms?.toFixed(3);
  const ae_spin_max = selectedSample?.features?.AE_spindle_maximum?.toFixed(3);
  const ae_table_rms = selectedSample?.features?.AE_table_rms?.toFixed(3);

  const predVB = selectedSample?.oof_prediction?.predicted_VB_mm ?? selectedSample?.predicted_VB_mm;
  const isHighWear = predVB !== undefined && predVB !== null && predVB >= 0.5;

  return (
    <div className="schematic-container">
      <div className="schematic-header">
        <div>
          <span className="section-eyebrow">PHYSICAL INSTRUMENTATION</span>
          <h4 className="schematic-title">CNC Milling Machine Sensor Architecture</h4>
        </div>
        <div className="schematic-legend">
          <span className="legend-item"><span className="dot dot-current"></span> Current Sensors</span>
          <span className="legend-item"><span className="dot dot-vib"></span> Vibration</span>
          <span className="legend-item"><span className="dot dot-ae"></span> Acoustic Emission</span>
        </div>
      </div>

      <div className="schematic-canvas-wrap">
        <svg viewBox="0 0 880 340" className="schematic-svg" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="metalGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#1e293b" />
              <stop offset="100%" stopColor="#0f172a" />
            </linearGradient>
            <linearGradient id="spindleGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#334155" />
              <stop offset="100%" stopColor="#1e293b" />
            </linearGradient>
            <linearGradient id="toolGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={isHighWear ? "#ef4444" : "#f59e0b"} />
              <stop offset="100%" stopColor={isHighWear ? "#b91c1c" : "#10b981"} />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Machine Column / Frame */}
          <rect x="80" y="20" width="120" height="290" rx="8" fill="url(#metalGrad)" stroke="#334155" strokeWidth="2" />
          <line x1="140" y1="20" x2="140" y2="310" stroke="#475569" strokeDasharray="4 4" strokeWidth="1" />
          <text x="140" y="55" fill="#64748b" fontSize="12" fontWeight="600" textAnchor="middle" letterSpacing="1">CNC COLUMN</text>

          {/* Spindle Motor & Housing */}
          <rect x="200" y="40" width="160" height="90" rx="6" fill="#1e293b" stroke="#3b82f6" strokeWidth="2" />
          <text x="280" y="65" fill="#93c5fd" fontSize="13" fontWeight="700" textAnchor="middle">AC/DC SPINDLE MOTOR</text>
          <text x="280" y="82" fill="#64748b" fontSize="11" textAnchor="middle">Direct Drive Spindle</text>

          {/* Current Sensor Hotspot 1: smcAC */}
          <circle cx="230" cy="105" r="7" fill="#0284c7" filter="url(#glow)" />
          <text x="245" y="109" fill="#38bdf8" fontSize="11" fontWeight="600">smcAC</text>

          {/* Current Sensor Hotspot 2: smcDC */}
          <circle cx="310" cy="105" r="7" fill="#0284c7" filter="url(#glow)" />
          <text x="325" y="109" fill="#38bdf8" fontSize="11" fontWeight="600">smcDC</text>

          {/* Spindle Shaft / Bearings */}
          <rect x="250" y="130" width="60" height="70" fill="url(#spindleGrad)" stroke="#475569" strokeWidth="1.5" />
          <rect x="260" y="145" width="40" height="40" rx="4" fill="#0f172a" stroke="#64748b" strokeWidth="1" />

          {/* Sensor Hotspot 3: vib_spindle */}
          <circle cx="240" cy="165" r="7" fill="#8b5cf6" filter="url(#glow)" />
          <line x1="240" y1="165" x2="260" y2="165" stroke="#8b5cf6" strokeWidth="1.5" />
          <text x="175" y="169" fill="#c4b5fd" fontSize="11" fontWeight="600">vib_spindle</text>

          {/* Sensor Hotspot 4: AE_spindle */}
          <circle cx="320" cy="165" r="7" fill="#10b981" filter="url(#glow)" />
          <line x1="300" y1="165" x2="320" y2="165" stroke="#10b981" strokeWidth="1.5" />
          <text x="335" y="169" fill="#6ee7b7" fontSize="11" fontWeight="600">AE_spindle</text>

          {/* Tool Holder & Cutting Tool (End Mill Insert) */}
          <polygon points="265,200 295,200 288,230 272,230" fill="#475569" stroke="#94a3b8" strokeWidth="1" />
          <rect x="277" y="230" width="6" height="25" fill="#cbd5e1" />
          {/* Cutting Edge / Flank Wear Location */}
          <polygon points="275,255 285,255 280,265" fill="url(#toolGrad)" stroke={isHighWear ? "#ef4444" : "#10b981"} strokeWidth="1.5" filter="url(#glow)" />
          <circle cx="280" cy="265" r="4" fill="#f59e0b" />
          <line x1="280" y1="265" x2="380" y2="265" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="3 3" />
          <text x="388" y="269" fill="#f59e0b" fontSize="12" fontWeight="700">Flank Wear (VB)</text>

          {/* Workpiece */}
          <rect x="220" y="265" width="120" height="22" rx="2" fill="#334155" stroke="#64748b" strokeWidth="1.5" />
          <text x="280" y="280" fill="#94a3b8" fontSize="10" fontWeight="600" textAnchor="middle">KC710 INSERT / WORKPIECE</text>

          {/* Machine Table (Base) */}
          <rect x="180" y="287" width="220" height="35" rx="4" fill="url(#metalGrad)" stroke="#334155" strokeWidth="2" />
          <text x="290" y="308" fill="#64748b" fontSize="11" fontWeight="600" textAnchor="middle">MACHINE TABLE (FEED AXIS)</text>

          {/* Sensor Hotspot 5: vib_table */}
          <circle cx="195" cy="305" r="7" fill="#8b5cf6" filter="url(#glow)" />
          <text x="125" y="309" fill="#c4b5fd" fontSize="11" fontWeight="600">vib_table</text>

          {/* Sensor Hotspot 6: AE_table */}
          <circle cx="385" cy="305" r="7" fill="#10b981" filter="url(#glow)" />
          <text x="400" y="309" fill="#6ee7b7" fontSize="11" fontWeight="600">AE_table</text>

          {/* Data Telemetry Cards (Right Side) */}
          <g transform="translate(480, 25)">
            <rect x="0" y="0" width="370" height="285" rx="8" fill="#0f172a" stroke="#1e293b" strokeWidth="1.5" />
            <text x="20" y="30" fill="#94a3b8" fontSize="12" fontWeight="700" letterSpacing="1">LIVE TELEMETRY CALLOUTS</text>

            {/* Current card */}
            <rect x="15" y="45" width="340" height="65" rx="6" fill="#1e293b" stroke="#0284c7" strokeWidth="1" />
            <text x="30" y="68" fill="#38bdf8" fontSize="12" fontWeight="700">MOTOR CURRENTS</text>
            <text x="30" y="88" fill="#94a3b8" fontSize="11">AC Current RMS: <tspan fill="#f1f5f9" fontWeight="600">{smcAC_rms ? `${smcAC_rms} A` : "1.53 A"}</tspan></text>
            <text x="180" y="88" fill="#94a3b8" fontSize="11">DC Current Mean: <tspan fill="#f1f5f9" fontWeight="600">{smcDC_mean ? `${smcDC_mean} A` : "5.44 A"}</tspan></text>
            <text x="30" y="102" fill="#64748b" fontSize="10">Sensitive to motor torque resistance from dull tool friction</text>

            {/* Vibration card */}
            <rect x="15" y="120" width="340" height="65" rx="6" fill="#1e293b" stroke="#8b5cf6" strokeWidth="1" />
            <text x="30" y="143" fill="#c4b5fd" fontSize="12" fontWeight="700">VIBRATION DYNAMICS</text>
            <text x="30" y="163" fill="#94a3b8" fontSize="11">Spindle RMS: <tspan fill="#f1f5f9" fontWeight="600">{vib_spin_rms ? `${vib_spin_rms} g` : "0.570 g"}</tspan></text>
            <text x="180" y="163" fill="#94a3b8" fontSize="11">Table RMS: <tspan fill="#f1f5f9" fontWeight="600">{vib_table_rms ? `${vib_table_rms} g` : "1.790 g"}</tspan></text>
            <text x="30" y="177" fill="#64748b" fontSize="10">Spindle vibration skewness & kurtosis rank in top 3 model features</text>

            {/* AE card */}
            <rect x="15" y="195" width="340" height="65" rx="6" fill="#1e293b" stroke="#10b981" strokeWidth="1" />
            <text x="30" y="218" fill="#6ee7b7" fontSize="12" fontWeight="700">ACOUSTIC EMISSION (AE)</text>
            <text x="30" y="238" fill="#94a3b8" fontSize="11">Spindle AE Peak: <tspan fill="#f1f5f9" fontWeight="600">{ae_spin_max ? `${ae_spin_max} V` : "0.429 V"}</tspan></text>
            <text x="180" y="238" fill="#94a3b8" fontSize="11">Table AE RMS: <tspan fill="#f1f5f9" fontWeight="600">{ae_table_rms ? `${ae_table_rms} V` : "0.178 V"}</tspan></text>
            <text x="30" y="252" fill="#64748b" fontSize="10">AE spindle peak has highest single model importance (13.94%)</text>
          </g>
        </svg>
      </div>
    </div>
  );
}
