"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useInspection, InspectionResult, Defect, AnomalyScore, API_URL } from "@/hooks/useInspection";

// ── Placeholder SVG ───────────────────────────────────────────────────────────
function makePlaceholder(): string {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 750 540'>
    <defs>
      <radialGradient id='bg'><stop offset='0%' stop-color='#1a2a1a'/><stop offset='100%' stop-color='#0c1410'/></radialGradient>
      <pattern id='g' width='10' height='10' patternUnits='userSpaceOnUse'>
        <path d='M10 0H0V10' fill='none' stroke='rgba(80,90,70,0.08)' stroke-width='0.5'/>
      </pattern>
    </defs>
    <rect width='750' height='540' fill='url(#bg)'/><rect width='750' height='540' fill='url(#g)'/>
    <text x='375' y='285' text-anchor='middle' fill='rgba(255,255,255,0.18)' font-size='14' font-family='monospace'>No image loaded</text>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}
const PLACEHOLDER_IMG = makePlaceholder();

// ── Sparkline ─────────────────────────────────────────────────────────────────
function Sparkline({ data, color = "var(--accent)" }: { data: number[]; color?: string }) {
  const W = 60, H = 24;
  const max = Math.max(...data), min = Math.min(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - ((v - min) / range) * (H - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      <polyline points={`${pts} ${W},${H} 0,${H}`} fill={color} opacity={0.1} stroke="none" />
    </svg>
  );
}

// ── CountUp ───────────────────────────────────────────────────────────────────
function useCountUp(target: number, decimals = 0) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let raf: number;
    const tStart = performance.now() + 80;
    const tick = (now: number) => {
      if (now < tStart) { raf = requestAnimationFrame(tick); return; }
      const p = Math.min(1, (now - tStart) / 900);
      setVal(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);
  return decimals > 0 ? val.toFixed(decimals) : Math.round(val).toLocaleString();
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function Stat({ label, value, unit, decimals, delta, deltaDir, deltaTone, spark, tint, sparkColor }: {
  label: string; value: number; unit?: string; decimals?: number;
  delta: string; deltaDir: "up" | "down" | "flat";
  deltaTone?: string; spark: number[]; tint?: string; sparkColor?: string;
}) {
  const animated = useCountUp(value, decimals);
  return (
    <div className={`card stat${tint ? " " + tint : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{animated}{unit && <span className="unit">{unit}</span>}</div>
      <div className="stat-foot">
        <span className={`delta ${deltaDir}${deltaTone ? " " + deltaTone : ""}`}>
          {deltaDir === "up" ? "↑" : deltaDir === "down" ? "↓" : "—"} {delta}
        </span>
        <span>vs. last hour</span>
      </div>
      <Sparkline data={spark} color={sparkColor || "var(--accent)"} />
    </div>
  );
}

// ── Score bar ─────────────────────────────────────────────────────────────────
function ScoreBar({ score, threshold, verdict }: { score: number; threshold: number; verdict: string }) {
  const pct = Math.min(100, score * 100);
  const thPct = Math.min(100, threshold * 100);
  const isNG = verdict === "NG";
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-header">
        <span className="score-bar-label">Anomaly score</span>
        <span className={`score-bar-val${isNG ? " ng" : " ok"}`}>{score.toFixed(4)}</span>
      </div>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${pct}%`, background: isNG ? "var(--red)" : "var(--green)" }}
        />
        <div className="score-bar-threshold" style={{ left: `${thPct}%` }}>
          <div className="score-bar-threshold-line" />
          <div className="score-bar-threshold-label">{threshold.toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
}

// ── Pipeline badge ─────────────────────────────────────────────────────────────
const PIPELINE_DISPLAY: Record<string, string> = {
  yolo_clip: "YOLO+CLIP",
  cnn: "CNN",
  patchcore: "PatchCore",
  efficientad: "EfficientAD",
  ensemble: "Ensemble",
};

// ── TopBar ────────────────────────────────────────────────────────────────────
function TopBar({ wsStatus, showStats, showMidrow, onToggleStats, onToggleMidrow }: {
  wsStatus: string; showStats: boolean; showMidrow: boolean;
  onToggleStats: () => void; onToggleMidrow: () => void;
}) {
  const [time, setTime] = useState<Date | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setTime(new Date());
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) {
          const data = await res.json();
          setPipelineStatus(data.pipelines ?? {});
        }
      } catch { /* offline */ }
    };
    fetchHealth();
    const id = setInterval(fetchHealth, 15_000);
    return () => clearInterval(id);
  }, []);

  const hh = time ? String(time.getHours()).padStart(2, "0") : "--";
  const mm = time ? String(time.getMinutes()).padStart(2, "0") : "--";
  const isLive = wsStatus === "connected";

  const PIPELINE_SHORT: Record<string, string> = {
    yolo_clip: "YOLO", cnn: "CNN", patchcore: "PatchCore",
    efficientad: "EfficientAD", ensemble: "Ensemble",
  };

  return (
    <div className="topbar">
      <div className="topbar-left">
        <div className="brand">
          <span className="brand-mark">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3.5" /><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
            </svg>
          </span>
          Vision Inspect
          <span className="brand-sub">Industrial AI v4.0</span>
        </div>
      </div>

      <div className="topbar-center">
        {Object.entries(PIPELINE_SHORT).map(([key, label]) => {
          const loaded = pipelineStatus[key];
          const tone = loaded === undefined ? "grey" : loaded ? "green" : "red";
          return (
            <div key={key} className="sysbadge" title={PIPELINE_DISPLAY[key]}>
              <span className={`dot ${tone}`} />
              <span className="lbl">{label}</span>
            </div>
          );
        })}
      </div>

      <div className="topbar-right">
        <button className={`view-toggle-btn${showStats ? " active" : ""}`} onClick={onToggleStats}>Stats</button>
        <button className={`view-toggle-btn${showMidrow ? " active" : ""}`} onClick={onToggleMidrow}>Panels</button>
        <span className="clock">{hh}:{mm}</span>
        <span className={`live-badge${isLive ? "" : " offline"}`}>
          <span className="dot" />
          {isLive ? "Live" : wsStatus === "connecting" ? "Connecting…" : "Offline"}
        </span>
      </div>
    </div>
  );
}

// ── StatsRow ──────────────────────────────────────────────────────────────────
function StatsRow({ stats }: { stats: { total: number; okRate: number; ngRate: number; defects: number; latency: number } }) {
  return (
    <div className="stats">
      <Stat label="Total inspections" value={stats.total}
        delta="—" deltaDir="flat"
        spark={[12,18,16,22,28,30,26,38,42,36,48,52,Math.min(stats.total,58)]}
        sparkColor="var(--ink-4)" />
      <Stat label="OK rate" value={stats.okRate} decimals={1} unit="%" tint="ok"
        delta={stats.okRate > 0 ? `${stats.okRate.toFixed(1)}%` : "—"} deltaDir="up"
        spark={[96,97,96.5,98,97.5,98.2,98.4,98.1,98.5,98.7,98.6,98.8,stats.okRate||98.8]}
        sparkColor="var(--green)" />
      <Stat label="NG rate" value={stats.ngRate} decimals={1} unit="%" tint="ng"
        delta={stats.ngRate > 0 ? `${stats.ngRate.toFixed(1)}%` : "—"} deltaDir="down" deltaTone="good"
        spark={[3.8,3.4,3.5,2.8,2.5,2.1,1.9,1.5,1.4,1.3,1.4,1.2,stats.ngRate||1.2]}
        sparkColor="var(--red)" />
      <Stat label="Total defects" value={stats.defects}
        delta="—" deltaDir="up" deltaTone="bad"
        spark={[24,28,30,26,32,38,42,40,44,46,48,52,Math.min(stats.defects,58)]}
        sparkColor="var(--red)" />
      <Stat label="Avg latency" value={stats.latency} unit="ms"
        delta="—" deltaDir="down" deltaTone="good"
        spark={[156,148,142,138,140,132,128,124,122,118,114,112,stats.latency||112]}
        sparkColor="var(--ink-4)" />
    </div>
  );
}

// ── CameraPanel ───────────────────────────────────────────────────────────────
function CameraPanel({ scanning, ngFlash, image }: { scanning: boolean; ngFlash: boolean; image: string }) {
  return (
    <div className={`card camera${scanning ? " scanning" : ""}${ngFlash ? " ng" : ""}`}>
      <div className="card-hd">
        <div className="card-hd-title"><span className="blink" />Live camera</div>
        <div className="card-hd-meta">Camera 03 · 1080p</div>
      </div>
      <div className="camera-feed">
        <img className="camera-img" src={image} alt="camera feed" />
        <div className="camera-corners">
          <span className="tl" /><span className="tr" />
          <span className="bl" /><span className="br" />
        </div>
        <div className="scanline" />
        <div className="scan-overlay"><span className="spinner" />Scanning…</div>
        <div className="cam-tag"><span className="dot" />Recording</div>
      </div>
    </div>
  );
}

// ── InspectPanel ──────────────────────────────────────────────────────────────
const PIPELINES = [
  { key: "yolo_clip",   label: "YOLO+CLIP",   desc: "Bounding box detection + CLIP classification" },
  { key: "cnn",         label: "CNN",          desc: "ResNet-18 binary classifier" },
  { key: "patchcore",   label: "PatchCore",    desc: "Memory-bank anomaly detection" },
  { key: "efficientad", label: "EfficientAD",  desc: "Student-teacher anomaly detection" },
  { key: "ensemble",    label: "Ensemble",     desc: "Fail-safe combination (any NG = NG)" },
];

function InspectPanel({ pipeline, setPipeline, scanning, filename, lastResult, onFileChange }: {
  pipeline: string; setPipeline: (p: string) => void;
  scanning: boolean; filename: string;
  lastResult: { verdict: string; score: number; defects: number; ms: number } | null;
  onFileChange: (file: File) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  return (
    <div className="card inspect-panel">
      <div className="card-hd">
        <div className="card-hd-title">Inspect</div>
        <div className="card-hd-meta">Manual mode</div>
      </div>
      <div className="inspect-body">
        <div
          className={`dropzone${isDragging ? " drag-over" : ""}${scanning ? " scanning" : ""}`}
          onClick={() => !scanning && fileRef.current?.click()}
          onDragEnter={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
          onDrop={(e) => {
            e.preventDefault(); setIsDragging(false);
            const f = e.dataTransfer.files[0];
            if (f && !scanning) onFileChange(f);
          }}
        >
          <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onFileChange(f); }} />
          <svg className="icon" width="22" height="22" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          {filename ? (
            <><div className="filename">{filename}</div><span className="hint">Drop a new image to re-inspect</span></>
          ) : (
            <><div style={{ fontWeight: 600, color: "var(--ink-1)" }}>Drop image to inspect</div>
            <span className="hint">Auto-inspects on drop · JPG · PNG · TIFF</span></>
          )}
          {scanning && <div className="dropzone-scanning-overlay"><span className="spinner" />Analyzing…</div>}
        </div>

        <div>
          <div className="field-label">Pipeline</div>
          <div className="pipeline">
            {PIPELINES.map((p) => (
              <button key={p.key} className={`pill${pipeline === p.key ? " active" : ""}`}
                onClick={() => setPipeline(p.key)} title={p.desc}>
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {lastResult && (
          <div className={`verdict${lastResult.verdict === "OK" ? " ok" : " ng"}`}>
            <div className="verdict-badge">{lastResult.verdict}</div>
            <div className="verdict-meta">
              <div className="row"><span className="k">Score</span><span className="v">{lastResult.score.toFixed(3)}</span></div>
              <div className="row"><span className="k">Defects</span><span className="v">{lastResult.defects}</span></div>
              <div className="row"><span className="k">Time</span><span className="v">{lastResult.ms}ms</span></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── FeedPanel ─────────────────────────────────────────────────────────────────
function FeedPanel({ results }: { results: InspectionResult[] }) {
  const feed = results.slice(0, 8);
  return (
    <div className="card feed-panel">
      <div className="card-hd">
        <div className="card-hd-title"><span className="blink" />Live feed</div>
        <div className="card-hd-meta">Last {feed.length || "—"}</div>
      </div>
      <div className="feed-list">
        {feed.length === 0 ? (
          <div style={{ padding: "24px 16px", textAlign: "center", color: "var(--ink-4)", fontSize: 12 }}>
            Waiting for results…
          </div>
        ) : feed.map((r, i) => {
          const ts = new Date(r.timestamp).toLocaleTimeString("en-US", {
            hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit",
          });
          const label = PIPELINE_DISPLAY[r.pipeline] ?? r.pipeline ?? "—";
          return (
            <div key={r.id || i} className={`feed-row${r.verdict === "NG" ? " ng" : ""}`}>
              <span className="ts">{ts}</span>
              <span className={`vbadge${r.verdict === "OK" ? " ok" : " ng"}`}>{r.verdict}</span>
              <span className="meta">
                <span className="pipe">{label}</span>
                {r.overall_score > 0 && (
                  <><span className="sep">·</span>
                  <span className={r.verdict === "NG" ? "defs" : "score-ok"}>{r.overall_score.toFixed(3)}</span></>
                )}
              </span>
              <span className="ms">{Math.round(r.processing_ms ?? 0)} ms</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── ResultViewer ──────────────────────────────────────────────────────────────
function ResultViewer({ image, heatmapUrl, defects, anomalyScores, verdict, score, threshold, pipeline, explanation, history, activeHistoryId, onHistorySelect }: {
  image: string; heatmapUrl: string | null;
  defects: Defect[]; anomalyScores: AnomalyScore[];
  verdict: "OK" | "NG" | "REVIEW" | null;
  score: number; threshold: number; pipeline: string;
  explanation: string;
  history: HistoryItem[];
  activeHistoryId: string | null;
  onHistorySelect: (item: HistoryItem) => void;
}) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const displayImg = showHeatmap && heatmapUrl ? heatmapUrl : image;
  const isAnomalyPipeline = pipeline === "efficientad" || pipeline === "patchcore";

  // Reset heatmap toggle when result changes
  useEffect(() => { setShowHeatmap(false); }, [image]);

  return (
    <div className="card bbox-card bbox-row">
      <div className="card-hd">
        <div className="card-hd-title">
          {verdict === null ? "Result viewer" : verdict === "OK" ? "Result · Pass" : `Result · ${isAnomalyPipeline ? "Anomaly detected" : `${defects.length} detection${defects.length !== 1 ? "s" : ""}`}`}
          {pipeline && <span className="pipeline-chip">{PIPELINE_DISPLAY[pipeline] ?? pipeline}</span>}
        </div>
        <div className="card-hd-meta" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {heatmapUrl && (
            <button className={`heatmap-toggle${showHeatmap ? " active" : ""}`} onClick={() => setShowHeatmap(v => !v)}>
              {showHeatmap ? "Original" : "Heatmap"}
            </button>
          )}
          {verdict === null ? "Awaiting inspection"
            : verdict === "OK" ? "Last frame · pass"
            : "Last frame · fail"}
        </div>
      </div>

      {history.length > 0 && (
        <div className="history-rail">
          <div className="history-rail-label">History</div>
          <div className="history-strip">
            {history.map((item, i) => {
              const isNG = item.result.verdict === "NG";
              const isActive = item.result.id === activeHistoryId;
              const ts = new Date(item.result.timestamp).toLocaleTimeString("en-US", {
                hour12: false, hour: "2-digit", minute: "2-digit",
              });
              return (
                <button
                  key={item.result.id || i}
                  className={`history-thumb${isNG ? " ng" : " ok"}${isActive ? " active" : ""}`}
                  onClick={() => onHistorySelect(item)}
                  title={`${item.result.verdict} · ${PIPELINE_DISPLAY[item.result.pipeline] ?? item.result.pipeline} · ${ts}`}
                >
                  <img src={item.imageUrl} alt="history" />
                  <span className={`history-badge${isNG ? " ng" : " ok"}`}>{item.result.verdict}</span>
                  <span className="history-meta">{ts}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="bbox-body">
        {/* Image viewer */}
        <div className="bbox-view">
          <img src={displayImg} alt="inspection frame" />
          {verdict === null ? (
            <div className="bbox-empty" style={{ background: "rgba(0,0,0,0.25)" }}>
              <div className="lg" style={{ opacity: 0.45, fontSize: "0.9rem", fontWeight: 400 }}>Upload an image to inspect</div>
            </div>
          ) : verdict === "OK" && !isAnomalyPipeline ? (
            <div className="bbox-empty">
              <div className="check">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </div>
              <div className="lg">No defects detected</div>
              <div className="sm">Surface integrity verified</div>
            </div>
          ) : verdict === "NG" && !isAnomalyPipeline ? (
            defects.map((d, i) => {
              const x1 = d.bbox_x1 ?? d.x1 ?? 0;
              const y1 = d.bbox_y1 ?? d.y1 ?? 0;
              const x2 = d.bbox_x2 ?? d.x2 ?? 1;
              const y2 = d.bbox_y2 ?? d.y2 ?? 1;
              const isCrit = d.confidence > 0.85;
              return (
                <div key={i} className={`bbox${isCrit ? " crit" : ""}`}
                  style={{
                    left: `${(x1 * 100).toFixed(1)}%`, top: `${(y1 * 100).toFixed(1)}%`,
                    width: `${((x2 - x1) * 100).toFixed(1)}%`, height: `${((y2 - y1) * 100).toFixed(1)}%`,
                    animationDelay: `${i * 80}ms`,
                  }}>
                  <div className="chip">
                    <span>{d.defect_class || d.defect_label || d.clip_label || "DEFECT"}</span>
                    <span className="conf">{(d.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              );
            })
          ) : null}
        </div>

        {/* Sidebar */}
        <div className="bbox-side">
          {/* Score bar — always shown when we have a verdict */}
          {verdict !== null && (
            <div className="bbox-side-score">
              <ScoreBar score={score} threshold={threshold} verdict={verdict} />
            </div>
          )}

          {isAnomalyPipeline ? (
            <>
              <div className="bbox-side-hd">
                <span>Model scores</span>
                <span className={`count${verdict === "OK" ? " ok" : ""}`}>{anomalyScores.length}</span>
              </div>
              <div className="bbox-side-list">
                {anomalyScores.length === 0 ? (
                  <div className="empty-defects">{verdict === null ? "No image loaded." : "No anomaly data."}</div>
                ) : anomalyScores.map((a, i) => (
                  <div key={i} className={`defect-item${!a.passed ? " crit" : ""}`}>
                    <div>
                      <div className="name"><span className="dot" />{a.model_name}</div>
                      <div className="sub" style={{ marginLeft: 16 }}>
                        threshold {a.threshold?.toFixed(3)} · {a.passed ? "passed" : "failed"}
                      </div>
                    </div>
                    <span className={`conf${!a.passed ? " ng" : ""}`}>{a.score.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <>
              <div className="bbox-side-hd">
                <span>Detections</span>
                <span className={`count${verdict === "OK" ? " ok" : ""}`}>{verdict === "NG" ? defects.length : 0}</span>
              </div>
              <div className="bbox-side-list">
                {verdict !== "NG" ? (
                  <div className="empty-defects">{verdict === null ? "No image loaded." : "No detections in this frame."}</div>
                ) : defects.map((d, i) => {
                  const isCrit = d.confidence > 0.85;
                  return (
                    <div key={i} className={`defect-item${isCrit ? " crit" : ""}`}>
                      <div>
                        <div className="name"><span className="dot" />{d.defect_class || d.defect_label || "Unknown"}</div>
                        <div className="sub" style={{ marginLeft: 16 }}>
                          {isCrit ? "Critical" : "Moderate"} · {d.detection_type ?? "defect"}
                        </div>
                      </div>
                      <span className="conf">{(d.confidence * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {/* VLM explanation */}
          {explanation && (
            <div className="explanation-block">
              <div className="explanation-hd">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
                </svg>
                VLM Analysis
              </div>
              <p className="explanation-text">{explanation}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── HistoryGallery ────────────────────────────────────────────────────────────
interface HistoryItem {
  imageUrl: string;
  result: InspectionResult;
}

function HistoryGallery({ items, activeId, onSelect }: {
  items: HistoryItem[];
  activeId: string | null;
  onSelect: (item: HistoryItem) => void;
}) {
  if (items.length === 0) return null;

  return (
    <div className="history-rail">
      <div className="history-rail-label">History</div>
      <div className="history-strip">
        {items.map((item, i) => {
          const isNG = item.result.verdict === "NG";
          const isActive = item.result.id === activeId;
          const ts = new Date(item.result.timestamp).toLocaleTimeString("en-US", {
            hour12: false, hour: "2-digit", minute: "2-digit",
          });
          const pipeLabel = PIPELINE_DISPLAY[item.result.pipeline] ?? item.result.pipeline ?? "—";
          return (
            <button
              key={item.result.id || i}
              className={`history-thumb${isNG ? " ng" : " ok"}${isActive ? " active" : ""}`}
              onClick={() => onSelect(item)}
              title={`${item.result.verdict} · ${pipeLabel} · ${ts}`}
            >
              <img src={item.imageUrl} alt="history frame" />
              <span className={`history-badge${isNG ? " ng" : " ok"}`}>{item.result.verdict}</span>
              <span className="history-meta">{ts}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
interface BackendStats {
  total_inspections: number; ok_rate: number; ng_rate: number;
  total_defects: number; avg_processing_ms: number;
}

export default function HomePage() {
  const { results, connectionStatus } = useInspection();

  const [backendStats, setBackendStats] = useState<BackendStats | null>(null);
  const [showStats, setShowStats] = useState(true);
  const [showMidrow, setShowMidrow] = useState(true);
  const [pipeline, setPipeline] = useState("yolo_clip");
  const pipelineRef = useRef("yolo_clip");
  const [scanning, setScanning] = useState(false);
  const [ngFlash, setNgFlash] = useState(false);
  const [filename, setFilename] = useState("");
  const [cameraImage, setCameraImage] = useState(PLACEHOLDER_IMG);
  const [lastResult, setLastResult] = useState<InspectionResult | null>(null);
  const [activeImageUrl, setActiveImageUrl] = useState(PLACEHOLDER_IMG);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_URL}/statistics`);
        if (res.ok) setBackendStats(await res.json());
      } catch { /* offline */ }
    };
    poll();
    const id = setInterval(poll, 10_000);
    return () => clearInterval(id);
  }, []);

  // Sync WS results → lastResult
  const wsResult = results[0];
  useEffect(() => {
    if (!wsResult) return;
    setLastResult(wsResult);
    if (wsResult.verdict === "NG") {
      setNgFlash(true);
      setTimeout(() => setNgFlash(false), 3000);
    }
  }, [wsResult]);

  const runInspect = useCallback(async (file: File, localUrl: string) => {
    if (scanning) return;
    setError(null);
    setScanning(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("pipeline_name", pipelineRef.current);
      const res = await fetch(`${API_URL}/inspect`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const raw = await res.json();
      const normalized: InspectionResult = {
        id: raw.inspection_id ?? raw.id ?? String(Date.now()),
        timestamp: raw.timestamp ?? new Date().toISOString(),
        verdict: raw.verdict,
        pipeline: raw.pipeline_name ?? raw.pipeline ?? "",
        defects: raw.detections ?? raw.defects ?? [],
        anomaly_scores: raw.anomaly_scores ?? [],
        overall_score: raw.overall_score ?? 0,
        threshold: raw.threshold ?? 0.5,
        processing_ms: raw.processing_ms ?? 0,
        total_defects: raw.total_defects ?? 0,
        image_path: raw.image_path,
        heatmap_path: raw.heatmap_path,
        explanation: raw.report?.summary ?? "",
      };
      setLastResult(normalized);
      setActiveImageUrl(localUrl);
      setHistory(prev => [{ imageUrl: localUrl, result: normalized }, ...prev].slice(0, 50));
      if (normalized.verdict === "NG") {
        setNgFlash(true);
        setTimeout(() => setNgFlash(false), 3000);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inspection failed");
    } finally {
      setScanning(false);
    }
  }, [scanning]);

  const handleFileChange = useCallback((file: File) => {
    const localUrl = URL.createObjectURL(file);
    setFilename(file.name);
    setCameraImage(localUrl);
    setActiveImageUrl(localUrl);
    setLastResult(null);
    setError(null);
    runInspect(file, localUrl);
  }, [runInspect]);

  useEffect(() => { pipelineRef.current = pipeline; }, [pipeline]);

  const stats = {
    total: backendStats?.total_inspections ?? 0,
    okRate: backendStats?.ok_rate ?? 0,
    ngRate: backendStats?.ng_rate ?? 0,
    defects: backendStats?.total_defects ?? 0,
    latency: Math.round(backendStats?.avg_processing_ms ?? 0),
  };

  const bboxDefects = (lastResult?.defects ?? []).filter(d => d.is_defect !== false);
  const bboxVerdict = lastResult ? (lastResult.verdict as "OK" | "NG" | "REVIEW") : null;
  const heatmapUrl = lastResult?.heatmap_path ? `${API_URL}/${lastResult.heatmap_path}` : null;

  const panelLastResult = lastResult ? {
    verdict: lastResult.verdict,
    score: lastResult.overall_score,
    defects: lastResult.total_defects,
    ms: Math.round(lastResult.processing_ms),
  } : null;

  const handleHistorySelect = useCallback((item: HistoryItem) => {
    setLastResult(item.result);
    setActiveImageUrl(item.imageUrl);
    setCameraImage(item.imageUrl);
  }, []);

  return (
    <div className={`app${showStats ? "" : " hide-stats"}${showMidrow ? "" : " hide-midrow"}`}>
      <TopBar wsStatus={connectionStatus} showStats={showStats} showMidrow={showMidrow}
        onToggleStats={() => setShowStats(s => !s)} onToggleMidrow={() => setShowMidrow(s => !s)} />

      {showStats && <StatsRow stats={stats} />}

      {showMidrow && (
        <div className="midrow">
          <CameraPanel scanning={scanning} ngFlash={ngFlash} image={cameraImage} />
          <InspectPanel pipeline={pipeline} setPipeline={setPipeline}
            scanning={scanning} filename={filename}
            lastResult={panelLastResult} onFileChange={handleFileChange} />
          <FeedPanel results={results} />
        </div>
      )}

      {error ? (
        <div className="error-bar">{error}</div>
      ) : (
        <ResultViewer
          image={activeImageUrl}
          heatmapUrl={heatmapUrl}
          defects={bboxDefects}
          anomalyScores={lastResult?.anomaly_scores ?? []}
          verdict={bboxVerdict}
          score={lastResult?.overall_score ?? 0}
          threshold={lastResult?.threshold ?? 0.5}
          pipeline={lastResult?.pipeline ?? pipeline}
          explanation={lastResult?.explanation ?? ""}
          history={history}
          activeHistoryId={lastResult?.id ?? null}
          onHistorySelect={handleHistorySelect}
        />
      )}
    </div>
  );
}
