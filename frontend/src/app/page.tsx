"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useInspection, InspectionResult, Defect, AnomalyScore, API_URL } from "@/hooks/useInspection";

// ── Placeholder ────────────────────────────────────────────────────────────────
function makePlaceholder(): string {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'>
    <rect width='400' height='300' fill='#0b0e13'/>
    <text x='200' y='158' text-anchor='middle' fill='rgba(255,255,255,0.12)' font-size='13' font-family='monospace'>No image loaded</text>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}
const PLACEHOLDER = makePlaceholder();

// ── SVG Icons ──────────────────────────────────────────────────────────────────
function IconSettings({ size = 15 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9"/>
    </svg>
  );
}
function IconUpload({ size = 22 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}
function IconCheck({ size = 18 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
}
function IconInfo({ size = 12 }: { size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="16" x2="12" y2="12"/>
      <line x1="12" y1="8" x2="12.01" y2="8"/>
    </svg>
  );
}

// ── Pipeline config ────────────────────────────────────────────────────────────
const PIPELINES = [
  { key: "yolo_clip",   label: "YOLO+CLIP" },
  { key: "cnn",         label: "CNN"        },
  { key: "patchcore",   label: "PatchCore"  },
  { key: "efficientad", label: "EfficientAD"},
  { key: "ensemble",    label: "Ensemble", full: true },
];

const PIPELINE_DISPLAY: Record<string, string> = {
  yolo_clip: "YOLO+CLIP", cnn: "CNN", patchcore: "PatchCore",
  efficientad: "EfficientAD", ensemble: "Ensemble",
};

// ── Clock hook ─────────────────────────────────────────────────────────────────
function useClock() {
  const [t, setT] = useState<Date | null>(null);
  useEffect(() => {
    setT(new Date());
    const id = setInterval(() => setT(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return t;
}

// ── History item ───────────────────────────────────────────────────────────────
interface HistoryItem {
  imageUrl: string;
  result: InspectionResult;
}

// ── TopBar ─────────────────────────────────────────────────────────────────────
function TopBar({ wsStatus }: { wsStatus: string }) {
  const time = useClock();
  const [pipeStatus, setPipeStatus] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const fetch_ = async () => {
      try {
        const r = await fetch(`${API_URL}/health`);
        if (r.ok) { const d = await r.json(); setPipeStatus(d.pipelines ?? {}); }
      } catch { /* offline */ }
    };
    fetch_();
    const id = setInterval(fetch_, 15_000);
    return () => clearInterval(id);
  }, []);

  const hh = time ? String(time.getHours()).padStart(2, "0") : "--";
  const mm = time ? String(time.getMinutes()).padStart(2, "0") : "--";
  const ss = time ? String(time.getSeconds()).padStart(2, "0") : "--";
  const isLive = wsStatus === "connected";

  const PIPE_SHORT: Record<string, string> = {
    yolo_clip: "YOLO+CLIP", cnn: "CNN", patchcore: "PatchCore",
    efficientad: "EfficientAD", ensemble: "Ensemble",
  };

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">VI</div>
        <span className="brand-name">Vision Inspect</span>
        <span className="brand-sub">v4.0</span>
      </div>

      <div className="pipes">
        {Object.entries(PIPE_SHORT).map(([key, label]) => {
          const loaded = pipeStatus[key];
          const tone = loaded === undefined ? "grey" : loaded ? "green" : "red";
          return (
            <div key={key} className="pipe-badge">
              <span className={`dot ${tone}`} />
              {label}
            </div>
          );
        })}
      </div>

      <div className="topright">
        <div className="clock">
          {hh}:{mm}<span className="sec">:{ss}</span>
        </div>
        <div className={`live-chip${isLive ? "" : " offline"}`}>
          <span className="pulse" />
          {isLive ? "Live" : wsStatus === "connecting" ? "Conn…" : "Offline"}
        </div>
        <button className="icon-btn" title="Settings">
          <IconSettings size={15} />
        </button>
      </div>
    </header>
  );
}

// ── LiveCamPanel ───────────────────────────────────────────────────────────────
function LiveCamPanel({ image, scanning }: { image: string; scanning: boolean }) {
  const time = useClock();
  const hh = time ? String(time.getHours()).padStart(2, "0") : "--";
  const mm = time ? String(time.getMinutes()).padStart(2, "0") : "--";
  const ss = time ? String(time.getSeconds()).padStart(2, "0") : "--";

  return (
    <section className="panel">
      <div className="panel-hd">
        <div className="panel-title">Live Camera</div>
        <div className="panel-meta">CAM-03 · 1080p</div>
      </div>
      <div className="panel-body" style={{ padding: "10px 14px 14px" }}>
        <div className={`livecam${scanning ? " scanning" : ""}`}>
          <img src={image} alt="live camera feed" />
          <div className="rec"><span className="rdot" />REC</div>
          <div className="ts">{hh}:{mm}:{ss}</div>
        </div>
      </div>
    </section>
  );
}

// ── InspectPanel ───────────────────────────────────────────────────────────────
function InspectPanel({
  pipeline, setPipeline, scanning, filename, lastResult, onFileChange,
}: {
  pipeline: string;
  setPipeline: (p: string) => void;
  scanning: boolean;
  filename: string;
  lastResult: { verdict: string; score: number; defects: number; ms: number } | null;
  onFileChange: (f: File) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  return (
    <section className="panel inspect-panel">
      <div className="panel-hd">
        <div className="panel-title">Inspect</div>
        <div className="panel-meta">Drop · Auto-run</div>
      </div>
      <div className="inspect-inner">
        {/* Drop zone */}
        <div
          className={`drop${drag ? " drag-over" : ""}${scanning ? " scanning" : ""}`}
          onClick={() => !scanning && fileRef.current?.click()}
          onDragEnter={e => { e.preventDefault(); setDrag(true); }}
          onDragOver={e => { e.preventDefault(); setDrag(true); }}
          onDragLeave={e => { e.preventDefault(); setDrag(false); }}
          onDrop={e => {
            e.preventDefault(); setDrag(false);
            const f = e.dataTransfer.files[0];
            if (f && !scanning) onFileChange(f);
          }}
        >
          <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
            onChange={e => { const f = e.target.files?.[0]; if (f) onFileChange(f); }} />
          <div className="drop-icon"><IconUpload size={22} /></div>
          {filename ? (
            <>
              <div className="drop-title" style={{ fontSize: 11.5, wordBreak: "break-all" }}>{filename}</div>
              <div className="drop-sub">Drop to re-inspect</div>
            </>
          ) : (
            <>
              <div className="drop-title">Drop image to inspect</div>
              <div className="drop-sub">JPG · PNG · TIFF</div>
            </>
          )}
          {scanning && (
            <div className="drop-scanning"><span className="spin" />Analyzing…</div>
          )}
        </div>

        {/* Pipeline */}
        <div className="field-label" style={{ margin: 0 }}>Pipeline</div>
        <div className="pipeline-grid">
          {PIPELINES.map(p => (
            <button
              key={p.key}
              className={`pipe-pill${pipeline === p.key ? " active" : ""}${p.full ? " full" : ""}`}
              onClick={() => setPipeline(p.key)}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Last result mini */}
        {lastResult && (
          <div className={`verdict-mini${lastResult.verdict === "OK" ? " ok" : lastResult.verdict === "NG" ? " ng" : " idle"}`}>
            <span className={`vtag${lastResult.verdict === "OK" ? " ok" : lastResult.verdict === "NG" ? " ng" : " idle"}`}>
              {lastResult.verdict === "OK" ? "PASS" : lastResult.verdict === "NG" ? "FAIL" : lastResult.verdict}
            </span>
            <div className="verdict-mini-meta">
              <span className="v">{lastResult.score.toFixed(3)} · {lastResult.defects} det · {lastResult.ms}ms</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

// ── ResultViewer ───────────────────────────────────────────────────────────────
function ResultViewer({
  image, heatmapUrl, defects, anomalyScores, verdict, score, threshold,
  pipeline, explanation, showBoxes, setShowBoxes, history, activeHistoryId, onHistorySelect,
}: {
  image: string;
  heatmapUrl: string | null;
  defects: Defect[];
  anomalyScores: AnomalyScore[];
  verdict: "OK" | "NG" | "REVIEW" | null;
  score: number;
  threshold: number;
  pipeline: string;
  explanation: string;
  showBoxes: boolean;
  setShowBoxes: (v: boolean) => void;
  history: HistoryItem[];
  activeHistoryId: string | null;
  onHistorySelect: (item: HistoryItem) => void;
}) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  useEffect(() => { setShowHeatmap(false); }, [image]);

  const displayImg = showHeatmap && heatmapUrl ? heatmapUrl : image;
  const isAnomaly = pipeline === "efficientad" || pipeline === "patchcore";
  const isNG = verdict === "NG";
  const isOK = verdict === "OK";
  const pipeLabel = PIPELINE_DISPLAY[pipeline] ?? pipeline ?? "—";

  // Active history ID for frame label
  const activeItem = history.find(h => h.result.id === activeHistoryId);
  const frameId = activeItem?.result.id?.slice(-8).toUpperCase() ?? "—";
  const frameTs = activeItem
    ? new Date(activeItem.result.timestamp).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "—";

  return (
    <section className="viewer">
      {/* Header */}
      <div className="viewer-hd">
        <div className="vhd-left">
          <span className="vhd-title">Result Viewer</span>
          <span className={`vhd-verdict${isNG ? " ng" : isOK ? " ok" : " idle"}`}>
            <span className="d" />
            {verdict === null ? "Awaiting" : verdict === "OK" ? "Pass" : verdict === "NG" ? "Fail" : verdict}
          </span>
          {verdict !== null && <span className="vhd-chip">{pipeLabel}</span>}
          {verdict !== null && (
            <span className="vhd-id">{frameId} · {frameTs}</span>
          )}
        </div>
        <div className="toolbar">
          <button className={`tool-btn${showBoxes ? " active" : ""}`} onClick={() => setShowBoxes(!showBoxes)}>
            Boxes
          </button>
          {heatmapUrl && (
            <button className={`tool-btn${showHeatmap ? " active" : ""}`} onClick={() => setShowHeatmap(v => !v)}>
              Heatmap
            </button>
          )}
        </div>
      </div>

      {/* Body */}
      <div className="viewer-body">
        {/* Stage */}
        <div className="stage">
          <div className="stage-frame">
            <img src={displayImg} alt="inspection frame" />

            {/* Bounding boxes */}
            {showBoxes && !isAnomaly && verdict === "NG" && defects.map((d, i) => {
              const x1 = d.bbox_x1 ?? d.x1 ?? 0;
              const y1 = d.bbox_y1 ?? d.y1 ?? 0;
              const x2 = d.bbox_x2 ?? d.x2 ?? 1;
              const y2 = d.bbox_y2 ?? d.y2 ?? 1;
              const isCrit = d.confidence > 0.75;
              return (
                <div key={i} className={`bbox${isCrit ? "" : " warn"}`} style={{
                  left: `${(x1 * 100).toFixed(2)}%`,
                  top: `${(y1 * 100).toFixed(2)}%`,
                  width: `${((x2 - x1) * 100).toFixed(2)}%`,
                  height: `${((y2 - y1) * 100).toFixed(2)}%`,
                  animationDelay: `${i * 70}ms`,
                }}>
                  <div className="bbox-label">
                    #{i + 1} {d.defect_class || d.defect_label || d.clip_label || "Defect"} · {(d.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              );
            })}

            {/* OK overlay */}
            {verdict === "OK" && !isAnomaly && (
              <div className="stage-overlay">
                <div className="ok-ring"><IconCheck size={24} /></div>
                <div className="ok-label">No defects detected</div>
                <div className="ok-sub">Surface integrity verified</div>
              </div>
            )}
          </div>
        </div>

        {/* Side panel */}
        <aside className="side">
          {/* Anomaly score */}
          <section className="side-sec">
            <div className="score-head">
              <div className="score-label">Anomaly Score</div>
              <div className={`score-num${isNG ? " ng" : isOK ? " ok" : " idle"}`}>
                {verdict !== null ? score.toFixed(3) : "—"}
              </div>
            </div>
            <div className="score-track">
              {verdict !== null && (
                <div
                  className={`seg${isNG ? " ng" : " ok"}`}
                  style={{ width: `${Math.min(100, score * 100).toFixed(1)}%` }}
                />
              )}
              <div className="thresh" style={{ left: `${(threshold * 100).toFixed(1)}%` }} />
            </div>
            <div className="score-foot">
              <span>0.000</span>
              <span>Threshold {threshold.toFixed(3)}</span>
              <span>1.000</span>
            </div>
          </section>

          {/* Detections / model scores */}
          <section className="side-sec">
            <div className="det-head">
              <div className="score-label">
                {isAnomaly ? "Model Scores" : "Detections"}
              </div>
              <div className="det-count">
                {isAnomaly
                  ? `${anomalyScores.length} model${anomalyScores.length !== 1 ? "s" : ""}`
                  : `${defects.filter(d => d.is_defect !== false).length} found`
                }
              </div>
            </div>

            {isAnomaly ? (
              <div className="det-list">
                {anomalyScores.length === 0 ? (
                  <div className="det-empty">
                    <div className="ok-icon"><IconCheck size={16} /></div>
                    No anomaly data
                  </div>
                ) : anomalyScores.map((a, i) => (
                  <div key={i} className={`det-row${!a.passed ? "" : " warn"}`}>
                    <div className="det-id">{i + 1}</div>
                    <div>
                      <div className="det-name">{a.model_name}</div>
                      <div className="det-meta">threshold {a.threshold?.toFixed(3)} · {a.passed ? "passed" : "failed"}</div>
                    </div>
                    <div className={`det-conf${!a.passed ? "" : ""}`} style={{ color: a.passed ? "var(--green)" : "var(--red)" }}>
                      {a.score.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="det-list">
                {defects.filter(d => d.is_defect !== false).length === 0 ? (
                  <div className="det-empty">
                    {verdict === "OK" ? (
                      <>
                        <div className="ok-icon"><IconCheck size={16} /></div>
                        No defects detected
                      </>
                    ) : (
                      <span style={{ color: "var(--ink-4)" }}>
                        {verdict === null ? "No image loaded." : "Awaiting inspection."}
                      </span>
                    )}
                  </div>
                ) : defects.filter(d => d.is_defect !== false).map((d, i) => {
                  const isCrit = d.confidence > 0.75;
                  return (
                    <div key={i} className={`det-row${isCrit ? "" : " warn"}`}>
                      <div className="det-id">{i + 1}</div>
                      <div>
                        <div className="det-name">{d.defect_class || d.defect_label || "Defect"}</div>
                        <div className="det-meta">
                          {isCrit ? "Critical" : "Moderate"} · {d.detection_type ?? "defect"}
                        </div>
                      </div>
                      <div className="det-conf">{(d.confidence * 100).toFixed(0)}%</div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          {/* VLM */}
          {(explanation || verdict !== null) && (
            <section className="side-sec">
              <div className="vlm-box">
                <div className="vlm-hd">
                  <IconInfo size={12} />
                  VLM Analysis
                </div>
                <p className="vlm-text">
                  {explanation || (verdict === "OK"
                    ? "No defects detected. Surface integrity verified."
                    : "Awaiting analysis…")}
                </p>
              </div>
            </section>
          )}
        </aside>
      </div>

      {/* History strip */}
      {history.length > 0 && (
        <div className="history">
          <div className="history-title">History</div>
          <div className="history-strip">
            {history.map((item, i) => {
              const isItemNG = item.result.verdict === "NG";
              const isActive = item.result.id === activeHistoryId;
              const ts = new Date(item.result.timestamp).toLocaleTimeString("en-US", {
                hour12: false, hour: "2-digit", minute: "2-digit",
              });
              return (
                <button
                  key={item.result.id || i}
                  className={`hthumb${isItemNG ? " ng" : ""}${isActive ? " active" : ""}`}
                  onClick={() => onHistorySelect(item)}
                  title={`${item.result.verdict} · ${PIPELINE_DISPLAY[item.result.pipeline] ?? item.result.pipeline} · ${ts}`}
                >
                  <img src={item.imageUrl} alt="history" />
                  <span className={`badge ${isItemNG ? "ng" : "ok"}`}>{item.result.verdict}</span>
                  <span className="ts">{ts}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

// ── Main ───────────────────────────────────────────────────────────────────────
export default function HomePage() {
  const { results, connectionStatus } = useInspection();

  const [pipeline, setPipeline] = useState("yolo_clip");
  const pipelineRef = useRef("yolo_clip");
  const [scanning, setScanning] = useState(false);
  const [filename, setFilename] = useState("");
  const [cameraImage, setCameraImage] = useState(PLACEHOLDER);
  const [activeImageUrl, setActiveImageUrl] = useState(PLACEHOLDER);
  const [lastResult, setLastResult] = useState<InspectionResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showBoxes, setShowBoxes] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Sync WS → lastResult (for external uploads / camera events)
  const wsResult = results[0];
  useEffect(() => {
    if (!wsResult) return;
    // Only sync WS results if they don't match the current manual inspection
    // (avoid clobbering manual result with duplicate ws broadcast)
    setLastResult(prev => {
      if (prev?.id === wsResult.id) return prev;
      return wsResult;
    });
  }, [wsResult]);

  useEffect(() => { pipelineRef.current = pipeline; }, [pipeline]);

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

  const handleHistorySelect = useCallback((item: HistoryItem) => {
    setLastResult(item.result);
    setActiveImageUrl(item.imageUrl);
    setCameraImage(item.imageUrl);
  }, []);

  const bboxDefects = (lastResult?.defects ?? []).filter(d => d.is_defect !== false);
  const bboxVerdict = lastResult ? (lastResult.verdict as "OK" | "NG" | "REVIEW") : null;
  const heatmapUrl = lastResult?.heatmap_path ? `${API_URL}/${lastResult.heatmap_path}` : null;

  const panelLastResult = lastResult ? {
    verdict: lastResult.verdict,
    score: lastResult.overall_score,
    defects: lastResult.total_defects,
    ms: Math.round(lastResult.processing_ms),
  } : null;

  return (
    <div className="app">
      <TopBar wsStatus={connectionStatus} />

      <div className="main">
        {/* Left rail */}
        <aside className="rail">
          <LiveCamPanel image={cameraImage} scanning={scanning} />
          <InspectPanel
            pipeline={pipeline}
            setPipeline={setPipeline}
            scanning={scanning}
            filename={filename}
            lastResult={panelLastResult}
            onFileChange={handleFileChange}
          />
        </aside>

        {/* Result viewer */}
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
          showBoxes={showBoxes}
          setShowBoxes={setShowBoxes}
          history={history}
          activeHistoryId={lastResult?.id ?? null}
          onHistorySelect={handleHistorySelect}
        />
      </div>

      {error && <div className="error-bar">{error}</div>}
    </div>
  );
}
