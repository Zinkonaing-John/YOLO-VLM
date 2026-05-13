"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useInspection, InspectionResult, Defect } from "@/hooks/useInspection";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Procedural PCB placeholder image ────────────────────────────────────────
function makePCB(seed: number): string {
  const r = (n: number) => {
    const x = Math.sin(seed * 9301 + n * 49297) * 233280;
    return x - Math.floor(x);
  };
  const traces = Array.from({ length: 18 }, (_, i) =>
    `<line x1="${r(i + 100) * 80}" y1="${30 + i * 30 + r(i) * 8}" x2="${720 - r(i + 200) * 80}" y2="${30 + i * 30 + r(i) * 8}" stroke="rgba(180,140,80,0.45)" stroke-width="${1 + r(i + 50) * 1.5}"/>`
  ).join("");
  const vias = Array.from({ length: 40 }, (_, i) =>
    `<circle cx="${30 + r(i + 300) * 720}" cy="${30 + r(i + 400) * 480}" r="${2 + r(i + 500) * 2.5}" fill="rgba(140,130,120,0.5)" stroke="rgba(60,55,50,0.7)" stroke-width="0.6"/>`
  ).join("");
  const ics = Array.from({ length: 4 }, (_, i) => {
    const w = 60 + r(i + 600) * 80;
    const h = 30 + r(i + 700) * 30;
    const x = 50 + r(i + 800) * 600;
    const y = 60 + r(i + 900) * 400;
    return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="rgba(30,28,25,0.85)" stroke="rgba(80,75,70,0.7)" stroke-width="0.8"/>`;
  }).join("");
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 750 540' preserveAspectRatio='xMidYMid slice'>
    <defs>
      <radialGradient id='bg'><stop offset='0%' stop-color='#1a2a1a'/><stop offset='100%' stop-color='#0c1410'/></radialGradient>
      <pattern id='g' width='10' height='10' patternUnits='userSpaceOnUse'>
        <path d='M10 0H0V10' fill='none' stroke='rgba(80,90,70,0.08)' stroke-width='0.5'/>
      </pattern>
    </defs>
    <rect width='750' height='540' fill='url(#bg)'/><rect width='750' height='540' fill='url(#g)'/>
    ${traces}${ics}${vias}
    <rect x='1' y='1' width='748' height='538' fill='none' stroke='rgba(245,158,11,0.08)' stroke-width='1'/>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

const PLACEHOLDER_IMG = makePCB(3);

// ── Sparkline ────────────────────────────────────────────────────────────────
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

// ── useCountUp ───────────────────────────────────────────────────────────────
function useCountUp(target: number, decimals = 0) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let raf: number;
    const tStart = performance.now() + 80;
    const tick = (now: number) => {
      if (now < tStart) { raf = requestAnimationFrame(tick); return; }
      const p = Math.min(1, (now - tStart) / 900);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(target * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);
  return decimals > 0 ? val.toFixed(decimals) : Math.round(val).toLocaleString();
}

// ── Stat card ────────────────────────────────────────────────────────────────
function Stat({
  label, value, unit, decimals, delta, deltaDir, deltaTone, spark, tint, sparkColor,
}: {
  label: string; value: number; unit?: string; decimals?: number;
  delta: string; deltaDir: "up" | "down" | "flat";
  deltaTone?: string; spark: number[]; tint?: string; sparkColor?: string;
}) {
  const animated = useCountUp(value, decimals);
  return (
    <div className={`card stat${tint ? " " + tint : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {animated}
        {unit && <span className="unit">{unit}</span>}
      </div>
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

// ── TopBar ───────────────────────────────────────────────────────────────────
function TopBar({ wsStatus }: { wsStatus: string }) {
  const [time, setTime] = useState<Date | null>(null);
  useEffect(() => {
    setTime(new Date());
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const hh = time ? String(time.getHours()).padStart(2, "0") : "--";
  const mm = time ? String(time.getMinutes()).padStart(2, "0") : "--";
  const isLive = wsStatus === "connected";

  return (
    <div className="topbar">
      <div className="topbar-left">
        <div className="brand">
          <span className="brand-mark">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3.5" />
              <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
            </svg>
          </span>
          Vision Inspect
          <span className="brand-sub">Industrial AI v3.0</span>
        </div>
      </div>
      <div className="topbar-center">
        <div className="sysbadge"><span className="dot green" /><span className="lbl">Database</span></div>
        <div className="sysbadge"><span className="dot green" /><span className="lbl">YOLO model</span></div>
        <div className="sysbadge"><span className="dot amber" /><span className="lbl">CLIP model</span></div>
      </div>
      <div className="topbar-right">
        <span className="clock">{hh}:{mm}</span>
        <span className="shift">Shift <span className="v">B</span></span>
        <span className={`live-badge${isLive ? "" : " offline"}`}>
          <span className="dot" />
          {isLive ? "Live" : wsStatus === "connecting" ? "Connecting…" : "Offline"}
        </span>
      </div>
    </div>
  );
}

// ── StatsRow ─────────────────────────────────────────────────────────────────
function StatsRow({ stats }: {
  stats: { total: number; okRate: number; ngRate: number; defects: number; latency: number };
}) {
  return (
    <div className="stats">
      <Stat
        label="Total inspections" value={stats.total}
        delta="—" deltaDir="flat"
        spark={[12, 18, 16, 22, 28, 30, 26, 38, 42, 36, 48, 52, Math.min(stats.total, 58)]}
        sparkColor="var(--ink-4)"
      />
      <Stat
        label="OK rate" value={stats.okRate} decimals={1} unit="%" tint="ok"
        delta={stats.okRate > 0 ? `${stats.okRate.toFixed(1)}%` : "—"} deltaDir="up"
        spark={[96, 97, 96.5, 98, 97.5, 98.2, 98.4, 98.1, 98.5, 98.7, 98.6, 98.8, stats.okRate || 98.8]}
        sparkColor="var(--green)"
      />
      <Stat
        label="NG rate" value={stats.ngRate} decimals={1} unit="%" tint="ng"
        delta={stats.ngRate > 0 ? `${stats.ngRate.toFixed(1)}%` : "—"} deltaDir="down" deltaTone="good"
        spark={[3.8, 3.4, 3.5, 2.8, 2.5, 2.1, 1.9, 1.5, 1.4, 1.3, 1.4, 1.2, stats.ngRate || 1.2]}
        sparkColor="var(--red)"
      />
      <Stat
        label="Total defects" value={stats.defects}
        delta="—" deltaDir="up" deltaTone="bad"
        spark={[24, 28, 30, 26, 32, 38, 42, 40, 44, 46, 48, 52, Math.min(stats.defects, 58)]}
        sparkColor="var(--red)"
      />
      <Stat
        label="Avg latency" value={stats.latency} unit="ms"
        delta="—" deltaDir="down" deltaTone="good"
        spark={[156, 148, 142, 138, 140, 132, 128, 124, 122, 118, 114, 112, stats.latency || 112]}
        sparkColor="var(--ink-4)"
      />
    </div>
  );
}

// ── CameraPanel ───────────────────────────────────────────────────────────────
function CameraPanel({ scanning, ngFlash, image }: {
  scanning: boolean; ngFlash: boolean; image: string;
}) {
  return (
    <div className={`card camera${scanning ? " scanning" : ""}${ngFlash ? " ng" : ""}`}>
      <div className="card-hd">
        <div className="card-hd-title">
          <span className="blink" />
          Live camera
        </div>
        <div className="card-hd-meta">Camera 03 · 1080p</div>
      </div>
      <div className="camera-feed">
        <img className="camera-img" src={image} alt="camera feed" />
        <div className="camera-corners">
          <span className="tl" /><span className="tr" />
          <span className="bl" /><span className="br" />
        </div>
        <div className="scanline" />
        <div className="scan-overlay">
          <span className="spinner" />
          Scanning…
        </div>
        <div className="cam-tag">
          <span className="dot" />
          Recording
        </div>
      </div>
    </div>
  );
}

// ── InspectPanel ──────────────────────────────────────────────────────────────
function InspectPanel({
  pipeline, setPipeline, scanning, onInspect,
  filename, lastResult, onFileChange,
}: {
  pipeline: string;
  setPipeline: (p: string) => void;
  scanning: boolean;
  onInspect: () => void;
  filename: string;
  lastResult: { verdict: string; defects: number; ms: number } | null;
  onFileChange: (file: File) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const pipelines = [
    { key: "yolo_clip", label: "YOLO+CLIP" },
    { key: "cnn",       label: "CNN" },
    { key: "ensemble",  label: "Ensemble" },
  ];

  return (
    <div className="card inspect-panel">
      <div className="card-hd">
        <div className="card-hd-title">Inspect</div>
        <div className="card-hd-meta">Manual mode</div>
      </div>
      <div className="inspect-body">
        {/* Dropzone */}
        <div
          className={`dropzone${isDragging ? " drag-over" : ""}`}
          onClick={() => fileRef.current?.click()}
          onDragEnter={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            const f = e.dataTransfer.files[0];
            if (f) onFileChange(f);
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onFileChange(f);
            }}
          />
          <svg
            className="icon" width="22" height="22" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" strokeWidth="1.8"
            strokeLinecap="round" strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          {filename ? (
            <>
              <div className="filename">{filename}</div>
              <span className="hint">Click to replace or drop a new image</span>
            </>
          ) : (
            <>
              <div style={{ fontWeight: 600, color: "var(--ink-1)" }}>Drop image or click to browse</div>
              <span className="hint">JPG · PNG · TIFF, up to 50 MB</span>
            </>
          )}
        </div>

        {/* Pipeline selector */}
        <div>
          <div className="field-label">Pipeline</div>
          <div className="pipeline">
            {pipelines.map((p) => (
              <button
                key={p.key}
                className={`pill${pipeline === p.key ? " active" : ""}`}
                onClick={() => setPipeline(p.key)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Inspect button */}
        <button
          className={`inspect-btn${scanning ? " scanning" : ""}`}
          onClick={onInspect}
          disabled={scanning}
        >
          {scanning ? (
            <>
              <span className="spinner-sm" />
              Scanning…
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              Run inspection
            </>
          )}
        </button>

        {/* Last verdict */}
        {lastResult && (
          <div className={`verdict${lastResult.verdict === "OK" ? " ok" : " ng"}`}>
            <div className="verdict-badge">{lastResult.verdict}</div>
            <div className="verdict-meta">
              <div className="row">
                <span className="k">Defects</span>
                <span className="v">{lastResult.defects}</span>
              </div>
              <div className="row">
                <span className="k">Time</span>
                <span className="v">{lastResult.ms}ms</span>
              </div>
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
        <div className="card-hd-title">
          <span className="blink" />
          Live feed
        </div>
        <div className="card-hd-meta">Last {feed.length || "—"}</div>
      </div>
      <div className="feed-list">
        {feed.length === 0 ? (
          <div style={{ padding: "24px 16px", textAlign: "center", color: "var(--ink-4)", fontSize: 12 }}>
            Waiting for results…
          </div>
        ) : (
          feed.map((r, i) => {
            const ts = new Date(r.timestamp).toLocaleTimeString("en-US", {
              hour12: false,
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            });
            const pipelineLabel =
              r.pipeline === "yolo_clip" ? "YOLO+CLIP"
              : r.pipeline === "cnn" ? "CNN"
              : r.pipeline === "ensemble" ? "Ensemble"
              : r.pipeline ?? "—";
            return (
              <div key={r.id || i} className={`feed-row${r.verdict === "NG" ? " ng" : ""}`}>
                <span className="ts">{ts}</span>
                <span className={`vbadge${r.verdict === "OK" ? " ok" : " ng"}`}>{r.verdict}</span>
                <span className="meta">
                  <span className="pipe">{pipelineLabel}</span>
                  {r.verdict === "NG" && r.total_defects > 0 && (
                    <>
                      <span className="sep">·</span>
                      <span className="defs">
                        {r.total_defects} defect{r.total_defects !== 1 ? "s" : ""}
                      </span>
                    </>
                  )}
                </span>
                <span className="ms">{Math.round(r.processing_ms ?? 0)} ms</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ── BboxViewer ────────────────────────────────────────────────────────────────
function BboxViewer({ image, defects, isOk }: {
  image: string; defects: Defect[]; isOk: boolean;
}) {
  return (
    <div className="card bbox-card bbox-row">
      <div className="card-hd">
        <div className="card-hd-title">Defect overlay</div>
        <div className="card-hd-meta">
          {isOk
            ? "Last frame · pass"
            : `Last frame · ${defects.length} detection${defects.length !== 1 ? "s" : ""}`}
        </div>
      </div>
      <div className="bbox-body">
        {/* Image + boxes */}
        <div className="bbox-view">
          <img src={image} alt="last inspected frame" />
          {isOk ? (
            <div className="bbox-empty">
              <div className="check">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </div>
              <div className="lg">No defects detected</div>
              <div className="sm">Surface integrity verified</div>
            </div>
          ) : (
            defects.map((d, i) => {
              const isCrit = d.confidence > 0.85;
              return (
                <div
                  key={i}
                  className={`bbox${isCrit ? " crit" : ""}`}
                  style={{
                    left: `${(d.bbox_x1 * 100).toFixed(1)}%`,
                    top: `${(d.bbox_y1 * 100).toFixed(1)}%`,
                    width: `${((d.bbox_x2 - d.bbox_x1) * 100).toFixed(1)}%`,
                    height: `${((d.bbox_y2 - d.bbox_y1) * 100).toFixed(1)}%`,
                    animationDelay: `${i * 80}ms`,
                  }}
                >
                  <div className="chip">
                    <span>{d.defect_class || d.clip_label || "DEFECT"}</span>
                    <span className="conf">{(d.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Sidebar */}
        <div className="bbox-side">
          <div className="bbox-side-hd">
            <span>Detections</span>
            <span className={`count${isOk ? " ok" : ""}`}>{isOk ? 0 : defects.length}</span>
          </div>
          <div className="bbox-side-list">
            {isOk ? (
              <div className="empty-defects">No detections in this frame.</div>
            ) : (
              defects.map((d, i) => {
                const isCrit = d.confidence > 0.85;
                return (
                  <div key={i} className={`defect-item${isCrit ? " crit" : ""}`}>
                    <div>
                      <div className="name">
                        <span className="dot" />
                        {d.defect_class || d.clip_label || "Unknown"}
                      </div>
                      <div className="sub" style={{ marginLeft: 16 }}>
                        {isCrit ? "Critical" : "Moderate"} · {d.detection_type ?? "defect"}
                      </div>
                    </div>
                    <span className="conf">{(d.confidence * 100).toFixed(0)}%</span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main dashboard ────────────────────────────────────────────────────────────
interface BackendStats {
  total_inspections: number;
  ok_rate: number;
  ng_rate: number;
  total_defects: number;
  avg_processing_ms: number;
}

export default function HomePage() {
  const { results, connectionStatus } = useInspection();

  const [backendStats, setBackendStats] = useState<BackendStats | null>(null);
  const [pipeline, setPipeline] = useState("yolo_clip");
  const [scanning, setScanning] = useState(false);
  const [ngFlash, setNgFlash] = useState(false);
  const [filename, setFilename] = useState("");
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [cameraImage, setCameraImage] = useState(PLACEHOLDER_IMG);
  const [lastResult, setLastResult] = useState<InspectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Poll /statistics every 10s
  useEffect(() => {
    const fetch_ = async () => {
      try {
        const res = await fetch(`${API_URL}/statistics`);
        if (res.ok) setBackendStats(await res.json());
      } catch { /* backend not reachable */ }
    };
    fetch_();
    const id = setInterval(fetch_, 10_000);
    return () => clearInterval(id);
  }, []);

  // Also update stats when new WS results arrive
  const wsResult = results[0];
  useEffect(() => {
    if (!wsResult) return;
    setLastResult(wsResult);
    if (wsResult.verdict === "NG") {
      setNgFlash(true);
      setTimeout(() => setNgFlash(false), 3000);
    }
  }, [wsResult]);

  const handleFileChange = useCallback((file: File) => {
    setCurrentFile(file);
    setFilename(file.name);
    const url = URL.createObjectURL(file);
    setCameraImage(url);
    setLastResult(null);
    setError(null);
  }, []);

  const runInspect = useCallback(async () => {
    if (scanning) return;
    if (!currentFile) {
      setError("Please select an image first.");
      return;
    }
    setError(null);
    setScanning(true);
    try {
      const form = new FormData();
      form.append("file", currentFile);
      form.append("pipeline", pipeline);
      const res = await fetch(`${API_URL}/inspect`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      const normalized: InspectionResult = {
        id: data.id ?? data.inspection_id ?? "",
        timestamp: data.timestamp ?? new Date().toISOString(),
        verdict: data.verdict,
        defects: data.defects ?? data.detections ?? [],
        processing_ms: data.processing_ms ?? 0,
        total_defects: data.total_defects ?? 0,
        pipeline: data.pipeline,
        image_path: data.image_path,
      };
      setLastResult(normalized);
      if (normalized.verdict === "NG") {
        setNgFlash(true);
        setTimeout(() => setNgFlash(false), 3000);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inspection failed");
    } finally {
      setScanning(false);
    }
  }, [scanning, currentFile, pipeline]);

  // Merge backend stats with live WS increments
  const stats = {
    total: backendStats?.total_inspections ?? 0,
    okRate: backendStats?.ok_rate ?? 0,
    ngRate: backendStats?.ng_rate ?? 0,
    defects: backendStats?.total_defects ?? 0,
    latency: Math.round(backendStats?.avg_processing_ms ?? 0),
  };

  const bboxDefects = (lastResult?.defects ?? []).filter((d) => d.is_defect !== false);
  const bboxIsOk = !lastResult || lastResult.verdict === "OK";

  const panelLastResult = lastResult
    ? { verdict: lastResult.verdict, defects: lastResult.total_defects, ms: Math.round(lastResult.processing_ms) }
    : null;

  return (
    <div className="app">
      <TopBar wsStatus={connectionStatus} />
      <StatsRow stats={stats} />

      <div className="midrow">
        <CameraPanel scanning={scanning} ngFlash={ngFlash} image={cameraImage} />
        <InspectPanel
          pipeline={pipeline}
          setPipeline={setPipeline}
          scanning={scanning}
          onInspect={runInspect}
          filename={filename}
          lastResult={panelLastResult}
          onFileChange={handleFileChange}
        />
        <FeedPanel results={results} />
      </div>

      {error ? (
        <div className="error-bar">{error}</div>
      ) : (
        <BboxViewer image={cameraImage} defects={bboxDefects} isOk={bboxIsOk} />
      )}
    </div>
  );
}
