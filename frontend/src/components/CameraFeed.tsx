"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { InspectionResult, Defect } from "@/hooks/useInspection";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Roi {
  x: number;
  y: number;
  w: number;
  h: number;
}

type TriggerState = "waiting" | "settling" | "inspecting" | "done" | "cooldown";

// Compare two ImageData arrays, return fraction of pixels that changed significantly
function computeMotion(prev: Uint8ClampedArray, curr: Uint8ClampedArray, threshold: number): number {
  let changed = 0;
  const total = prev.length / 4;
  for (let i = 0; i < prev.length; i += 4) {
    const dr = Math.abs(prev[i] - curr[i]);
    const dg = Math.abs(prev[i + 1] - curr[i + 1]);
    const db = Math.abs(prev[i + 2] - curr[i + 2]);
    if (dr + dg + db > threshold) changed++;
  }
  return changed / total;
}

export default function CameraFeed() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const motionRafRef = useRef<number | null>(null);
  const settleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const busyRef = useRef(false);
  const prevFrameRef = useRef<Uint8ClampedArray | null>(null);
  const triggerStateRef = useRef<TriggerState>("waiting");

  const [active, setActive] = useState(false);
  const pipeline = "ensemble";
  const [cameraReady, setCameraReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<InspectionResult | null>(null);
  const [processingMs, setProcessingMs] = useState<number | null>(null);
  const [triggerState, setTriggerState] = useState<TriggerState>("waiting");
  const [motionLevel, setMotionLevel] = useState(0);

  // ROI selection state (normalized 0-1 coordinates)
  const [roi, setRoi] = useState<Roi | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);
  const [selectMode, setSelectMode] = useState(false);

  // Sensitivity: fraction of pixels that must change to count as motion
  const MOTION_THRESHOLD = 60;   // per-pixel RGB diff sum to count as "changed"
  const MOTION_ENTER = 0.08;     // 8% of pixels changed → item arrived
  const MOTION_EXIT = 0.08;      // 8% of pixels changed after done → item removed
  const SETTLE_MS = 2000;        // wait 2s after motion before capturing
  const POLL_INTERVAL = 200;     // check for motion every 200ms

  // Convert mouse event to normalized 0-1 coordinates relative to container
  const toNormalized = useCallback((e: React.MouseEvent): { x: number; y: number } => {
    const rect = containerRef.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
    };
  }, []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!selectMode || !active) return;
      e.preventDefault();
      const pt = toNormalized(e);
      setDrawing(true);
      setDrawStart(pt);
      setDrawCurrent(pt);
    },
    [selectMode, active, toNormalized]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!drawing) return;
      e.preventDefault();
      setDrawCurrent(toNormalized(e));
    },
    [drawing, toNormalized]
  );

  const handleMouseUp = useCallback(() => {
    if (!drawing || !drawStart || !drawCurrent) {
      setDrawing(false);
      return;
    }
    const x = Math.min(drawStart.x, drawCurrent.x);
    const y = Math.min(drawStart.y, drawCurrent.y);
    const w = Math.abs(drawCurrent.x - drawStart.x);
    const h = Math.abs(drawCurrent.y - drawStart.y);

    if (w > 0.02 && h > 0.02) {
      setRoi({ x, y, w, h });
      // Reset motion detection for new ROI
      prevFrameRef.current = null;
      triggerStateRef.current = "waiting";
      setTriggerState("waiting");
      setLastResult(null);
    }
    setDrawing(false);
    setDrawStart(null);
    setDrawCurrent(null);
    setSelectMode(false);
  }, [drawing, drawStart, drawCurrent]);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraReady(true);
      }
    } catch {
      setError("Camera access denied or unavailable");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (motionRafRef.current) {
      clearInterval(motionRafRef.current as unknown as number);
      motionRafRef.current = null;
    }
    if (settleTimerRef.current) {
      clearTimeout(settleTimerRef.current);
      settleTimerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraReady(false);
    setLastResult(null);
    setProcessingMs(null);
    setTriggerState("waiting");
    setMotionLevel(0);
    busyRef.current = false;
    prevFrameRef.current = null;
    triggerStateRef.current = "waiting";
  }, []);

  // Grab a small snapshot of the ROI (or full frame) for motion comparison
  const grabRoiFrame = useCallback((): Uint8ClampedArray | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Use a small resolution for motion detection (fast)
    const sampleW = 80;
    const sampleH = 60;

    if (roi) {
      const sx = Math.round(roi.x * vw);
      const sy = Math.round(roi.y * vh);
      const sw = Math.round(roi.w * vw);
      const sh = Math.round(roi.h * vh);
      canvas.width = sampleW;
      canvas.height = sampleH;
      ctx.drawImage(video, sx, sy, sw, sh, 0, 0, sampleW, sampleH);
    } else {
      canvas.width = sampleW;
      canvas.height = sampleH;
      ctx.drawImage(video, 0, 0, sampleW, sampleH);
    }

    return ctx.getImageData(0, 0, sampleW, sampleH).data;
  }, [roi]);

  const captureAndInspect = useCallback(async () => {
    if (busyRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    busyRef.current = true;
    triggerStateRef.current = "inspecting";
    setTriggerState("inspecting");

    const ctx = canvas.getContext("2d");
    if (!ctx) { busyRef.current = false; return; }

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    if (roi) {
      const sx = Math.round(roi.x * vw);
      const sy = Math.round(roi.y * vh);
      const sw = Math.round(roi.w * vw);
      const sh = Math.round(roi.h * vh);
      canvas.width = sw;
      canvas.height = sh;
      ctx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
    } else {
      canvas.width = vw;
      canvas.height = vh;
      ctx.drawImage(video, 0, 0);
    }

    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.85)
      );
      if (!blob) { busyRef.current = false; return; }

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      formData.append("pipeline", pipeline);
      const t0 = performance.now();
      const res = await fetch(`${API_URL}/inspect`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        // Backend returns "detections", normalize to "defects" for frontend
        const defects: Defect[] = data.detections ?? data.defects ?? [];
        const result: InspectionResult = {
          ...data,
          defects,
        };
        setLastResult(result);
        setProcessingMs(performance.now() - t0);
        drawOverlay(defects);
      }
    } catch {
      // skip
    } finally {
      busyRef.current = false;
      // Move to "done" – wait for item to leave before re-arming
      triggerStateRef.current = "done";
      setTriggerState("done");
      // Capture new baseline after inspection
      prevFrameRef.current = grabRoiFrame();
    }
  }, [roi, pipeline, grabRoiFrame]);

  const drawOverlay = useCallback((defects: Defect[]) => {
    const video = videoRef.current;
    const overlay = overlayCanvasRef.current;
    if (!video || !overlay) return;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw ROI outline
    if (roi) {
      const rx = roi.x * overlay.width;
      const ry = roi.y * overlay.height;
      const rw = roi.w * overlay.width;
      const rh = roi.h * overlay.height;

      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(0, 0, overlay.width, overlay.height);
      ctx.clearRect(rx, ry, rw, rh);

      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);

      ctx.font = "bold 10px monospace";
      ctx.fillStyle = "rgba(59,130,246,0.85)";
      ctx.fillRect(rx, ry - 16, 30, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText("ROI", rx + 5, ry - 4);
    }

    if (defects.length === 0) return;

    const offsetX = roi ? roi.x * overlay.width : 0;
    const offsetY = roi ? roi.y * overlay.height : 0;
    const regionDisplayW = roi ? roi.w * overlay.width : overlay.width;
    const regionDisplayH = roi ? roi.h * overlay.height : overlay.height;

    defects.forEach((d) => {
      const x = d.bbox_x1 * regionDisplayW + offsetX;
      const y = d.bbox_y1 * regionDisplayH + offsetY;
      const w = (d.bbox_x2 - d.bbox_x1) * regionDisplayW;
      const h = (d.bbox_y2 - d.bbox_y1) * regionDisplayH;

      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const label = `${d.defect_class} ${(d.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 11px monospace";
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(239,68,68,0.85)";
      ctx.fillRect(x, y - 16, textW + 8, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 4, y - 4);
    });
  }, [roi]);

  // Redraw overlay when ROI changes
  useEffect(() => {
    if (active && cameraReady) {
      drawOverlay(lastResult?.defects ?? []);
    }
  }, [roi, active, cameraReady, drawOverlay, lastResult]);

  // Motion detection polling loop
  useEffect(() => {
    if (!active || !cameraReady) return;

    const pollId = setInterval(() => {
      const curr = grabRoiFrame();
      if (!curr) return;

      const prev = prevFrameRef.current;
      if (!prev) {
        // First frame — store as baseline
        prevFrameRef.current = curr;
        return;
      }

      const motion = computeMotion(prev, curr, MOTION_THRESHOLD);
      setMotionLevel(motion);

      const state = triggerStateRef.current;

      if (state === "waiting") {
        // Waiting for item to arrive
        if (motion >= MOTION_ENTER) {
          // Motion detected — start settling timer
          triggerStateRef.current = "settling";
          setTriggerState("settling");
          prevFrameRef.current = curr;

          if (settleTimerRef.current) clearTimeout(settleTimerRef.current);
          settleTimerRef.current = setTimeout(() => {
            // After 2s settle, fire inspection
            if (triggerStateRef.current === "settling") {
              captureAndInspect();
            }
          }, SETTLE_MS);
        } else {
          // Update baseline slowly when idle
          prevFrameRef.current = curr;
        }
      } else if (state === "settling") {
        // During settling, if there's still significant motion, reset the timer
        if (motion >= MOTION_ENTER) {
          prevFrameRef.current = curr;
          if (settleTimerRef.current) clearTimeout(settleTimerRef.current);
          settleTimerRef.current = setTimeout(() => {
            if (triggerStateRef.current === "settling") {
              captureAndInspect();
            }
          }, SETTLE_MS);
        }
        // If motion stopped, let the timer fire naturally
      } else if (state === "done") {
        // Waiting for item to leave
        if (motion >= MOTION_EXIT) {
          // Item is being removed — enter cooldown, don't re-arm yet
          prevFrameRef.current = curr;
          triggerStateRef.current = "cooldown";
          setTriggerState("cooldown");
          setLastResult(null);
          drawOverlay([]);
        }
      } else if (state === "cooldown") {
        // Wait for scene to become still before re-arming
        prevFrameRef.current = curr;
        if (motion < MOTION_ENTER) {
          // Scene is still — safe to start waiting for next item
          triggerStateRef.current = "waiting";
          setTriggerState("waiting");
        }
      }
      // "inspecting" state — do nothing, wait for inspection to finish
    }, POLL_INTERVAL);

    return () => {
      clearInterval(pollId);
      if (settleTimerRef.current) {
        clearTimeout(settleTimerRef.current);
        settleTimerRef.current = null;
      }
    };
  }, [active, cameraReady, grabRoiFrame, captureAndInspect, drawOverlay]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  const toggle = async () => {
    if (active) {
      stopCamera();
      setActive(false);
    } else {
      setActive(true);
      await startCamera();
    }
  };

  const isPassing = lastResult?.verdict === "OK";
  const defects = lastResult?.defects ?? [];

  const previewRoi =
    drawing && drawStart && drawCurrent
      ? {
          x: Math.min(drawStart.x, drawCurrent.x),
          y: Math.min(drawStart.y, drawCurrent.y),
          w: Math.abs(drawCurrent.x - drawStart.x),
          h: Math.abs(drawCurrent.y - drawStart.y),
        }
      : null;

  const stateLabel: Record<TriggerState, string> = {
    waiting: "Waiting for item...",
    settling: "Item detected — settling...",
    inspecting: "Inspecting...",
    done: "Done — remove item to re-arm",
    cooldown: "Item removing — standby...",
  };

  const stateColor: Record<TriggerState, string> = {
    waiting: "text-zinc-500",
    settling: "text-amber-400",
    inspecting: "text-blue-400",
    done: "text-emerald-400",
    cooldown: "text-orange-400",
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-zinc-200">Camera Feed</h3>
        <div className="flex items-center gap-2">
          {/* ROI select button */}
          {active && (
            <>
              <button
                onClick={() => setSelectMode((v) => !v)}
                className={`
                  flex items-center gap-1.5 text-[11px] px-2 py-1 rounded-full border transition-colors
                  ${selectMode
                    ? "bg-amber-500/15 border-amber-500/40 text-amber-400"
                    : "bg-zinc-800 border-zinc-700 text-zinc-500 hover:text-zinc-300"
                  }
                `}
                title="Draw a region of interest to inspect"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4" />
                </svg>
                {selectMode ? "Drawing..." : "Select ROI"}
              </button>
              {roi && (
                <button
                  onClick={() => {
                    setRoi(null);
                    setLastResult(null);
                    prevFrameRef.current = null;
                    triggerStateRef.current = "waiting";
                    setTriggerState("waiting");
                  }}
                  className="text-[11px] px-2 py-1 rounded-full border bg-zinc-800 border-zinc-700 text-zinc-500 hover:text-zinc-300 transition-colors"
                  title="Clear selection, inspect full frame"
                >
                  Full Frame
                </button>
              )}
            </>
          )}
          {/* Start/Stop */}
          <button
            onClick={toggle}
            className={`
              text-[11px] px-3 py-1 rounded-full border font-medium transition-colors
              ${active
                ? "bg-red-500/15 border-red-500/40 text-red-400 hover:bg-red-500/25"
                : "bg-emerald-500/15 border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/25"
              }
            `}
          >
            {active ? "Stop" : "Start"}
          </button>
        </div>
      </div>

      {/* Video container */}
      <div
        ref={containerRef}
        className={`relative rounded-lg overflow-hidden bg-zinc-900 aspect-video ${
          selectMode ? "cursor-crosshair" : ""
        }`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { if (drawing) handleMouseUp(); }}
      >
        <video
          ref={videoRef}
          muted
          playsInline
          className={`w-full h-full object-cover ${active ? "" : "hidden"}`}
        />
        <canvas ref={overlayCanvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
        <canvas ref={canvasRef} className="hidden" />

        {/* Drawing preview rect */}
        {previewRoi && (
          <div
            className="absolute border-2 border-dashed border-amber-400 bg-amber-400/10 pointer-events-none"
            style={{
              left: `${previewRoi.x * 100}%`,
              top: `${previewRoi.y * 100}%`,
              width: `${previewRoi.w * 100}%`,
              height: `${previewRoi.h * 100}%`,
            }}
          />
        )}

        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-500">
            <svg className="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z"
              />
            </svg>
            <p className="text-xs">Click Start to open camera</p>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/90">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {/* Live verdict overlay */}
        {active && lastResult && (
          <div className="absolute top-2 left-2 flex items-center gap-2">
            <span className={`text-xs font-bold px-2 py-0.5 rounded ${
              isPassing ? "bg-emerald-500/80 text-white" : "bg-red-500/80 text-white"
            }`}>
              {lastResult.verdict}
            </span>
            {processingMs !== null && (
              <span className="text-[10px] bg-zinc-900/80 text-zinc-300 px-1.5 py-0.5 rounded">
                {processingMs.toFixed(0)}ms
              </span>
            )}
            {roi && (
              <span className="text-[10px] bg-blue-900/80 text-blue-300 px-1.5 py-0.5 rounded">
                ROI
              </span>
            )}
          </div>
        )}

        {/* Select mode hint */}
        {selectMode && !drawing && (
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2">
            <span className="text-[11px] bg-amber-500/90 text-white px-3 py-1 rounded-full">
              Click and drag to select inspection area
            </span>
          </div>
        )}
      </div>

      {/* Trigger status bar */}
      {active && (
        <div className="flex items-center justify-between mt-2 px-1">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              triggerState === "waiting" ? "bg-zinc-500" :
              triggerState === "settling" ? "bg-amber-400 animate-pulse" :
              triggerState === "inspecting" ? "bg-blue-400 animate-pulse" :
              triggerState === "cooldown" ? "bg-orange-400 animate-pulse" :
              "bg-emerald-400"
            }`} />
            <span className={`text-[11px] ${stateColor[triggerState]}`}>
              {stateLabel[triggerState]}
            </span>
          </div>
          {/* Motion level indicator */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-600">motion</span>
            <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-200 ${
                  motionLevel > MOTION_ENTER ? "bg-amber-400" : "bg-zinc-600"
                }`}
                style={{ width: `${Math.min(100, motionLevel * 500)}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Defect list below video */}
      {active && defects.length > 0 && (
        <div className="mt-2 space-y-1">
          {defects.map((d, i) => (
            <div key={i} className="flex items-center justify-between text-xs px-1">
              <span className="text-red-400">{d.defect_class}</span>
              <span className="text-zinc-500 font-mono">{(d.confidence * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

    </div>
  );
}
