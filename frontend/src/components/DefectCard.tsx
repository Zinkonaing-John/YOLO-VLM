"use client";

import { useRef, useState, useCallback } from "react";
import { InspectionResult, Defect } from "@/hooks/useInspection";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Draw bounding boxes + labels onto the image and trigger a download. */
function downloadWithBoxes(src: string, defects: Defect[], filename: string) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(img, 0, 0);

    const w = canvas.width;
    const h = canvas.height;
    const lineW = Math.max(2, Math.round(Math.min(w, h) / 200));
    const fontSize = Math.max(12, Math.round(Math.min(w, h) / 40));

    defects.forEach((d) => {
      const x1 = d.bbox_x1 * w;
      const y1 = d.bbox_y1 * h;
      const bw = (d.bbox_x2 - d.bbox_x1) * w;
      const bh = (d.bbox_y2 - d.bbox_y1) * h;
      const color = d.detection_type === "defect" ? "#ef4444" : "#22c55e";

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = lineW;
      ctx.strokeRect(x1, y1, bw, bh);

      // Label
      const label = `${d.defect_class} ${(d.confidence * 100).toFixed(0)}%`;
      ctx.font = `bold ${fontSize}px monospace`;
      const tw = ctx.measureText(label).width;
      const labelH = fontSize + 6;
      const labelY = y1 - labelH > 0 ? y1 - labelH : y1;

      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY, tw + 10, labelH);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 + 5, labelY + fontSize);
    });

    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }, "image/jpeg", 0.92);
  };
  img.src = src;
}

interface DefectCardProps {
  result: InspectionResult;
  onClick?: () => void;
}

/**
 * Compute the rendered image rect inside an object-contain container.
 * Returns { offsetX, offsetY, width, height } in pixels relative to the container.
 */
function getContainedRect(
  containerW: number,
  containerH: number,
  naturalW: number,
  naturalH: number
) {
  const containerAR = containerW / containerH;
  const imageAR = naturalW / naturalH;

  let width: number, height: number, offsetX: number, offsetY: number;

  if (imageAR > containerAR) {
    // Image wider than container → letterbox top/bottom
    width = containerW;
    height = containerW / imageAR;
    offsetX = 0;
    offsetY = (containerH - height) / 2;
  } else {
    // Image taller than container → pillarbox left/right
    height = containerH;
    width = containerH * imageAR;
    offsetX = (containerW - width) / 2;
    offsetY = 0;
  }

  return { offsetX, offsetY, width, height };
}

export default function DefectCard({ result, onClick }: DefectCardProps) {
  const isOK = result.verdict === "OK";
  const timestamp = new Date(result.timestamp);
  const rawSrc = result.image_url ?? result.image_path ?? null;
  const imageSrc = rawSrc
    ? rawSrc.startsWith("http")
      ? rawSrc
      : `${API_URL}${rawSrc}`
    : null;
  const allDetections = result.defects ?? [];
  const objects = allDetections.filter((d) => d.detection_type === "object");
  const defects = allDetections.filter((d) => d.detection_type === "defect");
  const shortId = (result.id ?? "").slice(0, 8) || "--------";
  const formattedTime = timestamp.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const [imgRect, setImgRect] = useState<{
    offsetX: number;
    offsetY: number;
    width: number;
    height: number;
  } | null>(null);

  const handleImgLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    const container = containerRef.current;
    if (!container) return;
    const rect = getContainedRect(
      container.clientWidth,
      container.clientHeight,
      img.naturalWidth,
      img.naturalHeight
    );
    setImgRect(rect);
  }, []);

  return (
    <div
      onClick={onClick}
      className={`
        card-hover cursor-pointer overflow-hidden group
        ${isOK ? "hover:border-emerald-500/30" : "hover:border-red-500/30"}
      `}
    >
      {/* Image area */}
      <div
        ref={containerRef}
        className="relative w-full aspect-video bg-black rounded-md mb-3 overflow-hidden"
      >
        {imageSrc ? (
          <img
            src={imageSrc}
            alt={`Inspection ${result.id}`}
            className="w-full h-full object-contain"
            onLoad={handleImgLoad}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-zinc-600">
            <svg
              className="w-12 h-12"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z"
              />
            </svg>
          </div>
        )}

        {/* Bounding box overlays — positioned within the actual image rect */}
        {imgRect &&
          allDetections.map((defect, i) => {
            const { bbox_x1: x1, bbox_y1: y1, bbox_x2: x2, bbox_y2: y2 } = defect;
            const isDefect = defect.detection_type === "defect";
            const borderColor = isDefect
              ? "border-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"
              : "border-emerald-500";
            const label = defect.defect_class;
            const score = (defect.confidence * 100).toFixed(0);
            return (
              <div
                key={i}
                className={`absolute border-[2.5px] ${borderColor} pointer-events-none`}
                style={{
                  left: imgRect.offsetX + x1 * imgRect.width,
                  top: imgRect.offsetY + y1 * imgRect.height,
                  width: (x2 - x1) * imgRect.width,
                  height: (y2 - y1) * imgRect.height,
                }}
              >
                <span
                  className={`absolute -top-5 left-0 text-[9px] font-semibold ${
                    isDefect ? "bg-red-500" : "bg-emerald-500"
                  } text-white px-1.5 py-0.5 rounded whitespace-nowrap`}
                >
                  {label} {score}%
                </span>
              </div>
            );
          })}

        {/* Download + Verdict badges */}
        <div className="absolute top-2 right-2 flex items-center gap-1.5">
          {imageSrc && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                downloadWithBoxes(imageSrc, result.defects ?? [], `inspection_${shortId}.jpg`);
              }}
              className="opacity-0 group-hover:opacity-100 transition-opacity bg-zinc-900/80 hover:bg-zinc-800 text-zinc-300 hover:text-white p-1.5 rounded-md"
              title="Download image with bounding boxes"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
              </svg>
            </button>
          )}
          <span className={isOK ? "badge-pass" : "badge-fail"}>
            {result.verdict}
          </span>
        </div>
      </div>

      {/* Info */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-zinc-500 font-mono">
            #{shortId}
          </span>
          <span className="text-xs text-zinc-500">{formattedTime}</span>
        </div>

        {/* Detections list: defects first, then objects */}
        {allDetections.length > 0 ? (
          <div className="space-y-1">
            {defects.slice(0, 2).map((d, i) => (
              <div key={`d-${i}`} className="flex items-center justify-between text-xs">
                <span className="text-red-400 truncate mr-2">
                  {d.defect_class}
                </span>
                <span className="text-zinc-500 font-mono whitespace-nowrap">
                  {(d.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))}
            {objects.slice(0, defects.length > 0 ? 1 : 3).map((d, i) => (
              <div key={`o-${i}`} className="flex items-center justify-between text-xs">
                <span className="text-emerald-400 truncate mr-2">
                  {d.defect_class}
                </span>
                <span className="text-zinc-500 font-mono whitespace-nowrap">
                  {(d.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))}
            {allDetections.length > 3 && (
              <p className="text-[10px] text-zinc-600">
                +{allDetections.length - 3} more
              </p>
            )}
          </div>
        ) : (
          <p className="text-xs text-zinc-500">
            No detections
          </p>
        )}

        {/* Processing time + counts */}
        <div className="flex items-center justify-between text-[10px] text-zinc-600">
          <span>{(result.processing_ms ?? 0).toFixed(0)}ms</span>
          <span className="font-mono">
            {objects.length} obj · {defects.length} defect{defects.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>
    </div>
  );
}
