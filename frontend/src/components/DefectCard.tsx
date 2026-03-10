"use client";

import { InspectionResult } from "@/hooks/useInspection";

interface DefectCardProps {
  result: InspectionResult;
  onClick?: () => void;
}

export default function DefectCard({ result, onClick }: DefectCardProps) {
  const isPassing = result.verdict === "PASS";
  const timestamp = new Date(result.timestamp);
  const imageSrc = result.image_url ?? result.image_path ?? null;
  const defects = result.defects ?? [];
  const shortId = (result.id ?? "").slice(0, 8) || "--------";
  const formattedTime = timestamp.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div
      onClick={onClick}
      className={`
        card-hover cursor-pointer overflow-hidden group
        ${isPassing ? "hover:border-emerald-500/30" : "hover:border-red-500/30"}
      `}
    >
      {/* Image area */}
      <div className="relative w-full aspect-video bg-zinc-900 rounded-md mb-3 overflow-hidden">
        {imageSrc ? (
          <img
            src={imageSrc}
            alt={`Inspection ${result.id}`}
            className="w-full h-full object-contain bg-black"
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

        {/* Bounding box overlays */}
        {defects.map((defect, i) => {
          const { bbox_x1: x1, bbox_y1: y1, bbox_x2: x2, bbox_y2: y2 } = defect;
          return (
            <div
              key={i}
              className="absolute border-2 border-red-500 rounded-sm pointer-events-none"
              style={{
                left: `${x1 * 100}%`,
                top: `${y1 * 100}%`,
                width: `${(x2 - x1) * 100}%`,
                height: `${(y2 - y1) * 100}%`,
              }}
            >
              <span className="absolute -top-5 left-0 text-[9px] bg-red-500 text-white px-1 rounded whitespace-nowrap">
                {defect.defect_class} {(defect.confidence * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}

        {/* Verdict badge */}
        <div className="absolute top-2 right-2">
          <span className={isPassing ? "badge-pass" : "badge-fail"}>
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

        {/* Defects list */}
        {defects.length > 0 ? (
          <div className="space-y-1">
            {defects.slice(0, 3).map((defect, i) => (
              <div
                key={i}
                className="flex items-center justify-between text-xs"
              >
                <span className="text-zinc-300 truncate mr-2">
                  {defect.defect_class}
                </span>
                <span className="text-zinc-500 font-mono whitespace-nowrap">
                  {(defect.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))}
            {defects.length > 3 && (
              <p className="text-[10px] text-zinc-600">
                +{defects.length - 3} more defects
              </p>
            )}
          </div>
        ) : (
          <p className="text-xs text-zinc-500">
            {result.vlm_response
              ?? defects.find((d) => d.vlm_description)?.vlm_description
              ?? (isPassing ? "No defects detected" : "Failed inspection")}
          </p>
        )}

        {/* VLM Description */}
        {defects.some((d) => d.vlm_description) && (
          <div className="pt-1 border-t border-zinc-800">
            <p className="text-[11px] text-zinc-400 line-clamp-2">
              {defects.find((d) => d.vlm_description)?.vlm_description}
            </p>
          </div>
        )}

        {/* Processing time */}
        <div className="flex items-center gap-1 text-[10px] text-zinc-600">
          <span>{(result.processing_ms ?? 0).toFixed(0)}ms</span>
        </div>
      </div>
    </div>
  );
}
