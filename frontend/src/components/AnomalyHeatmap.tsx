"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface AnomalyHeatmapProps {
  imageUrl: string;
  heatmapUrl: string | null;
  anomalyScore?: number | null;
}

export default function AnomalyHeatmap({
  imageUrl,
  heatmapUrl,
  anomalyScore,
}: AnomalyHeatmapProps) {
  const [opacity, setOpacity] = useState(0.5);

  const fullImageUrl = imageUrl.startsWith("http")
    ? imageUrl
    : `${API_URL}${imageUrl}`;
  const fullHeatmapUrl =
    heatmapUrl && (heatmapUrl.startsWith("http")
      ? heatmapUrl
      : `${API_URL}${heatmapUrl}`);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-zinc-500 uppercase tracking-wider">
          Anomaly Heatmap
        </span>
        {anomalyScore != null && (
          <span
            className={`text-sm font-mono font-bold ${
              anomalyScore >= 0.5
                ? "text-red-400"
                : anomalyScore >= 0.3
                ? "text-amber-400"
                : "text-emerald-400"
            }`}
          >
            Score: {anomalyScore.toFixed(4)}
          </span>
        )}
      </div>

      <div className="relative w-full aspect-square bg-zinc-900 rounded-lg overflow-hidden">
        {/* Base image */}
        <img
          src={fullImageUrl}
          alt="Inspection"
          className="absolute inset-0 w-full h-full object-contain"
        />

        {/* Heatmap overlay */}
        {fullHeatmapUrl && (
          <img
            src={fullHeatmapUrl}
            alt="Anomaly heatmap"
            className="absolute inset-0 w-full h-full object-contain mix-blend-screen"
            style={{ opacity }}
          />
        )}

        {/* No heatmap message */}
        {!fullHeatmapUrl && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs text-zinc-600">No heatmap available</span>
          </div>
        )}
      </div>

      {/* Opacity slider */}
      {fullHeatmapUrl && (
        <div className="flex items-center gap-3">
          <span className="text-xs text-zinc-500 w-16">Overlay</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1 h-1 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <span className="text-xs text-zinc-400 w-10 text-right">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}

      {/* Color scale legend */}
      {fullHeatmapUrl && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-500">Normal</span>
          <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-blue-600 via-green-500 via-yellow-400 to-red-500" />
          <span className="text-[10px] text-zinc-500">Anomalous</span>
        </div>
      )}
    </div>
  );
}
