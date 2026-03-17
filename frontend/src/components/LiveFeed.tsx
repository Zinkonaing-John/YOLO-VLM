"use client";

import { useInspection, InspectionResult } from "@/hooks/useInspection";
import { useEffect, useRef, useState } from "react";

function FeedItem({ result, isNew }: { result: InspectionResult; isNew: boolean }) {
  const isPassing = result.verdict === "OK";
  const shortId = result.id ? result.id.slice(0, 8) : "--------";
  const defects = result.defects ?? [];
  const time = new Date(result.timestamp).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div
      className={`
        flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-all duration-300
        ${isPassing ? "border-zinc-800" : "border-zinc-800"}
        ${isNew && isPassing ? "animate-flash-green" : ""}
        ${isNew && !isPassing ? "animate-flash-red" : ""}
      `}
    >
      {/* Status indicator */}
      <div
        className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
          isPassing ? "bg-emerald-500 glow-green" : "bg-red-500 glow-red"
        }`}
      />

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono text-zinc-400">
            #{shortId}
          </span>
          <span className={`text-xs font-bold ${isPassing ? "text-emerald-400" : "text-red-400"}`}>
            {result.verdict}
          </span>
        </div>
        <div className="flex items-center justify-between mt-0.5">
          <span className="text-[11px] text-zinc-500 truncate">
            {defects.length > 0
              ? `${defects.filter(d => d.detection_type === "object").length} obj, ${defects.filter(d => d.detection_type === "defect").length} defects: ${defects.map((d) => d.defect_class).join(", ")}`
              : isPassing ? "No detections" : "Failed inspection"}
          </span>
          <span className="text-[10px] text-zinc-600 ml-2 flex-shrink-0">{time}</span>
        </div>
      </div>
    </div>
  );
}

const STATUS_COLORS: Record<string, string> = {
  connected: "bg-emerald-500",
  connecting: "bg-yellow-500 animate-pulse",
  disconnected: "bg-zinc-500",
  error: "bg-red-500",
};

const STATUS_LABELS: Record<string, string> = {
  connected: "Live",
  connecting: "Connecting...",
  disconnected: "Offline",
  error: "Error",
};

export default function LiveFeed() {
  const { results, connectionStatus } = useInspection();
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const prevCountRef = useRef(0);
  const displayResults = results.slice(0, 10);

  useEffect(() => {
    if (results.length > prevCountRef.current) {
      const newCount = results.length - prevCountRef.current;
      const ids = new Set(
        results
          .slice(0, newCount)
          .map((r) => r.id)
          .filter((id): id is string => Boolean(id)),
      );
      setNewIds(ids);

      const timer = setTimeout(() => setNewIds(new Set()), 700);
      return () => clearTimeout(timer);
    }
    prevCountRef.current = results.length;
  }, [results]);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-zinc-200">Live Feed</h3>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${STATUS_COLORS[connectionStatus]}`} />
          <span className="text-[11px] text-zinc-500">
            {STATUS_LABELS[connectionStatus]}
          </span>
        </div>
      </div>

      <div className="space-y-1.5 max-h-[500px] overflow-y-auto">
        {displayResults.length > 0 ? (
          displayResults.map((result, index) => {
            const key = result.id ?? `${result.timestamp}-${index}`;
            return (
              <FeedItem
                key={key}
                result={result}
                isNew={result.id ? newIds.has(result.id) : false}
              />
            );
          })
        ) : (
          <div className="text-center py-8">
            <p className="text-sm text-zinc-500">Waiting for inspection results...</p>
            <p className="text-xs text-zinc-600 mt-1">
              {connectionStatus === "connected"
                ? "Connected and listening"
                : "Attempting to connect to backend"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
