"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import DefectCard from "@/components/DefectCard";
import { InspectionResult } from "@/hooks/useInspection";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const PAGE_SIZE = 20;

type VerdictFilter = "ALL" | "OK" | "NG";
type DetailTab = "original" | "heatmap" | "clip";

function getContainedRect(cW: number, cH: number, nW: number, nH: number) {
  const cAR = cW / cH;
  const iAR = nW / nH;
  let w: number, h: number, ox: number, oy: number;
  if (iAR > cAR) {
    w = cW; h = cW / iAR; ox = 0; oy = (cH - h) / 2;
  } else {
    h = cH; w = cH * iAR; ox = (cW - w) / 2; oy = 0;
  }
  return { offsetX: ox, offsetY: oy, width: w, height: h };
}

interface ClipDetail {
  defect_id: string;
  defect_class: string;
  confidence: number;
  bbox: [number, number, number, number];
  clip_label: string;
  clip_score: number;
  is_defect: boolean;
}

function InspectionDetailModal({
  result,
  onClose,
}: {
  result: InspectionResult;
  onClose: () => void;
}) {
  const [tab, setTab] = useState<DetailTab>("original");
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapError, setHeatmapError] = useState<string | null>(null);
  const [anomalyScore, setAnomalyScore] = useState<number | null>(null);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.5);
  const [clipDetails, setClipDetails] = useState<ClipDetail[]>([]);
  const [clipLoading, setClipLoading] = useState(false);
  const [highlightIdx, setHighlightIdx] = useState<number | null>(null);
  const imgContainerRef = useRef<HTMLDivElement>(null);
  const [imgRect, setImgRect] = useState<{ offsetX: number; offsetY: number; width: number; height: number } | null>(null);

  const handleImgLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    const container = imgContainerRef.current;
    if (!container) return;
    setImgRect(getContainedRect(container.clientWidth, container.clientHeight, img.naturalWidth, img.naturalHeight));
  }, []);

  const allDetections = result.defects ?? [];
  const shortId = (result.id ?? "").slice(0, 12) || "---";
  const processingTimeMs = result.processing_ms ?? 0;

  const imageUrl = result.image_url
    ? result.image_url.startsWith("http")
      ? result.image_url
      : `${API_URL}${result.image_url}`
    : null;

  // Fetch heatmap when tab switches
  useEffect(() => {
    if (tab === "heatmap" && !heatmapUrl && !heatmapLoading && result.id) {
      setHeatmapLoading(true);
      setHeatmapError(null);
      fetch(`${API_URL}/inspections/${result.id}/heatmap`)
        .then(async (res) => {
          if (!res.ok) {
            const detail = await res.json().catch(() => ({}));
            throw new Error(detail.detail || `Error ${res.status}`);
          }
          const score = res.headers.get("X-Anomaly-Score");
          if (score) setAnomalyScore(parseFloat(score));
          const blob = await res.blob();
          setHeatmapUrl(URL.createObjectURL(blob));
        })
        .catch((err) => setHeatmapError(err.message))
        .finally(() => setHeatmapLoading(false));
    }
  }, [tab, heatmapUrl, heatmapLoading, result.id]);

  // Fetch CLIP details when tab switches
  useEffect(() => {
    if (tab === "clip" && clipDetails.length === 0 && !clipLoading && result.id) {
      setClipLoading(true);
      fetch(`${API_URL}/inspections/${result.id}/clip-details`)
        .then(async (res) => {
          if (!res.ok) throw new Error(`Error ${res.status}`);
          const data = await res.json();
          setClipDetails(data.clip_details ?? []);
        })
        .catch(() => setClipDetails([]))
        .finally(() => setClipLoading(false));
    }
  }, [tab, clipDetails.length, clipLoading, result.id]);

  // Cleanup blob URL
  useEffect(() => {
    return () => {
      if (heatmapUrl) URL.revokeObjectURL(heatmapUrl);
    };
  }, [heatmapUrl]);

  const tabs: { key: DetailTab; label: string }[] = [
    { key: "original", label: "Detections" },
    { key: "heatmap", label: "Heatmap" },
    { key: "clip", label: "CLIP Analysis" },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-2xl w-full p-6 max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold text-zinc-200">
              Inspection #{shortId}
            </h3>
            <span className={result.verdict === "OK" ? "badge-pass" : "badge-fail"}>
              {result.verdict}
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-200 text-lg leading-none"
          >
            x
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-4 bg-zinc-800/50 rounded-lg p-1">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex-1 text-xs py-1.5 px-3 rounded-md font-medium transition-colors ${
                tab === t.key
                  ? "bg-zinc-700 text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "original" && (
          <div>
            {/* Image with bounding boxes */}
            <div ref={imgContainerRef} className="relative w-full aspect-video bg-black rounded-lg mb-4 overflow-hidden">
              {imageUrl ? (
                <img src={imageUrl} alt="Inspection" className="w-full h-full object-contain" onLoad={handleImgLoad} />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-zinc-600 text-sm">
                  No image available
                </div>
              )}
              {imgRect && allDetections.map((defect, i) => {
                const x1 = defect.bbox_x1 ?? 0;
                const y1 = defect.bbox_y1 ?? 0;
                const x2 = defect.bbox_x2 ?? 0;
                const y2 = defect.bbox_y2 ?? 0;
                if (!x2 && !y2) return null;
                const isHighlight = highlightIdx === i;
                const color = defect.is_defect ? "border-red-500" : "border-emerald-500";
                return (
                  <div
                    key={i}
                    className={`absolute border-2 ${color} rounded-sm transition-all ${
                      isHighlight ? "ring-2 ring-yellow-400 ring-offset-1 ring-offset-black" : ""
                    }`}
                    style={{
                      left: imgRect.offsetX + x1 * imgRect.width,
                      top: imgRect.offsetY + y1 * imgRect.height,
                      width: (x2 - x1) * imgRect.width,
                      height: (y2 - y1) * imgRect.height,
                    }}
                  >
                    <span
                      className={`absolute -top-5 left-0 text-[9px] font-semibold ${
                        defect.is_defect ? "bg-red-500" : "bg-emerald-500"
                      } text-white px-1.5 py-0.5 rounded whitespace-nowrap`}
                    >
                      {defect.defect_class} {(defect.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Detections list */}
            {allDetections.length > 0 ? (
              <div className="space-y-1.5">
                <h4 className="text-xs font-semibold text-zinc-400 uppercase">
                  Detections ({allDetections.length})
                </h4>
                {allDetections.map((defect, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-2 bg-zinc-800 rounded-lg border border-zinc-700 cursor-pointer hover:border-zinc-600 transition-colors"
                    onMouseEnter={() => setHighlightIdx(i)}
                    onMouseLeave={() => setHighlightIdx(null)}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${defect.is_defect ? "bg-red-500" : "bg-emerald-500"}`} />
                      <span className="text-sm text-zinc-200">{defect.defect_class}</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-zinc-400 font-mono">
                      {defect.clip_label && (
                        <span className="text-zinc-500">{defect.clip_label}</span>
                      )}
                      <span>{(defect.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-zinc-500">No detections</p>
            )}
          </div>
        )}

        {tab === "heatmap" && (
          <div>
            {/* Heatmap overlay on original */}
            <div className="relative w-full aspect-video bg-zinc-800 rounded-lg mb-4 overflow-hidden">
              {imageUrl && (
                <img src={imageUrl} alt="Original" className="absolute inset-0 w-full h-full object-contain bg-black" />
              )}
              {heatmapLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                  <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                </div>
              )}
              {heatmapError && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                  <p className="text-sm text-red-400">{heatmapError}</p>
                </div>
              )}
              {heatmapUrl && (
                <img
                  src={heatmapUrl}
                  alt="Anomaly heatmap"
                  className="absolute inset-0 w-full h-full object-contain mix-blend-screen"
                  style={{ opacity: heatmapOpacity }}
                />
              )}
            </div>

            {/* Controls */}
            <div className="space-y-3">
              {anomalyScore != null && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-zinc-500 uppercase tracking-wider">Anomaly Score</span>
                  <span
                    className={`text-sm font-mono font-bold ${
                      anomalyScore >= 0.5 ? "text-red-400" : anomalyScore >= 0.3 ? "text-amber-400" : "text-emerald-400"
                    }`}
                  >
                    {anomalyScore.toFixed(4)}
                  </span>
                </div>
              )}

              {heatmapUrl && (
                <>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-zinc-500 w-16">Overlay</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={heatmapOpacity}
                      onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                      className="flex-1 h-1 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                    <span className="text-xs text-zinc-400 w-10 text-right">
                      {Math.round(heatmapOpacity * 100)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-zinc-500">Normal</span>
                    <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-blue-600 via-green-500 via-yellow-400 to-red-500" />
                    <span className="text-[10px] text-zinc-500">Anomalous</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {tab === "clip" && (
          <div>
            {/* Image with boxes for context */}
            <div className="relative w-full aspect-video bg-black rounded-lg mb-4 overflow-hidden">
              {imageUrl && (
                <img src={imageUrl} alt="Original" className="w-full h-full object-contain" onLoad={handleImgLoad} />
              )}
              {imgRect && clipDetails.map((cd, i) => {
                const [x1, y1, x2, y2] = cd.bbox;
                const isHighlight = highlightIdx === i;
                return (
                  <div
                    key={i}
                    className={`absolute border-2 rounded-sm transition-all ${
                      cd.is_defect ? "border-red-500" : "border-emerald-500"
                    } ${isHighlight ? "ring-2 ring-yellow-400 ring-offset-1 ring-offset-black" : ""}`}
                    style={{
                      left: imgRect.offsetX + x1 * imgRect.width,
                      top: imgRect.offsetY + y1 * imgRect.height,
                      width: (x2 - x1) * imgRect.width,
                      height: (y2 - y1) * imgRect.height,
                    }}
                  />
                );
              })}
            </div>

            {clipLoading && (
              <div className="flex items-center justify-center py-8">
                <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <span className="ml-3 text-sm text-zinc-400">Running CLIP analysis on each ROI...</span>
              </div>
            )}

            {!clipLoading && clipDetails.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-zinc-400 uppercase">
                  CLIP Classification per Detection
                </h4>
                {clipDetails.map((cd, i) => (
                  <div
                    key={i}
                    className={`p-3 bg-zinc-800 rounded-lg border transition-colors cursor-pointer ${
                      highlightIdx === i ? "border-yellow-500/50" : "border-zinc-700"
                    }`}
                    onMouseEnter={() => setHighlightIdx(i)}
                    onMouseLeave={() => setHighlightIdx(null)}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${cd.is_defect ? "bg-red-500" : "bg-emerald-500"}`} />
                        <span className="text-sm text-zinc-200 font-medium">
                          {cd.defect_class}
                        </span>
                      </div>
                      <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                        cd.is_defect ? "bg-red-500/20 text-red-400" : "bg-emerald-500/20 text-emerald-400"
                      }`}>
                        {cd.is_defect ? "DEFECT" : "OK"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-zinc-400">
                      <span>CLIP: <span className="text-zinc-300 font-medium">{cd.clip_label}</span></span>
                      <span className="font-mono">{(cd.clip_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="mt-1 w-full bg-zinc-700 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full ${cd.is_defect ? "bg-red-500" : "bg-emerald-500"}`}
                        style={{ width: `${cd.clip_score * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {!clipLoading && clipDetails.length === 0 && allDetections.length === 0 && (
              <p className="text-sm text-zinc-500 text-center py-4">
                No detections to analyze with CLIP
              </p>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between mt-4 pt-3 border-t border-zinc-800">
          <span className="text-xs text-zinc-500">
            {processingTimeMs.toFixed(0)}ms | {new Date(result.timestamp).toLocaleString()}
          </span>
          <span className="text-xs text-zinc-600 font-mono">
            {allDetections.length} detection{allDetections.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>
    </div>
  );
}

export default function GalleryPage() {
  const [results, setResults] = useState<InspectionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>("ALL");
  const [classFilter, setClassFilter] = useState<string>("ALL");
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [allClasses, setAllClasses] = useState<string[]>([]);
  const [selectedResult, setSelectedResult] = useState<InspectionResult | null>(null);
  const [deleting, setDeleting] = useState(false);

  const fetchResults = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.set("limit", String(PAGE_SIZE));
      params.set("offset", String(page * PAGE_SIZE));
      if (verdictFilter !== "ALL") {
        params.set("verdict", verdictFilter);
      }

      const res = await fetch(`${API_URL}/inspections?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        const items: InspectionResult[] = Array.isArray(data) ? data : data.items || [];
        setResults(items);
        setHasMore(items.length === PAGE_SIZE);

        const classes = new Set<string>();
        items.forEach((item) =>
          (item.defects ?? [])
            .filter((d) => d.is_defect)
            .forEach((d) => classes.add(d.clip_label ?? d.defect_class))
        );
        setAllClasses((prev) => {
          const combined = new Set([...prev, ...classes]);
          return Array.from(combined).sort();
        });
      }
    } catch {
      // Backend unreachable
    } finally {
      setLoading(false);
    }
  }, [verdictFilter, page]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  useEffect(() => {
    setPage(0);
  }, [verdictFilter]);

  const filteredResults =
    classFilter === "ALL"
      ? results
      : results.filter((r) =>
          (r.defects ?? []).some(
            (d) => d.is_defect && (d.clip_label ?? d.defect_class) === classFilter
          )
        );

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-zinc-100">Defect Gallery</h2>
          <p className="text-sm text-zinc-500 mt-1">
            Browse and filter inspection results
          </p>
        </div>
        <button
          onClick={async () => {
            if (!confirm("Delete ALL inspections? This cannot be undone.")) return;
            setDeleting(true);
            try {
              const res = await fetch(`${API_URL}/inspections`, { method: "DELETE" });
              if (res.ok) {
                setResults([]);
                setPage(0);
                setHasMore(false);
                setAllClasses([]);
                setSelectedResult(null);
              }
            } catch {
              // ignore
            } finally {
              setDeleting(false);
            }
          }}
          disabled={deleting || (results.length === 0 && !loading)}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors disabled:opacity-30 disabled:pointer-events-none"
        >
          {deleting ? (
            <div className="w-3 h-3 border-2 border-red-400 border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          )}
          Delete All
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500 uppercase">Verdict:</span>
          {(["ALL", "OK", "NG"] as VerdictFilter[]).map((v) => (
            <button
              key={v}
              onClick={() => setVerdictFilter(v)}
              className={
                verdictFilter === v ? "filter-btn-active" : "filter-btn-inactive"
              }
            >
              {v}
            </button>
          ))}
        </div>

        {allClasses.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500 uppercase">Defect:</span>
            <select
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value)}
              className="input-field text-xs py-1"
            >
              <option value="ALL">All Types</option>
              {allClasses.map((cls) => (
                <option key={cls} value={cls}>
                  {cls}
                </option>
              ))}
            </select>
          </div>
        )}

        <span className="text-xs text-zinc-600 ml-auto">
          {filteredResults.length} result{filteredResults.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Grid */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : filteredResults.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredResults.map((result) => (
            <DefectCard
              key={result.id}
              result={result}
              onClick={() => setSelectedResult(result)}
            />
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center h-48 text-zinc-500">
          <p className="text-sm">No results found</p>
          <p className="text-xs mt-1">Try adjusting your filters</p>
        </div>
      )}

      {/* Pagination */}
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={() => setPage((p) => Math.max(0, p - 1))}
          disabled={page === 0}
          className="btn-secondary text-xs disabled:opacity-30"
        >
          Previous
        </button>
        <span className="text-xs text-zinc-500">Page {page + 1}</span>
        <button
          onClick={() => setPage((p) => p + 1)}
          disabled={!hasMore}
          className="btn-secondary text-xs disabled:opacity-30"
        >
          Next
        </button>
      </div>

      {/* Detail Modal */}
      {selectedResult && (
        <InspectionDetailModal
          result={selectedResult}
          onClose={() => setSelectedResult(null)}
        />
      )}
    </div>
  );
}
