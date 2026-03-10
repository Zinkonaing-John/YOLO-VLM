"use client";

import { useEffect, useState, useCallback } from "react";
import DefectCard from "@/components/DefectCard";
import { InspectionResult } from "@/hooks/useInspection";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const PAGE_SIZE = 20;

type VerdictFilter = "ALL" | "PASS" | "FAIL";

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

        // Collect unique defect classes
        const classes = new Set<string>();
        items.forEach((item) =>
          (item.defects ?? []).forEach((d) => classes.add(d.class_name))
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

  // Apply client-side class filter
  const filteredResults =
    classFilter === "ALL"
      ? results
      : results.filter((r) =>
          (r.defects ?? []).some((d) => d.class_name === classFilter)
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
        {/* Verdict filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500 uppercase">Verdict:</span>
          {(["ALL", "PASS", "FAIL"] as VerdictFilter[]).map((v) => (
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

        {/* Class filter */}
        {allClasses.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500 uppercase">Class:</span>
            <select
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value)}
              className="input-field text-xs py-1"
            >
              <option value="ALL">All Classes</option>
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
        (() => {
          const defects = selectedResult.defects ?? [];
          const processingTimeMs = selectedResult.processing_time_ms ?? 0;
          const shortId = (selectedResult.id ?? "").slice(0, 12) || "—";
          return (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setSelectedResult(null)}
        >
          <div
            className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-lg w-full p-6 max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-zinc-200">
                Inspection #{shortId}
              </h3>
              <button
                onClick={() => setSelectedResult(null)}
                className="text-zinc-500 hover:text-zinc-200 text-lg"
              >
                x
              </button>
            </div>

            {/* Image */}
            <div className="relative w-full aspect-video bg-zinc-800 rounded-lg mb-4 overflow-hidden">
              {selectedResult.image_url ? (
                <img
                  src={selectedResult.image_url}
                  alt="Inspection"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-zinc-600 text-sm">
                  No image available
                </div>
              )}
              {selectedResult.defects.map((defect, i) => {
                const x1 = defect.bbox_x1 ?? 0;
                const y1 = defect.bbox_y1 ?? 0;
                const x2 = defect.bbox_x2 ?? 0;
                const y2 = defect.bbox_y2 ?? 0;
                if (!x2 && !y2) return null;
                return (
                  <div
                    key={i}
                    className="absolute border-2 border-red-500 rounded-sm"
                    style={{
                      left: `${x1 * 100}%`,
                      top: `${y1 * 100}%`,
                      width: `${(x2 - x1) * 100}%`,
                      height: `${(y2 - y1) * 100}%`,
                    }}
                  />
                );
              })}
            </div>

            {/* Verdict */}
            <div className="flex items-center justify-between mb-3">
              <span
                className={
                  selectedResult.verdict === "PASS" ? "badge-pass" : "badge-fail"
                }
              >
                {selectedResult.verdict}
              </span>
              <span className="text-xs text-zinc-500">
                {processingTimeMs.toFixed(0)}ms |{" "}
                {new Date(selectedResult.timestamp).toLocaleString()}
              </span>
            </div>

            {/* Defects */}
            {defects.length > 0 ? (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-zinc-400 uppercase">
                  Defects ({selectedResult.defects.length})
                </h4>
                {defects.map((defect, i) => (
                  <div
                    key={i}
                    className="p-2 bg-zinc-800 rounded-lg border border-zinc-700"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-zinc-200 font-medium">
                        {defect.class_name}
                      </span>
                      <span className="text-xs text-zinc-400 font-mono">
                        {(defect.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    {defect.vlm_description && (
                      <p className="text-xs text-zinc-400 mt-1">
                        {defect.vlm_description}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-zinc-500">
                {selectedResult.vlm_response
                  ?? (selectedResult.verdict === "PASS" ? "No defects detected" : "Failed inspection")}
              </p>
            )}
          </div>
        </div>
          );
        })()
      )}
    </div>
  );
}
