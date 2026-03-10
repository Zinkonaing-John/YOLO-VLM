"use client";

import { useEffect, useState, useCallback } from "react";
import { InspectionResult } from "@/hooks/useInspection";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ReportsPage() {
  const [results, setResults] = useState<InspectionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedResult, setSelectedResult] = useState<InspectionResult | null>(null);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const pageSize = 25;

  const fetchResults = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.set("limit", String(pageSize));
      params.set("offset", String(page * pageSize));

      const res = await fetch(`${API_URL}/inspections?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        const items: InspectionResult[] = Array.isArray(data) ? data : data.items || [];
        setResults(items);
        setHasMore(items.length === pageSize);
      }
    } catch {
      // Backend unreachable
    } finally {
      setLoading(false);
    }
  }, [page]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-zinc-100">Inspection Reports</h2>
        <p className="text-sm text-zinc-500 mt-1">
          Complete inspection history and detail records
        </p>
      </div>

      {/* Table */}
      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 bg-zinc-900/80">
                <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                  ID
                </th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                  Verdict
                </th>
                <th className="text-right px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                  Defects
                </th>
                <th className="text-right px-4 py-3 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                  Time (ms)
                </th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={5} className="text-center py-12">
                    <div className="inline-flex items-center gap-2">
                      <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                      <span className="text-zinc-500">Loading reports...</span>
                    </div>
                  </td>
                </tr>
              ) : results.length > 0 ? (
                results.map((result, index) => {
                  const isPassing = result.verdict === "PASS";
                  const defects = result.defects ?? [];
                  const processingTimeMs = result.processing_time_ms ?? 0;
                  const key = result.id ?? `${result.timestamp}-${index}`;
                  return (
                    <tr
                      key={key}
                      onClick={() => setSelectedResult(result)}
                      className="table-row"
                    >
                      <td className="px-4 py-3 font-mono text-xs text-zinc-400">
                        {(result.id ?? "").slice(0, 12) || "—"}
                      </td>
                      <td className="px-4 py-3 text-xs text-zinc-300">
                        {new Date(result.timestamp).toLocaleString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })}
                      </td>
                      <td className="px-4 py-3">
                        <span className={isPassing ? "badge-pass" : "badge-fail"}>
                          {result.verdict}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-xs text-zinc-300 font-mono">
                        {defects.length}
                      </td>
                      <td className="px-4 py-3 text-right text-xs text-zinc-400 font-mono">
                        {processingTimeMs.toFixed(0)}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={5} className="text-center py-12 text-zinc-500 text-sm">
                    No inspection records found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

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
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setSelectedResult(null)}
        >
          <div
            className="bg-zinc-900 border border-zinc-800 rounded-xl max-w-2xl w-full p-6 max-h-[85vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold text-zinc-200">
                  Inspection Report
                </h3>
                <p className="text-xs text-zinc-500 font-mono mt-0.5">
                  {selectedResult.id}
                </p>
              </div>
              <button
                onClick={() => setSelectedResult(null)}
                className="text-zinc-500 hover:text-zinc-200 text-lg leading-none"
              >
                x
              </button>
            </div>

            {/* Summary */}
            {(() => {
              const defects = selectedResult.defects ?? [];
              const processingTimeMs = selectedResult.processing_time_ms ?? 0;
              return (
                <>
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="bg-zinc-800 rounded-lg p-3">
                <span className="text-[10px] text-zinc-500 uppercase">Verdict</span>
                <div className="mt-1">
                  <span
                    className={
                      selectedResult.verdict === "PASS" ? "badge-pass" : "badge-fail"
                    }
                  >
                    {selectedResult.verdict}
                  </span>
                </div>
              </div>
              <div className="bg-zinc-800 rounded-lg p-3">
                <span className="text-[10px] text-zinc-500 uppercase">Defects Found</span>
                <p className="text-lg font-bold text-zinc-100 mt-1">
                  {defects.length}
                </p>
              </div>
              <div className="bg-zinc-800 rounded-lg p-3">
                <span className="text-[10px] text-zinc-500 uppercase">Processing</span>
                <p className="text-lg font-bold text-zinc-100 mt-1">
                  {processingTimeMs.toFixed(0)}ms
                </p>
              </div>
            </div>
                </>
              );
            })()}

            {/* Timestamp */}
            <div className="text-xs text-zinc-500 mb-4">
              Inspected at{" "}
              {new Date(selectedResult.timestamp).toLocaleString("en-US", {
                weekday: "long",
                year: "numeric",
                month: "long",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
            </div>

            {/* Defects */}
            {(selectedResult.defects ?? []).length > 0 ? (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-zinc-400 uppercase">
                  Detected Defects
                </h4>
                {(selectedResult.defects ?? []).map((defect, i) => (
                  <div
                    key={i}
                    className="p-3 bg-zinc-800 rounded-lg border border-zinc-700"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-zinc-200 font-medium">
                        {defect.class_name}
                      </span>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-zinc-400 font-mono">
                          conf: {(defect.confidence * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs text-zinc-500 font-mono">
                          bbox: [{[defect.bbox_x1, defect.bbox_y1, defect.bbox_x2, defect.bbox_y2].map((v) => (v ?? 0).toFixed(3)).join(", ")}]
                        </span>
                      </div>
                    </div>
                    {defect.vlm_description && (
                      <div className="mt-2 pt-2 border-t border-zinc-700">
                        <span className="text-[10px] text-zinc-500 uppercase">
                          VLM Analysis
                        </span>
                        <p className="text-xs text-zinc-300 mt-0.5">
                          {defect.vlm_description}
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-zinc-500">
                <p className="text-sm">
                  {selectedResult.vlm_response
                    ?? (selectedResult.verdict === "PASS" ? "No defects detected" : "Failed inspection")}
                </p>
                {!selectedResult.vlm_response && selectedResult.verdict === "PASS" && (
                  <p className="text-xs mt-1">This part passed quality inspection</p>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
