"use client";

import { useState, useRef, useCallback } from "react";
import { InspectionResult } from "@/hooks/useInspection";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function InspectUpload() {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const pipeline = "ensemble";
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file");
      return;
    }

    setError(null);
    setResult(null);
    setPreviewUrl(URL.createObjectURL(file));
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      formData.append("pipeline", pipeline);
      const response = await fetch(`${API_URL}/inspect`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Inspection failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      // Backend returns "detections" and "inspection_id", normalize to frontend types
      const normalized: InspectionResult = {
        ...data,
        id: data.id ?? data.inspection_id ?? "",
        timestamp: data.timestamp ?? new Date().toISOString(),
        defects: data.defects ?? data.detections ?? [],
      };
      setResult(normalized);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inspection failed");
    } finally {
      setIsLoading(false);
    }
  }, [pipeline]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleClick = () => fileInputRef.current?.click();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const reset = () => {
    setResult(null);
    setError(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const isPassing = result?.verdict === "OK";
  const defects = result?.defects ?? [];

  return (
    <div className="card">
      <h3 className="text-sm font-semibold text-zinc-200 mb-3">Manual Inspection</h3>

      {/* Upload area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={`
          relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
          transition-colors duration-150
          ${isDragging ? "border-blue-500 bg-blue-500/10" : "border-zinc-700 hover:border-zinc-500"}
          ${isLoading ? "pointer-events-none opacity-50" : ""}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleInputChange}
          className="hidden"
        />

        {isLoading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-zinc-400">Processing inspection...</p>
          </div>
        ) : previewUrl && !result ? (
          <div className="flex flex-col items-center gap-2">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-h-32 rounded"
            />
            <p className="text-sm text-zinc-400">Processing...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <svg
              className="w-10 h-10 text-zinc-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
            <p className="text-sm text-zinc-400">
              Drop an image here or <span className="text-blue-400">browse</span>
            </p>
            <p className="text-xs text-zinc-600">PNG, JPG, BMP up to 10MB</p>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className={`mt-3 p-3 rounded-lg border ${
          isPassing
            ? "bg-emerald-500/10 border-emerald-500/30"
            : "bg-red-500/10 border-red-500/30"
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className={isPassing ? "badge-pass" : "badge-fail"}>
              {result.verdict}
            </span>
            <span className="text-xs text-zinc-400">
              {(result.processing_ms ?? 0).toFixed(0)}ms
            </span>
          </div>

          {defects.length > 0 ? (
            <div className="space-y-1 mt-2">
              {defects.map((defect, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className={defect.detection_type === "defect" ? "text-red-400" : "text-emerald-400"}>
                    {defect.defect_class}
                  </span>
                  <span className="text-zinc-500 font-mono">
                    {(defect.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-zinc-400">
              {isPassing ? "No detections" : "Failed inspection"}
            </p>
          )}

          <button
            onClick={(e) => {
              e.stopPropagation();
              reset();
            }}
            className="mt-3 w-full btn-secondary text-xs"
          >
            Inspect Another Image
          </button>
        </div>
      )}
    </div>
  );
}
