"use client";

import { useEffect, useState } from "react";
import StatsChart from "@/components/StatsChart";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface DailyDataPoint {
  date: string;
  pass_count: number;
  fail_count: number;
}

interface DefectDistItem {
  class_name: string;
  count: number;
}

interface Statistics {
  total_inspections: number;
  pass_rate: number;
  total_defects: number;
  avg_processing_time_ms: number;
  defect_distribution?: DefectDistItem[];
}

export default function DashboardPage() {
  const [dailyData, setDailyData] = useState<DailyDataPoint[]>([]);
  const [defectDist, setDefectDist] = useState<DefectDistItem[]>([]);
  const [stats, setStats] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [dailyRes, statsRes] = await Promise.allSettled([
          fetch(`${API_URL}/statistics/daily`),
          fetch(`${API_URL}/statistics`),
        ]);

        if (dailyRes.status === "fulfilled" && dailyRes.value.ok) {
          const data = await dailyRes.value.json();
          setDailyData(Array.isArray(data) ? data : data.daily || []);
        }

        if (statsRes.status === "fulfilled" && statsRes.value.ok) {
          const data: Statistics = await statsRes.value.json();
          setStats(data);
          if (data.defect_distribution) {
            setDefectDist(data.defect_distribution);
          }
        }
      } catch {
        // Backend unreachable
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-zinc-100">Dashboard</h2>
        <p className="text-sm text-zinc-500 mt-1">
          Inspection analytics and defect trends
        </p>
      </div>

      {/* Summary stats row */}
      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {(() => {
            const totalInspections = stats.total_inspections ?? 0;
            const passRate = stats.pass_rate ?? 0;
            const totalDefects = stats.total_defects ?? 0;
            const avgTimeMs = stats.avg_processing_time_ms ?? 0;

            return (
              <>
                <div className="stat-card">
                  <span className="text-xs text-zinc-500 uppercase tracking-wider">Inspections</span>
                  <span className="text-2xl font-bold text-zinc-100">
                    {totalInspections.toLocaleString()}
                  </span>
                </div>
                <div className="stat-card">
                  <span className="text-xs text-zinc-500 uppercase tracking-wider">Pass Rate</span>
                  <span className={`text-2xl font-bold ${passRate >= 95 ? "text-emerald-400" : "text-amber-400"}`}>
                    {passRate.toFixed(1)}%
                  </span>
                </div>
                <div className="stat-card">
                  <span className="text-xs text-zinc-500 uppercase tracking-wider">Total Defects</span>
                  <span className="text-2xl font-bold text-red-400">
                    {totalDefects.toLocaleString()}
                  </span>
                </div>
                <div className="stat-card">
                  <span className="text-xs text-zinc-500 uppercase tracking-wider">Avg Time</span>
                  <span className="text-2xl font-bold text-blue-400">
                    {avgTimeMs.toFixed(0)}ms
                  </span>
                </div>
              </>
            );
          })()}
        </div>
      )}

      {/* Charts */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-zinc-500">Loading analytics...</p>
          </div>
        </div>
      ) : (
        <StatsChart dailyData={dailyData} defectDistribution={defectDist} />
      )}
    </div>
  );
}
