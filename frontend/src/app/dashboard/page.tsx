"use client";

import { useEffect, useState } from "react";
import StatsChart from "@/components/StatsChart";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface DailyDataPoint {
  date: string;
  ok_count: number;
  ng_count: number;
}

interface DefectDistItem {
  defect_class: string;
  count: number;
}

interface Statistics {
  total_inspections: number;
  ok_rate: number;
  ng_rate: number;
  total_defects: number;
  avg_processing_ms: number;
  defect_type_distribution?: DefectDistItem[];
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
          setDailyData(Array.isArray(data) ? data : data.items || []);
        }

        if (statsRes.status === "fulfilled" && statsRes.value.ok) {
          const data: Statistics = await statsRes.value.json();
          setStats(data);
          if (data.defect_type_distribution) {
            setDefectDist(data.defect_type_distribution);
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
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
          <div className="stat-card">
            <span className="text-xs text-zinc-500 uppercase tracking-wider">Inspections</span>
            <span className="text-2xl font-bold text-zinc-100">
              {stats.total_inspections.toLocaleString()}
            </span>
          </div>
          <div className="stat-card">
            <span className="text-xs text-zinc-500 uppercase tracking-wider">OK Rate</span>
            <span className={`text-2xl font-bold ${stats.ok_rate >= 95 ? "text-emerald-400" : "text-amber-400"}`}>
              {stats.ok_rate.toFixed(1)}%
            </span>
          </div>
          <div className="stat-card">
            <span className="text-xs text-zinc-500 uppercase tracking-wider">NG Rate</span>
            <span className={`text-2xl font-bold ${stats.ng_rate <= 5 ? "text-emerald-400" : "text-red-400"}`}>
              {stats.ng_rate.toFixed(1)}%
            </span>
          </div>
          <div className="stat-card">
            <span className="text-xs text-zinc-500 uppercase tracking-wider">Total Defects</span>
            <span className="text-2xl font-bold text-red-400">
              {stats.total_defects.toLocaleString()}
            </span>
          </div>
          <div className="stat-card">
            <span className="text-xs text-zinc-500 uppercase tracking-wider">Avg Time</span>
            <span className="text-2xl font-bold text-blue-400">
              {(stats.avg_processing_ms ?? 0).toFixed(0)}ms
            </span>
          </div>
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
