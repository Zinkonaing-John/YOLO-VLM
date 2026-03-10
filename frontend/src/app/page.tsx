"use client";

import { useEffect, useState } from "react";
import LiveFeed from "@/components/LiveFeed";
import InspectUpload from "@/components/InspectUpload";
import CameraFeed from "@/components/CameraFeed";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Statistics {
  total_inspections: number;
  pass_rate: number;
  total_defects: number;
  avg_processing_time_ms: number;
}

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color: string;
}) {
  return (
    <div className="stat-card">
      <span className="text-xs text-zinc-500 uppercase tracking-wider">{label}</span>
      <span className={`text-2xl font-bold ${color}`}>{value}</span>
      {sub && <span className="text-[11px] text-zinc-500">{sub}</span>}
    </div>
  );
}

export default function HomePage() {
  const [stats, setStats] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStats() {
      try {
        const res = await fetch(`${API_URL}/statistics`);
        if (res.ok) {
          const data = await res.json();
          setStats(data);
        }
      } catch {
        // Backend not reachable; show placeholder
      } finally {
        setLoading(false);
      }
    }
    fetchStats();
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const totalInspections = stats?.total_inspections ?? 0;
  const passRate = stats?.pass_rate ?? 0;
  const totalDefects = stats?.total_defects ?? 0;
  const avgTime = stats?.avg_processing_time_ms ?? 0;

  return (
    <div className="space-y-6">
      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Inspections"
          value={loading ? "--" : totalInspections.toLocaleString()}
          sub="all time"
          color="text-zinc-100"
        />
        <StatCard
          label="Pass Rate"
          value={loading ? "--" : `${passRate.toFixed(1)}%`}
          sub={passRate >= 95 ? "within target" : "below 95% target"}
          color={passRate >= 95 ? "text-emerald-400" : "text-amber-400"}
        />
        <StatCard
          label="Total Defects"
          value={loading ? "--" : totalDefects.toLocaleString()}
          sub="detected"
          color="text-red-400"
        />
        <StatCard
          label="Avg Processing"
          value={loading ? "--" : `${avgTime.toFixed(0)}ms`}
          sub="per inspection"
          color="text-blue-400"
        />
      </div>

      {/* Camera feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <CameraFeed />
        </div>
        <div className="space-y-6">
          <InspectUpload />
          <LiveFeed />
        </div>
      </div>
    </div>
  );
}
