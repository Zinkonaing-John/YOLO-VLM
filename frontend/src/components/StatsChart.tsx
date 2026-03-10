"use client";

interface DailyDataPoint {
  date: string;
  pass_count: number;
  fail_count: number;
}

interface DefectDistItem {
  defect_class: string;
  count: number;
}

interface StatsChartProps {
  dailyData: DailyDataPoint[];
  defectDistribution: DefectDistItem[];
}

const DONUT_COLORS = [
  "#3b82f6", // blue
  "#ef4444", // red
  "#f59e0b", // amber
  "#10b981", // emerald
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#06b6d4", // cyan
  "#f97316", // orange
];

function BarChart({ data }: { data: DailyDataPoint[] }) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-zinc-500 text-sm">
        No data available
      </div>
    );
  }

  const maxVal = Math.max(...data.map((d) => d.pass_count + d.fail_count), 1);

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-zinc-300">Daily Pass / Fail</h4>
      <div className="flex items-end gap-2 h-48">
        {data.map((point, i) => {
          const total = point.pass_count + point.fail_count;
          const passH = (point.pass_count / maxVal) * 100;
          const failH = (point.fail_count / maxVal) * 100;
          const dayLabel = new Date(point.date).toLocaleDateString("en-US", {
            weekday: "short",
          });

          return (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <span className="text-[10px] text-zinc-500 font-mono">{total}</span>
              <div className="w-full flex flex-col justify-end h-40 gap-0.5">
                {point.fail_count > 0 && (
                  <div
                    className="w-full bg-red-500/80 rounded-t-sm transition-all duration-500"
                    style={{ height: `${failH}%` }}
                    title={`Fail: ${point.fail_count}`}
                  />
                )}
                {point.pass_count > 0 && (
                  <div
                    className="w-full bg-emerald-500/80 rounded-t-sm transition-all duration-500"
                    style={{ height: `${passH}%` }}
                    title={`Pass: ${point.pass_count}`}
                  />
                )}
              </div>
              <span className="text-[10px] text-zinc-500">{dayLabel}</span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center justify-center gap-4 text-xs text-zinc-400">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-emerald-500/80" />
          <span>Pass</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-red-500/80" />
          <span>Fail</span>
        </div>
      </div>
    </div>
  );
}

function DonutChart({ data }: { data: DefectDistItem[] }) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-zinc-500 text-sm">
        No defect data
      </div>
    );
  }

  const total = data.reduce((sum, d) => sum + d.count, 0);
  let cumulativePercent = 0;

  const segments = data.map((item, i) => {
    const percent = (item.count / total) * 100;
    const startPercent = cumulativePercent;
    cumulativePercent += percent;
    return {
      ...item,
      percent,
      startPercent,
      color: DONUT_COLORS[i % DONUT_COLORS.length],
    };
  });

  // Build conic gradient
  const gradientStops = segments
    .map((seg) => `${seg.color} ${seg.startPercent}% ${seg.startPercent + seg.percent}%`)
    .join(", ");

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-zinc-300">Defect Distribution</h4>
      <div className="flex items-center gap-6">
        {/* Donut */}
        <div className="relative w-36 h-36 flex-shrink-0">
          <div
            className="w-full h-full rounded-full"
            style={{
              background: `conic-gradient(${gradientStops})`,
            }}
          />
          {/* Inner circle for donut effect */}
          <div className="absolute inset-0 m-auto w-20 h-20 rounded-full bg-zinc-900 flex items-center justify-center">
            <div className="text-center">
              <span className="text-lg font-bold text-zinc-100">{total}</span>
              <p className="text-[9px] text-zinc-500">Total</p>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex-1 space-y-1.5 max-h-36 overflow-y-auto">
          {segments.map((seg, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <div
                className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                style={{ backgroundColor: seg.color }}
              />
              <span className="text-zinc-300 truncate flex-1">{seg.defect_class}</span>
              <span className="text-zinc-500 font-mono">
                {seg.count} ({seg.percent.toFixed(0)}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function StatsChart({ dailyData, defectDistribution }: StatsChartProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="card">
        <BarChart data={dailyData} />
      </div>
      <div className="card">
        <DonutChart data={defectDistribution} />
      </div>
    </div>
  );
}
