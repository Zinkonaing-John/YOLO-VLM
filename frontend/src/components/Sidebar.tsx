"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

interface NavItem {
  label: string;
  href: string;
  icon: string;
}

const navItems: NavItem[] = [
  { label: "Overview", href: "/", icon: "\u2302" },
  { label: "Dashboard", href: "/dashboard", icon: "\u2261" },
  { label: "Gallery", href: "/gallery", icon: "\u25A6" },
  { label: "Reports", href: "/reports", icon: "\u2637" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <>
      {/* Mobile toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="fixed top-3 left-3 z-50 md:hidden bg-zinc-800 border border-zinc-700
                   rounded-lg p-2 text-zinc-400 hover:text-zinc-100 transition-colors"
        aria-label="Toggle navigation"
      >
        {collapsed ? "\u2715" : "\u2630"}
      </button>

      {/* Overlay for mobile */}
      {collapsed && (
        <div
          className="fixed inset-0 bg-black/60 z-30 md:hidden"
          onClick={() => setCollapsed(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed top-0 left-0 z-40 h-full w-56 bg-zinc-900 border-r border-zinc-800
          flex flex-col transition-transform duration-200 ease-in-out
          md:translate-x-0 md:static md:z-auto
          ${collapsed ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
        `}
      >
        {/* Logo area */}
        <div className="px-4 py-5 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold text-sm">
              AI
            </div>
            <div>
              <h1 className="text-sm font-bold text-zinc-100 leading-tight">Vision</h1>
              <p className="text-[10px] text-zinc-500 leading-tight">Inspector v1.0</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setCollapsed(false)}
                className={`
                  flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                  transition-all duration-150
                  ${
                    isActive
                      ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                      : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800"
                  }
                `}
              >
                <span className="text-lg w-5 text-center">{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-zinc-800">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-slow" />
            <span className="text-xs text-zinc-500">System Online</span>
          </div>
        </div>
      </aside>
    </>
  );
}
