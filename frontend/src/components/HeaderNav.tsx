"use client";

import { useState } from "react";
import Modal from "./Modal";
import DashboardPage from "@/app/dashboard/page";
import GalleryPage from "@/app/gallery/page";
import ReportsPage from "@/app/reports/page";

interface NavPopup {
  key: string;
  label: string;
  icon: string;
  component: React.ComponentType;
}

const navPopups: NavPopup[] = [
  { key: "dashboard", label: "Dashboard", icon: "≡", component: DashboardPage },
  { key: "gallery", label: "Gallery", icon: "▦", component: GalleryPage },
  { key: "reports", label: "Reports", icon: "☷", component: ReportsPage },
];

export default function HeaderNav() {
  const [activePopup, setActivePopup] = useState<string | null>(null);

  const togglePopup = (key: string) => {
    setActivePopup((prev) => (prev === key ? null : key));
  };

  return (
    <>
      <header className="sticky top-0 z-20 bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800 px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Logo + Title */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold text-sm">
              AI
            </div>
            <h1 className="text-lg font-bold text-gradient">AI Vision Inspector</h1>
            <span className="hidden sm:inline-flex text-[10px] bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-full border border-zinc-700">
              Industrial QC
            </span>
          </div>

          {/* Nav Icons + Status */}
          <div className="flex items-center gap-2">
            {navPopups.map((item) => {
              const isActive = activePopup === item.key;
              return (
                <button
                  key={item.key}
                  onClick={() => togglePopup(item.key)}
                  title={item.label}
                  className={`
                    w-9 h-9 flex items-center justify-center rounded-lg text-lg
                    transition-all duration-150 cursor-pointer
                    ${
                      isActive
                        ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                        : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 border border-transparent"
                    }
                  `}
                >
                  {item.icon}
                </button>
              );
            })}

            <div className="w-px h-5 bg-zinc-800 mx-1" />

            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span className="hidden sm:inline">Backend Connected</span>
            </div>
          </div>
        </div>
      </header>

      {/* Modals */}
      {navPopups.map((item) => (
        <Modal
          key={item.key}
          isOpen={activePopup === item.key}
          onClose={() => setActivePopup(null)}
          title={item.label}
        >
          <item.component />
        </Modal>
      ))}
    </>
  );
}
