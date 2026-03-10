import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "AI Vision Inspector",
  description: "Industrial AI Vision Engineering - Defect Detection & Quality Control",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="flex min-h-screen">
        <Sidebar />
        <div className="flex-1 flex flex-col min-h-screen md:ml-0">
          {/* Header */}
          <header className="sticky top-0 z-20 bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800 px-6 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3 ml-10 md:ml-0">
                <h1 className="text-lg font-bold text-gradient">AI Vision Inspector</h1>
                <span className="hidden sm:inline-flex text-[10px] bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-full border border-zinc-700">
                  Industrial QC
                </span>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <span className="hidden sm:inline">Backend Connected</span>
                </div>
              </div>
            </div>
          </header>

          {/* Main content */}
          <main className="flex-1 p-4 md:p-6 overflow-y-auto">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
