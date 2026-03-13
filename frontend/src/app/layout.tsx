import type { Metadata } from "next";
import "./globals.css";
import HeaderNav from "@/components/HeaderNav";

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
      <body className="min-h-screen flex flex-col">
        <HeaderNav />
        <main className="flex-1 p-4 md:p-6 overflow-y-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
