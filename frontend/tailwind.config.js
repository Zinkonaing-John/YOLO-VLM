/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        industrial: {
          50: "#f8fafc",
          100: "#f1f5f9",
          200: "#e2e8f0",
          300: "#cbd5e1",
          400: "#94a3b8",
          500: "#64748b",
          600: "#475569",
          700: "#334155",
          800: "#1e293b",
          900: "#0f172a",
          950: "#020617",
        },
        pass: {
          DEFAULT: "#10b981",
          light: "#34d399",
          dark: "#059669",
        },
        fail: {
          DEFAULT: "#ef4444",
          light: "#f87171",
          dark: "#dc2626",
        },
        warn: {
          DEFAULT: "#f59e0b",
          light: "#fbbf24",
          dark: "#d97706",
        },
      },
      animation: {
        "flash-green": "flashGreen 0.6s ease-out",
        "flash-red": "flashRed 0.6s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        flashGreen: {
          "0%": { backgroundColor: "rgba(16, 185, 129, 0.3)" },
          "100%": { backgroundColor: "transparent" },
        },
        flashRed: {
          "0%": { backgroundColor: "rgba(239, 68, 68, 0.3)" },
          "100%": { backgroundColor: "transparent" },
        },
      },
    },
  },
  plugins: [],
};
