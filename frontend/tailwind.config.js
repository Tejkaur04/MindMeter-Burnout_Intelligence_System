/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: "#0F111A",
          card: "rgba(22, 26, 41, 0.65)",
          border: "rgba(255, 255, 255, 0.06)",
          indigo: "#6366F1",
          blue: "#3B82F6",
          teal: "#14B8A6",
          lavender: "#A78BFA",
          emerald: "#10B981",
          amber: "#F59E0B",
          coral: "#F43F5E"
        }
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"]
      },
      backdropBlur: {
        xs: "2px"
      }
    },
  },
  plugins: [],
}