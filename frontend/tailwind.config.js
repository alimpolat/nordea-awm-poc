/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ivory: "#FAF9F5",
        paper: "#FFFFFF",
        slate: "#141413",
        clay: "#D97757",
        oat: "#E3DACC",
        olive: "#788C5D",
        rust: "#B04A3F",
        amber: "#C7A35F",
        "amber-dark": "#8C7038",
        "gray-100": "#F7F4EC",
        "gray-150": "#F0EEE6",
        "gray-300": "#D1CFC5",
        "gray-500": "#87867F",
        "gray-700": "#3D3D3A",
      },
      fontFamily: {
        serif: ["Newsreader", "Georgia", "serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
