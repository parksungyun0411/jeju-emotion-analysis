import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        jeju: {
          DEFAULT: "#0d9488",
          dark: "#0f766e",
          light: "#ccfbf1",
        },
      },
    },
  },
  plugins: [],
};

export default config;
