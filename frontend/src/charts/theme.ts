/**
 * House ECharts theme — spreads into every chart option.
 * Palette mirrors Tailwind config exactly.
 */
import type { EChartsOption } from "echarts";

// ── Palette consts ────────────────────────────────────────────────────────────
export const CLAY = "#D97757";
export const OLIVE = "#788C5D";
export const AMBER = "#C7A35F";
export const RUST = "#B04A3F";
export const SLATE = "#141413";
export const IVORY = "#FAF9F5";
export const PAPER = "#FFFFFF";
export const OAT = "#E3DACC";
export const AMBER_DARK = "#8C7038";
export const GRAY_100 = "#F7F4EC";
export const GRAY_150 = "#F0EEE6";
export const GRAY_300 = "#D1CFC5";
export const GRAY_500 = "#87867F";
export const GRAY_700 = "#3D3D3A";

export const MONO = "'JetBrains Mono', ui-monospace, monospace";
export const SERIF = "Newsreader, Georgia, serif";

/** Six-tone series colour ramp */
export const SERIES_PALETTE = [CLAY, OLIVE, AMBER, RUST, SLATE, GRAY_500];

// ── Shared axis defaults (cartesian charts spread these in) ───────────────────
export const axisDefaults = {
  axisLabel: {
    color: GRAY_500,
    fontFamily: MONO,
    fontSize: 11,
  },
  axisLine: {
    lineStyle: { color: GRAY_300 },
  },
  splitLine: {
    lineStyle: { color: GRAY_300, type: "dashed" as const },
  },
};

// ── Base option every chart spreads ─────────────────────────────────────────
export const houseChartBase: EChartsOption = {
  color: SERIES_PALETTE,
  textStyle: {
    fontFamily: SERIF,
    color: SLATE,
  },
  animation: true,
  animationEasing: "cubicOut",
  animationDuration: 900,
  tooltip: {
    backgroundColor: "rgba(255,255,255,0.96)",
    borderColor: GRAY_300,
    borderWidth: 1,
    textStyle: {
      color: SLATE,
      fontFamily: SERIF,
      fontSize: 14,
    },
    extraCssText:
      "box-shadow:0 4px 12px rgba(0,0,0,0.08);border-radius:8px;",
  },
};
