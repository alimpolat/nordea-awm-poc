/**
 * DriftBars — horizontal bar chart of allocation drift in percentage points.
 * Severity colouring: |drift| ≤ 2 → olive, 2–4 → amber, > 4 → rust.
 * Zero-centred axis; positive = overweight (right), negative = underweight (left).
 */
import ReactECharts from "echarts-for-react";
import { houseChartBase, axisDefaults, OLIVE, AMBER, RUST, SLATE, GRAY_300, MONO, SERIF } from "./theme";

export interface DriftEntry {
  asset_class: string;
  drift_pp: number;
}

interface Props {
  drift: DriftEntry[];
  height?: number;
}

function driftColor(v: number): string {
  const abs = Math.abs(v);
  if (abs <= 2) return OLIVE;
  if (abs <= 4) return AMBER;
  return RUST;
}

export default function DriftBars({ drift, height = 260 }: Props) {
  // Sort by drift value so the chart reads logically
  const sorted = [...drift].sort((a, b) => a.drift_pp - b.drift_pp);

  const labels = sorted.map((d) => d.asset_class);
  const values = sorted.map((d) => d.drift_pp);
  const colors = values.map(driftColor);

  const option = {
    ...houseChartBase,
    grid: { left: 140, right: 40, top: 16, bottom: 28 },
    xAxis: {
      type: "value" as const,
      name: "pp",
      nameTextStyle: { fontFamily: MONO, fontSize: 10, color: "#87867F" },
      ...axisDefaults,
      splitLine: {
        lineStyle: { color: GRAY_300, type: "dashed" as const },
      },
      axisLabel: {
        ...axisDefaults.axisLabel,
        formatter: (v: number) => (v > 0 ? `+${v}` : `${v}`),
      },
    },
    yAxis: {
      type: "category" as const,
      data: labels,
      axisLabel: {
        color: SLATE,
        fontFamily: MONO,
        fontSize: 11,
        width: 130,
        overflow: "truncate" as const,
      },
      axisLine: { lineStyle: { color: GRAY_300 } },
      splitLine: { show: false },
    },
    series: [
      {
        type: "bar" as const,
        data: values.map((v, i) => ({
          value: v,
          itemStyle: { color: colors[i], borderRadius: [0, 3, 3, 0] },
          label: {
            show: true,
            position: v >= 0 ? ("right" as const) : ("left" as const),
            formatter: (p: { value: number }) =>
              `${p.value > 0 ? "+" : ""}${p.value.toFixed(1)}pp`,
            fontFamily: MONO,
            fontSize: 10,
            color: SLATE,
          },
        })),
        barMaxWidth: 22,
      },
    ],
    tooltip: {
      ...houseChartBase.tooltip,
      formatter: (params: { name: string; value: number }) =>
        `<span style="font-family:${SERIF};font-size:13px"><b>${params.name}</b><br/>Drift: <b>${params.value > 0 ? "+" : ""}${params.value.toFixed(1)} pp</b></span>`,
    },
  };

  return (
    <ReactECharts
      option={option}
      style={{ width: "100%", height }}
      opts={{ renderer: "svg" }}
    />
  );
}
