/**
 * NbaProjection — 12-month projected portfolio value.
 * withAction = olive filled area; withoutAction = gray dashed line.
 * Animated fill-in on mount.
 */
import ReactECharts from "echarts-for-react";
import { houseChartBase, axisDefaults, OLIVE, GRAY_500, SLATE, MONO, SERIF } from "./theme";

interface Props {
  months?: number;
  withAction: number[];
  withoutAction: number[];
  height?: number;
}

export default function NbaProjection({
  months = 12,
  withAction,
  withoutAction,
  height = 280,
}: Props) {
  const xData = Array.from({ length: months }, (_, i) => {
    const d = new Date();
    d.setMonth(d.getMonth() + i + 1);
    return d.toLocaleString("en-SE", { month: "short", year: "2-digit" });
  });

  const fmt = (v: number) =>
    v >= 1_000_000
      ? `${(v / 1_000_000).toFixed(0)}M`
      : v >= 1_000
      ? `${(v / 1_000).toFixed(0)}k`
      : `${v}`;

  const option = {
    ...houseChartBase,
    grid: { left: 56, right: 20, top: 24, bottom: 36 },
    xAxis: {
      type: "category" as const,
      data: xData,
      boundaryGap: false,
      ...axisDefaults,
    },
    yAxis: {
      type: "value" as const,
      axisLabel: {
        ...axisDefaults.axisLabel,
        formatter: fmt,
      },
      axisLine: axisDefaults.axisLine,
      splitLine: axisDefaults.splitLine,
    },
    legend: {
      top: 0,
      right: 8,
      textStyle: { fontFamily: MONO, fontSize: 11, color: SLATE },
      itemWidth: 16,
      itemHeight: 3,
    },
    series: [
      {
        name: "With action",
        type: "line" as const,
        data: withAction,
        smooth: true,
        symbol: "none",
        lineStyle: { color: OLIVE, width: 2.5 },
        areaStyle: {
          color: {
            type: "linear" as const,
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: `${OLIVE}50` },
              { offset: 1, color: `${OLIVE}08` },
            ],
          },
        },
        itemStyle: { color: OLIVE },
      },
      {
        name: "Without action",
        type: "line" as const,
        data: withoutAction,
        smooth: true,
        symbol: "none",
        lineStyle: { color: GRAY_500, width: 1.5, type: "dashed" as const },
        itemStyle: { color: GRAY_500 },
      },
    ],
    tooltip: {
      ...houseChartBase.tooltip,
      trigger: "axis" as const,
      formatter: (params: { seriesName: string; value: number; axisValue: string }[]) => {
        const rows = params
          .map(
            (p) =>
              `<tr><td style="padding-right:12px;font-family:${MONO};font-size:11px;color:${GRAY_500}">${p.seriesName}</td><td style="font-weight:600">SEK ${fmt(p.value)}</td></tr>`
          )
          .join("");
        return `<span style="font-family:${SERIF};font-size:12px;color:${GRAY_500}">${params[0]?.axisValue}</span><table style="margin-top:4px">${rows}</table>`;
      },
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
